"""CPU-only safety/coverage fault injection; never imports GPU runtimes."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from length_audit import common, coordinator, budget_watchdog


def record(attempt='a', start=0, status='running'):
    value = {'attempt_id': attempt, 'status': status, 'started_unix': start,
             'n_gpus': 4, 'plan_sha256': 'frozen'}
    if status != 'running':
        value.update(finished_unix=start + 900, wall_seconds=900, allocated_gpu_hours=1.0)
    return value


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'evaluations').mkdir()

    def put(self, name, value):
        path = self.root / 'evaluations' / name / 'launcher_record.json'
        common.atomic(path, value)
        return path

    def test_read_retries_transient_missing_and_bad_json(self):
        with patch.object(Path, 'read_text', side_effect=[FileNotFoundError(), '{', '{"ok":1}']), patch.object(common.time, 'sleep'):
            self.assertEqual(common.read_json(Path('unused')), {'ok': 1})

    def test_read_permanent_failure_is_not_zero_cost(self):
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError()), patch.object(common.time, 'sleep'):
            with self.assertRaises(FileNotFoundError):
                common.read_json(Path('unused'))

    def test_archived_failures_and_running_are_both_charged(self):
        self.put('old.failed-a', record('a', status='failed'))
        self.put('current', record('b', start=900))
        self.assertEqual(common.CostLedger(self.root).snapshot(now=1800), (2.0, 2))

    def test_disappearing_known_cost_fails_closed(self):
        path = self.put('current', record())
        ledger = common.CostLedger(self.root)
        ledger.snapshot(now=900)
        path.unlink()
        with self.assertRaises(OSError):
            ledger.snapshot(now=900)

    def test_cost_archive_rename_preserves_identity(self):
        path = self.put('current', record(status='failed'))
        ledger = common.CostLedger(self.root)
        self.assertEqual(ledger.snapshot(now=1000)[0], 1.0)
        path.parent.rename(path.parent.with_name('current.failed-a'))
        self.assertEqual(ledger.snapshot(now=1000)[0], 1.0)

    def test_duplicate_attempt_rejected(self):
        self.put('first', record())
        self.put('second', record())
        with self.assertRaises(ValueError):
            common.CostLedger(self.root).snapshot(now=900)

    def test_current_plan_has_no_budget_or_phase_timeout(self):
        self.assertIsNone(common.CAP)
        self.assertIsNone(common.THRESHOLD)
        for charged in (0.0, 6.0, 100.0):
            self.assertIsNone(common.phase_timeout(charged))

    def test_running_cannot_be_retried_or_overwritten(self):
        path = self.put('plain-opd', record())
        with self.assertRaises(RuntimeError):
            coordinator.existing(path.parent, True, 'frozen')
        self.assertEqual(common.read_json(path)['status'], 'running')

    def test_failed_retry_archives_evidence(self):
        path = self.put('plain-opd', record(status='failed'))
        with self.assertRaises(RuntimeError):
            coordinator.existing(path.parent, False, 'frozen')
        self.assertFalse(coordinator.existing(path.parent, True, 'frozen'))
        archived = path.parent.with_name('plain-opd.failed-a') / path.name
        self.assertEqual(common.read_json(archived)['allocated_gpu_hours'], 1.0)

    def test_pid_reuse_never_signals(self):
        with patch.object(common, 'process_identity', return_value={'pid': 42, 'start_ticks': 200}), patch.object(common.os, 'kill') as kill:
            self.assertFalse(common.stop_verified_owner({'pid': 42, 'start_ticks': 100}, '/task.py'))
            kill.assert_not_called()

    def test_failed_spawn_settles_original_record(self):
        ledger = common.CostLedger(self.root)
        with patch.object(coordinator, 'free_gpus'), patch.object(coordinator, 'ensure_watchdog'), patch.object(coordinator.subprocess, 'Popen', side_effect=OSError('spawn failed')):
            with self.assertRaises(OSError):
                coordinator.evaluate(self.root, {'name': 'plain-opd', 'model': '/fixed/model'},
                    'frozen', [], {'pid': 42}, ledger, False, lambda stage: None)
        saved = common.read_json(self.root / 'evaluations/plain-opd/launcher_record.json')
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['pids'], [])
        self.assertFalse(saved['cap_enabled'])
        self.assertIsNone(saved['gpu_hour_cap'])
        self.assertIsNone(saved['max_wall_seconds'])
        self.assertGreaterEqual(saved['allocated_gpu_hours'], 0)
        self.assertEqual(ledger.snapshot()[1], 1)

    def test_worker_failure_preserves_pid_evidence_and_cost(self):
        launched = []
        folder = self.root / 'evaluations/plain-opd'

        def spawn(*args, **kwargs):
            before = common.read_json(folder / 'launcher_record.json')
            self.assertEqual(before['status'], 'running')
            self.assertEqual(len(before['pids']), len(launched))
            pid = 100 + len(launched)
            child = SimpleNamespace(pid=pid, returncode=1, poll=lambda: 1)
            launched.append(child)
            return child

        def identity(pid):
            return {'pid': pid, 'start_ticks': pid + 1000, 'session': pid, 'pgrp': pid, 'argv': ['worker']}

        with patch.object(coordinator, 'free_gpus'), patch.object(coordinator, 'ensure_watchdog'), patch.object(coordinator.subprocess, 'Popen', side_effect=spawn), patch.object(coordinator, 'process_identity', side_effect=identity), patch.object(coordinator, 'cleanup') as cleanup:
            with self.assertRaises(RuntimeError):
                coordinator.evaluate(self.root, {'name': 'plain-opd', 'model': '/fixed/model'},
                    'frozen', [], {'pid': 42}, common.CostLedger(self.root), False, lambda stage: None)
        cleanup.assert_called_once()
        saved = common.read_json(folder / 'launcher_record.json')
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['pids'], [100, 101, 102, 103])
        self.assertEqual(len(saved['workers']), 4)
        self.assertAlmostEqual(saved['allocated_gpu_hours'], 4 * saved['wall_seconds'] / 3600)

    def test_frozen_input_mutation_rejected(self):
        path = self.root / 'frozen-input'
        path.write_text('frozen')
        plan = {'frozen_files': [{'path': str(path), 'sha256': common.sha256(path)}]}
        self.assertEqual(len(common.verify_frozen_files(plan)), 1)
        path.write_text('changed')
        with self.assertRaises(RuntimeError):
            common.verify_frozen_files(plan)

    def test_guard_does_not_stop_at_six_or_one_hundred_gpu_hours(self):
        owner = {'pid': 42, 'start_ticks': 100}
        state = {'status': 'running', 'stage': 'evaluate_plain-opd', 'pid': 42, 'coordinator_start_ticks': 100}
        for charged in (6.0, 100.0):
            with patch.object(budget_watchdog, 'read_json', side_effect=[state, dict(state, stage='analysis')]), patch.object(budget_watchdog, 'same_process', return_value=True), patch.object(common.CostLedger, 'snapshot', return_value=(charged, 2)), patch.object(budget_watchdog, 'stop_verified_owner') as stop, patch.object(budget_watchdog.time, 'sleep'), patch.object(budget_watchdog, 'atomic', wraps=common.atomic) as write:
                budget_watchdog.monitor(self.root, owner, {'pid': 50})
            stop.assert_not_called()
            watching = write.call_args_list[0].args[1]
            self.assertEqual(watching['status'], 'watching')
            self.assertEqual(watching['estimated_allocated_gpu_hours'], charged)
            self.assertFalse(watching['cap_enabled'])
            self.assertIsNone(watching['ceiling_gpu_hours'])
            self.assertIsNone(watching['termination_threshold_gpu_hours'])

    def test_guard_persistent_read_failure_stops_only_verified_owner(self):
        with patch.object(budget_watchdog, 'read_json', side_effect=FileNotFoundError('fault')), patch.object(budget_watchdog.time, 'sleep'), patch.object(budget_watchdog, 'stop_verified_owner', return_value=True) as stop:
            budget_watchdog.monitor(self.root, {'pid': 42, 'start_ticks': 100}, {'pid': 50})
        stop.assert_called_once()
        self.assertEqual(common.read_json(self.root / 'budget_watchdog_state.json')['status'], 'guard_error_stop_sent')

    def test_coverage_complete_then_duplicate_and_wrong_hash_rejected(self):
        folder = self.root / 'output'
        folder.mkdir()
        entries = [{'index': i, 'question_sha256': 'hash-' + str(i)} for i in range(512)]
        for rank in range(4):
            rows = [{'suite': 'pilot_test', 'index': i, 'question_sha256': 'hash-' + str(i),
                     'sample': sample, 'sampling_seed': 42 + sample, 'score': 0,
                     'response_tokens': 4096}
                    for i in range(rank, 512, 4) for sample in range(4)]
            (folder / f'rank_{rank}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
            common.atomic(folder / f'rank_{rank}.done.json', {'status': 'completed', 'rank': rank,
                'rows': 512, 'samples': 4, 'model': '/fixed/model'})
        self.assertEqual(common.validate_coverage(folder, entries, '/fixed/model'), 2048)
        path = folder / 'rank_0.jsonl'
        original = path.read_text()
        path.write_text(original + original.splitlines()[0] + '\n')
        with self.assertRaises(ValueError):
            common.validate_coverage(folder, entries)
        path.write_text(original.replace('hash-0', 'tampered', 1))
        with self.assertRaises(ValueError):
            common.validate_coverage(folder, entries)


if __name__ == '__main__':
    unittest.main()
