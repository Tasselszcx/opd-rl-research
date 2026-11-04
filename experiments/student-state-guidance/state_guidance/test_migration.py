"""CPU tests for migration recovery, rollback evidence, cadence and cost accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from state_guidance.control import atomic, ledger, read
from state_guidance.migration import checkpoint_files, load_continuation, prepare, sha
from state_guidance.run_phase import command


class MigrationTests(unittest.TestCase):
    def fixture(self, root, task='countdown', origin=0, saved=50, last=82, mode='plain'):
        name = f'{task}-s42-reference' if origin == 0 else f'{task}-s42-state{origin}-{mode}'
        folder = root / 'runs' / name
        (folder / 'checkpoints').mkdir(parents=True)
        (root / 'migration-20261102').mkdir(exist_ok=True)
        old = {'status': 'failed', 'task': task, 'mode': mode, 'seed': 42, 'n_gpus': 4,
               'started_unix': 1000., 'finished_unix': 2000., 'allocated_gpu_hours': 12.5,
               'start_step': origin, 'end_step': origin + 150, 'check': False,
               'error': "RuntimeError('Interrupted by signal 15')"}
        atomic(folder / 'launcher_record.json', old)
        atomic(folder / 'request.json', {'config': {'data.seed': 42 if origin == 0 else 1000042}})
        rows = [{'step': step, 'additional_step': step - origin, 'payload': f'original-{step}'}
                for step in range(origin + 1, last + 1)]
        (folder / 'checkpoints/metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        inventory = []
        for step in ([30, saved] if origin == 0 else [saved]):
            checkpoint = folder / 'checkpoints' / f'global_step_{step}'
            for path in checkpoint_files(checkpoint):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((str(path.relative_to(root)) + ':test-saved-state').encode())
                inventory.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        (root / 'migration-20261102/checkpoint-sha256.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in inventory))
        for step in range(origin, last + 1, 25):
            val = folder / 'val_logs' / name / 'epoch_0' / f'step_{step}'
            atomic(val / 'metrics.json', {'step': step})
            (val / 'val.jsonl').write_text('preserved validation\n')
        route = {'name': name, 'original_attempt': 1000., 'original_start_step': origin,
                 'original_end_step': origin + 150, 'checkpoint_global_step': saved,
                 'last_metric_global_step': last, 'data_seed': 42 if origin == 0 else 1000042}
        return folder, route, old, rows

    def test_countdown_retains_prefix_and_archives_uncheckpointed_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, old, original = self.fixture(root)
            before = ledger(root)['allocated_gpu_hours']
            spec = prepare(root, route)
            archived = root / spec['archive_relative_path']
            kept = [json.loads(line) for line in (folder / 'checkpoints/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(kept, original[:50])
            self.assertEqual(len((archived / 'checkpoints/metrics.jsonl').read_text().splitlines()), 82)
            self.assertEqual(spec['uncheckpointed_updates'], list(range(51, 83)))
            self.assertEqual(read(archived / 'launcher_record.json'), old)
            self.assertEqual([read(p)['step'] for p in sorted((folder / 'val_logs').rglob('metrics.json'))], [0, 25, 50])
            self.assertTrue((folder / 'checkpoints/global_step_30/data.pt').exists())
            self.assertEqual(ledger(root)['allocated_gpu_hours'], before)

    def test_preserves_scientific_attempt_but_charges_new_segment_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, old, _ = self.fixture(root, 'kk', 30, 80, 80, 'error_targeted')
            spec = prepare(root, route)
            record = read(folder / 'launcher_record.json')
            self.assertEqual(record['scientific_attempt_started_unix'], old['started_unix'])
            self.assertEqual(spec['uncheckpointed_updates'], [])
            record.update(status='completed', n_gpus=4, started_unix=3000, allocated_gpu_hours=3.75)
            atomic(folder / 'launcher_record.json', record)
            self.assertEqual(ledger(root)['allocated_gpu_hours'], 16.25)

    def test_prepare_is_idempotent_and_does_not_duplicate_charges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, _, _ = self.fixture(root)
            first = prepare(root, route)
            self.assertEqual(prepare(root, route), first)
            self.assertEqual(len(list((root / 'runs').glob('*.failed-*'))), 1)
            self.assertEqual(ledger(root)['allocated_gpu_hours'], 12.5)

    def test_corrupted_saved_state_refuses_before_touching_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, old, _ = self.fixture(root)
            (folder / 'checkpoints/global_step_50/data.pt').write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):
                prepare(root, route)
            self.assertEqual(read(folder / 'launcher_record.json'), old)
            self.assertFalse(list((root / 'runs').glob('*.failed-*')))

    def test_running_or_unrelated_failure_is_never_archived(self):
        for change in [{'status': 'running'}, {'error': "RuntimeError('GPU fault')"}, {'started_unix': 999}]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder, route, old, _ = self.fixture(root)
                old.update(change)
                atomic(folder / 'launcher_record.json', old)
                with self.assertRaises(AssertionError):
                    prepare(root, route)
                self.assertEqual(read(folder / 'launcher_record.json'), old)

    def test_missing_checkpoint_dataloader_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, _, _ = self.fixture(root)
            prepare(root, route)
            (folder / 'checkpoints/global_step_50/data.pt').unlink()
            with self.assertRaises(AssertionError):
                load_continuation(folder, 'countdown', 'plain', 42)

    def test_second_rename_crash_is_recoverable_without_recopy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, _, _ = self.fixture(root)
            original = Path.rename
            def interrupted(path, destination):
                if path.name.startswith('staging-'):
                    raise OSError('simulated crash during commit')
                return original(path, destination)
            with patch.object(Path, 'rename', interrupted), self.assertRaises(OSError):
                prepare(root, route)
            self.assertFalse(folder.exists())
            spec = prepare(root, route)
            self.assertEqual(spec['checkpoint_step'], 50)
            self.assertEqual(ledger(root)['allocated_gpu_hours'], 12.5)

    def test_continuation_command_retains_original_seed_horizon_and_cadence(self):
        for task, origin, saved, mode in [('countdown', 0, 50, 'plain'), ('kk', 30, 80, 'error_targeted')]:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder, route, _, _ = self.fixture(root, task, origin, saved, saved, mode)
                spec = prepare(root, route)
                resume = folder / 'checkpoints' / f'global_step_{saved}'
                _, cfg, start, end = command(root, task, mode, 42, folder, resume, continuation=spec)
                self.assertEqual((start, end), (origin, origin + 150))
                self.assertEqual(cfg['trainer.total_training_steps'], origin + 150)
                self.assertEqual(cfg['trainer.g3_origin_step'], origin)
                self.assertFalse(cfg['trainer.reset_dataloader_on_fork'])
                self.assertTrue(cfg['trainer.require_dataloader_resume'])
                self.assertFalse(cfg['trainer.val_before_train'])
                self.assertEqual(cfg['data.seed'], route['data_seed'])
                self.assertEqual(cfg['trainer.extra_save_steps'], [30, 90] if origin == 0 else [])
                self.assertEqual(cfg['trainer.resume_mode'], str(resume))
                self.assertEqual([step for step in range(saved + 1, end + 1) if (step-origin) % 25 == 0],
                                 [origin + 75, origin + 100, origin + 125, origin + 150])

    def test_new_forks_still_reset_to_common_stream(self):
        _, cfg, start, end = command(Path('/unit'), 'kk', 'plain', 43, Path('/unit/run'), Path('/ref/global_step_90'))
        self.assertEqual((start, end), (90, 240))
        self.assertEqual(cfg['data.seed'], 1000043)
        self.assertTrue(cfg['trainer.reset_dataloader_on_fork'])
        self.assertTrue(cfg['trainer.val_before_train'])

    def test_wrong_phase_cannot_consume_a_prepared_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, _, _ = self.fixture(root)
            prepare(root, route)
            with self.assertRaises(AssertionError):
                load_continuation(folder, 'countdown', 'answer_only', 42)
            with self.assertRaises(AssertionError):
                load_continuation(folder, 'countdown', 'plain', 43)

    def test_checkpoint_references_work_when_shared_storage_rejects_hardlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, _, _ = self.fixture(root)
            original = {str(path.relative_to(folder)): sha(path)
                        for path in checkpoint_files(folder / 'checkpoints/global_step_50')}
            with patch('os.link', side_effect=PermissionError('shared storage EPERM')):
                spec = prepare(root, route)
            archive = root / spec['archive_relative_path']
            for relative, digest in original.items():
                path = folder / relative
                self.assertTrue(path.is_symlink())
                self.assertTrue(path.resolve().is_relative_to(archive.resolve()))
                self.assertEqual(sha(path), digest)
                self.assertFalse((archive / relative).is_symlink())
            self.assertEqual(ledger(root)['allocated_gpu_hours'], 12.5)

    def test_derived_checkpoint_metadata_is_independent_of_archived_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, _, _ = self.fixture(root)
            config = folder / 'checkpoints/global_step_50/actor/huggingface/config.json'
            atomic(config, {'original': True})
            spec = prepare(root, route)
            archive = root / spec['archive_relative_path']
            current = folder / config.relative_to(folder)
            self.assertFalse(current.is_symlink())
            current.write_text(json.dumps({'derived': True}))
            self.assertEqual(read(archive / config.relative_to(folder)), {'original': True})

    def health_fixture(self, root):
        folder, route, old, rows = self.fixture(root, saved=100, last=147)
        old['scientific_attempt_started_unix'] = 500.
        atomic(folder / 'launcher_record.json', old)
        atomic(root / 'study_state.json', {
            'status': 'failed', 'active': {}, 'identity': {'pid': 2147483647},
            'failed_runs': [folder.name],
            'error': "RuntimeError('Watchdog health remained invalid: stale')"})
        return folder, route, old, rows

    def test_health_recovery_retains_science_id_prefix_and_all_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, old, rows = self.health_fixture(root)
            spec = prepare(root, route, failure_kind='watchdog_stale')
            self.assertEqual(spec['scientific_attempt_started_unix'], 500.)
            self.assertEqual(spec['source_attempt_started_unix'], 1000.)
            self.assertEqual(spec['uncheckpointed_updates'], list(range(101, 148)))
            self.assertIn('.failed-health-', spec['archive_relative_path'])
            kept = [json.loads(x) for x in (folder / 'checkpoints/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(kept, rows[:100])
            self.assertEqual(ledger(root)['allocated_gpu_hours'], 12.5)
            self.assertEqual(prepare(root, route, failure_kind='watchdog_stale'), spec)

    def test_health_recovery_rejects_active_or_unrelated_coordinator_failure(self):
        for change in [{'active': {'A': {}}}, {'status': 'running'}, {'error': 'unrelated GPU error'}]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder, route, old, _ = self.health_fixture(root)
                state = read(root / 'study_state.json'); state.update(change)
                atomic(root / 'study_state.json', state)
                with self.assertRaises(AssertionError):
                    prepare(root, route, failure_kind='watchdog_stale')
                self.assertEqual(read(folder / 'launcher_record.json'), old)

    def test_health_recovery_second_rename_preserves_prior_science_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, route, _, _ = self.health_fixture(root)
            original = Path.rename
            def interrupted(path, destination):
                if path.name.startswith('staging-health-'):
                    raise OSError('simulated commit interruption')
                return original(path, destination)
            with patch.object(Path, 'rename', interrupted), self.assertRaises(OSError):
                prepare(root, route, failure_kind='watchdog_stale')
            spec = prepare(root, route, failure_kind='watchdog_stale')
            self.assertEqual(spec['scientific_attempt_started_unix'], 500.)
            self.assertEqual(ledger(root)['allocated_gpu_hours'], 12.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
