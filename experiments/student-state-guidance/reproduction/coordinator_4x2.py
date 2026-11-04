"""Run the authorized four K&K arms in two isolated four-GPU slots.

Use a separate REPRO_ROOT. Existing eight-GPU attempts remain intact and are
not accepted as completed four-GPU phases.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
import argparse
import fcntl
import hashlib
import os
import subprocess
import sys
import threading
import time
import traceback

import coordinator
from coordinator import read, validate_phase, merge, evaluate, costs, status, verify_models
from run_phase import ROOT, REPO, atomic


SLOTS = {'A': (0, 1, 2, 3), 'B': (4, 5, 6, 7)}
LEGACY_ROOT = Path(os.environ.get('REPRO_LEGACY_ROOT', str(REPO / '.local-runs' / 'sequential')))
ALLOW_RETRY = False
RECORD_LOCK = threading.Lock()


def study_status(stage, **extra):
    # Archiving a failed folder and enumerating its cost records must not race.
    with RECORD_LOCK:
        path = ROOT / 'prior_cost_ledger.json'
        prior = sum(event.get('allocated_gpu_hours', 0)
                    for event in read(path)['events']) if path.exists() else 0
        current = costs()
        status(stage, current_schedule_gpu_hours=current,
               prior_schedule_and_setup_gpu_hours=prior,
               total_recorded_gpu_hours=prior + current, **extra)


def free_slot(gpu_ids):
    """Wait only for the assigned devices; never terminate GPU processes."""
    wanted = set(gpu_ids)
    for _ in range(60):
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used',
             '--format=csv,noheader,nounits'], text=True)
        used = {}
        for line in output.splitlines():
            index, memory = line.split(',')
            used[int(index.strip())] = int(memory.strip())
        if not wanted.issubset(used):
            raise RuntimeError('Assigned GPUs are not available: ' + str(gpu_ids))
        if all(used[index] < 100 for index in wanted):
            return
        time.sleep(5)
    raise RuntimeError('Assigned GPUs are occupied; no processes were terminated: '
                       + str(gpu_ids))


def archive_failed(folder):
    with RECORD_LOCK:
        record = read(folder / 'launcher_record.json')
        if not ALLOW_RETRY or record['status'] != 'failed':
            raise RuntimeError('Review existing attempt; explicit --retry-failed required: '
                               + str(folder))
        target = folder.with_name(folder.name + '.failed-' + str(time.time_ns()))
        if target.exists():
            raise RuntimeError('Archive target already exists: ' + str(target))
        folder.rename(target)


def validate_four_gpu_phase(folder, method, end_step, resume=None):
    checkpoint = validate_phase(folder, method, end_step, resume)
    record_path = folder / 'launcher_record.json'
    record = read(record_path)
    if record['n_gpus'] != 4:
        message = 'An eight-GPU or other-world-size phase cannot satisfy this study'
        record.update(status='failed', failure_kind='phase_validation',
                      validation_status='failed', validation_error=message,
                      validated_unix=time.time())
        atomic(record_path, record)
        raise RuntimeError(message + ': ' + str(folder))
    return checkpoint


def phase(method, name, end_step, gpu_ids, resume=None, smoke=False, cancel_event=None):
    folder = ROOT / ('smoke' if smoke else 'runs') / name
    record_path = folder / 'launcher_record.json'
    if record_path.exists():
        record = read(record_path)
        if record['status'] == 'completed':
            return validate_four_gpu_phase(folder, method, end_step, resume)
        archive_failed(folder)
    free_slot(gpu_ids)
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError('Launch canceled after a failure in the other GPU slot: ' + name)
    args = [sys.executable, str(REPO / 'reproduction/run_phase.py'),
            '--method', method, '--name', name, '--end-step', str(end_step),
            '--gpu-ids', ','.join(map(str, gpu_ids))]
    if smoke:
        args.append('--smoke')
    if resume:
        args += ['--resume', str(resume)]
    # Each run_phase creates its own Ray cluster and durable launcher record.
    # Never mutate os.environ here: two worker threads launch concurrently.
    subprocess.run(args, check=True)
    return validate_four_gpu_phase(folder, method, end_step, resume)


def job(method, name, end_step, slot, *, resume=None, smoke=False):
    return {'method': method, 'name': name, 'end_step': end_step,
            'gpu_ids': SLOTS[slot], 'resume': resume, 'smoke': smoke}


def job_snapshots(lanes):
    snapshots = []
    for lane in lanes:
        for spec in lane:
            folder = ROOT / ('smoke' if spec['smoke'] else 'runs') / spec['name']
            item = {'name': spec['name'], 'method': spec['method'],
                    'gpu_ids': list(spec['gpu_ids']), 'status': 'pending'}
            record_path = folder / 'launcher_record.json'
            try:
                with RECORD_LOCK:
                    record = read(record_path) if record_path.exists() else {}
            except FileNotFoundError:
                record = {}
            if record:
                for key in ['status', 'pid', 'started_unix', 'finished_unix',
                            'training_batches', 'last_logged_step',
                            'allocated_gpu_hours', 'validation_status']:
                    if key in record:
                        item[key] = record[key]
            snapshots.append(item)
    return snapshots


def run_lanes(stage, lanes):
    """Advance each lane independently; a failure stops new work on both lanes.

    Already-running work in the other slot is allowed to finish and is retained.
    Only this main thread writes study_state.json.
    """
    stop_new_work = threading.Event()

    def worker(specs):
        results = {}
        for spec in specs:
            if stop_new_work.is_set():
                break
            try:
                results[spec['method']] = phase(**spec, cancel_event=stop_new_work)
            except BaseException:
                stop_new_work.set()
                raise
        return results

    results, errors = {}, []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='four-gpu-slot') as pool:
        pending = {pool.submit(worker, specs) for specs in lanes}
        while pending:
            study_status(stage, jobs=job_snapshots(lanes),
                         waiting_for_healthy_jobs_after_failure=stop_new_work.is_set())
            done, pending = wait(pending, timeout=15)
            for future in done:
                try:
                    results.update(future.result())
                except BaseException as exc:
                    errors.append(exc)
                    stop_new_work.set()
    if errors:
        raise RuntimeError(
            'Four-GPU lane failed; other in-flight jobs were allowed to finish. '
            'Inspect launcher records before --retry-failed. '
            + '; '.join(type(exc).__name__ + ': ' + str(exc) for exc in errors)
        ) from errors[0]
    return results


def frozen_plan():
    data = read(ROOT / 'data/manifest.json')
    assert data['train_overlap_counts']['unseen_test'] == 0
    plan = {
        'methods': ['grpo', 'opd', 'opd_then_rl', 'kdrl_anneal'],
        'total_batches': 150, 'switch_step': 60, 'train_batch': 128,
        'rollouts': 8, 'response_limit': 4096,
        'student': 'Qwen3-1.7B-Base', 'teacher': 'Qwen3-8B',
        'n_gpus': 4, 'concurrent_training_jobs': 2,
        'gpu_slots': {name: list(ids) for name, ids in SLOTS.items()},
        'runtime_change': 'User authorized two concurrent four-GPU jobs; '
                          'all formal arms start fresh in a separate study root',
        'schedule': {'A': ['grpo', 'kdrl_anneal'], 'B': ['opd', 'opd_then_rl']},
        'teacher_dispatch': 'author code retained, including unused teacher scoring during GRPO',
        'evaluation': {
            'suites': ['author_test', 'unseen_test'], 'n': 32, 'questions': 512,
            'temperature': 0.7, 'top_p': 0.8, 'top_k': 20,
            'sample_seeds': list(range(42, 74)), 'n_gpus': 8,
        },
        'selection': 'fixed final step 150; no best-test checkpoint or hyperparameter search',
        'source_author_commit': 'f5fd333736495608acd9415a60cd7568593aa1d4',
        'data_manifest_sha256': hashlib.sha256(
            (ROOT / 'data/manifest.json').read_bytes()).hexdigest(),
    }
    path = ROOT / 'study_plan.json'
    if path.exists():
        assert read(path) == plan, 'Frozen four-GPU protocol changed'
    else:
        atomic(path, plan)


def main():
    global ALLOW_RETRY
    parser = argparse.ArgumentParser()
    parser.add_argument('--retry-failed', action='store_true')
    ALLOW_RETRY = parser.parse_args().retry_failed
    coordinator.ALLOW_RETRY = ALLOW_RETRY
    if not os.environ.get('REPRO_ROOT') or ROOT.resolve() == LEGACY_ROOT.resolve():
        raise RuntimeError('Set REPRO_ROOT to a separate four-GPU study directory')
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT / 'coordinator.pid').write_text(str(os.getpid()) + '\n')
        try:
            study_status('verify_models', gpu_slots={k: list(v) for k, v in SLOTS.items()})
            verify_models()
            frozen_plan()
            smoke = run_lanes('parallel_smoke', [
                [job('opd', 'paper-opd-001', 1, 'A', smoke=True)],
                [job('kdrl_anneal', 'paper-joint-001', 1, 'B', smoke=True)],
            ])
            study_status('paper-resume-001')
            phase(**job('opd_then_rl', 'paper-resume-001', 2, 'A',
                        resume=smoke['opd'], smoke=True))
            prefix = ROOT / 'runs/opd-001/checkpoints/global_step_60'
            checkpoints = run_lanes('parallel_training', [
                [job('grpo', 'grpo-001', 150, 'A'),
                 job('kdrl_anneal', 'kdrl-anneal-001', 150, 'A')],
                [job('opd', 'opd-001', 150, 'B'),
                 job('opd_then_rl', 'opd-then-rl-001', 150, 'B', resume=prefix)],
            ])
            assert set(checkpoints) == {'grpo', 'opd', 'opd_then_rl', 'kdrl_anneal'}
            # Hold both test suites until all four fixed training paths finish.
            # The original evaluation implementation then uses all eight GPUs.
            for name in ['grpo', 'opd', 'opd_then_rl', 'kdrl_anneal']:
                evaluate(merge(checkpoints[name]), name)
            atomic(ROOT / 'final_checkpoints.json',
                   {name: str(checkpoints[name]) for name in checkpoints})
            study_status('training_and_evaluation_complete', status='completed')
            subprocess.run([sys.executable, str(REPO / 'reproduction/analyze_results.py')],
                           check=True)
            study_status('completed', status='completed',
                         report=str(ROOT / 'REPRODUCTION_RESULTS.md'))
        except BaseException as exc:
            study_status('needs_attention', status='failed',
                         error=type(exc).__name__ + ': ' + str(exc))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
