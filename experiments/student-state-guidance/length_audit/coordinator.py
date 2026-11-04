"""Run the frozen six-checkpoint length diagnostic, without changing any training."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from length_audit.common import (CAP, GPUS, CostLedger, atomic, phase_timeout,
    process_identity, read_json, same_process, sha256, signal_owned_group,
    validate_coverage, validate_plan, verify_frozen_files)


def existing(folder, retry, plan_sha):
    # Directory absence is established by its parent listing, not a lossy exists().
    names = {entry.name for entry in folder.parent.iterdir()}
    if folder.name not in names:
        return False
    record = read_json(folder / 'launcher_record.json')
    if record.get('plan_sha256') != plan_sha:
        raise RuntimeError('Existing allocation belongs to another frozen plan')
    if record['status'] == 'completed':
        return True
    if record['status'] != 'failed' or not retry:
        raise RuntimeError('Existing attempt requires diagnosis; only failed attempts allow --retry-failed: ' + str(folder))
    destination = folder.with_name(folder.name + '.failed-' + record['attempt_id'])
    if destination.name in names:
        raise RuntimeError('Attempt archive already exists; preserve both records')
    folder.rename(destination)
    return False


def ensure_watchdog(root, owner, max_age=30):
    state = read_json(root / 'budget_watchdog_state.json')
    if state.get('status') != 'watching' or time.time() - state['updated_unix'] > max_age:
        raise RuntimeError('Independent budget guard is not fresh and watching')
    if (state.get('coordinator_pid') != owner['pid'] or
            state.get('coordinator_start_ticks') != owner['start_ticks']):
        raise RuntimeError('Budget guard belongs to a different coordinator')
    if not same_process(state['identity'], REPO / 'length_audit/budget_watchdog.py'):
        raise RuntimeError('Independent budget guard process is not alive')


def wait_watchdog(root, owner):
    deadline = time.monotonic() + 120
    while True:
        try:
            ensure_watchdog(root, owner)
            return
        except (OSError, ValueError, KeyError, RuntimeError):
            if time.monotonic() >= deadline:
                raise RuntimeError('Independent watchdog did not become ready; no GPU work launched')
            time.sleep(2)


def free_gpus():
    for _ in range(36):
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=20).splitlines()
        if len(rows) == GPUS and all(int(row.split(',')[0]) < 1000 and
                int(row.split(',')[1]) <= 10 for row in rows):
            return
        time.sleep(5)
    raise RuntimeError('GPUs occupied; no unrelated job was terminated')


def cleanup(children, identities):
    errors = []
    for identity in identities:
        try:
            signal_owned_group(identity, signal.SIGTERM)
        except (OSError, RuntimeError) as exc:
            errors.append(repr(exc))
    deadline = time.monotonic() + 30
    while any(child.poll() is None for child in children) and time.monotonic() < deadline:
        time.sleep(0.2)
    for identity in identities:
        try:
            signal_owned_group(identity, signal.SIGKILL)
        except (OSError, RuntimeError) as exc:
            errors.append(repr(exc))
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            errors.append(repr(exc))
    if errors:
        raise RuntimeError('Owned worker cleanup did not fully verify: ' + '; '.join(errors))


def evaluate(root, endpoint, plan_sha, entries, owner, ledger, retry, set_status):
    name, model = endpoint['name'], endpoint['model']
    folder = root / 'evaluations' / name
    if existing(folder, retry, plan_sha):
        validate_coverage(folder, entries, model)
        return
    free_gpus()
    ensure_watchdog(root, owner)
    charged, _ = ledger.snapshot()
    ceiling = phase_timeout(charged)
    set_status('evaluate_' + name)
    folder.mkdir()
    started = time.time()
    monotonic_start = time.monotonic()
    record = {'status': 'running', 'attempt_id': uuid.uuid4().hex, 'name': name,
        'started_unix': started, 'n_gpus': GPUS, 'model': model,
        'questions': 512, 'samples': 4, 'max_tokens': 4096,
        'plan_sha256': plan_sha, 'max_wall_seconds': ceiling,
        'gpu_hour_cap': CAP, 'cap_enabled': CAP is not None,
        'coordinator': owner, 'workers': [], 'pids': []}
    # Persist chargeable start before creating the first GPU process.
    atomic(folder / 'launcher_record.json', record)
    children, identities, logs = [], [], []
    error = None
    try:
        for rank in range(GPUS):
            log = (folder / f'rank_{rank}.log').open('w')
            logs.append(log)
            args = [sys.executable, str(REPO / 'length_audit/eval_worker.py'), '--model', model,
                '--output', str(folder), '--rank', str(rank), '--world-size', '4',
                '--samples', '4', '--count', '512', '--max-tokens', '4096']
            # Do not allow a stop signal between spawn and recording ownership.
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            try:
                child = subprocess.Popen(args, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank)),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask))
                children.append(child)
                identity = process_identity(child.pid)
                if identity is None:
                    child.wait(timeout=5)
                    raise RuntimeError('Evaluation worker exited during launch')
                identities.append(identity)
                if identity['session'] != child.pid or identity['pgrp'] != child.pid:
                    raise RuntimeError('Worker did not receive its own session')
                record['workers'].append(dict(identity, rank=rank))
                record['pids'].append(child.pid)
                atomic(folder / 'launcher_record.json', record)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError('Evaluation worker failed')
            if ceiling is not None and time.monotonic() - monotonic_start >= ceiling:
                raise TimeoutError('GPU budget cleanup reserve reached')
            ensure_watchdog(root, owner)
            time.sleep(3)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError('Evaluation worker failed')
        record['responses'] = validate_coverage(folder, entries, model)
        record['status'] = 'completed'
    except BaseException as exc:
        error = exc
        record.update(status='failed', error=repr(exc))
    finally:
        # A second interrupt must not bypass process cleanup and cost settlement.
        old_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            try:
                cleanup(children, identities)
            except BaseException as exc:
                record.update(status='failed', cleanup_error=repr(exc))
                error = error or exc
            for log in logs:
                log.close()
            finished = time.time()
            elapsed = max(0, finished - started)
            record.update(finished_unix=finished, wall_seconds=elapsed,
                allocated_gpu_hours=GPUS * elapsed / 3600,
                returncodes=[child.poll() for child in children])
            atomic(folder / 'launcher_record.json', record)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
    if error is not None:
        raise error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    root = Path(os.environ['REPRO_ROOT']).resolve()
    (root / 'evaluations').mkdir(parents=True, exist_ok=True)
    script = REPO / 'length_audit/coordinator.py'
    with (root / '.length_audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = process_identity(os.getpid(), script)
        atomic(root / 'coordinator_identity.json', owner)
        (root / 'coordinator.pid').write_text(str(os.getpid()) + '\n')
        ledger = CostLedger(root)

        def set_status(stage, **extra):
            charged, count = ledger.snapshot()
            atomic(root / 'study_state.json', {'status': 'running', 'stage': stage,
                'pid': owner['pid'], 'coordinator_start_ticks': owner['start_ticks'],
                'updated_unix': time.time(), 'gpu_hour_cap': CAP,
                'cap_enabled': CAP is not None,
                'charged_gpu_hours': charged, 'cost_records': count, **extra})

        def interrupted(signum, frame):
            raise RuntimeError('Coordinator interrupted by signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            plan_path = root / 'plan.json'
            plan = read_json(plan_path)
            validate_plan(plan)
            plan_sha = sha256(plan_path)
            set_status('preflight')
            checked = verify_frozen_files(plan)
            entries = [json.loads(line) for line in (root / 'data/pilot_test.jsonl').read_text().splitlines()]
            test_path = (root / 'data/pilot_test.jsonl').resolve()
            if test_path not in {Path(item['path']).resolve() for item in plan['frozen_files']}:
                raise RuntimeError('pilot_test is not included in the frozen file manifest')
            if sha256(plan_path) != plan_sha:
                raise RuntimeError('Plan changed during preflight')
            atomic(root / 'preflight_receipt.json', {'status': 'passed', 'plan_sha256': plan_sha,
                'source_commit': plan['source_commit'], 'files': checked, 'checked_unix': time.time(),
                'control_source_sha256': {p.name: sha256(p) for p in (REPO / 'length_audit').glob('*.py')}})
            pending = []
            for endpoint in plan['endpoints']:
                folder = root / 'evaluations' / endpoint['name']
                if endpoint['name'] in {p.name for p in folder.parent.iterdir()}:
                    old = read_json(folder / 'launcher_record.json')
                    if old['status'] == 'running':
                        raise RuntimeError('Unsettled running attempt requires explicit diagnosis: ' + endpoint['name'])
                    if old['status'] == 'completed':
                        if old.get('plan_sha256') != plan_sha:
                            raise RuntimeError('Completed endpoint belongs to another plan')
                        validate_coverage(folder, entries, endpoint['model'])
                        continue
                pending.append(endpoint)
            if pending:
                set_status('waiting_budget_guard')
                wait_watchdog(root, owner)
            for endpoint in pending:
                if sha256(plan_path) != plan_sha:
                    raise RuntimeError('Frozen plan changed after preflight')
                evaluate(root, endpoint, plan_sha, entries, owner, ledger, args.retry_failed, set_status)
            set_status('analysis')
            subprocess.run([sys.executable, str(REPO / 'length_audit/analyze.py')], check=True, timeout=600)
            set_status('completed', status='completed')
        except BaseException as exc:
            try:
                set_status('needs_attention', status='failed', error=repr(exc))
            except BaseException as accounting_error:
                atomic(root / 'study_state.json', {'status': 'failed', 'stage': 'needs_attention',
                    'pid': owner['pid'], 'error': repr(exc), 'accounting_error': repr(accounting_error),
                    'updated_unix': time.time(), 'gpu_hour_cap': CAP,
                    'cap_enabled': CAP is not None})
            raise


if __name__ == '__main__':
    main()
