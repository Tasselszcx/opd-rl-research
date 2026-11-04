"""Independent G2 health/accounting guard; the current plan has no budget cap."""
from pathlib import Path
import fcntl
import json
import os
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from length_audit.common import (CAP, THRESHOLD, CostLedger, atomic, process_identity,
    read_json, same_process, stop_verified_owner, validate_plan)


def monitor(root, owner, own_identity):
    output = root / 'budget_watchdog_state.json'
    ledger = CostLedger(root)
    script = REPO / 'length_audit/coordinator.py'
    failures = 0
    while True:
        try:
            state = read_json(root / 'study_state.json')
            if state.get('pid') != owner['pid']:
                raise RuntimeError('Study state identifies a different owner')
            if not same_process(owner, script):
                atomic(output, {'status': 'owner_not_alive', 'identity': own_identity,
                    'coordinator_pid': owner['pid'], 'updated_unix': time.time()})
                return
            if state['status'] != 'running':
                atomic(output, {'status': 'finished', 'study_status': state['status'],
                    'identity': own_identity, 'updated_unix': time.time()})
                return
            if state['stage'] == 'analysis':
                atomic(output, {'status': 'gpu_work_finished', 'identity': own_identity,
                    'reason': 'Only CPU analysis remains', 'updated_unix': time.time()})
                return
            if state.get('coordinator_start_ticks') != owner['start_ticks']:
                raise RuntimeError('Study state owner start time changed')
            charged, count = ledger.snapshot()
            result = {'status': 'watching', 'pid': own_identity['pid'], 'identity': own_identity,
                'coordinator_pid': owner['pid'], 'coordinator_start_ticks': owner['start_ticks'],
                'estimated_allocated_gpu_hours': charged, 'cost_records': count,
                'ceiling_gpu_hours': CAP, 'termination_threshold_gpu_hours': THRESHOLD,
                'cap_enabled': CAP is not None,
                'updated_unix': time.time()}
            if THRESHOLD is not None and charged >= THRESHOLD:
                result.update(status='budget_stop_sent', stop_sent=stop_verified_owner(owner, script),
                    reason='Preserve cleanup time within the explicitly configured GPU-hour ceiling')
                atomic(output, result)
                return
            atomic(output, result)
            failures = 0
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            failures += 1
            print(json.dumps({'event': 'guard_snapshot_error', 'failure_count': failures,
                'error': repr(exc), 'time': time.time()}), flush=True)
            if failures >= 3:
                try:
                    stopped = stop_verified_owner(owner, script)
                except (OSError, RuntimeError):
                    stopped = False
                atomic(output, {'status': 'guard_error_stop_sent', 'identity': own_identity,
                    'coordinator_pid': owner['pid'], 'stop_sent': stopped,
                    'error': repr(exc), 'updated_unix': time.time()})
                return
        time.sleep(5)


def main():
    root = Path(os.environ['REPRO_ROOT']).resolve()
    with (root / '.budget_watchdog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_plan(read_json(root / 'plan.json'))
        owner = read_json(root / 'coordinator_identity.json')
        if not same_process(owner, REPO / 'length_audit/coordinator.py'):
            raise RuntimeError('No matching live coordinator')
        own_identity = process_identity(os.getpid(), REPO / 'length_audit/budget_watchdog.py')
        (root / 'budget_watchdog.pid').write_text(str(os.getpid()) + '\n')
        monitor(root, owner, own_identity)


if __name__ == '__main__':
    main()
