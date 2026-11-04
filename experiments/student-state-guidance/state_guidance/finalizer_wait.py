"""Bounded, stop-aware tolerance for transient shared-filesystem state reads."""
import time
from state_guidance.control import read


def training_snapshot(path, stop, report, *, read_state=read, clock=time.monotonic,
                      retry_seconds=5., max_wait_seconds=120.):
    started = clock()
    failures = 0
    while not stop.is_set():
        try:
            state = read_state(path)
            if not isinstance(state, dict) or not isinstance(state.get('status'), str):
                raise ValueError('Training snapshot has no valid status')
            return state
        except (OSError, ValueError) as error:
            failures += 1
            elapsed = clock() - started
            stage = 'failed' if elapsed >= max_wait_seconds else 'waiting_training_snapshot_retry'
            try:
                report(stage, snapshot_error=repr(error), snapshot_read_failures=failures,
                       snapshot_retry_elapsed_seconds=elapsed)
            except OSError as report_error:
                # A wider storage outage may also prevent the status write.
                # Keep CPU-only retries bounded and retain the diagnostic in stdout.
                print('Finalizer snapshot/status storage error:', repr(error), repr(report_error), flush=True)
            if elapsed >= max_wait_seconds:
                raise RuntimeError('Training snapshot unavailable for bounded retry window') from error
            if stop.wait(min(retry_seconds, max_wait_seconds - elapsed)):
                break
    return None
