"""Debounce transient shared-storage health reads; never accept an unhealthy gate."""
import json
import time
from state_guidance.control import read, same_process


def event(root, kind, **details):
    line=json.dumps({'event': kind, 'unix': time.time(), **details}, ensure_ascii=False)
    try:
        with (root/'health_events.jsonl').open('a') as stream:stream.write(line+'\n')
    except OSError:
        # Event logging must not turn a transient shared-storage error into a stop.
        try:print(line,flush=True)
        except OSError:pass


def wait_healthy(root, owner, grace_seconds=60, clock=time, reader=read, matches=same_process, emit=event):
    deadline = clock.monotonic()+grace_seconds
    first = True
    while True:
        detail = {}
        try:
            info = reader(root/'health_watchdog_state.json')
            age = clock.time()-info['updated_unix']
            detail = {'status': info.get('status'), 'age_seconds': age, 'watchdog_identity': info.get('identity'),
                      'coordinator_identity': info.get('coordinator'), 'error': info.get('error')}
            healthy = (info['status'] == 'watching' and -5 <= age < 40
                       and info['coordinator'] == owner and matches(info['identity']))
        except (OSError, ValueError, KeyError) as exc:
            healthy = False; detail = {'read_exception': repr(exc)}
        if healthy:
            if not first: emit(root, 'coordinator_health_recovered', **detail)
            return
        if first:
            emit(root, 'coordinator_health_retry', grace_seconds=grace_seconds, **detail)
            first = False
        if clock.monotonic() >= deadline:
            emit(root, 'coordinator_health_failed', **detail)
            raise RuntimeError('Watchdog health remained invalid: '+json.dumps(detail))
        clock.sleep(min(2, max(0, deadline-clock.monotonic())))


def protection_due(consecutive_errors, first_error_monotonic, now):
    return consecutive_errors >= 3 and now-first_error_monotonic >= 60
