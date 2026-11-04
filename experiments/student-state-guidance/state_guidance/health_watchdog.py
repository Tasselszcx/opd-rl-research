"""Independent cost/identity health monitor. G3 has no automatic cost cap."""
import fcntl
import os
from pathlib import Path
import time
from state_guidance.control import atomic,identity,ledger,read,same_process,stop_owned
from state_guidance.health_gate import event,protection_due


def main():
    root=Path(os.environ['REPRO_ROOT']);lock=(root/'health_watchdog.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=read(root/'coordinator_identity.json');me=identity(os.getpid())
    (root/'health_watchdog.pid').write_text(str(os.getpid())+'\n');failures=0;first_error=None;last_success=None
    while True:
        state={'identity':me,'coordinator':owner,'updated_unix':time.time(),'budget_cap':None,'cap_enabled':False}
        try:
            costs=ledger(root);atomic(root/'cost_ledger.json',costs)
            study=read(root/'study_state.json')
            if study['status'] in ('completed','failed','stopped','training_complete'):
                state.update(status='finished',study_status=study['status'],allocated_gpu_hours=costs['allocated_gpu_hours'])
                atomic(root/'health_watchdog_state.json',state);return
            if not same_process(owner):
                state.update(status='coordinator_dead',error='Coordinator identity no longer alive; inspect any unsettled workers')
                atomic(root/'health_watchdog_state.json',state);return
            if failures:event(root,'watchdog_read_recovered',consecutive_errors=failures)
            last_success=time.time()
            state.update(status='watching',allocated_gpu_hours=costs['allocated_gpu_hours'],last_success_unix=last_success)
            failures=0;first_error=None
        except Exception as exc:
            if first_error is None:first_error=time.monotonic()
            failures+=1;state.update(status='read_error',consecutive_errors=failures,error=repr(exc),last_success_unix=last_success)
            event(root,'watchdog_read_error',consecutive_errors=failures,error=repr(exc))
            if protection_due(failures,first_error,time.monotonic()):
                event(root,'watchdog_protection_stop',coordinator=owner,error=repr(exc))
                stop_owned(owner);state['status']='protection_stop'
        state['updated_unix']=time.time()
        atomic(root/'health_watchdog_state.json',state)
        if state['status']=='protection_stop':return
        time.sleep(10)


if __name__=='__main__':main()
