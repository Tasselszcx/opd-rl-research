"""G3 fixed 4-task x 2-state x 5-guidance x 3-seed training scheduler."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from state_guidance.control import atomic,identity,ledger,read,same_process,stop_owned
from state_guidance.run_phase import validate
from state_guidance.health_gate import wait_healthy
from state_guidance.scope import load_scope

REPO=Path(__file__).resolve().parents[1]
TASKS=['kk','gsm8k','zebra','countdown']
MODES=['plain','answer_only','full_reasoning','key_steps','error_targeted']
SLOTS={'A':'0,1,2,3','B':'4,5,6,7'}


def phases(root):
    result=[]
    for seed in [42,43,44]:
        for task in TASKS:
            result.append(dict(name=f'{task}-s{seed}-reference',task=task,seed=seed,mode='plain',steps=150,resume=None,start=0,dependency=None))
        for task in TASKS:
            for start in [30,90]:
                modes=list(MODES)
                # Counterbalance condition execution order without changing scientific RNG.
                from random import Random
                Random(seed+start+TASKS.index(task)).shuffle(modes)
                for mode in modes:
                    reference=f'{task}-s{seed}-reference'
                    result.append(dict(name=f'{task}-s{seed}-state{start}-{mode}',task=task,seed=seed,
                                       mode=mode,steps=150,start=start,dependency=reference,
                                       resume=str(root/'runs'/reference/'checkpoints'/f'global_step_{start}')))
    assert len(result)==132 and len({p['name'] for p in result})==132
    scope=load_scope(root)
    result=[p for p in result if p['task'] in scope['tasks'] and p['seed'] in scope['training_seeds']
            and (p['resume'] is None or p['mode'] in scope['modes'])]
    if scope.get('ordered_modes'):
        assert set(scope['ordered_modes'])==set(scope['modes'])
        result.sort(key=lambda p:(p['seed'],scope['tasks'].index(p['task']),p['start'],
                                 scope['ordered_modes'].index(p['mode'])))
    return result


def state_candidates(candidates,all_phases,completed,seed,enabled):
    if not enabled:return candidates
    remaining=[p['start'] for p in all_phases if p['seed']==seed and p['name'] not in completed]
    return [p for p in candidates if remaining and p['start']==min(remaining)]


def next_candidate(candidates,active):
    counts={}
    for item in active.values():
        task=item['phase']['task'];counts[task]=counts.get(task,0)+1
    index=min(range(len(candidates)),key=lambda i:counts.get(candidates[i]['task'],0))
    return candidates.pop(index)


def ready(root,task):
    data=root/'data'/task
    return all((data/f).exists() for f in ['manifest.json','guidance_manifest.json','prompt_audit.json'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--resume-migration',action='store_true')
    p.add_argument('--adopt-running',type=Path);a=p.parse_args()
    assert not (a.retry_failed and a.resume_migration), 'Migration continuation and fresh failed-attempt retry are separate operations'
    root=Path(os.environ['REPRO_ROOT']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=identity(os.getpid());atomic(root/'coordinator_identity.json',owner)
    (root/'coordinator.pid').write_text(str(os.getpid())+'\n')
    for name in ['runs','checks','logs','evaluations','locks']: (root/name).mkdir(exist_ok=True)
    plan_path=root/'study_plan.json';plan=read(plan_path);plan_hash=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert plan['training_seeds']==[42,43,44] and plan['branch_additional_steps']==150
    assert plan['formal_branches']==120 and plan['reference_runs']==12
    scope=load_scope(root)
    if scope['scope_sha256'] is not None:assert scope['baseline_plan_sha256']==plan_hash
    all_phases=phases(root);active={};completed=set();failed=[];watch=None
    if a.adopt_running:
        from state_guidance.adoption import adoption_items
        manifest=read(a.adopt_running)
        assert manifest['scope_sha256']==scope['scope_sha256']
        active=adoption_items(root,manifest,all_phases,SLOTS)
    state={'status':'running','stage':'preparing','identity':owner,'source':str(REPO),'plan_sha256':plan_hash,
           'scope_sha256':scope['scope_sha256'],'active_scope':scope,'formal_route_count':len(all_phases)}
    amendment=root/'PROTOCOL_AMENDMENT.json'
    amendment_hash=hashlib.sha256(amendment.read_bytes()).hexdigest() if amendment.exists() else None
    state['protocol_amendment_sha256']=amendment_hash
    def save(stage,**extra):
        if hashlib.sha256(plan_path.read_bytes()).hexdigest()!=plan_hash:raise RuntimeError('Frozen study plan changed')
        if load_scope(root)['scope_sha256']!=scope['scope_sha256']:raise RuntimeError('Approved scope changed')
        if amendment_hash is not None and hashlib.sha256(amendment.read_bytes()).hexdigest()!=amendment_hash:
            raise RuntimeError('Approved protocol amendment changed')
        state.update(stage=stage,updated_unix=time.time(),completed_runs=sorted(completed),failed_runs=list(failed),
                     active={k:{'name':v['phase']['name'],'identity':v['identity']} for k,v in active.items()},**extra)
        atomic(root/'study_state.json',state)
    save('preparing')
    with (root/'health_watchdog.log').open('a') as log:
        watch=subprocess.Popen([sys.executable,'-m','state_guidance.health_watchdog'],cwd=REPO,
                               stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    def healthy():
        wait_healthy(root,owner)
    def inspect_or_archive(phase,check=False):
        folder=root/('checks' if check else 'runs')/phase['name'];record=folder/'launcher_record.json'
        if not folder.exists():return False
        old=read(record)
        if old['status']=='completed':
            validate(folder,phase['start'],phase['start']+phase['steps'],phase['mode']);return True
        if any(item['phase']['name']==phase['name'] for item in active.values()):return False
        if old['status']=='migration_ready':
            from state_guidance.migration import load_continuation
            assert a.resume_migration and not check, 'Explicit --resume-migration required'
            load_continuation(folder,phase['task'],phase['mode'],phase['seed'],phase['start'],phase['start']+phase['steps'])
            return False
        if old['status']!='failed' or not a.retry_failed:
            raise RuntimeError('Unsettled existing attempt requires diagnosis: '+str(folder))
        folder.rename(folder.with_name(folder.name+'.failed-'+str(time.time_ns())))
        return False
    def launch(phase,slot,check=False):
        healthy()
        args=[sys.executable,'-m','state_guidance.run_phase','--task',phase['task'],'--mode',phase['mode'],
              '--seed',str(phase['seed']),'--name',phase['name'],'--steps',str(phase['steps']),'--gpu-ids',SLOTS[slot]]
        if phase['resume']:args+=['--resume',phase['resume']]
        if check:args+=['--check']
        record=root/('checks' if check else 'runs')/phase['name']/'launcher_record.json'
        if record.exists() and read(record)['status']=='migration_ready':
            assert a.resume_migration
            args+=['--continue-migration']
        log=(root/'logs'/(phase['name']+'.launcher.log')).open('a')
        child=subprocess.Popen(args,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        active[slot]={'phase':phase,'process':child,'identity':identity(child.pid),'log':log,'check':check,'started':time.time()}
        save('checks' if check else 'training')
    def collect():
        for slot,item in list(active.items()):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();phase=item['phase'];folder=root/('checks' if item['check'] else 'runs')/phase['name']
            rec=folder/'launcher_record.json'
            if not rec.exists():
                atomic(rec,{'status':'failed','failure_kind':'before_gpu_launch','n_gpus':0,
                            'started_unix':item['started'],'finished_unix':time.time(),'allocated_gpu_hours':0,
                            'return_code':code,'error':'Inspect logs/'+phase['name']+'.launcher.log'})
            record=read(rec)
            if code==0 and record['status']=='completed':completed.add(phase['name'])
            else:failed.append(phase['name'])
            del active[slot]
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        deadline=time.monotonic()+120
        while True:
            try:healthy();break
            except (OSError,ValueError,KeyError,AssertionError,RuntimeError):
                if time.monotonic()>deadline:raise RuntimeError('Health watcher did not start')
                time.sleep(2)
        while not (ready(root,'kk') and (Path(os.environ['G3_MODEL_ROOT'])/'CACHE_MANIFEST.json').exists()):
            healthy();save('waiting_prepared_inputs');time.sleep(10)
        checks=[dict(name='opd-update',task='kk',seed=42,mode='full_reasoning',steps=1,start=0,resume=None),
                dict(name='opd-resume',task='kk',seed=42,mode='key_steps',steps=1,start=1,
                     resume=str(root/'checks/opd-update/checkpoints/global_step_1'))]
        for check in checks:
            if inspect_or_archive(check,True):completed.add(check['name']);continue
            launch(check,'A',True)
            while active:collect();save('checks');time.sleep(5)
            if failed:raise RuntimeError('New OPD continuation check failed')
        atomic(root/'preflight/training_chain_acceptance.json',{'status':'passed','checks':[p['name'] for p in checks],
               'scope':'Full batch128, rollout8, response4096; OPD checkpoint resumed into OPD; not a research score',
               'checked_unix':time.time()})
        for phase in all_phases:
            if inspect_or_archive(phase):completed.add(phase['name'])
        while any(p['name'] not in completed for p in all_phases):
            collect();healthy()
            if all(p['name'] in completed for p in all_phases):
                break
            if failed:
                if not active:raise RuntimeError('Training failure: '+','.join(failed))
                save('failure_waiting_other_lane');time.sleep(10);continue
            # Finish the entire paired-seed matrix before advancing training seed.
            current_seed=min(p['seed'] for p in all_phases if p['name'] not in completed)
            running_names={v['phase']['name'] for v in active.values()}
            candidates=[p for p in all_phases if p['seed']==current_seed and p['name'] not in completed|running_names
                        and (p['dependency'] is None or p['dependency'] in completed) and ready(root,p['task'])]
            candidates=state_candidates(candidates,all_phases,completed,current_seed,scope.get('sequential_student_states',False))
            used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in used.splitlines()}
            for slot in SLOTS:
                free=all(memory[int(g)]<2048 for g in SLOTS[slot].split(','))
                if slot not in active and candidates and free:launch(next_candidate(candidates,active),slot)
            save('training' if active else 'waiting_prepared_inputs');time.sleep(10)
        save('training_complete',status='training_complete',allocation=ledger(root),
             next_action='Frozen final evaluations and analysis only; no new training scope')
    except BaseException as exc:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        for item in active.values():stop_owned(item['identity'])
        for item in active.values():
            try:item['process'].wait(timeout=90)
            except subprocess.TimeoutExpired:pass
        collect();save('stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',
                       status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),allocation=ledger(root))
        raise


if __name__=='__main__':main()
