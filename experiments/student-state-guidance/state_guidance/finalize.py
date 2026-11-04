"""Wait for the fixed training matrix, then run diagnostics and full endpoint evaluations."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from state_guidance.control import atomic,identity,ledger,read,same_process,stop_owned
from state_guidance.coordinator import phases,SLOTS,TASKS
from state_guidance.eval_worker import entries_for,suite_settings
from state_guidance.data import score_answer
from state_guidance.finalizer_wait import training_snapshot
from state_guidance.run_phase import validate as validate_training
from state_guidance.scope import load_scope

REPO=Path(__file__).resolve().parents[1]


def training_finished(root,state):
    complete=state['status']=='training_complete'
    if not complete and (state['status']!='failed' or state.get('error')!="ValueError('min() arg is an empty sequence')"):return False
    # A known pre-fix scheduler may collect its last successful job before calling min(empty).
    # Preserve the failed controller record; independently validate every actual training run.
    expected=phases(root)
    scope=load_scope(root)
    if scope['scope_sha256'] is not None:
        assert state.get('scope_sha256')==scope['scope_sha256'], 'Training/evaluation scope mismatch'
    if not all((root/'runs'/p['name']/'launcher_record.json').exists() and
               read(root/'runs'/p['name']/'launcher_record.json')['status']=='completed' for p in expected):return False
    for phase in expected:
        validate_training(root/'runs'/phase['name'],phase['start'],phase['start']+150,phase['mode'])
    atomic(root/'training_completion_audit.json',{'status':'passed','runs':len(expected),
        'scope_sha256':scope['scope_sha256'],
        'controller_error_preserved':state.get('error'),'checked_unix':time.time(),
        'reason':'All fixed training batches, finite gradients and required checkpoints independently verified'})
    return True


def jobs(root):
    out=[];scope=load_scope(root)
    for seed in scope['training_seeds']:
        for task in scope['tasks']:
            for start in scope['states']:
                out.append({'name':f'diagnostic-{task}-s{seed}-state{start}','task':task,'seed':seed,'start':start,
                    'mode':'reference','samples':16,'suites':['diagnostic'],
                    'checkpoint':str(root/'runs'/f'{task}-s{seed}-reference'/'checkpoints'/f'global_step_{start}')})
    for phase in phases(root):
        if phase['resume'] is None:continue
        out.append({**phase,'samples':32,'suites':['test','math500','diagnostic'] if phase['task']=='gsm8k' else ['test','diagnostic'],
                    'checkpoint':str(root/'runs'/phase['name']/'checkpoints'/f'global_step_{phase["start"]+150}')})
    assert len(out)==len(scope['tasks'])*len(scope['training_seeds'])*len(scope['states'])*(1+len(scope['modes']))
    return out


def validate(root,job,folder,rescore=False):
    expected={s:{e['index']:e for e in entries_for(root,job['task'],s)} for s in job['suites']};seen=set()
    for rank in range(4):
        assert read(folder/f'rank_{rank}.done.json')['status']=='completed'
        with (folder/f'rank_{rank}.jsonl').open() as f:
            for line in f:
                x=json.loads(line);key=(x['suite'],x['index'],x['sample']);assert key not in seen
                row=expected[x['suite']][x['index']]
                assert x['question_sha256']==row['question_sha256'] and x['index']%4==rank
                samples,seed_start=suite_settings(job['samples'],x['suite'])
                assert x['sampling_seed']==seed_start+x['sample'] and 0<=x['sample']<samples
                assert x['score'] in (0.,1.) and len(x['token_ids'])==x['response_tokens']
                if rescore:assert score_answer(x['predicted_answer'],row)==x['score']
                seen.add(key)
    assert len(seen)==sum(len(v)*suite_settings(job['samples'],s)[0] for s,v in expected.items())
    return len(seen)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--retry-failed',action='store_true');a=parser.parse_args()
    root=Path(os.environ['REPRO_ROOT']);lock=(root/'finalizer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    me=identity(os.getpid());atomic(root/'finalizer_identity.json',me);(root/'finalizer.pid').write_text(str(os.getpid())+'\n')
    scope=load_scope(root)
    active={};mutex=threading.Lock();stop=threading.Event();failure_halt=threading.Event();finished=[];errors=[]
    def status(stage,**extra):
        with mutex:atomic(root/'evaluation_state.json',{'stage':stage,'identity':me,'updated_unix':time.time(),
                    'completed':list(finished),'active':dict(active),'errors':list(errors),
                    'scope_sha256':scope['scope_sha256'],**extra})
    def signal_stop(signum,frame):stop.set()
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    while True:
        if stop.is_set():status('stopped');return
        assert load_scope(root)['scope_sha256']==scope['scope_sha256'], 'Evaluation scope changed'
        training=training_snapshot(root/'study_state.json',stop,status)
        if training is None:status('stopped');return
        if training['status']=='stopped':status('stopped');return
        if training_finished(root,training):break
        status('waiting_training',training_status=training['status']);stop.wait(30)
    all_jobs=jobs(root)
    plan=read(root/'study_plan.json')
    for task in scope['tasks']:
        for split,record in plan['data'][task]['splits'].items():
            assert hashlib.sha256((root/'data'/task/(split+'.jsonl')).read_bytes()).hexdigest()==record['sha256']
    source=read(root/'source_data/source_manifest.json')
    for record in source['files']:
        assert hashlib.sha256((root/'source_data'/record['file']).read_bytes()).hexdigest()==record['sha256']
    # Grader acceptance is done before model generation, never silently substitute a broken parser.
    for row in entries_for(root,'gsm8k','math500'):assert score_answer(row['answer'],row)==1.0
    atomic(root/'evaluation_plan.json',{'jobs':all_jobs,'source':str(REPO),'scope_sha256':scope['scope_sha256'],
                                     'plan_sha256':hashlib.sha256((root/'study_plan.json').read_bytes()).hexdigest()})

    def execute(job,slot):
        if stop.is_set():return
        folder=root/'evaluations'/job['name'];record_path=folder/'launcher_record.json'
        if folder.exists():
            old=read(record_path)
            if old['status']=='completed':validate(root,job,folder);finished.append(job['name']);return
            if old['status']!='failed' or not a.retry_failed:raise RuntimeError('Unsettled existing evaluation: '+job['name'])
            folder.rename(folder.with_name(folder.name+'.failed-'+str(time.time_ns())))
        checkpoint=Path(job['checkpoint']);model=checkpoint/'actor/huggingface'
        if not list(model.glob('*.safetensors')):
            with (checkpoint/'merge.log').open('w') as log:
                subprocess.run([sys.executable,str(REPO/'verl/scripts/model_merger.py'),'--local_dir',str(checkpoint/'actor')],
                               env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
        lane=(root/'locks'/('lane-'+SLOTS[slot].split(',')[0]+'.lock')).open('a')
        fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
        used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(x.split(',')[0]):int(x.split(',')[1]) for x in used.splitlines()};gpu_ids=[int(x) for x in SLOTS[slot].split(',')]
        assert all(memory[g]<2048 for g in gpu_ids),'Evaluation lane is occupied'
        folder.mkdir(parents=True);record={'status':'running','n_gpus':4,'started_unix':time.time(),'job':job,
                    'gpu_ids':gpu_ids,'source_path':str(REPO),'workers':[]};atomic(record_path,record)
        children=[];logs=[];identities=[]
        with mutex:active[slot]=job['name']
        status('evaluating')
        try:
            prompt_cap=read(root/'data'/job['task']/'prompt_audit.json')['max_prompt_length']
            for rank,gpu in enumerate(gpu_ids):
                log=(folder/f'rank_{rank}.log').open('w');logs.append(log)
                args=[sys.executable,'-m','state_guidance.eval_worker','--model',str(model),'--task',job['task'],
                      '--output',str(folder),'--rank',str(rank),'--samples',str(job['samples']),
                      '--prompt-cap',str(prompt_cap),'--suites',*job['suites']]
                child=subprocess.Popen(args,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child);identities.append(identity(child.pid));record['workers']=identities;atomic(record_path,record)
            while any(c.poll() is None for c in children):
                if stop.is_set() or any(c.poll() not in (None,0) for c in children):raise RuntimeError('Evaluation stopped or worker failed')
                time.sleep(5)
            assert all(c.returncode==0 for c in children)
            record.update(status='completed',responses=validate(root,job,folder))
        except BaseException as exc:
            record.update(status='failed',error=repr(exc));errors.append(job['name']);failure_halt.set()
            raise
        finally:
            for saved in identities:
                if saved:stop_owned(saved)
            for child in children:
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
            for log in logs:log.close()
            record.update(finished_unix=time.time(),wall_seconds=time.time()-record['started_unix'])
            record['allocated_gpu_hours']=4*record['wall_seconds']/3600;atomic(record_path,record)
            with mutex:active.pop(slot,None)
            fcntl.flock(lane,fcntl.LOCK_UN);lane.close()
        finished.append(job['name']);status('evaluating',allocation=ledger(root))

    try:
        # Two fixed lane queues avoid oversubscription; every checkpoint is predetermined.
        def worker(slot,queue):
            for job in queue:
                if stop.is_set() or failure_halt.is_set():return
                execute(job,slot)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,slot,all_jobs[i::2]) for i,slot in enumerate(SLOTS)]
            for future in futures:future.result()
        if stop.is_set():status('stopped' if not errors else 'failed');return
        status('scoring_audit')
        for job in all_jobs:validate(root,job,root/'evaluations'/job['name'],rescore=True)
        status('analysis')
        subprocess.run([sys.executable,'-m','state_guidance.analyze'],cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        status('completed',allocation=ledger(root))
    except BaseException as exc:
        stop.set();status('failed',error=repr(exc));raise


if __name__=='__main__':main()
