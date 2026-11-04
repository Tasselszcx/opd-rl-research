"""Run exactly three frozen guidance arms under a shared hard GPU-hour ceiling."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT=Path(os.environ['REPRO_ROOT'])
REPO=Path(__file__).resolve().parents[1]
MIGRATION=Path(os.environ['MIGRATION_ROOT'])
sys.path.insert(0,str(REPO/'reproduction'))
from run_phase import atomic
from coordinator import validate_phase
from guidance.costs import charged_gpu_hours
CAP=24.0
MODES=['plain','key_step','full_solution']

def read(p):return json.loads(p.read_text())

def charged():
    return charged_gpu_hours(ROOT,MIGRATION)

def status(stage, **extra):
    atomic(ROOT/'study_state.json',{'status':'running','stage':stage,'pid':os.getpid(),
        'updated_unix':time.time(),'gpu_hour_cap':CAP,'charged_gpu_hours_completed':charged(),**extra})

def free_gpus():
    for _ in range(36):
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(rows)==4 and all(int(x.split(',')[0])<1000 and int(x.split(',')[1])<=10 for x in rows):return
        time.sleep(5)
    raise RuntimeError('GPUs occupied; no unrelated process was terminated')

def wait_owned(child, timeout):
    try:
        return child.wait(timeout=timeout)
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def remaining_seconds():
    value=(CAP-charged())*3600/4
    if value<60:raise RuntimeError('GPU-hour budget exhausted; do not expand automatically')
    return value

def existing(folder, retry):
    p=folder/'launcher_record.json'
    if not p.exists():return False
    record=read(p)
    if record['status']=='completed':return True
    if record['status']!='failed' or not retry:
        raise RuntimeError('Existing non-completed attempt requires diagnosis and explicit --retry-failed: '+str(folder))
    folder.rename(folder.with_name(folder.name+'.failed-'+str(int(time.time()))))
    return False

def phase(mode, kind, name, end, resume=None, smoke=False, retry=False):
    folder=ROOT/('smoke' if smoke else 'runs')/name
    method='opd' if kind=='opd' else 'opd_then_rl'
    if existing(folder,retry):
        record=read(folder/'launcher_record.json')
        assert record['guidance_mode']==mode and record['phase']==kind
        return validate_phase(folder,method,end,resume)
    free_gpus();status(name)
    ceiling=min(remaining_seconds(),1800 if smoke else 4*3600)
    args=[sys.executable,str(REPO/'guidance/run_phase.py'),'--mode',mode,'--phase',kind,
          '--name',name,'--end-step',str(end),'--max-wall-seconds',str(ceiling)]
    if resume:args+=['--resume',str(resume)]
    if smoke:args+=['--smoke']
    child=subprocess.Popen(args,start_new_session=True)
    if wait_owned(child,ceiling+45)!=0:raise RuntimeError('Phase failed: '+name)
    return validate_phase(folder,method,end,resume)

def merge(checkpoint):
    output=checkpoint/'actor/huggingface'
    if list(output.glob('*.safetensors')):return output
    status('merge_'+checkpoint.parent.parent.name+'_'+checkpoint.name)
    with (checkpoint/'merge.log').open('w') as log:
        subprocess.run([sys.executable,str(REPO/'verl/scripts/model_merger.py'),'--local_dir',str(checkpoint/'actor')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    assert list(output.glob('*.safetensors'))
    return output

def evaluate(model, name, retry):
    folder=ROOT/'evaluations'/name
    if existing(folder,retry):return
    free_gpus();status('evaluate_'+name)
    ceiling=remaining_seconds();folder.mkdir(parents=True)
    start=time.time();record={'status':'running','started_unix':start,'n_gpus':4,
        'model':str(model),'questions':512,'samples':4,'max_wall_seconds':ceiling}
    atomic(folder/'launcher_record.json',record)
    children=[];logs=[];error=None
    try:
        for rank in range(4):
            log=(folder/f'rank_{rank}.log').open('w');logs.append(log)
            args=[sys.executable,str(REPO/'guidance/eval_worker.py'),'--model',str(model),
                  '--output',str(folder),'--rank',str(rank),'--world-size','4','--samples','4','--count','512','--suites','pilot_test']
            children.append(subprocess.Popen(args,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Evaluation worker failed')
            if time.time()-start>ceiling:raise TimeoutError('Study GPU-hour cap reached')
            time.sleep(5)
        assert all(c.returncode==0 for c in children)
        seen=set()
        for f in folder.glob('rank_*.jsonl'):
            for line in f.open():
                x=json.loads(line);key=(x['index'],x['sample']);assert key not in seen
                assert x['suite']=='pilot_test' and x['sampling_seed']==42+x['sample'] and x['score'] in (0,1)
                seen.add(key)
        assert seen=={(i,j) for i in range(512) for j in range(4)}
        record.update(status='completed',responses=len(seen))
    except BaseException as exc:
        error=exc;record.update(status='failed',error=repr(exc))
    finally:
        for c in children:
            if c.poll() is None:
                os.killpg(c.pid,signal.SIGTERM)
                try:c.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
        for log in logs:log.close()
        elapsed=time.time()-start;record.update(finished_unix=time.time(),wall_seconds=elapsed,allocated_gpu_hours=4*elapsed/3600)
        atomic(folder/'launcher_record.json',record)
    if error:raise error

def main():
    p=argparse.ArgumentParser();p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    def interrupted(signum,frame):raise RuntimeError('Coordinator interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (ROOT/'.pilot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (ROOT/'coordinator.pid').write_text(str(os.getpid())+'\n')
        try:
            wait_started=time.time()
            while True:
                migration=read(MIGRATION/'migration/training_smoke_state.json')
                if migration['status']=='passed':break
                if migration['status']!='running':raise RuntimeError('Migration training-chain validation failed; diagnose first')
                pid=migration['pid'];proc=Path('/proc')/str(pid)
                if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':
                    raise RuntimeError('Migration smoke state is stale; owner process is not alive')
                if time.time()-wait_started>2400:raise TimeoutError('Migration validation wait exceeded 40 minutes')
                status('waiting_migration_training_validation');time.sleep(10)
            data=read(ROOT/'data/manifest.json');assert data['zero_overlap']
            for name,record in data['datasets'].items():
                assert hashlib.sha256((ROOT/'data'/(name+'.jsonl')).read_bytes()).hexdigest()==record['sha256'], 'Frozen data changed: '+name
            plan={'modes':MODES,'opd_batches':20,'rl_batches':20,'batch':32,'rollouts':4,
                  'mini_batch':16,'prompt':1024,'response':2048,'lr':1e-6,'n_gpus':4,'gpu_hour_cap':CAP,
                  'test_questions':512,'test_samples':4,'data_manifest_sha256':hashlib.sha256((ROOT/'data/manifest.json').read_bytes()).hexdigest(),
                  'source_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()}
            if (ROOT/'study_plan.json').exists():
                original=read(ROOT/'study_plan.json')
                assert {k:v for k,v in original.items() if k!='source_commit'}=={k:v for k,v in plan.items() if k!='source_commit'}
            else:atomic(ROOT/'study_plan.json',plan)
            with (ROOT/'source_runs.jsonl').open('a') as history:
                history.write(json.dumps({'time':time.time(),'source_commit':plan['source_commit'],'retry_failed':a.retry_failed})+'\n')
            warm=phase('full_solution','opd','guidance-full-check',1,smoke=True,retry=a.retry_failed)
            phase('full_solution','rl','guidance-resume-check',2,warm,smoke=True,retry=a.retry_failed)
            checkpoints={}
            for mode in MODES:
                warm=phase(mode,'opd',mode+'-opd',20,retry=a.retry_failed)
                final=phase(mode,'rl',mode+'-rl',40,warm,retry=a.retry_failed)
                checkpoints[mode]={'opd':str(warm),'rl':str(final)}
            atomic(ROOT/'final_checkpoints.json',checkpoints)
            # Test scores are not opened until all three fixed paths complete.
            for mode in MODES:
                for endpoint in ['opd','rl']:
                    evaluate(merge(Path(checkpoints[mode][endpoint])),mode+'-'+endpoint,a.retry_failed)
            status('analysis')
            subprocess.run([sys.executable,str(REPO/'guidance/analyze.py')],check=True)
            status('completed',status='completed')
        except BaseException as exc:
            status('needs_attention',status='failed',error=repr(exc));raise

if __name__=='__main__':main()
