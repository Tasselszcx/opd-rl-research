"""One bounded, separately recorded phase of the teacher-guidance pilot."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'reproduction'))
from run_phase import argv_for, atomic
from coordinator import validate_phase
ROOT = Path(os.environ['REPRO_ROOT'])

def make_command(mode, phase, folder, end_step, resume, smoke):
    method = 'opd' if phase == 'opd' else 'opd_then_rl'
    args, cfg = argv_for(method, folder, end_step, resume, smoke, n_gpus=4)
    overrides = {'data.train_batch_size':32,'data.val_batch_size':32,
        'data.max_prompt_length':1024,'data.max_response_length':2048,'actor_rollout_ref.rollout.n':4,
        'actor_rollout_ref.actor.ppo_mini_batch_size':16,
        'reasoning_gym.val_dataset_size':8 if smoke else 128,
        'data.frozen_train_file':str(ROOT/'data/train.jsonl'),
        'data.frozen_validation_file':str(ROOT/'data/meta_val.jsonl'),
        'algorithm.teacher_guidance':mode if phase=='opd' else 'plain',
        'algorithm.teacher_guidance_file':str(ROOT/'data/guidance.jsonl'),
        'algorithm.guidance_compare_once':bool(smoke and phase=='opd' and mode!='plain'),
        'trainer.save_freq':1 if smoke else 20,'trainer.test_freq':1 if smoke else 20}
    for key,value in overrides.items():
        args=[x for x in args if not x.startswith('++'+key+'=')]
        args.append('++'+key+'='+json.dumps(value));cfg[key]=value
    return method,args,cfg

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=['plain','key_step','full_solution'])
    parser.add_argument('--phase',required=True,choices=['opd','rl'])
    parser.add_argument('--name',required=True)
    parser.add_argument('--end-step',type=int,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--max-wall-seconds',type=float,required=True)
    a=parser.parse_args()
    assert a.max_wall_seconds>0
    assert (a.phase=='rl') == (a.resume is not None)
    folder=ROOT/('smoke' if a.smoke else 'runs')/a.name
    method,args,cfg=make_command(a.mode,a.phase,folder,a.end_step,a.resume,a.smoke)
    if a.dry_run:
        print(json.dumps({'argv':args,'config':cfg},indent=2));return
    if (folder/'launcher_record.json').exists():raise RuntimeError('Existing attempt retained; explicit diagnosis/archive required')
    if a.resume:assert (a.resume/'data.pt').is_file()
    for f in ['train.jsonl','meta_val.jsonl','guidance.jsonl']:assert (ROOT/'data'/f).is_file()
    folder.mkdir(parents=True,exist_ok=True)
    atomic(folder/'request.json',{'argv':args,'config':cfg,'method':method,'guidance_mode':a.mode,
        'phase':a.phase,'source_resume':str(a.resume) if a.resume else None,'gpu_ids':[0,1,2,3],
        'smoke':a.smoke,'max_wall_seconds':a.max_wall_seconds,
        'source_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()})
    key=hashlib.sha256(str(folder).encode()).hexdigest()[:10]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',REPRO_RAY_ISOLATE='1',
        REPRO_RAY_NUM_GPUS='4',REPRO_RAY_NUM_CPUS='32',RAY_TMPDIR='/tmp/kkg-'+key,
        TRITON_CACHE_DIR='/tmp/kkg-cache-'+key+'/triton',TORCHINDUCTOR_CACHE_DIR='/tmp/kkg-cache-'+key+'/inductor')
    for k in ['RAY_ADDRESS','RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(k,None)
    start=time.time();record={'status':'running','started_unix':start,'n_gpus':4,'method':method,
        'end_step':a.end_step,'guidance_mode':a.mode,'phase':a.phase,'smoke':a.smoke,'gpu_ids':[0,1,2,3]}
    child=None;error=None;rc=None
    def interrupted(signum,frame):raise RuntimeError('Interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        with (folder/'train.log').open('w') as log:
            child=subprocess.Popen(args,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=child.pid;atomic(folder/'launcher_record.json',record)
            rc=child.wait(timeout=a.max_wall_seconds)
            if rc:raise RuntimeError('Training exit code '+str(rc))
    except BaseException as exc:error=repr(exc)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        metrics=folder/'checkpoints/metrics.jsonl'
        rows=[json.loads(x) for x in metrics.read_text().splitlines()] if metrics.exists() else []
        checkpoint=folder/'checkpoints'/('global_step_'+str(a.end_step))
        passed=error is None and rc==0 and rows and rows[-1]['step']==a.end_step
        elapsed=time.time()-start
        record.update(status='completed' if passed else 'failed',return_code=rc,error=error,
            finished_unix=time.time(),wall_seconds=elapsed,allocated_gpu_hours=4*elapsed/3600,
            first_logged_step=rows[0]['step'] if rows else None,last_logged_step=rows[-1]['step'] if rows else None,
            training_batches=len(rows),checkpoint=str(checkpoint) if passed else None)
        atomic(folder/'launcher_record.json',record)
    if not passed:raise RuntimeError('Phase failed; logs and costs retained: '+str(folder))
    validate_phase(folder,method,a.end_step,a.resume)
    try:
        for row in rows:
            m=row['metrics'];assert m['guidance/response_alignment_ok']==1
            assert m['reproduction/teacher_used_in_objective']==(1 if a.phase=='opd' else 0)
            assert m['guidance/conditioned_sequences']==(128 if a.phase=='opd' and a.mode!='plain' else 0)
            if a.smoke and a.phase=='opd' and a.mode!='plain':
                assert m['guidance/ref_logprob_delta_mean_abs']>0
    except Exception as exc:
        record.update(status='failed',validation_status='failed',failure_kind='guidance_validation',error=repr(exc))
        atomic(folder/'launcher_record.json',record)
        raise
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
