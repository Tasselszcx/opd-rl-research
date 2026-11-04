"""Launch one G3 reference/fork in an exclusive four-GPU lane, with a durable ledger."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'reproduction'))
from run_phase import argv_for, atomic, hydra_value


def command(root,task,mode,seed,folder,resume=None,steps=150,check=False,continuation=None):
    restored=int(resume.name.split('global_step_')[-1]) if resume else 0
    start=continuation['original_start_step'] if continuation else restored
    end=start+steps
    if continuation:
        assert not check and steps==150 and start < restored < end
        assert continuation['checkpoint_step']==restored and continuation['original_end_step']==end
        assert (continuation['task'],continuation['mode'],continuation['seed'])==(task,mode,seed)
    args,cfg=argv_for('opd',folder,end,resume,False,n_gpus=4)
    data=root/'data'/task
    audit_path=data/'prompt_audit.json'
    prompt_limit=json.loads(audit_path.read_text())['max_prompt_length'] if audit_path.exists() else 4096
    values={'data.g3_frozen_task':task,'data.seed':seed+1000000 if start>0 else seed,
            'data.frozen_train_file':str(data/'train.jsonl'),
            'data.frozen_validation_file':str(data/'dev.jsonl'),
            'data.max_prompt_length':prompt_limit,'data.max_response_length':8192 if task=='gsm8k' else 4096,
            'data.cot_suffix':None if task=='gsm8k' else "Let's think step by step.\n",
            'reasoning_gym.val_dataset_size':8 if check else 512,
            'algorithm.teacher_guidance':mode,'algorithm.teacher_guidance_file':str(data/'guidance.jsonl'),
            'algorithm.guidance_compare_once':check and mode!='plain',
            'actor_rollout_ref.rollout.engine_seed':seed,
            'actor_rollout_ref.rollout.val_kwargs.n':1 if check else 8,
            'actor_rollout_ref.rollout.val_kwargs.seed':4200,
            'actor_rollout_ref.rollout.val_kwargs.temperature':0.7,
            'actor_rollout_ref.rollout.val_kwargs.top_p':0.8,
            'actor_rollout_ref.rollout.val_kwargs.top_k':20,
            'actor_rollout_ref.model.path':str(Path(os.environ.get('G3_MODEL_ROOT', str(REPO / '.model-cache')))/'Qwen3-1.7B-Base'),
            'actor_rollout_ref.ref.path':str(Path(os.environ.get('G3_MODEL_ROOT', str(REPO / '.model-cache')))/'Qwen3-8B'),
            'trainer.project_name':'g3-fixed-teacher-student-state',
            'trainer.total_epochs':1000,'trainer.reset_dataloader_on_fork':bool(resume) and not continuation,
            'trainer.del_local_ckpt_after_load':False,
            'trainer.save_freq':1 if check else 50,'trainer.test_freq':1 if check else 25,
            'trainer.extra_save_steps':[30,90] if start==0 and not check else [],
            'trainer.remove_previous_ckpt_in_save':False}
    values.update({'hydra.job.chdir':False,'hydra.run.dir':str(folder/'hydra')})
    amendment=root/'PROTOCOL_AMENDMENT.json'
    if amendment.exists():
        approved=json.loads(amendment.read_text())
        assert approved['status']=='approved_user_repair' and approved['method_revision']=='semantic_eos_v1'
        values.update({'algorithm.opd_eos_mode':'semantic',
                       'actor_rollout_ref.ref.eos_semantic_student_id':151643,
                       'actor_rollout_ref.ref.eos_semantic_teacher_ids':[151643,151645]})
    output_policy=root/'TEACHER_OUTPUT_POLICY.json'
    if output_policy.exists():
        policy=json.loads(output_policy.read_text())
        from state_guidance.teacher_inputs import KK_OUTPUT_CONTRACT
        assert task=='kk' and policy['status']=='validated_for_formal'
        assert policy['kind']=='kk_teacher_completion_contract_v1'
        assert policy['reminder_sha256']==hashlib.sha256(KK_OUTPUT_CONTRACT.encode()).hexdigest()
        # The same format requirement applies to every condition, including plain.
        # Only the non-plain conditions receive answers/reasoning references.
        values['algorithm.kk_teacher_output_contract']=True
    storage=root/'STORAGE_POLICY.json'
    if storage.exists() and not check and steps==150 and start in (0,30,90):
        policy=json.loads(storage.read_text())
        assert policy['status']=='approved_by_user' and policy['policy']=='fork_states_full_terminal_model_only'
        values.update({'trainer.save_freq':150,'trainer.g3_terminal_model_only':True})
    if continuation:
        assert continuation['data_seed']==values['data.seed']
        values.update({'trainer.g3_origin_step':start,'trainer.require_dataloader_resume':True,
                       'trainer.val_before_train':False})
    for key,value in values.items():
        args=[v for v in args if not v.startswith('++'+key+'=')]
        args.append('++'+key+'='+hydra_value(value));cfg[key]=value
    return args,cfg,start,end


def proc_identity(pid):
    stat=Path(f'/proc/{pid}/stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    return {'pid':pid,'start_ticks':int(tail[19]),'pgid':os.getpgid(pid)}


def validate(folder,start,end,mode):
    rows=[json.loads(x) for x in (folder/'checkpoints/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(start+1,end+1))
    assert all(len(r['prompt_indices'])==128 for r in rows)
    assert all(r['metrics']['reproduction/teacher_used_in_objective']==1 for r in rows)
    assert all(r['metrics']['guidance/response_alignment_ok']==1 for r in rows)
    assert all(math.isfinite(r['metrics']['actor/grad_norm']) for r in rows)
    assert all(r['metrics']['guidance/conditioned_sequences']==(0 if mode=='plain' else 1024) for r in rows)
    if start in (30,90) and (folder.parent.parent/'PROTOCOL_AMENDMENT.json').exists():
        assert all(r['metrics'].get('guidance/eos_semantic_enabled')==1 for r in rows)
        assert json.loads((folder/'request.json').read_text())['config']['algorithm.opd_eos_mode']=='semantic'
    final=folder/'checkpoints'/f'global_step_{end}'
    assert (final/'data.pt').is_file() and (final/'actor').is_dir()
    checkpoints=[final]
    if start==0 and end==150:
        checkpoints += [folder/'checkpoints'/f'global_step_{step}' for step in [30,90]]
    for checkpoint in checkpoints:
        assert (checkpoint/'data.pt').is_file()
        marker=checkpoint/'checkpoint_kind.json'
        model_only=marker.exists() and json.loads(marker.read_text()).get('kind')=='model_only'
        if model_only:
            assert checkpoint==final, 'Branch start states must retain optimizer and RNG'
            policy=json.loads((folder.parent.parent/'STORAGE_POLICY.json').read_text())
            assert policy['policy']=='fork_states_full_terminal_model_only' and policy['status']=='approved_by_user'
            assert (checkpoint/'actor/huggingface/config.json').is_file()
            assert json.loads((checkpoint/'actor/checkpoint_contents.json').read_text())['model_only'] is True
        for kind in (['model'] if model_only else ['model','optim','extra_state']):
            for rank in range(4):
                file=checkpoint/'actor'/f'{kind}_world_size_4_rank_{rank}.pt'
                assert file.is_file() and file.stat().st_size>0,str(file)
    return final,len(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['kk','zebra','countdown','gsm8k'])
    p.add_argument('--mode',required=True,choices=['plain','answer_only','full_reasoning','key_steps','error_targeted'])
    p.add_argument('--seed',type=int,required=True);p.add_argument('--name',required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--steps',type=int,default=150)
    p.add_argument('--gpu-ids',required=True);p.add_argument('--check',action='store_true')
    p.add_argument('--continue-migration',action='store_true')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    root=Path(os.environ['REPRO_ROOT']);folder=root/('checks' if a.check else 'runs')/a.name
    continuation=None
    if a.continue_migration:
        from state_guidance.migration import load_continuation
        assert not a.check
        continuation=load_continuation(folder,a.task,a.mode,a.seed)
        a.resume=folder/'checkpoints'/f'global_step_{continuation["checkpoint_step"]}'
    args,cfg,start,end=command(root,a.task,a.mode,a.seed,folder,a.resume,a.steps,a.check,continuation)
    if continuation:
        parent=json.loads((folder/'parent_request.json').read_text())['config']
        relocated={'data.frozen_train_file','data.frozen_validation_file','algorithm.teacher_guidance_file',
                   'actor_rollout_ref.model.path','actor_rollout_ref.ref.path','trainer.default_local_dir',
                   'trainer.resume_mode','trainer.reset_dataloader_on_fork','trainer.val_before_train','hydra.run.dir'}
        for key,value in parent.items():
            if key not in relocated:assert cfg.get(key)==value, 'Scientific configuration changed: '+key
    if a.dry_run:print(json.dumps({'argv':args,'config':cfg,'start':start,'end':end},indent=2));return
    assert a.steps==150 or a.check
    gpu_ids=[int(x) for x in a.gpu_ids.split(',')];assert gpu_ids in ([0,1,2,3],[4,5,6,7])
    (root/'locks').mkdir(exist_ok=True);lock=(root/'locks'/('lane-'+str(gpu_ids[0])+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'launcher_record.json').exists():
        old=json.loads((folder/'launcher_record.json').read_text())
        if not continuation or old['status']!='migration_ready':
            raise RuntimeError('Existing attempt retained; explicit retry required')
    manifest=json.loads((root/'data'/a.task/'manifest.json').read_text())
    assert (root/'data'/a.task/'prompt_audit.json').exists(), 'All-guidance input audit must precede reference training'
    for split in ['train','dev']:
        path=root/'data'/a.task/(split+'.jsonl')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['splits'][split]['sha256']
    if a.mode!='plain':
        guide=root/'data'/a.task/'guidance.jsonl';gm=json.loads(guide.with_name('guidance_manifest.json').read_text())
        assert hashlib.sha256(guide.read_bytes()).hexdigest()==gm['sha256']
        assert (guide.parent/'prompt_audit.json').exists()
    if a.resume:assert (a.resume/'data.pt').exists()
    assert (Path(os.environ['G3_MODEL_ROOT'])/'CACHE_MANIFEST.json').exists()
    # Never treat someone else's current GPU use as ours to terminate.
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    occupied={int(v.split(',')[0]):int(v.split(',')[1]) for v in query.splitlines()}
    assert all(occupied[g]<2048 for g in gpu_ids), 'Exclusive lane not free: '+str(occupied)
    folder.mkdir(parents=True,exist_ok=True)
    request={'argv':args,'config':cfg,'task':a.task,'mode':a.mode,'seed':a.seed,'gpu_ids':gpu_ids,
             'source_resume':str(a.resume) if a.resume else None,'source_path':str(REPO),
             'start_step':start,'additional_steps':a.steps,'end_step':end,'check':a.check}
    if continuation:request['migration']=continuation
    atomic(folder/'request.json',request)
    key=hashlib.sha256(str(folder).encode()).hexdigest()[:12]
    plasma_dir=Path('/tmp')/('g3-plasma-'+key)
    plasma_dir.mkdir(mode=0o700,exist_ok=True)
    assert plasma_dir.is_dir() and plasma_dir.stat().st_uid==os.getuid()
    assert shutil.disk_usage(plasma_dir).free>8*1024**3,'Private local Ray backing store lacks free space'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu_ids,G3_TRAIN_SEED=str(a.seed),
             REPRO_RAY_ISOLATE='1',REPRO_RAY_NUM_GPUS='4',REPRO_RAY_NUM_CPUS='32',
             REPRO_RAY_PLASMA_DIR=str(plasma_dir),
             RAY_TMPDIR='/tmp/g3-ray-'+key,TRITON_CACHE_DIR='/tmp/g3-cache-'+key+'/triton',
             TORCHINDUCTOR_CACHE_DIR='/tmp/g3-cache-'+key+'/inductor')
    record={'status':'starting','task':a.task,'mode':a.mode,'seed':a.seed,'started_unix':time.time(),
            'n_gpus':4,'gpu_ids':gpu_ids,'start_step':start,'end_step':end,'check':a.check,
            'launcher_identity':proc_identity(os.getpid()),'source_path':str(REPO)}
    if continuation:
        record.update(migration=continuation,scientific_attempt_started_unix=continuation['scientific_attempt_started_unix'],
                      resume_step=continuation['checkpoint_step'])
    atomic(folder/'launcher_record.json',record);child=None;error=None;rc=None
    def stop(signum,frame):raise RuntimeError('Interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with (folder/'train.log').open('w') as log:
            child=subprocess.Popen(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(status='running',pid=child.pid,child_identity=proc_identity(child.pid))
            atomic(folder/'launcher_record.json',record)
            rc=child.wait()
        if rc:raise RuntimeError('Training failed with exit '+str(rc))
        final,count=validate(folder,start,end,a.mode)
        record.update(status='completed',checkpoint=str(final),training_batches=count)
        if continuation:record['training_batches_this_segment']=end-continuation['checkpoint_step']
    except BaseException as exc:
        error=repr(exc);record.update(status='failed',error=error)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        record.update(return_code=rc,finished_unix=time.time(),wall_seconds=time.time()-record['started_unix'])
        record['allocated_gpu_hours']=4*record['wall_seconds']/3600
        atomic(folder/'launcher_record.json',record)
    print(json.dumps(record),flush=True)
    if error:raise SystemExit(1)


if __name__=='__main__':main()
