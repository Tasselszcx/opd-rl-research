"""User-authorized 12-update paired recovery experiment; never launches the formal queue."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
from state_guidance.control import atomic,read,identity,ledger
from state_guidance.run_phase import command,REPO

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['control','semantic'],required=True)
    ap.add_argument('--updates',type=int,choices=[12,36],default=12);a=ap.parse_args()
    root=Path(os.environ['REPRO_ROOT']);assert read(root/'study_state.json')['status']=='stopped'
    assert read(root/'TRAINING_HOLD.json')['status']=='paused_for_diagnosis'
    folder=root/'evaluations'/(f'diagnostic-eos{a.updates}-'+a.arm)
    assert not (folder/'launcher_record.json').exists(),'Never duplicate or overwrite an attempt'
    gpu='0,1,2,3' if a.arm=='control' else '4,5,6,7'
    lock=(root/'locks'/('lane-'+gpu[0]+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    mem=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(x.split(',')[0]):int(x.split(',')[1]) for x in mem.splitlines()}
    assert all(used[int(i)]<2048 for i in gpu.split(','))
    resume=root/'runs/kk-s42-state30-full_reasoning/checkpoints/global_step_180'
    args,cfg,start,end=command(root,'kk','full_reasoning',42,folder,resume,steps=a.updates)
    overrides={'data.train_batch_size':32,'data.val_batch_size':32,
        'reasoning_gym.val_dataset_size':128,'actor_rollout_ref.rollout.n':4,
        'actor_rollout_ref.rollout.val_kwargs.n':4,'trainer.test_freq':12,'trainer.save_freq':36 if a.updates==36 else 0,
        'trainer.reset_dataloader_on_fork':True,'trainer.val_before_train':True,
        'trainer.project_name':'g3-eos-recovery-diagnostic',
        'algorithm.opd_eos_mode':'semantic' if a.arm=='semantic' else 'legacy_mask'}
    if a.arm=='semantic':overrides.update({'actor_rollout_ref.ref.eos_semantic_student_id':151643,
                                         'actor_rollout_ref.ref.eos_semantic_teacher_ids':[151643,151645]})
    local_checkpoint=None
    if a.updates==36:
        local_checkpoint=Path('/tmp')/('g3-eos36-'+a.arm)/'checkpoints'
        assert not local_checkpoint.exists(), 'Preserve any previous local checkpoint attempt'
        overrides['trainer.default_local_dir']=str(local_checkpoint)
    for k,v in overrides.items():
        args=[x for x in args if not x.startswith('++'+k+'=')];args.append('++'+k+'='+json.dumps(v));cfg[k]=v
    folder.mkdir(parents=True)
    if local_checkpoint:
        local_checkpoint.mkdir(parents=True);(folder/'checkpoints').symlink_to(local_checkpoint,target_is_directory=True)
        (folder/'val_logs').symlink_to(local_checkpoint.parent/'val_logs',target_is_directory=True)
    atomic(folder/'request.json',{'argv':args,'config':cfg,'arm':a.arm,'source_resume':str(resume),
        'scientific_scope':f'Diagnostic only: same optimizer/RNG checkpoint, same reset sampler; {a.updates} updates, batch32 rollout4, dev128x4. No final test or formal queue.',
        'selection':'First 128 frozen dev questions chosen before running; all frozen training questions available.',
        'no_final_checkpoint':a.updates==12})
    key=hashlib.sha256(str(folder).encode()).hexdigest()[:12];plasma=Path('/tmp')/('g3-eos-plasma-'+key);plasma.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,G3_TRAIN_SEED='42',REPRO_RAY_ISOLATE='1',REPRO_RAY_NUM_GPUS='4',REPRO_RAY_NUM_CPUS='32',REPRO_RAY_PLASMA_DIR=str(plasma),RAY_TMPDIR='/tmp/g3-eos-ray-'+key,TRITON_CACHE_DIR='/tmp/g3-eos-cache-'+key+'/triton',TORCHINDUCTOR_CACHE_DIR='/tmp/g3-eos-cache-'+key+'/inductor')
    rec={'status':'starting','n_gpus':4,'started_unix':time.time(),'diagnostic_only':True,'arm':a.arm,'source_path':str(REPO),'identity':identity(os.getpid()),'end_step':end}
    atomic(folder/'launcher_record.json',rec);child=None
    def stop(signum,frame):raise RuntimeError('Diagnostic interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with (folder/'train.log').open('w') as log:
            child=subprocess.Popen(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rec.update(status='running',child_identity=identity(child.pid));atomic(folder/'launcher_record.json',rec)
            rc=child.wait(timeout=7200)
        assert rc==0,'Driver exit '+str(rc)
        rows=[json.loads(x) for x in (folder/'checkpoints/metrics.jsonl').read_text().splitlines()]
        assert [x['step'] for x in rows]==list(range(181,end+1))
        assert all(x['metrics']['guidance/conditioned_sequences']==128 for x in rows)
        if a.arm=='semantic':assert all(x['metrics']['guidance/eos_semantic_enabled']==1 for x in rows)
        if a.updates==36:
            from state_guidance.migration import checkpoint_files
            assert all(p.is_file() and p.stat().st_size>0 for p in checkpoint_files(folder/'checkpoints'/f'global_step_{end}'))
            from state_guidance.checkpoint_export import export_verified
            rec['phase']='copy_checkpoint_to_shared';atomic(folder/'launcher_record.json',rec)
            staging=folder/'checkpoint_export.partial'
            rec['exported_bytes']=export_verified(local_checkpoint,staging)
            (folder/'checkpoints').unlink();staging.rename(folder/'checkpoints')
            val_staging=folder/'validation_export.partial'
            export_verified(local_checkpoint.parent/'val_logs',val_staging)
            (folder/'val_logs').unlink();val_staging.rename(folder/'val_logs')
            rec['checkpoint']=str(folder/'checkpoints'/f'global_step_{end}')
        rec.update(status='completed',return_code=rc,training_batches=len(rows))
    except BaseException as e:rec.update(status='failed',error=repr(e))
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        rec.update(finished_unix=time.time());rec['allocated_gpu_hours']=4*(rec['finished_unix']-rec['started_unix'])/3600
        atomic(folder/'launcher_record.json',rec);atomic(root/'cost_ledger.json',ledger(root));print(json.dumps(rec),flush=True)
    if rec['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
