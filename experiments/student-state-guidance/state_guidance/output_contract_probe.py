"""Bounded teacher-format ablation, never starts or resumes the formal queue."""
import argparse,fcntl,hashlib,json,math,os,signal,subprocess,time
from pathlib import Path
from state_guidance.control import atomic,read,identity,ledger
from state_guidance.run_phase import command,REPO,hydra_value


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['control','contract'],required=True);a=ap.parse_args()
    root=Path(os.environ['REPRO_ROOT'])
    assert read(root/'study_state.json')['status']=='stopped'
    assert read(root/'TRAINING_HOLD.json')['status']=='paused_for_debug'
    folder=root/'evaluations'/('diagnostic-output-contract12-'+a.arm)
    assert not folder.exists(),'Never overwrite an attempt'
    gpu='0,1,2,3' if a.arm=='control' else '4,5,6,7'
    lock=(root/'locks'/('lane-'+gpu[0]+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    used={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
    assert all(used[int(i)]<2048 for i in gpu.split(','))
    resume=root/'runs/kk-s42-reference/checkpoints/global_step_30';assert (resume/'data.pt').is_file()
    args,cfg,start,end=command(root,'kk','answer_only',42,folder,resume,steps=12)
    overrides={'data.train_batch_size':32,'data.val_batch_size':32,'reasoning_gym.val_dataset_size':128,
        'actor_rollout_ref.rollout.n':4,'actor_rollout_ref.rollout.val_kwargs.n':4,
        'trainer.test_freq':12,'trainer.save_freq':0,'trainer.extra_save_steps':[],
        'trainer.reset_dataloader_on_fork':True,'trainer.val_before_train':True,
        'trainer.project_name':'g3-output-contract-diagnostic',
        'algorithm.kk_teacher_output_contract':a.arm=='contract'}
    for k,v in overrides.items():
        args=[x for x in args if not x.startswith('++'+k+'=')];args.append('++'+k+'='+hydra_value(v));cfg[k]=v
    assert cfg['algorithm.opd_eos_mode']=='semantic' and cfg['data.cot_suffix'].endswith('\n')
    folder.mkdir(parents=True)
    atomic(folder/'request.json',{'argv':args,'config':cfg,'arm':a.arm,'source_resume':str(resume),
        'source_path':str(REPO),'additional_steps':12,'start_step':30,'end_step':42,
        'scientific_scope':'Diagnostic only. Same state30 optimizer/RNG, sampler, real-newline input, answer hint, semantic EOS; only teacher output-format reminder differs. Batch32 rollout4, first128 frozen dev questions x4 before/after. No checkpoint or formal final test.',
        'official_scorer_unchanged':True,'automatic_formal_resume':False})
    key=hashlib.sha256(str(folder).encode()).hexdigest()[:12]
    plasma=Path('/tmp')/('g3-format-plasma-'+key);plasma.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,G3_TRAIN_SEED='42',REPRO_RAY_ISOLATE='1',REPRO_RAY_NUM_GPUS='4',REPRO_RAY_NUM_CPUS='32',
        REPRO_RAY_PLASMA_DIR=str(plasma),RAY_TMPDIR='/tmp/g3-format-ray-'+key,
        TRITON_CACHE_DIR='/tmp/g3-format-cache-'+key+'/triton',TORCHINDUCTOR_CACHE_DIR='/tmp/g3-format-cache-'+key+'/inductor')
    rec={'status':'starting','n_gpus':4,'started_unix':time.time(),'diagnostic_only':True,'arm':a.arm,
         'identity':identity(os.getpid()),'source_path':str(REPO),'start_step':30,'end_step':42}
    record=folder/'launcher_record.json';atomic(record,rec);child=None
    def stop(signum,frame):raise RuntimeError('Diagnostic interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with (folder/'train.log').open('w') as log:
            child=subprocess.Popen(args,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rec.update(status='running',child_identity=identity(child.pid));atomic(record,rec)
            rc=child.wait(timeout=3600)
        assert rc==0,'Driver exit '+str(rc)
        rows=[json.loads(x) for x in (folder/'checkpoints/metrics.jsonl').read_text().splitlines()]
        assert [x['step'] for x in rows]==list(range(31,43))
        for row in rows:
            m=row['metrics'];assert len(row['prompt_indices'])==32
            assert math.isfinite(m['actor/grad_norm']) and m['guidance/response_alignment_ok']==1
            assert m['guidance/conditioned_sequences']==128 and m['guidance/eos_semantic_enabled']==1
            assert m['guidance/output_contract_sequences']==(128 if a.arm=='contract' else 0)
        vals=[read(p) for p in (folder/'val_logs').rglob('metrics.json')]
        assert sorted(x['step'] for x in vals)==[30,42]
        rec.update(status='completed',return_code=0,training_batches=12)
    except BaseException as exc:
        rec.update(status='failed',error=repr(exc))
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        rec['finished_unix']=time.time();rec['allocated_gpu_hours']=4*(rec['finished_unix']-rec['started_unix'])/3600
        atomic(record,rec);atomic(root/'cost_ledger.json',ledger(root));print(json.dumps(rec),flush=True)
    if rec['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
