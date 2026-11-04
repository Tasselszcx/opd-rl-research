"""One bounded paper-config training phase, with durable launch/cost records."""
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
ROOT = Path(os.environ.get('REPRO_ROOT', str(REPO / '.local-runs' / 'sequential')))
MODEL_ROOT = Path(os.environ.get('G3_MODEL_ROOT', str(REPO / '.model-cache')))


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def argv_for(method, folder, end_step, resume=None, smoke=False, n_gpus=8):
    adv = {'grpo': 'grpo', 'opd': 'on_policy_distillation', 'opd_then_rl': 'grpo',
           'kdrl_anneal': 'grpo_opd'}[method]
    cfg = {
        'data.train_batch_size': 128, 'data.val_batch_size': 128,
        'data.max_prompt_length': 4096, 'data.max_response_length': 4096,
        'data.enable_thinking': False,
        'data.seed': 1, 'reasoning_gym.dataset_size': 20000,
        'reasoning_gym.val_dataset_size': 8 if smoke else 1000,
        'algorithm.adv_estimator': adv,
        'actor_rollout_ref.model.path': str(MODEL_ROOT / 'Qwen3-1.7B-Base'),
        'actor_rollout_ref.ref.path': str(MODEL_ROOT / 'Qwen3-8B'),
        'actor_rollout_ref.model.use_remove_padding': False,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.optim.lr': 1e-6,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 16,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.use_kl_loss': True,
        'actor_rollout_ref.actor.kl_loss_coef': 0.0,
        'actor_rollout_ref.actor.entropy_coeff': 0.0,
        'actor_rollout_ref.actor.grad_clip': 1.0,
        'actor_rollout_ref.actor.clip_ratio': 0.2,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.max_num_batched_tokens': 16384,
        'actor_rollout_ref.rollout.n': 8,
        'actor_rollout_ref.rollout.temperature': 1.0,
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.6,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': 1,
        # Online development validation retains the published code's seed-2
        # split and one-sample settings. Final evaluation is a separate job.
        'actor_rollout_ref.rollout.val_kwargs.n': 1,
        'actor_rollout_ref.rollout.val_kwargs.do_sample': True,
        'trainer.n_gpus_per_node': n_gpus, 'trainer.nnodes': 1,
        'trainer.project_name': 'sequential-kk-reproduction',
        'trainer.experiment_name': folder.name,
        'trainer.default_local_dir': str(folder / 'checkpoints'),
        'trainer.logger': ['console'], 'trainer.val_before_train': True,
        'trainer.resume_mode': str(resume) if resume else 'disable',
        'trainer.total_epochs': 2, 'trainer.total_training_steps': end_step,
        'trainer.save_freq': 1 if smoke else 5,
        'trainer.test_freq': 1 if smoke else 5,
        'trainer.remove_previous_ckpt_in_save': False,
    }
    if method == 'kdrl_anneal':
        cfg.update({'algorithm.opd_beta_init': 0.2, 'algorithm.opd_beta_min': 0.002,
                    'algorithm.opd_beta_delta': 0.0016})
    args = [sys.executable, '-u', str(REPO / 'logic/training/train_grpo.py'),
            '--config-path', str(REPO / 'logic/training/configs/gkd_configs'), '--config-name', 'kk']
    args += ['++' + k + '=' + json.dumps(v, ensure_ascii=False) for k, v in cfg.items()]
    return args, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['grpo', 'opd', 'opd_then_rl', 'kdrl_anneal'])
    p.add_argument('--name', required=True)
    p.add_argument('--end-step', required=True, type=int)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--compat-models', action='store_true', help='Smoke only: existing V1 models for runtime debugging')
    p.add_argument('--gpu-ids', help='Exclusive physical GPU IDs, e.g. 0,1,2,3; starts an isolated local Ray cluster')
    a = p.parse_args()
    gpu_ids = [int(x) for x in a.gpu_ids.split(',')] if a.gpu_ids else list(range(8))
    assert gpu_ids and len(set(gpu_ids)) == len(gpu_ids)
    assert all(0 <= x < 8 for x in gpu_ids)
    n_gpus = len(gpu_ids)
    assert 16 % n_gpus == 0, 'Global mini-batch must divide evenly across GPUs'
    folder = ROOT / ('smoke' if a.smoke else 'runs') / a.name
    args, cfg = argv_for(a.method, folder, a.end_step, a.resume, a.smoke, n_gpus=n_gpus)
    child_env = dict(os.environ)
    if a.gpu_ids:
        slot_key = hashlib.sha256(str(folder).encode()).hexdigest()[:10]
        child_env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpu_ids)),
                         REPRO_RAY_ISOLATE='1', REPRO_RAY_NUM_GPUS=str(n_gpus),
                         REPRO_RAY_NUM_CPUS='32', RAY_TMPDIR='/tmp/kk4-' + slot_key,
                         TRITON_CACHE_DIR='/tmp/kk4-cache-' + slot_key + '/triton',
                         TORCHINDUCTOR_CACHE_DIR='/tmp/kk4-cache-' + slot_key + '/inductor')
        for key in ['RAY_ADDRESS', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
            child_env.pop(key, None)
    if a.compat_models:
        assert a.smoke, 'Compatibility models cannot be used for the paper experiment'
        replacements = {
            'actor_rollout_ref.model.path': '/tmp/opd-meta-teaching-v1-models/Qwen3-1.7B',
            'actor_rollout_ref.ref.path': '/tmp/opd-meta-teaching-v1-models/Qwen3-4B',
        }
        for key, value in replacements.items():
            args = [x for x in args if not x.startswith('++' + key + '=')]
            args.append('++' + key + '=' + json.dumps(value)); cfg[key] = value
    if a.dry_run:
        print(json.dumps({'argv': args, 'config': cfg}, indent=2)); return
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / 'launcher_record.json'
    if record_path.exists():
        raise RuntimeError('Refusing duplicate launch: ' + str(folder))
    if a.resume:
        assert a.resume.is_dir() and (a.resume / 'data.pt').is_file(), 'Missing checkpoint/dataloader state'
    for key in ['actor_rollout_ref.model.path', 'actor_rollout_ref.ref.path']:
        model = Path(cfg[key])
        assert (model / 'config.json').is_file()
        assert list(model.glob('*.safetensors'))
    atomic(folder / 'request.json', {'argv': args, 'config': cfg, 'method': a.method,
                                   'source_resume': str(a.resume) if a.resume else None,
                                   'smoke': a.smoke, 'compatibility_models_only': a.compat_models,
                                   'gpu_ids': gpu_ids, 'isolated_ray': bool(a.gpu_ids)})
    start = time.time()
    record = {'status': 'starting', 'started_unix': start, 'n_gpus': n_gpus, 'method': a.method,
              'end_step': a.end_step, 'smoke': a.smoke, 'gpu_ids': gpu_ids}
    atomic(record_path, record)
    timeout = 3600 if a.smoke else 24 * 3600
    received_signal = None
    with (folder / 'train.log').open('w') as log:
        child = subprocess.Popen(args, cwd=folder, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(status='running', pid=child.pid); atomic(record_path, record)
        def stop_owned_child(signum, frame):
            nonlocal received_signal
            received_signal = signum
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        signal.signal(signal.SIGTERM, stop_owned_child)
        signal.signal(signal.SIGINT, stop_owned_child)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            code = 124
    elapsed = time.time() - start
    metrics_file = folder / 'checkpoints/metrics.jsonl'
    rows = [json.loads(line) for line in metrics_file.read_text().splitlines()] if metrics_file.exists() else []
    final = folder / 'checkpoints' / f'global_step_{a.end_step}'
    passed = received_signal is None and code == 0 and bool(rows) and rows[-1]['step'] == a.end_step and (final / 'data.pt').exists()
    record.update(status='completed' if passed else 'failed', return_code=code,
                  finished_unix=time.time(), wall_seconds=elapsed, allocated_gpu_hours=n_gpus * elapsed / 3600,
                  first_logged_step=rows[0]['step'] if rows else None,
                  last_logged_step=rows[-1]['step'] if rows else None,
                  training_batches=len(rows), checkpoint=str(final) if passed else None)
    if received_signal is not None:
        record.update(status='failed', interrupted_signal=received_signal,
                      failure_kind='launcher_interrupted')
    atomic(record_path, record)
    print(json.dumps(record, indent=2))
    if not passed: raise SystemExit(1)


if __name__ == '__main__':
    main()
