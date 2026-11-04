"""Run only the authorized four-arm K&K reproduction and its fixed evaluations."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback

from run_phase import ROOT, REPO, MODEL_ROOT, atomic
ALLOW_RETRY = False


def read(path):
    return json.loads(path.read_text())


def archive_failed(folder):
    record = read(folder / 'launcher_record.json')
    if not ALLOW_RETRY or record['status'] != 'failed':
        raise RuntimeError('Review existing attempt; explicit --retry-failed required: ' + str(folder))
    target = folder.with_name(folder.name + '.failed-' + str(int(time.time())))
    assert not target.exists()
    folder.rename(target)


def costs():
    return sum(read(p).get('allocated_gpu_hours', 0) for p in ROOT.glob('**/launcher_record.json'))


def status(stage, **extra):
    data = {'status': 'running', 'stage': stage, 'pid': os.getpid(),
            'updated_unix': time.time(), 'charged_gpu_hours_completed_jobs': costs(), **extra}
    atomic(ROOT / 'study_state.json', data)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def free_gpus():
    for _ in range(60):
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(lines) == 8 and all(int(v.strip()) < 100 for v in lines):
            return
        time.sleep(5)
    raise RuntimeError('GPUs are occupied; no processes were terminated')


def verify_models():
    for name in ['Qwen3-1.7B-Base', 'Qwen3-8B']:
        manifest = read(ROOT / 'models' / (name + '-manifest.json'))
        assert manifest['repo_id'] == 'Qwen/' + name
        dest = MODEL_ROOT / name
        dest.mkdir(parents=True, exist_ok=True)
        for filename, expected in manifest['files'].items():
            source = ROOT / 'models' / name / filename
            path = dest / filename
            if not path.exists() or path.stat().st_size != expected['bytes']:
                import shutil
                temporary = path.with_suffix(path.suffix + '.copying')
                shutil.copyfile(source, temporary)
                temporary.replace(path)
            assert path.stat().st_size == expected['bytes'], path
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
                    h.update(block)
            assert h.hexdigest() == expected['sha256'], path
    atomic(ROOT / 'model_staging.json', {'status': 'verified', 'directory': str(MODEL_ROOT), 'time': time.time()})


def validate_phase(folder, method, end_step, resume=None):
    """Validate every completed phase, including cached results after restart."""
    record_path = folder / 'launcher_record.json'
    record = read(record_path)

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        require(record['status'] == 'completed', 'Launcher did not complete')
        require(record['return_code'] == 0, 'Training process returned a nonzero code')
        require(record['method'] == method, 'Cached method differs from requested method')
        require(record['end_step'] == end_step, 'Cached terminal step differs from requested step')
        expected_first = int(resume.name.split('_')[-1]) + 1 if resume else 1
        expected_steps = list(range(expected_first, end_step + 1))
        require(bool(expected_steps), 'Phase has no requested training batches')
        require(record['first_logged_step'] == expected_first, 'Incorrect initial training step')
        require(record['last_logged_step'] == end_step, 'Incorrect final training step')
        require(record['training_batches'] == len(expected_steps), 'Incorrect training batch count')
        metrics_file = folder / 'checkpoints/metrics.jsonl'
        rows = [json.loads(line) for line in metrics_file.read_text().splitlines()]
        require([row['step'] for row in rows] == expected_steps, 'Noncontiguous or duplicated training steps')
        for row in rows:
            grad = row['metrics']['actor/grad_norm']
            require(math.isfinite(grad), f"Nonfinite gradient at step {row['step']}")
        checkpoint = folder / 'checkpoints' / f'global_step_{end_step}'
        require(Path(record['checkpoint']).resolve() == checkpoint.resolve(), 'Unexpected checkpoint path')
        n_gpus = record['n_gpus']
        require(isinstance(n_gpus, int) and n_gpus > 0, 'Invalid checkpoint world size')
        required = [checkpoint / 'data.pt']
        required += [checkpoint / 'actor' / f'{kind}_world_size_{n_gpus}_rank_{rank}.pt'
                     for kind in ['model', 'optim', 'extra_state'] for rank in range(n_gpus)]
        for path in required:
            require(path.is_file() and path.stat().st_size > 0, 'Missing or empty checkpoint file: ' + str(path))
    except Exception as exc:
        record.update(status='failed', failure_kind='phase_validation',
                      validation_status='failed', validation_error=type(exc).__name__ + ': ' + str(exc),
                      validated_unix=time.time())
        atomic(record_path, record)
        raise RuntimeError('Phase validation failed: ' + str(folder) + ': ' + str(exc)) from exc
    record.update(validation_status='passed', validated_unix=time.time())
    atomic(record_path, record)
    return checkpoint


def phase(method, name, end_step, resume=None, smoke=False):
    folder = ROOT / ('smoke' if smoke else 'runs') / name
    record_path = folder / 'launcher_record.json'
    if record_path.exists():
        record = read(record_path)
        if record['status'] == 'completed':
            return validate_phase(folder, method, end_step, resume)
        archive_failed(folder)
    free_gpus()
    status(name)
    args = [sys.executable, str(REPO / 'reproduction/run_phase.py'), '--method', method,
            '--name', name, '--end-step', str(end_step)]
    if smoke:
        args.append('--smoke')
    if resume:
        args += ['--resume', str(resume)]
    subprocess.run(args, check=True)
    return validate_phase(folder, method, end_step, resume)


def merge(checkpoint):
    target = checkpoint / 'actor/huggingface'
    if list(target.glob('*.safetensors')):
        return target
    status('merge_' + checkpoint.parent.parent.name)
    with (checkpoint / 'merge.log').open('w') as log:
        subprocess.run([sys.executable, str(REPO / 'verl/scripts/model_merger.py'),
                        '--local_dir', str(checkpoint / 'actor')], stdout=log, stderr=subprocess.STDOUT, check=True)
    assert list(target.glob('*.safetensors'))
    return target


def evaluate(model, name, samples=32, count=512, suites=None):
    suites = suites or ['author_test', 'unseen_test']
    folder = ROOT / 'evaluations' / name
    record_path = folder / 'launcher_record.json'
    if record_path.exists():
        if read(record_path)['status'] == 'completed':
            return
        archive_failed(folder)
    free_gpus()
    status('evaluate_' + name)
    folder.mkdir(parents=True, exist_ok=True)
    start = time.time()
    record = {'status': 'running', 'started_unix': start, 'n_gpus': 8, 'model': str(model),
              'samples': samples, 'question_count_per_suite': count, 'suites': suites}
    atomic(record_path, record)
    children, logs = [], []
    for rank in range(8):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank))
        # Standalone evaluation is independent of the training Ray cluster.
        for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
            env.pop(key, None)
        args = [sys.executable, str(REPO / 'reproduction/eval_worker.py'), '--model', str(model),
                '--output', str(folder), '--rank', str(rank), '--samples', str(samples),
                '--count', str(count), '--suites', *suites]
        log = (folder / f'rank_{rank}.log').open('w'); logs.append(log)
        children.append(subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True))
    deadline = start + 8 * 3600
    while any(child.poll() is None for child in children):
        failure = any(child.poll() not in (None, 0) for child in children)
        if failure or time.time() > deadline:
            # Only the process groups created by this evaluation are stopped.
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for child in children:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            break
        time.sleep(5)
    codes = [child.wait() for child in children]
    for log in logs:
        log.close()
    rows = sum(len(p.read_text().splitlines()) for p in folder.glob('rank_*.jsonl'))
    passed = all(code == 0 for code in codes) and rows == len(suites) * samples * count
    record.update(status='completed' if passed else 'failed', return_codes=codes,
                  finished_unix=time.time(), wall_seconds=time.time() - start,
                  allocated_gpu_hours=8 * (time.time() - start) / 3600, rows=rows)
    atomic(record_path, record)
    if not passed:
        raise RuntimeError('Evaluation failed: ' + name)


def main():
    global ALLOW_RETRY
    parser = argparse.ArgumentParser()
    parser.add_argument('--retry-failed', action='store_true')
    ALLOW_RETRY = parser.parse_args().retry_failed
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT / 'coordinator.pid').write_text(str(os.getpid()) + '\n')
        try:
            status('verify_models')
            verify_models()
            data = read(ROOT / 'data/manifest.json')
            assert data['train_overlap_counts']['unseen_test'] == 0
            plan = {'methods': ['grpo', 'opd', 'opd_then_rl', 'kdrl_anneal'], 'total_batches': 150,
                    'switch_step': 60, 'train_batch': 128, 'rollouts': 8, 'response_limit': 4096,
                    'student': 'Qwen3-1.7B-Base', 'teacher': 'Qwen3-8B', 'n_gpus': 8,
                    'teacher_dispatch': 'author code retained, including unused teacher scoring during GRPO',
                    'evaluation': {'suites': ['author_test', 'unseen_test'], 'n': 32, 'questions': 512,
                                   'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'sample_seeds': list(range(42, 74))},
                    'selection': 'fixed final step 150; no best-test checkpoint or hyperparameter search',
                    'source_author_commit': 'f5fd333736495608acd9415a60cd7568593aa1d4',
                    'data_manifest_sha256': hashlib.sha256((ROOT / 'data/manifest.json').read_bytes()).hexdigest()}
            if (ROOT / 'study_plan.json').exists():
                assert read(ROOT / 'study_plan.json') == plan, 'Frozen protocol changed'
            else:
                atomic(ROOT / 'study_plan.json', plan)
            # Before full training, test real paper models and checkpoint continuation.
            warm = phase('opd', 'paper-opd-001', 1, smoke=True)
            phase('opd_then_rl', 'paper-resume-001', 2, resume=warm, smoke=True)
            phase('kdrl_anneal', 'paper-joint-001', 1, smoke=True)
            evaluate(MODEL_ROOT / 'Qwen3-1.7B-Base', 'base-sanity', samples=4, count=64, suites=['author_dev'])
            evaluate(MODEL_ROOT / 'Qwen3-8B', 'teacher-sanity', samples=4, count=64, suites=['author_dev'])
            checkpoints = {}
            checkpoints['grpo'] = phase('grpo', 'grpo-001', 150)
            checkpoints['opd'] = phase('opd', 'opd-001', 150)
            prefix = checkpoints['opd'].parent / 'global_step_60'
            checkpoints['opd_then_rl'] = phase('opd_then_rl', 'opd-then-rl-001', 150, resume=prefix)
            checkpoints['kdrl_anneal'] = phase('kdrl_anneal', 'kdrl-anneal-001', 150)
            # Neither test suite is evaluated until all four fixed training paths end.
            for name, checkpoint in checkpoints.items():
                evaluate(merge(checkpoint), name)
            atomic(ROOT / 'final_checkpoints.json', {k: str(v) for k, v in checkpoints.items()})
            status('training_and_evaluation_complete', status='completed')
            subprocess.run([sys.executable, str(REPO / 'reproduction/analyze_results.py')], check=True)
            status('completed', status='completed', report=str(ROOT / 'REPRODUCTION_RESULTS.md'))
        except BaseException as exc:
            status('needs_attention', status='failed', error=type(exc).__name__ + ': ' + str(exc))
            traceback.print_exc(); raise


if __name__ == '__main__':
    main()
