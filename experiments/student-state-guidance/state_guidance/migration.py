"""Preserve interrupted attempts and prepare exact saved-state continuations.

This performs no GPU work. It runs only on the verified FMA copy, under the
coordinator lock. Uncheckpointed updates remain in the archived attempt.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from state_guidance.control import atomic, read, same_process


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(checkpoint):
    return [checkpoint / 'data.pt'] + [
        checkpoint / 'actor' / f'{kind}_world_size_4_rank_{rank}.pt'
        for kind in ['model', 'optim', 'extra_state'] for rank in range(4)]


def load_continuation(folder, task, mode, seed, start=None, end=None):
    data = read(folder / 'migration_resume.json')
    assert data['schema'] == 1 and data['name'] == folder.name
    assert (data['task'], data['mode'], data['seed']) == (task, mode, seed)
    origin, stop, saved = data['original_start_step'], data['original_end_step'], data['checkpoint_step']
    assert origin in [0, 30, 90] and stop == origin + 150 and origin < saved < stop
    assert (saved - origin) % 50 == 0
    if start is not None:
        assert (origin, stop) == (start, end)
    checkpoint = folder / 'checkpoints' / f'global_step_{saved}'
    assert all(p.is_file() and p.stat().st_size > 0 for p in checkpoint_files(checkpoint))
    rows = [json.loads(line) for line in (folder / 'checkpoints/metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(origin + 1, saved + 1))
    return data


def prepare(root, route, *, failure_kind='user_migration', inventory_path=None):
    name, saved = route['name'], route['checkpoint_global_step']
    assert '/' not in name and name not in ['.', '..']
    assert failure_kind in ('user_migration', 'watchdog_stale')
    if failure_kind == 'watchdog_stale':
        state = read(root / 'study_state.json')
        assert state['status'] == 'failed' and not state['active'] and not same_process(state['identity'])
        assert state.get('error', '').startswith("RuntimeError('Watchdog health remained invalid:")
        assert name in state['failed_runs']
    folder = root / 'runs' / name
    tag = 'migration' if failure_kind == 'user_migration' else 'health'
    archive = folder.with_name(name + '.failed-' + tag + '-' + str(route['original_attempt']))
    label = name if tag == 'migration' else tag + '-' + name + '-' + str(route['original_attempt'])
    staging = root / 'migration-20261102' / ('staging-' + label)
    journal = root / 'migration-20261102' / ('prepare-' + label + '.json')
    # A process crash between the two directory renames leaves a complete staging
    # directory and the untouched archived attempt. An explicit prepare can finish it.
    if not folder.exists() and archive.exists() and staging.exists():
        intent = read(staging / 'migration_resume.json')
        assert intent['archive_relative_path'] == str(archive.relative_to(root))
        assert intent.get('source_attempt_started_unix', intent['scientific_attempt_started_unix']) == route['original_attempt']
        assert read(staging / 'launcher_record.json')['status'] == 'migration_ready'
        staging.rename(folder)
    old = read(folder / 'launcher_record.json')
    if old['status'] == 'migration_ready':
        return load_continuation(folder, old['task'], old['mode'], old['seed'],
                                 route['original_start_step'], route['original_end_step'])
    assert old['status'] == 'failed' and old['started_unix'] == route['original_attempt']
    assert old.get('error') == "RuntimeError('Interrupted by signal 15')", 'Only the audited interruption is accepted'
    assert 'allocated_gpu_hours' in old and 'finished_unix' in old
    assert (old['start_step'], old['end_step']) == (route['original_start_step'], route['original_end_step'])
    assert not archive.exists() and not staging.exists(), 'Inspect retained migration staging before retrying'
    checkpoint = folder / 'checkpoints' / f'global_step_{saved}'
    inventory_path = inventory_path or root / 'migration-20261102/checkpoint-sha256.jsonl'
    inventory = {row['path']: row for row in
                 (json.loads(line) for line in inventory_path.read_text().splitlines())}
    for path in checkpoint_files(checkpoint):
        expected = inventory[str(path.relative_to(root))]
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], str(path)
    rows = [json.loads(line) for line in (folder / 'checkpoints/metrics.jsonl').read_text().splitlines()]
    origin = old['start_step']
    assert [row['step'] for row in rows] == list(range(origin + 1, rows[-1]['step'] + 1))
    assert rows[-1]['step'] == route['last_metric_global_step'] and saved <= rows[-1]['step']
    prefix = [row for row in rows if row['step'] <= saved]
    assert len(prefix) == saved - origin
    request = read(folder / 'request.json')
    assert request['config']['data.seed'] == route['data_seed']
    spec = {'schema': 1, 'name': name, 'task': old['task'], 'mode': old['mode'], 'seed': old['seed'],
            'original_start_step': origin, 'original_end_step': old['end_step'], 'checkpoint_step': saved,
            'data_seed': route['data_seed'],
            'scientific_attempt_started_unix': old.get('scientific_attempt_started_unix', old['started_unix']),
            'source_attempt_started_unix': old['started_unix'], 'interruption_kind': failure_kind,
            'archive_relative_path': str(archive.relative_to(root)),
            'previous_segment_gpu_hours': old['allocated_gpu_hours'],
            'retained_updates': len(prefix), 'uncheckpointed_updates': [row['step'] for row in rows if row['step'] > saved],
            'prepared_unix': time.time(), 'source_request_sha256': sha(folder / 'request.json'),
            'sampling_note': 'Restore saved optimizer, per-rank RNG and dataloader state. Cross-device rollout replay is not claimed bitwise identical.'}
    atomic(journal, {'status': 'staging', 'spec': spec})
    staging.mkdir(parents=True)
    (staging / 'checkpoints').mkdir()
    def preserve_file(source, target):
        source, target = Path(source), Path(target)
        relative = source.relative_to(folder)
        # The archived attempt keeps the already-verified immutable bytes.
        # Resolve links relative to the final canonical location, not staging.
        # Only derived metadata is copied, so later model merging cannot rewrite
        # the archived checkpoint/config files through a hard link.
        if source.suffix in ('.pt', '.safetensors') or source.name == 'val.jsonl':
            canonical = folder / relative
            backing = archive / relative
            link = os.path.relpath(backing, canonical.parent)
            assert (canonical.parent / link).resolve().is_relative_to(root.resolve())
            target.symlink_to(link)
        else:
            with source.open('rb') as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
        return str(target)
    for path in sorted((folder / 'checkpoints').glob('global_step_*')):
        step = int(path.name.removeprefix('global_step_'))
        if path.is_dir() and step <= saved:
            shutil.copytree(path, staging / 'checkpoints' / path.name, copy_function=preserve_file)
    (staging / 'checkpoints/metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in prefix))
    (staging / 'checkpoints/latest_checkpointed_iteration.txt').write_text(str(saved) + '\n')
    for path in sorted((folder / 'val_logs').rglob('metrics.json')):
        step = read(path)['step']
        if step <= saved:
            assert origin <= step and (step - origin) % 25 == 0
            shutil.copytree(path.parent, staging / path.parent.relative_to(folder), copy_function=preserve_file)
    atomic(staging / 'parent_request.json', request)
    atomic(staging / 'migration_resume.json', spec)
    atomic(staging / 'launcher_record.json', {
        'status': 'migration_ready', 'task': old['task'], 'mode': old['mode'], 'seed': old['seed'],
        'start_step': origin, 'end_step': old['end_step'], 'check': False, 'n_gpus': 0,
        'started_unix': spec['prepared_unix'], 'allocated_gpu_hours': 0,
        'scientific_attempt_started_unix': spec['scientific_attempt_started_unix'], 'migration': spec})
    folder.rename(archive)
    staging.rename(folder)
    verified = load_continuation(folder, old['task'], old['mode'], old['seed'], origin, old['end_step'])
    atomic(journal, {'status': 'prepared', 'spec': verified})
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        imported = read(root / 'migration-20261102/IMPORT_VERIFIED.json')
        assert imported['status'] == 'verified' and imported['target'] == str(root)
        assert Path(imported['source']) != root, 'Preserve the original shared source'
        state = read(root / 'study_state.json')
        assert state['status'] == 'stopped' and not state['active']
        manifest = read(root / 'migration-20261102/MIGRATION_RESUME_PLAN.json')
        results = [prepare(root, route) for route in manifest['resume_routes']]
        atomic(root / 'migration-20261102/CONTINUATIONS_PREPARED.json', results)
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
