"""Portable, CPU-only preparation for the approved 18-route study.

This module never launches training or rewrites an existing run configuration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_once(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError('Existing file differs; retained without changes: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently replacing another preparation process.
    with path.open('xb') as stream:
        stream.write(content)


def initialize(root):
    root = root.resolve()
    plan = (CONFIGS / 'protocol-baseline.json').read_bytes()
    scope = json.loads((CONFIGS / 'core18-scope.json').read_text())
    scope['baseline_plan_sha256'] = hashlib.sha256(plan).hexdigest()
    scope_bytes = (json.dumps(scope, indent=2) + '\n').encode()
    targets = [(root / 'study_plan.json', plan), (root / 'ACTIVE_SCOPE.json', scope_bytes)]
    # Check every conflict before writing either file.
    for path, raw in targets:
        if path.exists() and path.read_bytes() != raw:
            raise RuntimeError('Use a fresh run directory; existing metadata retained: ' + str(path))
    if any((root / p).exists() for p in ['study_state.json', 'coordinator.pid', 'evaluation_state.json']):
        raise RuntimeError('Existing execution state retained; init is for fresh runs only')
    for path, raw in targets:
        write_once(path, raw)
    return root


def metadata(root):
    raw = (root / 'study_plan.json').read_bytes()
    plan = json.loads(raw)
    scope = json.loads((root / 'ACTIVE_SCOPE.json').read_text())
    required = {'tasks': ['kk', 'gsm8k'], 'training_seeds': [42], 'states': [30, 90],
                'modes': ['plain', 'answer_only', 'full_reasoning', 'error_targeted'],
                'training_routes': 18, 'branch_additional_steps': 150}
    for key, value in required.items():
        if scope.get(key) != value:
            raise ValueError('Not the published core18 preset: ' + key)
    if scope['baseline_plan_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('Scope does not match the frozen protocol hash')
    if plan['branch_additional_steps'] != 150 or plan['reference_checkpoints'] != [30, 90]:
        raise ValueError('Training horizons differ from the published protocol')
    return plan, scope


def fetch_file(url, destination, expected_hash, opener=urlopen):
    if destination.exists():
        if digest(destination) != expected_hash:
            raise RuntimeError('Existing input differs; refusing replacement: ' + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=destination.name + '.', suffix='.partial',
                                     dir=destination.parent, delete=False) as stream:
        partial = Path(stream.name)
        with opener(url, timeout=90) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(chunk)
    if digest(partial) != expected_hash:
        raise RuntimeError('Downloaded input hash differs; partial retained: ' + str(partial))
    if destination.exists():
        if digest(destination) != expected_hash:
            raise RuntimeError('Input appeared concurrently and differs; partial retained')
        partial.unlink()
    else:
        partial.replace(destination)


def fetch_sources(root):
    manifest = json.loads((CONFIGS / 'public-data-sources.json').read_text())
    for item in manifest['files']:
        fetch_file(item['url'], root / 'source_data' / item['file'], item['sha256'])
    write_once(root / 'source_data/source_manifest.json',
               (json.dumps(manifest, indent=2) + '\n').encode())


def check_inputs(root):
    plan, _ = metadata(root)
    frozen = json.loads((CONFIGS / 'frozen-inputs.json').read_text())
    for task in ['kk', 'gsm8k']:
        folder = root / 'data' / task
        for split, item in plan['data'][task]['splits'].items():
            if digest(folder / (split + '.jsonl')) != item['sha256']:
                raise ValueError('Frozen split differs: ' + task + '/' + split)
        guidance = json.loads((folder / 'guidance_manifest.json').read_text())
        if not guidance['teacher_only'] or digest(folder / 'guidance.jsonl') != frozen[task]['guidance_sha256']:
            raise ValueError('Teacher-only guidance differs: ' + task)
        if guidance['sha256'] != frozen[task]['guidance_sha256']:
            raise ValueError('Guidance manifest differs: ' + task)
        prompt = json.loads((folder / 'prompt_audit.json').read_text())
        if prompt != frozen[task]['prompt_audit']:
            raise ValueError('Prompt audit differs: ' + task)
    source = json.loads((CONFIGS / 'public-data-sources.json').read_text())
    for item in source['files']:
        if digest(root / 'source_data' / item['file']) != item['sha256']:
            raise ValueError('Public input differs: ' + item['file'])


def check_model_cache(cache, expected, rehash=False):
    record = json.loads((cache / 'CACHE_MANIFEST.json').read_text())
    if record.get('status') != 'completed':
        raise ValueError('Model cache has not completed SHA verification')
    recorded = {(x['model'], x['file']): x for x in record['files']}
    for item in expected:
        key = item['model'], item['file']
        if recorded.get(key) != item:
            raise ValueError('Model verification record differs: ' + '/'.join(key))
        path = cache / item['model'] / item['file']
        if path.stat().st_size != item['bytes'] or (rehash and digest(path) != item['sha256']):
            raise ValueError('Model cache differs: ' + '/'.join(key))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init', 'fetch-data', 'plan', 'check'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--models', type=Path)
    p.add_argument('--hash-models', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == 'init':
        initialize(root)
    elif a.action == 'fetch-data':
        metadata(root)
        fetch_sources(root)
    elif a.action == 'plan':
        metadata(root)
        from state_guidance.coordinator import phases
        rows = phases(root)
        if len(rows) != 18:
            raise ValueError('Expected exactly 18 routes')
        print(json.dumps({'routes': [{k: x[k] for k in ['name', 'task', 'seed', 'mode', 'start', 'steps']}
                                     for x in rows]}, indent=2))
        return
    else:
        if a.models is None:
            p.error('check requires --models')
        check_inputs(root)
        expected = json.loads((CONFIGS / 'model-files.json').read_text())['files']
        check_model_cache(a.models.resolve(), expected, a.hash_models)
    print(json.dumps({'status': a.action + '_passed', 'training_started': False,
                      'scope': '18 routes; fixed teacher; no independent RL or guidance generator'}))


if __name__ == '__main__':
    main()
