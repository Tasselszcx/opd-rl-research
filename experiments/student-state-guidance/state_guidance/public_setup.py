"""Portable, CPU-only preparation for K&K's eight comparisons and shared reference.

This module never launches training or rewrites an existing run configuration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
SOURCE_COMMIT = json.loads((CONFIGS/'source-snapshot-files.json').read_text())['commit']


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
    scope = json.loads((CONFIGS / 'kk8-scope.json').read_text())
    scope['baseline_plan_sha256'] = hashlib.sha256(plan).hexdigest()
    clean_path = root/'CLEAN_REFERENCE_REQUIREMENT.json'
    clean = json.loads(clean_path.read_text()) if clean_path.exists() else {
        'status':'clean_reference_required', 'approved_unix':time.time(),
        'training_release':Path(__file__).resolve().parents[1].name,
        'canonical_training_source_commit':SOURCE_COMMIT,
        'source_identity_policy':'Public checkout directory; canonical hashes recorded in publication provenance.',
        'reference_source':'Official Qwen3-1.7B-Base; checkpoint resume disabled',
        'reference_steps':150, 'full_fork_steps':[30,90], 'legacy_reference_allowed':False}
    if clean.get('status')!='clean_reference_required' or clean.get('canonical_training_source_commit')!=SOURCE_COMMIT:
        raise RuntimeError('Existing reference policy differs; use a fresh run directory')
    clean_bytes=(json.dumps(clean,indent=2)+'\n').encode()
    scope['clean_reference_requirement_sha256']=hashlib.sha256(clean_bytes).hexdigest()
    scope['legacy_reference_allowed']=False
    scope_bytes = (json.dumps(scope, indent=2) + '\n').encode()
    repair = {'status':'approved_user_repair','method_revision':'semantic_eos_v1',
              'teacher_eos_ids':[151643,151645],'student_eos_id':151643,
              'source_release':SOURCE_COMMIT,
              'reference_states':'Clean-v4 starts from official Base and trains repaired reference states from step0; legacy parent checkpoints are forbidden.',
              'clean_reference_from_step0':True,
              'development_seed_forwarding':'4200 explicitly forwarded from warmup and across all eight branches',
              'extra_teacher_format_reminder':False}
    storage = {'status':'approved_by_user','policy':'fork_states_full_terminal_model_only'}
    suffix = "Let's think step by step.\n"
    prompt = {'status':'frozen_training_runtime','cot_suffix_by_task':{'kk':suffix},
              'cot_suffix_sha256':hashlib.sha256(suffix.encode()).hexdigest(),
              'source':'Corrected Hydra string serialization; shared by training and final evaluation.'}
    targets = [(root / 'study_plan.json', plan), (root / 'ACTIVE_SCOPE.json', scope_bytes)]
    targets.append((clean_path,clean_bytes))
    targets += [(root/name,(json.dumps(value,indent=2)+'\n').encode()) for name,value in
                [('PROTOCOL_AMENDMENT.json',repair),('STORAGE_POLICY.json',storage),('PROMPT_FORMAT.json',prompt)]]
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
    required = {'tasks': ['kk'], 'training_seeds': [42], 'states': [30, 90],
                'modes': ['plain', 'answer_only', 'full_reasoning', 'error_targeted'],
                'training_routes': 9, 'formal_branches':8,'branch_additional_steps': 150,
                'ordered_modes':['plain','answer_only','full_reasoning','error_targeted'],
                'sequential_student_states':True}
    for key, value in required.items():
        if scope.get(key) != value:
            raise ValueError('Not the current K&K eight-comparison preset: ' + key)
    if scope['baseline_plan_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('Scope does not match the frozen protocol hash')
    if plan['branch_additional_steps'] != 150 or plan['reference_checkpoints'] != [30, 90]:
        raise ValueError('Training horizons differ from the published protocol')
    repair=json.loads((root/'PROTOCOL_AMENDMENT.json').read_text())
    prompt=json.loads((root/'PROMPT_FORMAT.json').read_text())
    storage=json.loads((root/'STORAGE_POLICY.json').read_text())
    if repair.get('method_revision')!='semantic_eos_v1' or prompt['cot_suffix_by_task']['kk']!="Let's think step by step.\n":
        raise ValueError('Current EOS/newline repair is required')
    if storage.get('policy')!='fork_states_full_terminal_model_only':
        raise ValueError('Current storage policy is required')
    from state_guidance.clean_reference import requirement
    clean=requirement(root)
    if clean is None or clean.get('canonical_training_source_commit')!=SOURCE_COMMIT or scope.get('legacy_reference_allowed') is not False:
        raise ValueError('Fresh repaired reference policy is required')
    if (root/'TEACHER_OUTPUT_POLICY.json').exists():
        raise ValueError('The extra teacher-format reminder is experimental, not the selected current protocol')
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
    plan, scope = metadata(root)
    frozen = json.loads((CONFIGS / 'frozen-inputs.json').read_text())
    for task in scope['tasks']:
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
        if len(rows) != 9:
            raise ValueError('Expected one shared reference and eight comparisons')
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
                      'scope': 'K&K: eight comparisons plus one shared reference; fixed teacher'}))


if __name__ == '__main__':
    main()
