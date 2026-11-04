"""CPU-only, read-only collection of completed G3 development validations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

KEYS = ['critic/score/mean', 'actor/kl_loss', 'actor/grad_norm', 'response_length/mean',
        'response_length/clip_ratio', 'timing_s/step']


def stable(path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('file changing during read')
    return raw


def collect(root):
    result = {'checked_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'scope': 'development only; 512 questions x 8 samples; no final test results',
              'runs': [], 'pending_or_invalid': [], 'source_sha256': {}}
    scope_path=root/'ACTIVE_SCOPE.json'
    scope=json.loads(stable(scope_path)) if scope_path.exists() else None
    result['active_scope']=scope
    result['scope_sha256']=hashlib.sha256(stable(scope_path)).hexdigest() if scope else None
    result['outside_active_scope']=[]
    expected = {}
    for task in (scope['tasks'] if scope else ['kk', 'gsm8k', 'zebra', 'countdown']):
        raw = stable(root/'data'/task/'dev.jsonl')
        rows = [json.loads(line) for line in raw.split(b'\n') if line.strip()]
        assert len(rows) == 512
        expected[task] = {i: row['question_sha256'] for i, row in enumerate(rows)}
        result['source_sha256']['data/'+task+'/dev.jsonl'] = hashlib.sha256(raw).hexdigest()
    for record_path in sorted((root/'runs').glob('*/launcher_record.json')):
        folder = record_path.parent
        if '.failed-' in folder.name:
            continue  # Archived costs/evidence stay intact; formal curves use the current attempt.
        record = json.loads(stable(record_path))
        if scope and (record.get('task') not in scope['tasks'] or record.get('seed') not in scope['training_seeds']
                      or (record.get('start_step',0)>0 and record.get('mode') not in scope['modes'])):
            result['outside_active_scope'].append({'run':folder.name,'status':record['status']})
            continue
        if record.get('check') or record.get('end_step', 0)-record.get('start_step', 0) != 150:
            continue
        run = {'run': folder.name, 'task': record['task'], 'training_seed': record['seed'],
               'attempt_id': str(record.get('scientific_attempt_started_unix', record['started_unix'])),
               'start_step': record['start_step'], 'guidance': record['mode'], 'status': record['status'],
               'train': [], 'dev': []}
        path = folder/'checkpoints/metrics.jsonl'
        if path.exists():
            raw = path.read_bytes()
            lines = raw.split(b'\n')
            if lines[-1]:
                result['pending_or_invalid'].append({'run': folder.name, 'file': 'metrics.jsonl', 'reason': 'partial final line deferred'})
                lines = lines[:-1]
            rows = [json.loads(line) for line in lines if line.strip()]
            assert [x['step'] for x in rows] == list(range(record['start_step']+1, record['start_step']+len(rows)+1))
            for row in rows:
                run['train'].append({'global_step': row['step'], 'additional_step': row['step']-record['start_step'],
                                     **{key: row['metrics'].get(key) for key in KEYS}})
            result['source_sha256'][str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        seen_steps = set()
        for meta_path in sorted((folder/'val_logs').rglob('metrics.json')):
            path = meta_path.parent/'val.jsonl'
            try:
                meta = json.loads(stable(meta_path)); raw = stable(path)
                step = meta['step']-record['start_step']
                assert 0 <= step <= 150 and step % 25 == 0
                assert step not in seen_steps, 'duplicate validation step'
                rows = [json.loads(line) for line in raw.split(b'\n') if line.strip()]
                assert len(rows) == 4096 and meta['num_questions'] == 512 and meta['n_sampling'] == 8
                lookup = {}; hashes = {}; counts = [0]*512
                for row in rows:
                    i, sample = row['dataset_index'], row['rollout_idx']
                    assert i in expected[record['task']] and 0 <= sample < 8
                    assert row['question_sha256'] == expected[record['task']][i]
                    assert row['correctness_score'] in (0, 1), 'non-binary score must not be called success'
                    assert (i, sample) not in lookup, 'duplicate question/sample'
                    lookup[i, sample] = row['correctness_score']; hashes[i] = row['question_sha256']
                    counts[i] += int(row['correctness_score'])
                assert len(lookup) == 4096 and len(hashes) == 512
                curve = {str(k): sum(1 if 8-c < k else 1-math.comb(8-c, k)/math.comb(8, k) for c in counts)/512 for k in range(1, 9)}
                assert round(curve['1']*100, 1) == meta['pass_at_1'], 'display pass@1 differs from raw'
                assert round(curve['8']*100, 1) == meta['pass_at_k'], 'display pass@8 differs from raw'
                point = {'additional_step': step, 'global_step': meta['step'], 'questions': 512, 'samples': 8,
                         'pass_at_1': curve['1'], 'pass_at_8': curve['8'], 'pass_at_k': curve,
                         'mean_response_tokens': sum(r['response_tokens'] for r in rows)/4096,
                         'output_limit_fraction': sum(bool(r['at_output_limit']) for r in rows)/4096,
                         'question_scores': [{'index': i, 'question_sha256': hashes[i], 'correct_count': counts[i]} for i in range(512)]}
                run['dev'].append(point); seen_steps.add(step)
                result['source_sha256'][str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
            except (OSError, ValueError, AssertionError, RuntimeError) as exc:
                result['pending_or_invalid'].append({'run': folder.name, 'file': str(path.relative_to(folder)), 'reason': str(exc)})
        run['dev'].sort(key=lambda x: x['additional_step'])
        result['runs'].append(run)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root.resolve()), ensure_ascii=False))
