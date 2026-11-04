"""Freeze pilot splits and exact teacher-only guidance before any pilot scores."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import reasoning_gym
from reasoning_gym.composite import DatasetSpec
from guidance.proofs import make_guidance

def freeze(path, content):
    if path.exists():
        assert path.read_bytes() == content, 'Frozen file changed: ' + str(path)
    else:
        path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--train-source', type=Path, required=True)
    a = p.parse_args()
    start = time.time()
    data = a.root / 'data'
    data.mkdir(parents=True, exist_ok=True)
    content = a.train_source.read_bytes()
    records = {'train': {'sha256': freeze(data/'train.jsonl', content)}}
    train = [json.loads(x) for x in content.decode().splitlines()]
    assert len(train) == 20000 and [x['index'] for x in train] == list(range(20000))
    cfg = {'n_people': 6, 'depth_constraint': 2, 'width_constraint': 2}
    specs = [DatasetSpec(name='knights_knaves', weight=1, config=cfg)]
    datasets = {
        'meta_val': reasoning_gym.create_dataset('composite', seed=2000000, size=128, datasets=specs),
        'pilot_test': reasoning_gym.create_dataset('knights_knaves', seed=3000000, size=512, **cfg),
    }
    hashes = {'train': {x['question_sha256'] for x in train}}
    for name, dataset in datasets.items():
        rows = []
        for i in range(len(dataset)):
            entry = dataset[i]
            assert dataset.score_answer(entry['answer'], entry) == 1
            assert dataset.score_answer('', entry) == 0
            rows.append({'index': i, 'question_sha256': hashlib.sha256(entry['question'].encode()).hexdigest(), **entry})
        hashes[name] = {x['question_sha256'] for x in rows}
        assert len(hashes[name]) == len(rows)
        body = ('\n'.join(json.dumps(x, sort_keys=True) for x in rows)+'\n').encode()
        records[name] = {'count': len(rows), 'sha256': freeze(data/(name+'.jsonl'), body)}
    assert not hashes['train'] & hashes['meta_val']
    assert not hashes['train'] & hashes['pilot_test']
    assert not hashes['meta_val'] & hashes['pilot_test']
    guide = []
    for entry in train:
        assert hashlib.sha256(entry['question'].encode()).hexdigest() == entry['question_sha256']
        guide.append({'index': entry['index'], 'question_sha256': entry['question_sha256'], **make_guidance(entry)})
    body = ('\n'.join(json.dumps(x, sort_keys=True) for x in guide)+'\n').encode()
    records['guidance'] = {'count': len(guide), 'sha256': freeze(data/'guidance.jsonl', body)}
    records['train']['count'] = len(train)
    manifest = {'datasets': records, 'configuration': cfg, 'meta_val_seed': 2000000,
                'pilot_test_seed': 3000000, 'zero_overlap': True,
                'teacher_only_guidance': True, 'guidance_on_training_questions_only': True,
                'selection': 'Fixed post-OPD and post-RL endpoints; no test-driven choice or tuning'}
    freeze(data/'manifest.json', (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode())
    print(json.dumps({'status': 'prepared', 'seconds': time.time()-start, **manifest}), flush=True)

if __name__ == '__main__':
    main()
