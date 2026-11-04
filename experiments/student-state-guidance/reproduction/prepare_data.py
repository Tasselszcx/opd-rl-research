"""Freeze author-code train/dev/test questions plus a separate unseen audit set."""
from pathlib import Path
import os
import hashlib
import json
import reasoning_gym
from reasoning_gym.composite import DatasetSpec

ROOT = Path(os.environ.get('REPRO_ROOT', str(Path(__file__).resolve().parents[1] / '.local-runs' / 'sequential'))) / 'data'
CFG = dict(n_people=6, depth_constraint=2, width_constraint=2)


def composite(seed, size):
    return reasoning_gym.create_dataset('composite', seed=seed, size=size,
        datasets=[DatasetSpec(name='knights_knaves', weight=1, config=CFG)])


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    datasets = {
        'train': composite(1, 20000), 'author_dev': composite(2, 1000),
        'author_test': reasoning_gym.create_dataset('knights_knaves', seed=42, size=512, **CFG),
        'unseen_test': reasoning_gym.create_dataset('knights_knaves', seed=1000000, size=512, **CFG),
    }
    hashes, records = {}, {}
    for name, dataset in datasets.items():
        rows, lookup = [], {}
        for index in range(len(dataset)):
            entry = dataset[index]
            digest = hashlib.sha256(entry['question'].encode()).hexdigest()
            assert dataset.score_answer(entry['answer'], entry) == 1.0
            row = {'index': index, 'question_sha256': digest, **entry}
            rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
            lookup[digest] = index
        content = ('\n'.join(rows) + '\n').encode()
        path = ROOT / f'{name}.jsonl'
        if path.exists():
            assert path.read_bytes() == content, 'Frozen generated dataset changed'
        else:
            path.write_bytes(content)
        hashes[name] = set(lookup)
        records[name] = {'count': len(rows), 'unique_question_hashes': len(lookup),
                         'sha256': hashlib.sha256(content).hexdigest()}
        print(name, records[name], flush=True)
    overlap = {name: len(hashes['train'] & values) for name, values in hashes.items() if name != 'train'}
    assert overlap['author_test'] == 512 and overlap['author_dev'] == 1000
    assert overlap['unseen_test'] == 0
    assert not hashes['unseen_test'] & hashes['author_dev']
    assert not hashes['unseen_test'] & hashes['author_test']
    manifest = {'config': CFG, 'datasets': records, 'train_overlap_counts': overlap,
                'author_test_seed': 42, 'unseen_test_seed': 1000000,
                'policy': 'Both final test suites are fixed before training; no test-based model or hyperparameter selection.',
                'warning': 'Released generator uses Random(seed+index). Author-code test/dev are contained in training; do not call these held out.'}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
