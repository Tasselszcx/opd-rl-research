"""An explicitly approved subset of the original, unchanged G3 protocol."""
import hashlib
from state_guidance.control import read

DEFAULT = {'tasks': ['kk', 'gsm8k', 'zebra', 'countdown'], 'training_seeds': [42, 43, 44],
           'modes': ['plain', 'answer_only', 'full_reasoning', 'key_steps', 'error_targeted'],
           'states': [30, 90]}


def load_scope(root):
    path = root / 'ACTIVE_SCOPE.json'
    if not path.exists():
        return {**DEFAULT, 'scope_sha256': None}
    scope = read(path)
    assert scope['schema'] == 1 and scope['authorization'] == 'explicit_user_scope_revision'
    for key in DEFAULT:
        values = scope[key]
        assert values and len(values) == len(set(values)) and set(values) <= set(DEFAULT[key]), key
    assert scope['states'] == [30, 90] and 'plain' in scope['modes'] and 'answer_only' in scope['modes']
    count = len(scope['tasks']) * len(scope['training_seeds']) * (1 + len(scope['states']) * len(scope['modes']))
    assert scope['training_routes'] == count
    return {**scope, 'scope_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def seed_summary(values, seeds):
    import numpy as np
    return {'interaction_by_training_seed': dict(zip(map(str, seeds), values)),
            'mean_interaction': float(np.mean(values)),
            'seed_std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'scope': 'one training seed; no training-seed variability estimate' if len(values) == 1
                     else 'paired training seeds; no precise population-level seed CI'}
