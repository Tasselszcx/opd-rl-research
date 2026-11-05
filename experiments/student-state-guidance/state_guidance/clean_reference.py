"""Enforce the user's fresh, repaired reference-state rebuild."""
import hashlib
from pathlib import Path
from state_guidance.control import read


def requirement(root):
    path = Path(root)/'CLEAN_REFERENCE_REQUIREMENT.json'
    if not path.exists():
        return None
    value = read(path)
    assert value['status'] == 'clean_reference_required'
    scope = read(Path(root)/'ACTIVE_SCOPE.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == scope['clean_reference_requirement_sha256']
    return value


def check_configuration(root, config, resume, check=False):
    rule = requirement(root)
    if rule is None or check:
        return
    assert config['algorithm.opd_eos_mode'] == 'semantic', 'Clean warmup requires repaired EOS'
    assert config['data.cot_suffix'] == "Let's think step by step.\n", 'Clean warmup requires a real newline'
    assert config['data.enable_thinking'] is False
    if resume is None:
        assert config['trainer.resume_mode'] == 'disable', 'Fresh reference must start from the official base'
        assert Path(config['actor_rollout_ref.model.path']).name == 'Qwen3-1.7B-Base'
        assert config['algorithm.teacher_guidance'] == 'plain'
    else:
        expected = Path(root)/'runs/kk-s42-reference/checkpoints'
        assert Path(resume).resolve().parent == expected.resolve(), 'Legacy or external reference forbidden'
        assert Path(resume).name in ['global_step_30', 'global_step_90']


def check_reference_origin(root):
    rule = requirement(root)
    if rule is None:
        return
    folder = Path(root)/'runs/kk-s42-reference'
    request, record = read(folder/'request.json'), read(folder/'launcher_record.json')
    check_configuration(root, request['config'], None)
    assert request.get('source_resume') is None
    assert record['start_step'] == 0 and record['end_step'] == 150
    assert not record.get('reuse_only'), 'Copied legacy reference forbidden'
    assert record['started_unix'] >= rule['approved_unix'], 'Reference predates the clean rebuild'
    assert Path(record['source_path']).name == rule['training_release'], 'Reference came from another release'


def check_fork_parent(root, resume):
    if requirement(root) is None:
        return
    check_reference_origin(root)
    folder = Path(root)/'runs/kk-s42-reference'
    assert read(folder/'launcher_record.json')['status'] == 'completed'
    parent = Path(resume)
    assert parent.resolve().parent == (folder/'checkpoints').resolve()
    required = [parent/'data.pt'] + [parent/'actor'/f'{kind}_world_size_4_rank_{rank}.pt'
        for kind in ['model', 'optim', 'extra_state'] for rank in range(4)]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required), 'Full model/optimizer/RNG parent required'
