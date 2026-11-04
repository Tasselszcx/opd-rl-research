"""Real CPU autograd checks against G-OPD's actual actor advantage branch/loss.

These are tiny differentiable models, not fake optimizer-call counters. They do
not validate CUDA, FSDP, vLLM, or model-quality improvements.
"""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import types

import pytest

np = pytest.importorskip("numpy", reason="optional CPU training-test dependency")
torch = pytest.importorskip("torch", reason="real gradient tests require PyTorch")
pytest.importorskip("ray", reason="real DataProto imports Ray")
pytest.importorskip("tensordict", reason="real DataProto requires TensorDict")
OmegaConf = pytest.importorskip("omegaconf", reason="actual policy loss uses OmegaConf").OmegaConf
pytest.importorskip("transformers", reason="verl utilities require Transformers")

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mvp_gradient_hooks", ROOT / "verl/verl/trainer/ppo/meta_teaching_hooks.py")
hooks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hooks)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Embedding(2, 5)
        torch.nn.init.zeros_(self.logits.weight)
    def response_log_probs(self, data):
        context = data.batch["input_ids"][:, :1].expand_as(data.batch["responses"])
        logits = self.logits(context)
        return logits.log_softmax(-1).gather(-1, data.batch["responses"].unsqueeze(-1)).squeeze(-1)


def sample_data(student, same_prompt=False):
    ids = torch.tensor([[0, 0, 1], [0 if same_prompt else 1, 2, 3]])
    data = DataProto.from_dict(tensors={
        "input_ids": ids, "responses": ids[:, 1:], "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(3).expand_as(ids), "response_mask": torch.ones(2, 2),
    })
    data.batch["old_log_probs"] = student.response_log_probs(data).detach().clone()
    return data


def actual_actor_advantages(old, ref, task_advantages, warmup):
    """Execute the real only_reverse_kl_advantages block in dp_actor.py."""
    source = ast.parse((ROOT / "verl/verl/workers/actor/dp_actor.py").read_text())
    block = next(node for node in ast.walk(source) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "self.config.policy_loss.only_reverse_kl_advantages")
    scope = {
        "self": types.SimpleNamespace(config=OmegaConf.create({"policy_loss": {"only_reverse_kl_advantages": warmup, "lambda_vals": 1, "multi_teacher_distill": False}})),
        "model_inputs": {"ref_log_prob": ref}, "old_log_prob": old,
        "advantages": task_advantages.clone(), "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), str(ROOT / "verl/verl/workers/actor/dp_actor.py"), "exec"), scope)
    return scope["advantages"]


def optimize_once(student, data, ref, task_advantages, warmup):
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.03, weight_decay=0)
    optimizer.zero_grad()
    old = data.batch["old_log_probs"]
    advantages = actual_actor_advantages(old, ref, task_advantages, warmup)
    loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old, log_prob=student.response_log_probs(data), advantages=advantages,
        response_mask=data.batch["response_mask"], loss_agg_mode="token-mean",
        config=OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "clip_ratio_c": 3}),
    )
    loss.backward()
    gradient = student.logits.weight.grad.detach().clone()
    optimizer.step()
    return gradient, deepcopy(student.state_dict()), advantages


def teacher_reference(student, prefer_first):
    teacher = TinyPolicy()
    with torch.no_grad():
        teacher.logits.weight[0, 0] = 3 if prefer_first else -3
        teacher.logits.weight[0, 1] = -3 if prefer_first else 3
        teacher.logits.weight[1, 4] = 20  # This unqueried row must never affect OPD.
    teacher.eval()
    data = sample_data(student)
    class CPUScoringWorker:
        world_size = 1
        def compute_ref_log_prob(self, selected):
            with torch.no_grad():
                scored = teacher.response_log_probs(selected)
            return DataProto.from_dict(tensors={"ref_log_prob": scored})
    data, counters = hooks.score_selected_teacher(
        data, CPUScoringWorker(), None,
        {"exposure": "plain", "query_fraction": 0.5, "gate": "uncertainty", "seed": 42}, 0,
    )
    assert counters["teacher_selected_queries"] == 1
    return data, data.batch["ref_log_prob"]


def test_teacher_distribution_changes_real_student_gradients_and_parameters():
    first, second = TinyPolicy(), TinyPolicy()
    data_a, ref_a = teacher_reference(first, prefer_first=True)
    data_b, ref_b = teacher_reference(second, prefer_first=False)
    # Intentionally large task advantages: actual warmup actor code must ignore them.
    task_adv = torch.full((2, 2), 1000.)
    grad_a, state_a, adv_a = optimize_once(first, data_a, ref_a, task_adv, warmup=True)
    grad_b, state_b, adv_b = optimize_once(second, data_b, ref_b, -task_adv, warmup=True)
    assert not torch.allclose(grad_a[0], grad_b[0])
    assert not torch.allclose(state_a["logits.weight"][0], state_b["logits.weight"][0])
    assert state_a["logits.weight"][0, 0] > state_b["logits.weight"][0, 0]
    assert torch.count_nonzero(grad_a[1]) == torch.count_nonzero(grad_b[1]) == 0
    assert torch.count_nonzero(state_a["logits.weight"][1]) == 0
    assert torch.count_nonzero(adv_a[1]) == torch.count_nonzero(adv_b[1]) == 0


def test_independent_rl_gradients_depend_on_grpo_reward_not_any_teacher_tensor():
    warmed = TinyPolicy()
    warm_data, ref = teacher_reference(warmed, prefer_first=True)
    optimize_once(warmed, warm_data, ref, torch.zeros(2, 2), warmup=True)
    first, second = deepcopy(warmed), deepcopy(warmed)
    data_a = sample_data(first, same_prompt=True)
    data_b = sample_data(second, same_prompt=True)
    rewards = torch.tensor([[0., 1.], [0., 0.]])
    grpo_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=data_a.batch["response_mask"],
        index=np.array(["same-task", "same-task"]), norm_adv_by_std_in_grpo=True,
    )
    grad_a, state_a, adv_a = optimize_once(first, data_a, torch.full((2, 2), -100.), grpo_adv, warmup=False)
    grad_b, state_b, adv_b = optimize_once(second, data_b, torch.full((2, 2), 100.), grpo_adv, warmup=False)
    assert torch.equal(adv_a, grpo_adv) and torch.equal(adv_b, grpo_adv)
    assert torch.count_nonzero(grad_a) > 0
    assert torch.equal(grad_a, grad_b)
    assert torch.equal(state_a["logits.weight"], state_b["logits.weight"])


@pytest.mark.parametrize("rank", [0, 7])
def test_actual_worker_rng_checkpoint_roundtrip_and_missing_file_rejection(tmp_path, rank):
    source = ast.parse((ROOT / "verl/verl/workers/fsdp_workers.py").read_text())
    worker_node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    methods = [deepcopy(node) for node in worker_node.body if isinstance(node, ast.FunctionDef) and node.name in {"save_checkpoint", "load_checkpoint"}]
    for method in methods:
        method.decorator_list = []
    cls = ast.ClassDef(name="CPUCheckpointWorker", bases=[], keywords=[], body=methods, decorator_list=[])
    scope = {"torch": torch, "os": __import__("os"), "dist": types.SimpleNamespace(barrier=lambda: None),
             "get_torch_device": lambda: types.SimpleNamespace(get_rng_state=torch.get_rng_state)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "fsdp_workers.py", "exec"), scope)
    worker = scope["CPUCheckpointWorker"]()
    worker.config = OmegaConf.create({"rollout": {"seed": 42}})
    worker.rank = rank
    worker._is_actor = worker._is_rollout = True
    worker._is_offload_param = worker._is_offload_optimizer = worker._is_lora = False
    worker.checkpoint_manager = types.SimpleNamespace(save_checkpoint=lambda **kwargs: None, load_checkpoint=lambda **kwargs: None)
    saved_rng = torch.get_rng_state().clone()
    worker.gen_random_states = saved_rng.clone()
    worker.save_checkpoint(str(tmp_path), global_step=19)
    expected = tmp_path / f"rollout_rng_rank_{rank}.pt"
    assert expected.is_file()
    worker.gen_random_states = torch.zeros_like(saved_rng)
    worker.load_checkpoint(str(tmp_path))
    assert torch.equal(worker.gen_random_states, saved_rng)
    expected.unlink()
    with pytest.raises(ValueError, match="rollout RNG state"):
        worker.load_checkpoint(str(tmp_path))
