"""CPU contract tests for the actual opt-in G-OPD integration hooks."""
import importlib.util
import json
from pathlib import Path
import types

import pytest

np = pytest.importorskip("numpy", reason="backend tensor tests require the optional CPU training dependencies")
torch = pytest.importorskip("torch", reason="backend tensor tests require PyTorch")
pytest.importorskip("ray", reason="real DataProto imports Ray (no Ray cluster is started)")
pytest.importorskip("tensordict", reason="real DataProto requires TensorDict")
pytest.importorskip("omegaconf", reason="verl config classes require OmegaConf")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mvp_backend_hooks", ROOT / "verl/verl/trainer/ppo/meta_teaching_hooks.py")
hooks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hooks)


def config(phase="warmup", output="/tmp/mvp-tests"):
    return {
        "meta_teaching": {"enabled": True, "phase": phase, "arm_id": "test", "output_dir": output,
                          "phase_steps": 2, "exposure": "plain", "query_fraction": 1},
        "actor_rollout_ref": {
            "actor": {"strategy": "fsdp", "use_kl_loss": phase == "warmup", "kl_loss_coef": 0,
                      "use_rollout_log_probs": True, "entropy_coeff": 0,
                      "policy_loss": {"only_reverse_kl_advantages": phase == "warmup", "lambda_vals": 1},
                      "optim": {"lr_scheduler_type": "constant", "lr_warmup_steps": 0}},
            "model": {"path": "student"}, "ref": {"model": {"path": "teacher" if phase == "warmup" else None}},
            "rollout": {"mode": "sync", "multi_turn": {"enable": False}},
        },
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "data": {"return_raw_chat": True},
        "trainer": {"val_only": phase == "eval"},
    }


@pytest.mark.parametrize("phase", ["warmup", "rl", "eval"])
def test_valid_phase_configs(phase):
    hooks.validate_phase_config(config(phase))


@pytest.mark.parametrize("phase,path,replacement", [
    ("rl", "actor_rollout_ref.actor.use_kl_loss", True),
    ("rl", "actor_rollout_ref.ref.model.path", "teacher"),
    ("rl", "actor_rollout_ref.ref.model.path", ""),
    ("rl", "actor_rollout_ref.model.base_model_path", ""),
    ("rl", "actor_rollout_ref.model.base_model_path", "base"),
    ("rl", "algorithm.critique_vllm_url", "http://teacher"),
    ("warmup", "actor_rollout_ref.actor.use_rollout_log_probs", False),
    ("warmup", "actor_rollout_ref.actor.policy_loss.lambda_vals", 1.25),
    ("warmup", "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages", False),
    ("warmup", "actor_rollout_ref.actor.entropy_coeff", 0.01),
    ("warmup", "actor_rollout_ref.rollout.multi_turn.enable", True),
    ("warmup", "actor_rollout_ref.actor.optim.lr_scheduler_type", "cosine"),
    ("warmup", "algorithm.use_ref_solution_distillation", True),
])
def test_fail_fast_on_uncontrolled_training_paths(phase, path, replacement):
    cfg = config(phase)
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = replacement
    with pytest.raises(ValueError):
        hooks.validate_phase_config(cfg)


def test_partial_removes_nested_boxed_and_answer_suffix_before_cutting():
    text = r"First reason carefully. The ratio is found. \boxed{\frac{123}{7}}"
    assert "123" not in hooks.partial_reference(text, 1)
    assert "boxed" not in hooks.partial_reference(text, 1)
    assert hooks.partial_reference("Reasoning.\n#### 123", 1) == "Reasoning."
    assert hooks.partial_reference("Steps. Final answer: 123", 1) == "Steps."
    with pytest.raises(ValueError):
        hooks.partial_reference(r"\boxed{123}", 0.5)


def test_same_tokenizer_checks_both_ids_and_template():
    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        chat_template = "template"
        def get_vocab(self):
            return {"a": 1, "b": 2}
    hooks.assert_same_tokenizer(Tokenizer(), Tokenizer())
    other = Tokenizer()
    other.chat_template = "other"
    with pytest.raises(ValueError, match="chat_template"):
        hooks.assert_same_tokenizer(Tokenizer(), other)


def test_gate_is_deterministic_and_uses_student_surprisal():
    old = torch.tensor([[-0.1, -99], [-0.8, -0.8], [-0.3, -0.3]])
    attention = torch.tensor([[1, 0], [1, 1], [1, 1]])
    assert hooks.choose_query_indices(old, attention, 1 / 3, "uncertainty", 0) == [1]
    assert hooks.choose_query_indices(old, attention, 0, "random", 0) == []
    assert hooks.choose_query_indices(old, attention, 0.5, "random", 42) == hooks.choose_query_indices(old, attention, 0.5, "random", 42)


class TinyTokenizer:
    pad_token_id = 0
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages) + "\nAssistant:"
    def encode(self, text, **kwargs):
        return [ord(char) + 1 for char in text]


def batch(n=3):
    from verl import DataProto
    responses = torch.tensor([[41, 42, 0]] * n)
    ids = torch.tensor([[21, 22, 41, 42, 0]] * n)
    attention = torch.tensor([[1, 1, 1, 1, 0]] * n)
    raw = np.empty(n, dtype=object)
    raw[:] = [[{"role": "user", "content": f"Question {i}"}] for i in range(n)]
    extra = np.empty(n, dtype=object)
    extra[:] = [{"reference_solution": f"First explain step {i}. Then derive the result. \\boxed{{12345}}"} for i in range(n)]
    return DataProto.from_dict(
        tensors={"input_ids": ids, "attention_mask": attention, "position_ids": attention.cumsum(-1) - 1,
                 "responses": responses, "response_mask": torch.tensor([[1, 0, 0]] * n),
                 "old_log_probs": torch.tensor([[-1., -2., 0.]] * n)},
        non_tensors={"raw_prompt": raw, "extra_info": extra},
    )


def test_teacher_conditioning_preserves_student_ids_and_response_attention():
    source = batch()
    original_ids = source.batch["input_ids"].clone()
    original_response = source.batch["responses"].clone()
    original_prompt = source.non_tensor_batch["raw_prompt"][0][0]["content"]
    hooks.prepare_teacher_inputs(source, TinyTokenizer(), {"exposure": "partial", "exposure_fraction": 1, "teacher_max_prompt_length": 1000})
    assert torch.equal(source.batch["input_ids"], original_ids)
    assert torch.equal(source.batch["responses"], original_response)
    assert torch.equal(source.batch["ref_input_ids"][:, -3:], original_response)
    # The second response token has loss-mask 0, but attention must stay 1.
    assert source.batch["ref_attention_mask"][0, -2].item() == 1
    assert source.batch["ref_attention_mask"][0, -1].item() == 0
    assert source.non_tensor_batch["raw_prompt"][0][0]["content"] == original_prompt
    rendered = "".join(chr(x - 1) for x in source.batch["ref_input_ids"][0, :-3].tolist() if x)
    assert "12345" not in rendered
    assert "Teacher-only reference" in rendered


def test_missing_reference_and_overlong_teacher_prompt_fail():
    source = batch()
    source.non_tensor_batch["extra_info"][0] = {}
    with pytest.raises(ValueError, match="reference_solution"):
        hooks.prepare_teacher_inputs(source, TinyTokenizer(), {"exposure": "full"})
    with pytest.raises(ValueError, match="exceeds"):
        hooks.prepare_teacher_inputs(batch(), TinyTokenizer(), {"exposure": "full", "teacher_max_prompt_length": 2})


def test_query_subset_is_padded_and_scattered_using_real_dataproto():
    from verl import DataProto
    source = batch()
    seen = []
    def score(selected):
        seen.append(selected)
        return DataProto.from_dict(tensors={"ref_log_prob": torch.full_like(selected.batch["old_log_probs"], -0.25)})
    worker = types.SimpleNamespace(world_size=4, compute_ref_log_prob=score)
    result, counters = hooks.score_selected_teacher(source, worker, TinyTokenizer(), {"exposure": "plain", "query_fraction": 1 / 3, "seed": 9}, 5)
    assert len(seen) == 1 and len(seen[0]) == 4
    changed = (result.batch["ref_log_prob"] != result.batch["old_log_probs"]).any(-1)
    assert int(changed.sum()) == 1
    assert torch.equal(result.batch["ref_log_prob"][~changed], result.batch["old_log_probs"][~changed])
    assert counters == {"teacher_queries": 4, "teacher_selected_queries": 1, "teacher_scored_tokens": 16, "teacher_response_tokens": 8}
    assert "ref_input_ids" not in source.batch


def test_empty_gate_makes_no_teacher_rpc():
    source = batch()
    def forbidden(*args):
        raise AssertionError("No teacher RPC is allowed for an empty gate")
    worker = types.SimpleNamespace(world_size=8, compute_ref_log_prob=forbidden)
    result, counters = hooks.score_selected_teacher(source, worker, TinyTokenizer(), {"exposure": "full", "query_fraction": 0}, 0)
    assert torch.equal(result.batch["ref_log_prob"], result.batch["old_log_probs"])
    assert counters["teacher_queries"] == counters["teacher_scored_tokens"] == 0


def test_summary_is_atomic_and_rejects_stale_validation(tmp_path):
    state = hooks.PhaseState({"phase": "rl", "phase_id": "round0/rl", "spec_fingerprint": "abc123",
                              "seed": 42, "source_checkpoint": "/source/global_step_17",
                              "arm_id": "a", "output_dir": str(tmp_path), "phase_steps": 3,
                              "deadline_unix": 100, "validation_reserve_seconds": 10})
    state.begin(17)
    assert state.target_step == 20
    assert not state.should_stop_for_budget(now=89)
    assert state.should_stop_for_budget(now=90)
    state.validated(17, {"meta_teaching/val_score": 0.5})
    state.end_step = 18
    state.write("failed", "interrupted")
    payload = json.loads((tmp_path / "phase_summary.json").read_text())
    assert payload["steps_completed"] == 1
    assert payload["phase_id"] == "round0/rl" and payload["spec_fingerprint"] == "abc123"
    assert payload["seed"] == 42 and payload["source_checkpoint"] == "/source/global_step_17"
    assert payload["validation_step"] == 17
    assert payload["val_score"] is None
    assert payload["teacher_enabled"] is False
    state.validated(18, {"meta_teaching/val_score": np.float64(0.75)})
    state.write("completed")
    assert json.loads((tmp_path / "phase_summary.json").read_text())["val_score"] == 0.75
    assert not list(tmp_path.glob(".phase_summary.*"))


def test_conditioned_teacher_microbatch_uses_augmented_length_without_mutating_student():
    from verl import DataProto
    source = batch()
    original = source.batch["input_ids"].clone()
    def score(selected):
        assert torch.equal(selected.batch["input_ids"], selected.batch["ref_input_ids"])
        assert torch.equal(selected.batch["attention_mask"], selected.batch["ref_attention_mask"])
        assert selected.batch["input_ids"].shape[-1] > original.shape[-1]
        return DataProto.from_dict(tensors={"ref_log_prob": torch.full_like(selected.batch["old_log_probs"], -0.25)})
    worker = types.SimpleNamespace(world_size=1, compute_ref_log_prob=score)
    hooks.score_selected_teacher(source, worker, TinyTokenizer(), {"exposure": "full", "query_fraction": 1, "teacher_max_prompt_length": 1000}, 0)
    assert torch.equal(source.batch["input_ids"], original)
    assert "ref_input_ids" not in source.batch


def test_rollout_dataclass_seed_is_optional_and_explicitly_configurable():
    from verl.workers.config.rollout import RolloutConfig
    assert RolloutConfig(name="vllm").seed is None
    assert RolloutConfig(name="vllm", seed=42).seed == 42
