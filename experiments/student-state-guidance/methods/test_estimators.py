"""Check the reference implementation against both verl forks, element-wise.

`methods/estimators.py` restates each estimator once, in isolation. The code
that produced the paper's numbers is an `elif` branch inside
`compute_advantage` in each of the two pinned verl forks. This test builds one
synthetic batch, runs it through both, and asserts the advantages agree to
floating-point tolerance -- so the readable version cannot silently drift from
the one that ran.

    pytest methods/test_estimators.py

verl is exercised in a subprocess so this file stays importable without it.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimators as ref  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# name -> (path to the verl checkout, whether grpo_opd clamps the teacher term).
# One tree now serves both halves of the paper.
FORKS = {
    "verl": (REPO / "verl", True),
}

BETA = 0.02
RLSD_EPS_W = 0.2
RLSD_LAMBDA = 1.0


def build_batch():
    """One small batch, fixed seed.

    Two prompt groups of three rollouts. Group 'a' has one success, group 'b'
    has none -- so the HDPO-cliff gate fires on exactly one group and the
    correct/incorrect split is non-trivial for KDRL-MASK and SRPO. Response
    lengths differ so the EOS-dropping in `opd_mask` is actually exercised.
    """
    torch.manual_seed(0)
    bsz, prompt_len, resp_len = 6, 3, 8
    index = ["a", "a", "a", "b", "b", "b"]
    correctness = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    valid_len = [8, 6, 5, 7, 4, 8]

    response_mask = torch.zeros(bsz, resp_len)
    for i, n in enumerate(valid_len):
        response_mask[i, :n] = 1.0
    attention_mask = torch.cat([torch.ones(bsz, prompt_len), response_mask], dim=1)

    # rule-based reward: +1 correct, -1 wrong, placed on the last valid token
    token_level_rewards = torch.zeros(bsz, resp_len)
    for i, n in enumerate(valid_len):
        token_level_rewards[i, n - 1] = 1.0 if correctness[i] >= 1.0 else -1.0

    return {
        "responses": torch.zeros(bsz, resp_len, dtype=torch.long),
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "token_level_rewards": token_level_rewards,
        "old_log_probs": -torch.rand(bsz, resp_len) * 3.0,
        "ref_log_prob": -torch.rand(bsz, resp_len) * 3.0,
        "correctness_scores": correctness,
        "index": index,
    }


def reference_advantages(b, clamp_grpo_opd):
    grpo_adv = ref.grpo_outcome_advantage(b["token_level_rewards"], b["response_mask"], b["index"])
    t, s, m = b["ref_log_prob"], b["old_log_probs"], b["response_mask"]
    c, idx = b["correctness_scores"], b["index"]
    return {
        "grpo": grpo_adv,
        "on_policy_distillation": ref.opd(t, s, m),
        "grpo_opd": ref.grpo_opd(grpo_adv, t, s, m, BETA, clamp=clamp_grpo_opd),
        "grpo_opd_mask": ref.grpo_opd_mask(grpo_adv, t, s, c, m, BETA),
        "grpo_opd_cliff": ref.grpo_opd_cliff(grpo_adv, t, s, c, idx, m, BETA),
        "grpo_srpo": ref.grpo_srpo(grpo_adv, t, s, c, m, BETA),
        "grpo_rlsd": ref.grpo_rlsd(grpo_adv, t, s, m, RLSD_EPS_W, RLSD_LAMBDA),
    }


# --------------------------------------------------------------------------
# Worker: runs inside a fork's checkout and reports that fork's advantages.
# --------------------------------------------------------------------------
def _run_in_fork(verl_root):
    import numpy as np
    from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage

    b = build_batch()

    class _Data:
        def __init__(self):
            self.batch = {}
            self.non_tensor_batch = {}
            self.meta_info = {}

    out = {}
    for est in [
        "grpo",
        "on_policy_distillation",
        "grpo_opd",
        "grpo_opd_mask",
        "grpo_opd_cliff",
        "grpo_srpo",
        "grpo_rlsd",
    ]:
        data = _Data()
        data.batch = {
            "responses": b["responses"],
            "attention_mask": b["attention_mask"],
            "token_level_rewards": b["token_level_rewards"],
            "old_log_probs": b["old_log_probs"],
            "ref_log_prob": b["ref_log_prob"],
            "correctness_scores": b["correctness_scores"],
        }
        data.non_tensor_batch = {"uid": np.array(b["index"], dtype=object)}
        compute_advantage(
            data,
            adv_estimator=AdvantageEstimator(est),
            opd_beta=BETA,
            rlsd_eps_w=RLSD_EPS_W,
            rlsd_lambda=RLSD_LAMBDA,
        )
        out[est] = data.batch["advantages"].reshape(-1).tolist()
    return out


def _fork_advantages(verl_root):
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        cwd=str(verl_root),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(f"could not run compute_advantage in {verl_root}:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("fork", sorted(FORKS))
def test_reference_matches_fork(fork):
    verl_root, clamp_grpo_opd = FORKS[fork]
    if not (verl_root / "verl" / "trainer" / "ppo" / "ray_trainer.py").exists():
        pytest.skip(f"verl not found at {verl_root}")

    got = _fork_advantages(verl_root)
    expected = reference_advantages(build_batch(), clamp_grpo_opd)

    assert sorted(got) == sorted(expected), "estimator sets differ"
    for name, want in expected.items():
        have = torch.tensor(got[name]).reshape(want.shape)
        assert torch.allclose(have, want, atol=1e-6), (
            f"{fork}/{name} differs from the reference implementation\n"
            f"  fork:      {have.reshape(-1)[:8]}\n"
            f"  reference: {want.reshape(-1)[:8]}"
        )


def test_grpo_opd_clamp_is_not_vacuous():
    """The clamp changes the result, so unifying it was a real decision.

    Both forks clip now, but the original logic-puzzle runs did not; this pins
    down that the two forms are genuinely different rather than equivalent.
    """
    b = build_batch()
    grpo_adv = ref.grpo_outcome_advantage(b["token_level_rewards"], b["response_mask"], b["index"])
    t, s, m = b["ref_log_prob"], b["old_log_probs"], b["response_mask"]
    # With a teacher-student gap far beyond +/-5 the two settings must differ.
    big_gap = t + 50.0
    clamped = ref.grpo_opd(grpo_adv, big_gap, s, m, BETA, clamp=True)
    unclamped = ref.grpo_opd(grpo_adv, big_gap, s, m, BETA, clamp=False)
    assert not torch.allclose(clamped, unclamped)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        print(json.dumps(_run_in_fork(Path.cwd())))


def test_trainer_call_matches_compute_advantage_signature():
    """The reasoning-gym trainer must not pass kwargs compute_advantage lacks.

    This is a static check because it is cheap and because the failure it guards
    against -- dropping a parameter from compute_advantage while a caller still
    passes it -- only shows up once a real training step runs, several minutes
    into a GPU job.
    """
    import ast

    verl_src = (REPO / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text()
    for node in ast.walk(ast.parse(verl_src)):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_advantage":
            accepted = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
            break
    else:
        pytest.fail("compute_advantage not found in verl")

    trainer_path = REPO / "logic" / "training" / "trainers" / "ray_grpo_trainer.py"
    if not trainer_path.exists():
        pytest.skip("reasoning-gym trainer not present")

    passed = set()
    for node in ast.walk(ast.parse(trainer_path.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "compute_advantage"):
            passed |= {kw.arg for kw in node.keywords if kw.arg}
    assert passed, "no compute_advantage call found in the trainer"
    unknown = passed - accepted
    assert not unknown, f"trainer passes kwargs compute_advantage does not accept: {sorted(unknown)}"
