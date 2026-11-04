# tests/test_grpo_opd_cliff_gate.py
"""Unit tests for the GRPO_OPD_CLIFF advantage estimator (HDPO-style cliff gate).

The cliff gate fires only on groups where every rollout failed
(correctness < 1.0 for all rollouts in the group). On all other groups
(any rollout succeeded), the OPD term is masked off entirely.
"""
import numpy as np
import torch


def _compute_cliff_gate_advantages(
    grpo_adv,         # (bs, T) GRPO advantage broadcast to tokens
    teacher_logprob,  # (bs, T)
    student_logprob,  # (bs, T)
    response_mask,    # (bs, T)
    correctness,      # (bs,)  in [0, 1]
    index,            # array-like length bs, group ids
    opd_beta=1.0,
):
    """Extracted math for unit testing — matches the GRPO_OPD_CLIFF branch."""
    opd_mask = response_mask.clone()
    lengths = opd_mask.long().sum(dim=1)
    last_idx = (lengths - 1).clamp(min=0)
    opd_mask[torch.arange(opd_mask.size(0)), last_idx] = 0

    # group max correctness
    id2max = {}
    for i, gid in enumerate(index):
        v = float(correctness[i].item())
        id2max[gid] = max(id2max.get(gid, -1.0), v)
    cliff_mask = torch.tensor(
        [1.0 if id2max[gid] < 1.0 else 0.0 for gid in index],
        dtype=correctness.dtype)

    opd_term = (teacher_logprob - student_logprob) * opd_mask
    advantages = grpo_adv + opd_beta * opd_term * cliff_mask.unsqueeze(-1)
    return advantages, cliff_mask, opd_term


SEED = 0


def _make_inputs(bs=6, T=8):
    # Seeded: these inputs feed float-equality assertions, and unseeded randn
    # made them depend on whatever consumed the global RNG earlier in the run.
    torch.manual_seed(SEED)
    teacher = torch.randn(bs, T)
    student = torch.randn(bs, T)
    response_mask = torch.ones(bs, T)
    grpo_adv = torch.randn(bs, 1).expand(-1, T).clone() * response_mask
    return teacher, student, response_mask, grpo_adv


def test_cliff_fires_only_on_all_wrong_groups():
    """Three groups: g0 all-wrong, g1 mixed, g2 all-correct."""
    teacher, student, response_mask, grpo_adv = _make_inputs(bs=6, T=8)
    # g0 (rows 0,1): both r=0 → cliff fires
    # g1 (rows 2,3): one r=1, one r=0 → no cliff
    # g2 (rows 4,5): both r=1 → no cliff
    correctness = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    index = np.array(['g0', 'g0', 'g1', 'g1', 'g2', 'g2'])

    advantages, cliff_mask, opd_term = _compute_cliff_gate_advantages(
        grpo_adv, teacher, student, response_mask, correctness, index, opd_beta=1.0)

    # cliff_mask is per-rollout (broadcast from group-level)
    assert cliff_mask.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_advantage_equals_grpo_outside_cliff_groups():
    """Outside cliff groups, advantage should equal GRPO advantage exactly."""
    teacher, student, response_mask, grpo_adv = _make_inputs(bs=4, T=6)
    # g0 mixed, g1 all-correct
    correctness = torch.tensor([1.0, 0.0, 1.0, 1.0])
    index = np.array(['g0', 'g0', 'g1', 'g1'])

    advantages, cliff_mask, _ = _compute_cliff_gate_advantages(
        grpo_adv, teacher, student, response_mask, correctness, index, opd_beta=2.0)

    # No group is all-wrong → cliff_mask is all zero → advantages == grpo_adv
    assert torch.allclose(cliff_mask, torch.zeros_like(cliff_mask))
    assert torch.allclose(advantages, grpo_adv)


def test_advantage_includes_opd_in_cliff_groups():
    """In cliff groups, advantage = grpo_adv + beta * opd_term."""
    teacher, student, response_mask, grpo_adv = _make_inputs(bs=2, T=5)
    correctness = torch.tensor([0.0, 0.0])  # both wrong → cliff fires
    index = np.array(['g0', 'g0'])
    opd_beta = 0.5

    advantages, cliff_mask, opd_term = _compute_cliff_gate_advantages(
        grpo_adv, teacher, student, response_mask, correctness, index, opd_beta=opd_beta)

    expected = grpo_adv + opd_beta * opd_term
    assert torch.allclose(advantages, expected)
    assert torch.allclose(cliff_mask, torch.ones_like(cliff_mask))


def test_eos_token_dropped_from_opd_mask():
    """OPD term is zeroed at the last valid token (EOS-like convention)."""
    bs, T = 2, 6
    teacher = torch.full((bs, T), 5.0)   # large gap so EOS contribution would be visible
    student = torch.zeros(bs, T)
    response_mask = torch.ones(bs, T)
    response_mask[:, 5] = 0              # last valid token is t=4 in both rows
    grpo_adv = torch.zeros(bs, T)
    correctness = torch.tensor([0.0, 0.0])
    index = np.array(['g', 'g'])

    advantages, _, opd_term = _compute_cliff_gate_advantages(
        grpo_adv, teacher, student, response_mask, correctness, index, opd_beta=1.0)

    # EOS position (t=4) of opd_term should be zero; t=5 was masked out earlier
    assert torch.allclose(opd_term[:, 4], torch.zeros(bs))
    # t=0..3 should carry the teacher-student gap
    assert (opd_term[:, :4] == 5.0).all()


def test_grpo_term_unchanged_in_cliff_groups():
    """Cliff gate only adds the OPD term — the GRPO term must be untouched."""
    teacher, student, response_mask, grpo_adv = _make_inputs(bs=2, T=5)
    correctness = torch.tensor([0.0, 0.0])
    index = np.array(['g0', 'g0'])

    advantages, _, opd_term = _compute_cliff_gate_advantages(
        grpo_adv, teacher, student, response_mask, correctness, index, opd_beta=1.0)

    # advantages - opd_term*1.0 should recover grpo_adv exactly
    assert torch.allclose(advantages - opd_term, grpo_adv)
