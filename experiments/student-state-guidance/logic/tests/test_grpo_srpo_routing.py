# tests/test_grpo_srpo_routing.py
"""Unit tests for the GRPO_SRPO advantage estimator (SRPO routing).

Routing semantics:
  - GRPO advantage contributes only on correct rollouts (r_i = 1).
  - OPD term contributes only on incorrect rollouts (r_i = 0).
  - The two objectives are disjoint — no rollout has both terms active.
"""
import numpy as np
import torch


def _compute_srpo_advantages(
    grpo_adv,         # (bs, T)
    teacher_logprob,  # (bs, T)
    student_logprob,  # (bs, T)
    response_mask,    # (bs, T)
    correctness,      # (bs,)  in [0, 1]
    opd_beta=1.0,
):
    """Extracted math for unit testing — matches the GRPO_SRPO branch."""
    opd_mask = response_mask.clone()
    lengths = opd_mask.long().sum(dim=1)
    last_idx = (lengths - 1).clamp(min=0)
    opd_mask[torch.arange(opd_mask.size(0)), last_idx] = 0

    correct_mask = (correctness >= 1.0).to(grpo_adv.dtype)
    incorrect_mask = 1.0 - correct_mask
    opd_term = (teacher_logprob - student_logprob) * opd_mask
    advantages = (grpo_adv * correct_mask.unsqueeze(-1)
                  + opd_beta * opd_term * incorrect_mask.unsqueeze(-1))
    return advantages, correct_mask, incorrect_mask, opd_term


def test_grpo_term_zeroed_on_incorrect():
    """For r=0 rollouts, the GRPO contribution must be zero."""
    bs, T = 4, 6
    grpo_adv = torch.full((bs, T), 3.0)            # nonzero GRPO advantage
    teacher = torch.zeros(bs, T)
    student = torch.zeros(bs, T)                   # zero gap → opd term zero
    response_mask = torch.ones(bs, T)
    correctness = torch.tensor([0.0, 0.0, 1.0, 1.0])

    advantages, correct_mask, incorrect_mask, opd_term = _compute_srpo_advantages(
        grpo_adv, teacher, student, response_mask, correctness)

    # Rows 0,1 are r=0: GRPO contribution zeroed, opd_term=0 → advantages=0
    assert torch.allclose(advantages[:2], torch.zeros(2, T))
    # Rows 2,3 are r=1: GRPO contribution kept
    assert torch.allclose(advantages[2:], grpo_adv[2:])


def test_opd_term_zeroed_on_correct():
    """For r=1 rollouts, the OPD contribution must be zero."""
    bs, T = 4, 6
    grpo_adv = torch.zeros(bs, T)
    teacher = torch.full((bs, T), 2.0)             # nonzero teacher-student gap
    student = torch.zeros(bs, T)
    response_mask = torch.ones(bs, T)
    correctness = torch.tensor([0.0, 0.0, 1.0, 1.0])

    advantages, _, _, opd_term = _compute_srpo_advantages(
        grpo_adv, teacher, student, response_mask, correctness, opd_beta=1.0)

    # Rows 2,3 are r=1: OPD contribution zeroed → advantages=0
    assert torch.allclose(advantages[2:], torch.zeros(2, T))
    # Rows 0,1 are r=0: should carry the OPD term (with last-valid-token dropped)
    expected_opd_only = opd_term[:2]
    assert torch.allclose(advantages[:2], expected_opd_only)


def test_disjoint_no_double_count():
    """No rollout should receive both GRPO and OPD contributions."""
    bs, T = 4, 5
    grpo_adv = torch.full((bs, T), 1.5)
    teacher = torch.full((bs, T), 2.0)
    student = torch.zeros(bs, T)
    response_mask = torch.ones(bs, T)
    correctness = torch.tensor([0.0, 1.0, 0.0, 1.0])

    _, correct_mask, incorrect_mask, _ = _compute_srpo_advantages(
        grpo_adv, teacher, student, response_mask, correctness)

    # The two masks must partition every rollout exactly once.
    assert torch.allclose(correct_mask + incorrect_mask, torch.ones(bs))
    assert (correct_mask * incorrect_mask == 0).all()


def test_partial_credit_treated_as_incorrect():
    """SRPO uses a strict `correctness >= 1.0` boundary — partial-credit
    rollouts (e.g. correctness=0.5) are routed to OPD, not GRPO."""
    bs, T = 3, 4
    grpo_adv = torch.full((bs, T), 2.0)
    teacher = torch.zeros(bs, T)
    student = torch.zeros(bs, T)
    response_mask = torch.ones(bs, T)
    correctness = torch.tensor([0.0, 0.5, 1.0])

    advantages, correct_mask, incorrect_mask, _ = _compute_srpo_advantages(
        grpo_adv, teacher, student, response_mask, correctness)

    assert correct_mask.tolist() == [0.0, 0.0, 1.0]
    assert incorrect_mask.tolist() == [1.0, 1.0, 0.0]
    # Only row 2 keeps a nonzero GRPO contribution
    assert torch.allclose(advantages[2], grpo_adv[2])
    assert torch.allclose(advantages[:2], torch.zeros(2, T))


def test_eos_dropped_in_opd_term():
    """OPD term should drop the last valid token (EOS-like convention)."""
    bs, T = 2, 5
    grpo_adv = torch.zeros(bs, T)
    teacher = torch.full((bs, T), 4.0)
    student = torch.zeros(bs, T)
    response_mask = torch.ones(bs, T)
    response_mask[1, 4] = 0     # row 1 ends at t=3
    correctness = torch.tensor([0.0, 0.0])

    advantages, _, _, opd_term = _compute_srpo_advantages(
        grpo_adv, teacher, student, response_mask, correctness)

    # row 0: last valid t=4 → opd_term[0,4] == 0
    assert opd_term[0, 4].item() == 0.0
    # row 1: last valid t=3 → opd_term[1,3] == 0; t=4 was already masked
    assert opd_term[1, 3].item() == 0.0
    assert opd_term[1, 4].item() == 0.0
