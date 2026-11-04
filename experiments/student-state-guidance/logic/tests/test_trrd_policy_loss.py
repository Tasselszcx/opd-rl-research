"""Unit tests for the TRRD patch to verl's compute_policy_loss.

TRRD (RLAD paper) folds the teacher policy into the PPO importance ratio:

    r_TRRD = pi_S / (pi_S_old ** alpha  *  pi_T ** (1 - alpha))

Equivalently, against a log-mixture anchor
    log pi_mix = alpha * log pi_S_old + (1 - alpha) * log pi_T

The patch adds two optional kwargs to verl.trainer.ppo.core_algos.compute_policy_loss;
when either is None the function must be bit-identical to the original.
"""

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss


def _fixture(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    bs, T = 4, 16
    # keep log-probs negative-ish and bounded so exp(ratio) doesn't explode
    log_prob = -torch.rand(bs, T, generator=g)
    old_log_prob = log_prob + 0.02 * torch.randn(bs, T, generator=g)
    teacher_log_prob = log_prob + 0.05 * torch.randn(bs, T, generator=g)
    advantages = torch.randn(bs, T, generator=g)
    eos_mask = torch.ones(bs, T)
    eos_mask[:, -2:] = 0.0  # mask last 2 tokens to exercise masked_mean
    return log_prob, old_log_prob, teacher_log_prob, advantages, eos_mask


def test_backward_compat_no_kwargs():
    log_prob, old_log_prob, _, advantages, eos_mask = _fixture()
    # Call without any TRRD kwargs -> must match the original path.
    pg, clipfrac, ppo_kl = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
    )
    assert torch.isfinite(pg)
    assert torch.isfinite(clipfrac)
    assert torch.isfinite(ppo_kl)


def test_alpha_one_is_vanilla():
    log_prob, old_log_prob, teacher_log_prob, advantages, eos_mask = _fixture()
    baseline = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
    )
    trrd = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
        teacher_log_prob=teacher_log_prob,
        trrd_alpha=1.0,
    )
    for a, b in zip(baseline, trrd):
        assert torch.allclose(a, b, atol=1e-7), f"alpha=1.0 must be bit-identical, got {a} vs {b}"


def test_alpha_zero_is_pure_teacher():
    log_prob, old_log_prob, teacher_log_prob, advantages, eos_mask = _fixture()
    # With alpha=0 the mixture = teacher_log_prob, so the ratio should be
    # exp(log_prob - teacher_log_prob). We compute what the policy loss would
    # be with "old_log_prob" swapped for teacher_log_prob, and compare.
    expected = compute_policy_loss(
        old_log_prob=teacher_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
    )
    actual = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
        teacher_log_prob=teacher_log_prob,
        trrd_alpha=0.0,
    )
    # pg_loss and pg_clipfrac should match (they depend on the ratio); ppo_kl
    # is reported against the *true* old policy and therefore differs.
    assert torch.allclose(expected[0], actual[0], atol=1e-6)
    assert torch.allclose(expected[1], actual[1], atol=1e-6)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_mixture_geometric_mean_identity(alpha):
    log_prob, old_log_prob, teacher_log_prob, _, _ = _fixture()
    mixture = alpha * old_log_prob + (1.0 - alpha) * teacher_log_prob
    ratio_trrd = torch.exp(log_prob - mixture)
    ratio_product = (torch.exp(log_prob - old_log_prob) ** alpha) * (
        torch.exp(log_prob - teacher_log_prob) ** (1.0 - alpha)
    )
    assert torch.allclose(ratio_trrd, ratio_product, atol=1e-5)


def test_ppo_kl_invariant_to_teacher():
    log_prob, old_log_prob, teacher_log_prob, advantages, eos_mask = _fixture()
    _, _, ppo_kl_vanilla = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
    )
    _, _, ppo_kl_trrd = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_mask,
        cliprange=0.2,
        teacher_log_prob=teacher_log_prob,
        trrd_alpha=0.5,
    )
    # ppo_kl must still measure student <-> true old policy, independent of TRRD.
    assert torch.allclose(ppo_kl_vanilla, ppo_kl_trrd, atol=1e-7)


def test_alpha_out_of_range_raises():
    log_prob, old_log_prob, teacher_log_prob, advantages, eos_mask = _fixture()
    with pytest.raises(AssertionError):
        compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=eos_mask,
            cliprange=0.2,
            teacher_log_prob=teacher_log_prob,
            trrd_alpha=1.5,
        )


def _patch_teacher_at_eos(teacher_log_prob, old_log_prob, response_mask):
    """Mirror the dp_actor EOS patch: teacher[eos] := old[eos].

    With teacher == old at the last valid response token, the TRRD mixture
    collapses to old at that token, so the importance ratio reduces to the
    vanilla PPO ratio there (no teacher influence on the EOS gradient).
    """
    patched = teacher_log_prob.clone()
    eos_idx = (response_mask.long().sum(dim=1) - 1).clamp(min=0)
    rows = torch.arange(response_mask.size(0), device=response_mask.device)
    patched[rows, eos_idx] = old_log_prob[rows, eos_idx]
    return patched


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.7, 1.0])
def test_eos_patch_collapses_to_vanilla_at_eos_positions(alpha):
    # Build a response_mask that is 1 from token 0 up to and including EOS
    # (the verl convention from attention_mask). Patch teacher==old at EOS,
    # then compute TRRD pg_loss with the mask narrowed to EOS positions only.
    # That must match vanilla PPO's pg_loss on the same EOS-only mask.
    log_prob, old_log_prob, teacher_log_prob, advantages, _ = _fixture()
    bs, T = log_prob.shape

    lengths = torch.tensor([T, T - 1, T - 3, T - 5])[:bs]
    response_mask = torch.zeros(bs, T)
    for i, L in enumerate(lengths.tolist()):
        response_mask[i, :L] = 1.0

    patched_teacher = _patch_teacher_at_eos(teacher_log_prob, old_log_prob, response_mask)

    # Narrow mask to EOS positions only — the loss reduction then sees only
    # tokens where teacher==old, so any alpha must give the vanilla result.
    eos_only_mask = torch.zeros_like(response_mask)
    rows = torch.arange(bs)
    eos_only_mask[rows, lengths - 1] = 1.0

    vanilla = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_only_mask,
        cliprange=0.2,
    )
    trrd = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=eos_only_mask,
        cliprange=0.2,
        teacher_log_prob=patched_teacher,
        trrd_alpha=alpha,
    )
    assert torch.allclose(vanilla[0], trrd[0], atol=1e-6), "pg_loss must match at EOS-only positions"
    assert torch.allclose(vanilla[1], trrd[1], atol=1e-6), "pg_clipfrac must match at EOS-only positions"


def test_eos_patch_preserves_teacher_at_non_eos_positions():
    # The patch must only touch the EOS column; all other positions keep teacher.
    log_prob, old_log_prob, teacher_log_prob, _, _ = _fixture()
    bs, T = log_prob.shape

    # Variable per-row response lengths so EOS sits at different columns.
    lengths = torch.tensor([T, T - 1, T - 3, T - 5])[:bs]
    response_mask = torch.zeros(bs, T)
    for i, L in enumerate(lengths.tolist()):
        response_mask[i, :L] = 1.0

    patched = _patch_teacher_at_eos(teacher_log_prob, old_log_prob, response_mask)

    eos_idx = lengths - 1
    rows = torch.arange(bs)
    # EOS position: teacher overwritten with old
    assert torch.allclose(patched[rows, eos_idx], old_log_prob[rows, eos_idx])
    # All other valid positions: teacher unchanged
    other_mask = response_mask.clone()
    other_mask[rows, eos_idx] = 0.0
    assert torch.allclose(patched * other_mask, teacher_log_prob * other_mask)
