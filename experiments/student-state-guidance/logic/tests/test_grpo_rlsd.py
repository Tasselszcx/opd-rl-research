# tests/test_grpo_rlsd.py
"""Unit tests for the GRPO_RLSD advantage estimator.

Implements arXiv 2604.03128 ("Self-Distilled RLVR") Algorithm 1, lines 14-17,
with a fixed mixing coefficient λ (no step schedule):

    Δ_t  = sg(log P_T(y_t) − log P_S(y_t))
    w_t  = exp(sign(A) · Δ_t)
    Â_t  = A · ((1 − λ) + λ · clip(w_t, 1 − ε_w, 1 + ε_w))
"""
import torch


SEED = 0


def _compute_rlsd_advantages(
    grpo_adv,         # (bs, T) per-token GRPO advantage (broadcast from seq scalar)
    teacher_logprob,  # (bs, T)
    student_logprob,  # (bs, T)
    response_mask,    # (bs, T)
    eps_w=0.2,
    rlsd_lambda=1.0,
):
    """Extracted math for unit testing — matches the GRPO_RLSD branch."""
    seq_adv = grpo_adv[:, 0]
    sign_a = torch.sign(seq_adv).to(student_logprob.dtype)   # ∈ {−1, 0, +1}
    delta = teacher_logprob - student_logprob                # (bs, T)
    w = torch.exp(sign_a.unsqueeze(-1) * delta)              # (bs, T)
    w_clip = w.clamp(min=1.0 - eps_w, max=1.0 + eps_w)
    blend = (1.0 - rlsd_lambda) + rlsd_lambda * w_clip       # (bs, T)
    advantages = (grpo_adv * blend) * response_mask
    return advantages, w, w_clip, blend, sign_a


def _make_grpo_adv(bs, T, scalars):
    """Build a (bs, T) GRPO advantage that broadcasts a scalar value per row."""
    adv = torch.tensor(scalars, dtype=torch.float32).unsqueeze(-1).expand(-1, T).clone()
    return adv


def test_zero_gap_implies_w_one():
    """Δ_t = 0 ⇒ w_t = 1 ⇒ Â_t = A regardless of λ."""
    bs, T = 3, 5
    teacher = torch.zeros(bs, T)
    student = torch.zeros(bs, T)
    grpo_adv = _make_grpo_adv(bs, T, [1.5, -2.0, 0.5])
    response_mask = torch.ones(bs, T)

    for lam in [0.0, 0.5, 1.0]:
        advantages, w, _, _, _ = _compute_rlsd_advantages(
            grpo_adv, teacher, student, response_mask, eps_w=0.2, rlsd_lambda=lam)
        assert torch.allclose(w, torch.ones_like(w))
        assert torch.allclose(advantages, grpo_adv)


def test_lambda_zero_gives_pure_grpo():
    """λ = 0 ⇒ Â = A (no teacher influence) for any Δ."""
    # Seeded: unseeded randn made these float-equality assertions depend on
    # whatever consumed the global RNG earlier in the run.
    torch.manual_seed(SEED)
    bs, T = 4, 6
    teacher = torch.randn(bs, T)
    student = torch.randn(bs, T)
    grpo_adv = _make_grpo_adv(bs, T, [1.0, -1.0, 2.0, -3.0])
    response_mask = torch.ones(bs, T)

    advantages, _, _, _, _ = _compute_rlsd_advantages(
        grpo_adv, teacher, student, response_mask, eps_w=0.2, rlsd_lambda=0.0)
    assert torch.allclose(advantages, grpo_adv)


def test_sign_flips_ratio_direction():
    """For the same Δ, sign(A) > 0 gives w = exp(Δ); sign(A) < 0 gives w = exp(−Δ)."""
    bs, T = 2, 4
    delta_vals = torch.tensor([[0.5, -0.3, 0.1, 0.2],
                               [0.5, -0.3, 0.1, 0.2]])
    teacher = delta_vals
    student = torch.zeros_like(delta_vals)
    response_mask = torch.ones(bs, T)
    # Row 0: A > 0 → w = exp(+Δ). Row 1: A < 0 → w = exp(−Δ).
    grpo_adv = _make_grpo_adv(bs, T, [+2.0, -2.0])

    _, w, _, _, sign_a = _compute_rlsd_advantages(
        grpo_adv, teacher, student, response_mask, eps_w=10.0, rlsd_lambda=1.0)

    assert sign_a.tolist() == [1.0, -1.0]
    assert torch.allclose(w[0], torch.exp(delta_vals[0]))
    assert torch.allclose(w[1], torch.exp(-delta_vals[1]))


def test_clip_bounds_are_symmetric_around_one():
    """clip range is [1 − ε_w, 1 + ε_w], NOT [1/c_max, c_max]."""
    bs, T = 2, 3
    # Make Δ huge in both directions to force clipping
    teacher = torch.tensor([[10.0, -10.0, 10.0],
                            [10.0, -10.0, 10.0]])
    student = torch.zeros(bs, T)
    response_mask = torch.ones(bs, T)
    grpo_adv = _make_grpo_adv(bs, T, [+1.0, -1.0])
    eps_w = 0.3

    _, _, w_clip, _, _ = _compute_rlsd_advantages(
        grpo_adv, teacher, student, response_mask, eps_w=eps_w, rlsd_lambda=1.0)

    assert (w_clip >= 1.0 - eps_w - 1e-6).all()
    assert (w_clip <= 1.0 + eps_w + 1e-6).all()


def test_sign_preservation():
    """sign(Â_t) == sign(A) for every t, even with extreme Δ and λ=1."""
    # Seeded: unseeded randn made these float-equality assertions depend on
    # whatever consumed the global RNG earlier in the run.
    torch.manual_seed(SEED)
    bs, T = 6, 8
    teacher = torch.randn(bs, T) * 5.0    # extreme teacher
    student = torch.randn(bs, T) * 5.0
    response_mask = torch.ones(bs, T)
    seq_scalars = [3.0, -2.0, 1.0, -4.0, 0.5, -0.1]
    grpo_adv = _make_grpo_adv(bs, T, seq_scalars)

    # ε_w < 1 guarantees the blend factor stays strictly positive.
    advantages, _, _, _, _ = _compute_rlsd_advantages(
        grpo_adv, teacher, student, response_mask, eps_w=0.5, rlsd_lambda=1.0)

    sign_seq = torch.sign(torch.tensor(seq_scalars))
    sign_adv = torch.sign(advantages)
    # check every token shares the row's sign
    for i in range(bs):
        assert (sign_adv[i] == sign_seq[i]).all(), \
            f"row {i}: expected sign {sign_seq[i]}, got {sign_adv[i].unique()}"


def test_zero_advantage_stays_zero():
    """sign(0) = 0 ⇒ w = exp(0·Δ) = 1 ⇒ Â = 0 · blend = 0."""
    bs, T = 2, 4
    teacher = torch.tensor([[1.0, -1.0, 0.5, -0.5],
                            [0.0,  0.0, 0.0,  0.0]])
    student = torch.zeros(bs, T)
    response_mask = torch.ones(bs, T)
    grpo_adv = _make_grpo_adv(bs, T, [0.0, 0.0])

    advantages, w, _, _, sign_a = _compute_rlsd_advantages(
        grpo_adv, teacher, student, response_mask, eps_w=0.2, rlsd_lambda=1.0)

    assert sign_a.tolist() == [0.0, 0.0]
    assert torch.allclose(w, torch.ones_like(w))      # exp(0 · Δ) = 1
    assert torch.allclose(advantages, torch.zeros_like(advantages))


def test_response_mask_zeroes_padding():
    """Â outside the response mask must be zero."""
    bs, T = 2, 6
    teacher = torch.full((bs, T), 1.0)
    student = torch.zeros(bs, T)
    response_mask = torch.ones(bs, T)
    response_mask[:, 4:] = 0           # last two tokens are padding
    grpo_adv = _make_grpo_adv(bs, T, [+2.0, -2.0]) * response_mask

    advantages, _, _, _, _ = _compute_rlsd_advantages(
        grpo_adv, teacher, student, response_mask, eps_w=0.2, rlsd_lambda=1.0)

    assert torch.allclose(advantages[:, 4:], torch.zeros(bs, 2))


def test_lambda_one_matches_paper_eq15():
    """At λ = 1, Â_t = A · clip(w_t, 1−ε_w, 1+ε_w) — exactly Eq. 15."""
    bs, T = 2, 4
    teacher = torch.tensor([[0.1, -0.2, 0.05, 0.0],
                            [0.3, -0.1, 0.1, -0.05]])
    student = torch.zeros(bs, T)
    response_mask = torch.ones(bs, T)
    grpo_adv = _make_grpo_adv(bs, T, [+1.0, -1.0])
    eps_w = 0.5

    advantages, _, w_clip, _, _ = _compute_rlsd_advantages(
        grpo_adv, teacher, student, response_mask, eps_w=eps_w, rlsd_lambda=1.0)

    expected = grpo_adv * w_clip * response_mask
    assert torch.allclose(advantages, expected)
