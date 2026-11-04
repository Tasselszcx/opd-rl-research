"""Reference implementation of the OPD-family advantage estimators.

This module is documentation, not the code that produced the paper's numbers.
Those live in the two pinned verl forks:

    third_party/DeepMath/verl              -> bakdop/verl @ opd-release
    third_party/rgym-opd/third_party/verl  -> bakdop/verl-rgym @ opd-release

where each estimator is an ``elif`` branch inside
``verl/trainer/ppo/ray_trainer.py::compute_advantage``. Those two forks target
different verl bases, so the same estimator is written out twice there, wrapped
in boilerplate that recovers the same tensors each time. This file states each
method once, without the boilerplate, so the actual difference between them is
readable. ``test_estimators.py`` checks it against both forks element-wise.

Shared notation, all tensors ``(batch, response_len)`` unless noted:

    A_grpo      group-normalised GRPO advantage, broadcast along the response.
                Constant along the response, so ``A_grpo[:, 0]`` is the
                per-sequence value.
    log pi_T    teacher log-probs (the frozen reference policy). No grad.
    log pi_S    student log-probs at rollout time (``old_log_probs``). No grad.
    c           per-sequence correctness in {0, 1} from the rule-based reward.
    m           response mask with the last valid token zeroed (see `opd_mask`).

and the teacher term that every method is built from:

    opd_term = (log pi_T - log pi_S) * m

Both log-probs come from no-grad forward passes, so ``opd_term`` is detached and
enters the update only through the advantage.
"""

from collections import defaultdict

import torch

OPD_CLAMP = 5.0


def opd_mask(response_mask: torch.Tensor) -> torch.Tensor:
    """Response mask with the last valid token of each sequence zeroed.

    That position is the EOS token, and the student and the teacher do not agree
    on what it is: the student is a base model and the teacher an instruct model,
    so their end-of-sequence tokens differ. Comparing their log-probs there
    compares two different tokens, so every method drops that position from the
    teacher term. This is deliberate, not an off-by-one.
    """
    mask = response_mask.clone()
    lengths = mask.long().sum(dim=1)
    last_idx = (lengths - 1).clamp(min=0)
    mask[torch.arange(mask.size(0), device=mask.device), last_idx] = 0
    return mask


def opd_term(teacher_logprob, student_logprob, response_mask):
    """(log pi_T - log pi_S) * m -- the raw teacher signal, per token."""
    return (teacher_logprob - student_logprob) * opd_mask(response_mask)


def opd(teacher_logprob, student_logprob, response_mask):
    """OPD -- on-policy distillation. The teacher term is the whole advantage.

        A_t = clip(log pi_T - log pi_S, -5, 5) * m

    No verifiable reward enters the update.
    """
    return opd_term(teacher_logprob, student_logprob, response_mask).clamp(-OPD_CLAMP, OPD_CLAMP)


def grpo_opd(grpo_adv, teacher_logprob, student_logprob, response_mask, beta, clamp=True):
    """KDRL -- the teacher term is added to every rollout.

        A_t = A_grpo + beta * (log pi_T - log pi_S) * m

    ``beta`` is either fixed (KDRL-naive) or annealed linearly with the global
    step (KDRL-Anneal): ``beta_t = max(init - delta * step, floor)``.

    The teacher term is clipped to [-5, 5] for the same reason ``opd`` clips it:
    a single token where teacher and student disagree by orders of magnitude
    would otherwise dominate the sequence's advantage.

    NOTE: the original logic-puzzle runs did not clip here -- this was the one
    estimator whose two ports disagreed, and both now clip. ``clamp=False``
    reproduces those original runs.
    """
    term = opd_term(teacher_logprob, student_logprob, response_mask)
    if clamp:
        term = term.clamp(-OPD_CLAMP, OPD_CLAMP)
    return grpo_adv + beta * term


def grpo_opd_mask(grpo_adv, teacher_logprob, student_logprob, correctness, response_mask, beta):
    """KDRL-MASK -- the teacher term is gated to rollouts that got it wrong.

        A_t = A_grpo + beta * opd_term * 1[c < 1]

    A rollout that already solved the task is left to the verifiable reward
    alone, so the teacher cannot overwrite a correct solution with its own style.
    """
    incorrect = (correctness < 1.0).to(grpo_adv.dtype).unsqueeze(-1)
    return grpo_adv + beta * opd_term(teacher_logprob, student_logprob, response_mask) * incorrect


def grpo_opd_cliff(grpo_adv, teacher_logprob, student_logprob, correctness, index, response_mask, beta):
    """HDPO-cliff -- the teacher term fires only on groups where everything failed.

        A_t = A_grpo + beta * opd_term * 1[max_group(c) < 1]

    When no rollout in a group succeeds, the group-normalised GRPO advantage is
    identically zero and RL has no gradient at all. That is the cliff, and the
    teacher fills it. On every other group the teacher stays out.

    ``index`` is the per-sequence group id (``uid``), length ``batch``.
    """
    group_max = {}
    for gid, c in zip(index, correctness.tolist()):
        key = gid.item() if hasattr(gid, "item") else gid
        group_max[key] = max(group_max.get(key, -1.0), c)
    cliff = torch.tensor(
        [1.0 if group_max[gid.item() if hasattr(gid, "item") else gid] < 1.0 else 0.0 for gid in index],
        dtype=grpo_adv.dtype,
        device=grpo_adv.device,
    ).unsqueeze(-1)
    return grpo_adv + beta * opd_term(teacher_logprob, student_logprob, response_mask) * cliff


def grpo_srpo(grpo_adv, teacher_logprob, student_logprob, correctness, response_mask, beta):
    """SRPO -- the two signals are disjoint over samples.

        A_t = A_grpo * 1[c = 1] + beta * opd_term * 1[c < 1]

    Correct rollouts are updated by the verifiable reward alone, incorrect ones
    by the teacher alone. Unlike KDRL-MASK, a correct rollout's GRPO advantage
    is the only thing acting on it and an incorrect rollout's is zeroed out.
    """
    correct = (correctness >= 1.0).to(grpo_adv.dtype).unsqueeze(-1)
    incorrect = 1.0 - correct
    return grpo_adv * correct + beta * opd_term(teacher_logprob, student_logprob, response_mask) * incorrect


def grpo_rlsd(grpo_adv, teacher_logprob, student_logprob, response_mask, eps_w=0.2, lam=1.0):
    """RLSD -- the teacher reshapes magnitude, the reward keeps the sign.

        d_t = log pi_T - log pi_S
        w_t = exp(sign(A_grpo) * d_t)
        A_t = A_grpo * ((1 - lam) + lam * clip(w_t, 1 - eps_w, 1 + eps_w))

    Because the multiplier is positive, no token's advantage can change sign:
    the teacher can only say which tokens within an already-rewarded (or
    already-penalised) sequence deserve more weight. Note this method does NOT
    drop the EOS token -- it uses the plain response mask, since d_t enters as a
    weight rather than as an additive signal.
    """
    sign_a = torch.sign(grpo_adv[:, 0]).to(student_logprob.dtype).unsqueeze(-1)
    delta = teacher_logprob - student_logprob
    w = torch.exp(sign_a * delta).clamp(min=1.0 - eps_w, max=1.0 + eps_w)
    blend = (1.0 - lam) + lam * w
    return (grpo_adv * blend) * response_mask


def trrd_policy_loss_ratio(log_prob, old_log_prob, teacher_log_prob, alpha):
    """TRRD -- the teacher anchors the PPO importance ratio, not the advantage.

        log pi_mix = alpha * log pi_old + (1 - alpha) * clamp_log pi_T
        ratio      = exp(log pi - log pi_mix)

    ``log(pi_S / pi_T)`` is clamped to [-1, 1] before mixing (RLAD Appendix B)
    so the ratio stays finite when teacher and student are far apart; the clamp
    is written as a shift on the teacher's log-prob so gradient still flows
    through ``log_prob``. Advantages are plain GRPO. Reported ``ppo_kl`` is still
    taken against the true old policy.

    This is the only method that touches ``compute_policy_loss`` rather than
    ``compute_advantage``. It is implemented for the FSDP actor only, which is
    what every run in the paper used.

    ``teacher_log_prob`` arrives with the last valid response token pinned to
    ``old_log_prob``, so the ratio there is the plain PPO one -- the same reason
    every estimator drops that token from the teacher term. The paper's math-side
    TRRD runs did not pin it; that was an inconsistency between the two ports and
    both pin it now.
    """
    clamped_teacher = log_prob - torch.clamp(log_prob - teacher_log_prob, -1.0, 1.0)
    mixture = float(alpha) * old_log_prob + (1.0 - float(alpha)) * clamped_teacher
    return torch.exp(log_prob - mixture)


def grpo_outcome_advantage(token_level_rewards, response_mask, index, eps=1e-6):
    """Group-normalised outcome advantage, as verl computes it.

    Included so the reference implementation is runnable end to end. The
    sequence reward is normalised within its prompt group and broadcast to every
    response token; groups of size one get mean 0 and std 1.
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    for i, gid in enumerate(index):
        id2score[gid].append(scores[i])
    id2mean, id2std = {}, {}
    for gid, vals in id2score.items():
        if len(vals) == 1:
            id2mean[gid] = torch.tensor(0.0, device=scores.device)
            id2std[gid] = torch.tensor(1.0, device=scores.device)
        else:
            stacked = torch.stack(vals)
            id2mean[gid] = stacked.mean()
            id2std[gid] = stacked.std()
    normed = torch.stack([(scores[i] - id2mean[gid]) / (id2std[gid] + eps) for i, gid in enumerate(index)])
    return normed.unsqueeze(-1) * response_mask
