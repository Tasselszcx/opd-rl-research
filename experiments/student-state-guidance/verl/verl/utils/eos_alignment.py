"""Fold teacher termination symbols into the student's single EOS event."""
import torch


def semantic_eos_log_probs(logits, labels, log_probs, student_eos_id, teacher_eos_ids):
    if log_probs.dtype not in (torch.float32, torch.float64):
        raise ValueError('Semantic EOS requires float32 log probabilities (flash-attn cross entropy)')
    ids = sorted(set([int(student_eos_id)] + [int(x) for x in teacher_eos_ids]))
    if not ids or min(ids) < 0 or max(ids) >= logits.shape[-1]:
        raise ValueError('EOS alias outside the model vocabulary')
    # Reuse the exact normalizer implicit in the existing log-probability path.
    # This avoids materializing another full float32 vocabulary tensor.
    selected = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()
    log_z = selected - log_probs.float()
    alias_logits = logits.index_select(-1, torch.tensor(ids, device=logits.device)).float()
    stop_log_prob = torch.logsumexp(alias_logits, dim=-1) - log_z
    result = torch.where(labels == student_eos_id, stop_log_prob, log_probs.float())
    # The transferred mass cannot also remain on the old teacher EOS class.
    # A finite floor avoids inf * zero in the disabled KL-loss calculation.
    for alias in ids:
        if alias != student_eos_id:
            result = torch.where(labels == alias, torch.full_like(result, -80.), result)
    return result
