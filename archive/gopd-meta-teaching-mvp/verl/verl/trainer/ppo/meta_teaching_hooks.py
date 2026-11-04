# Copyright 2026 Meta-Teaching MVP contributors
# Licensed under the Apache License, Version 2.0 (see verl/LICENSE).

"""Small, opt-in hooks for the meta-teaching MVP.

The existing trainer remains the execution backend. This module handles teacher-only
conditioning, pre-query gating, phase budgets and machine-readable results. Heavy
training dependencies are imported only by the functions that need them.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from collections.abc import Mapping


def value(config, path, default=None):
    current = config
    for part in path.split("."):
        if current is None or not hasattr(current, "get"):
            return default
        current = current.get(part, default)
    return current


def enabled(config):
    return bool(value(config, "meta_teaching.enabled", False))


def validate_phase_config(config):
    """Fail before starting Ray if an experimental invariant is violated."""
    if not enabled(config):
        return
    meta = config.get("meta_teaching")
    phase = meta.get("phase")
    if phase not in {"warmup", "rl", "eval"}:
        raise ValueError("meta_teaching.phase must be warmup, rl or eval")
    if meta.get("exposure", "plain") not in {"plain", "partial", "full"}:
        raise ValueError("exposure must be plain, partial or full")
    exposure_fraction = float(meta.get("exposure_fraction", 0.5))
    if not 0 <= exposure_fraction <= 1:
        raise ValueError("exposure_fraction must be in [0, 1]")
    if meta.get("exposure", "plain") == "partial" and not 0 < exposure_fraction < 1:
        raise ValueError("partial exposure_fraction must be strictly between 0 and 1")
    if not 0 <= float(meta.get("query_fraction", 1.0)) <= 1:
        raise ValueError("query_fraction must be in [0, 1]")
    if meta.get("gate", "random") not in {"random", "uncertainty"}:
        raise ValueError("gate must be random or uncertainty")
    if int(meta.get("phase_steps", 0)) < 0:
        raise ValueError("phase_steps must be nonnegative")
    if int(meta.get("teacher_max_prompt_length", 4096)) < 1:
        raise ValueError("teacher_max_prompt_length must be positive")
    if not os.path.isabs(str(meta.get("output_dir", ""))):
        raise ValueError("meta_teaching.output_dir must be an absolute path")
    if float(meta.get("validation_reserve_seconds", 0)) < 0:
        raise ValueError("validation_reserve_seconds must be nonnegative")
    deadline = meta.get("deadline_unix")
    if deadline is not None and (not math.isfinite(float(deadline)) or float(deadline) <= 0):
        raise ValueError("deadline_unix must be a positive finite Unix timestamp")
    if value(config, "actor_rollout_ref.actor.strategy") not in {"fsdp", "fsdp2"}:
        raise ValueError("MVP supports FSDP/FSDP2 only")
    if value(config, "actor_rollout_ref.rollout.multi_turn.enable", False):
        raise ValueError("MVP is single-turn only; multi-turn requires separate attention/loss handling")
    if value(config, "actor_rollout_ref.rollout.mode", "sync") != "sync":
        raise ValueError("MVP supports synchronous rollout only")
    if value(config, "actor_rollout_ref.model.lora_rank", 0) or value(config, "actor_rollout_ref.model.lora_adapter_path") is not None:
        raise ValueError("MVP uses a separate teacher and full student checkpoints; LoRA is not supported")
    if value(config, "reward_model.enable", False):
        raise ValueError("MVP requires a rule-based task reward, not an extra reward model")
    if value(config, "algorithm.adv_estimator") != "grpo":
        raise ValueError("MVP uses GRPO for the independent RL phase")
    if value(config, "algorithm.use_kl_in_reward", False):
        raise ValueError("MVP does not allow a reference KL term in task reward")
    for key in ("actor_rollout_ref.model.base_model_path", "actor_rollout_ref.ref.model.base_model_path"):
        if value(config, key) is not None:
            raise ValueError(f"MVP forbids corrected-reward base-model loading: {key}")
    if value(config, "algorithm.critique_vllm_url") is not None or value(config, "algorithm.use_ref_solution_distillation", False):
        raise ValueError("Use meta_teaching.exposure; disable upstream critique/ref-solution paths")
    if value(config, "actor_rollout_ref.actor.policy_loss.multi_teacher_distill", False):
        raise ValueError("MVP has one frozen teacher per phase")
    if float(value(config, "actor_rollout_ref.actor.kl_loss_coef", 0)) != 0:
        raise ValueError("MVP requires actor.kl_loss_coef=0")
    if phase == "warmup":
        if value(config, "actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla") != "vanilla":
            raise ValueError("MVP warmup uses the vanilla sampled-token OPD policy loss")
        if not value(config, "actor_rollout_ref.actor.use_kl_loss", False):
            raise ValueError("warmup requires use_kl_loss=true to instantiate G-OPD's reference worker")
        if not value(config, "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages", False):
            raise ValueError("warmup must use only_reverse_kl_advantages=true")
        if float(value(config, "actor_rollout_ref.actor.policy_loss.lambda_vals", 1)) != 1:
            raise ValueError("warmup uses vanilla sampled-token OPD: lambda_vals must be 1")
        if not value(config, "data.return_raw_chat", False):
            raise ValueError("warmup requires data.return_raw_chat=true")
        if float(value(config, "actor_rollout_ref.actor.entropy_coeff", 0)) != 0:
            raise ValueError("warmup requires entropy_coeff=0 to keep skipped samples free of loss terms")
        if not value(config, "actor_rollout_ref.actor.use_rollout_log_probs", False):
            raise ValueError("warmup requires actor.use_rollout_log_probs=true to retain the stored old_log_probs anchor")
    else:
        if value(config, "actor_rollout_ref.actor.use_kl_loss", False):
            raise ValueError("RL/eval must set use_kl_loss=false: no reference worker is allowed")
        if value(config, "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages", False):
            raise ValueError("RL/eval must disable only_reverse_kl_advantages")
        if value(config, "actor_rollout_ref.ref.model.path") is not None:
            raise ValueError("RL/eval must clear ref.model.path so no teacher tokenizer/model is loaded")
    if phase == "eval" and not value(config, "trainer.val_only", False):
        raise ValueError("eval requires trainer.val_only=true")
    if phase != "eval" and value(config, "trainer.val_only", False):
        raise ValueError("training phases must set trainer.val_only=false")
    for operation in ("save_contents", "load_contents"):
        contents = value(config, f"actor_rollout_ref.actor.checkpoint.{operation}", ["model", "optimizer", "extra"])
        if not {"model", "optimizer", "extra"}.issubset(set(contents)):
            raise ValueError(f"MVP requires full checkpoint {operation}: model, optimizer, extra (RNG/scheduler)")
    if value(config, "trainer.del_local_ckpt_after_load", False):
        raise ValueError("MVP preserves branch source checkpoints; del_local_ckpt_after_load must be false")
    if value(config, "trainer.default_hdfs_dir") is not None:
        raise ValueError("MVP phase manifests require local/shared-filesystem checkpoints")
    if value(config, "actor_rollout_ref.actor.optim.lr_scheduler_type", "constant") != "constant":
        raise ValueError("MVP requires a constant LR so phase-local horizons do not change resumed scheduler semantics")
    if value(config, "actor_rollout_ref.actor.optim.warmup_style") not in {None, "constant"}:
        raise ValueError("MVP requires constant LR (including the deprecated warmup_style override)")
    if float(value(config, "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio", 0)) != 0 or int(value(config, "actor_rollout_ref.actor.optim.lr_warmup_steps", 0) or 0) > 0:
        raise ValueError("MVP requires zero LR warmup; optimizer and scheduler state are preserved on resume")


def assert_same_tokenizer(student, teacher):
    """Sampled-token KL requires identical token IDs and text encoding."""
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("MVP requires identical student/teacher token vocabularies; cross-tokenizer OPD is unsupported")
    for attribute in ("bos_token_id", "eos_token_id", "pad_token_id", "chat_template"):
        if getattr(student, attribute, None) != getattr(teacher, attribute, None):
            raise ValueError(f"MVP requires identical student/teacher {attribute}")
    # Equal vocabularies alone do not guarantee equal pre-tokenization/normalization.
    left = getattr(student, "backend_tokenizer", None)
    right = getattr(teacher, "backend_tokenizer", None)
    if left is not None and right is not None:
        left = json.loads(left.to_str())
        right = json.loads(right.to_str())
        for section in ("model", "normalizer", "pre_tokenizer", "post_processor", "added_tokens"):
            if left.get(section) != right.get(section):
                raise ValueError(f"MVP requires identical tokenizer {section}")


def strip_explicit_final_answer(solution):
    """Remove boxed/fboxed results and explicit final-answer tails before prefixing.

    This does not claim that reasoning contains no information about the answer;
    it prevents an explicit final-result suffix from entering the partial arm.
    """
    text = str(solution)
    markers = list(re.finditer(r"(?im)(?:^\s*####|\bfinal\s+answer\s*[:：]|\bthe\s+answer\s+is\b|最终答案\s*[:：]|答案是)", text))
    if markers:
        text = text[:markers[0].start()]
    pattern = re.compile(r"\\(?:boxed|fbox)\s*\{")
    while match := pattern.search(text):
        depth = 1
        end = match.end()
        while end < len(text) and depth:
            if text[end] == "{" and (end == 0 or text[end - 1] != "\\"):
                depth += 1
            elif text[end] == "}" and (end == 0 or text[end - 1] != "\\"):
                depth -= 1
            end += 1
        text = text[:match.start()] + text[end:]
    return text.strip()


def partial_reference(solution, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("partial exposure fraction must be in (0, 1]")
    text = strip_explicit_final_answer(solution)
    if not text:
        raise ValueError("reference_solution has no reasoning after removing the explicit final answer")
    # Character-prefix truncation is deterministic and works for single-line and
    # multilingual derivations, unlike assumptions about English sentence splits.
    return text[:max(1, math.ceil(len(text) * fraction))].rstrip()


def choose_query_indices(old_log_probs, response_attention, fraction, gate, seed):
    """Select using only student-side information, before any teacher RPC."""
    n = int(old_log_probs.shape[0])
    if not 0 <= fraction <= 1:
        raise ValueError("query fraction must be in [0, 1]")
    count = min(n, math.ceil(n * fraction))
    if count == 0:
        return []
    if gate == "random":
        return sorted(random.Random(seed).sample(range(n), count))
    if gate == "uncertainty":
        mask = response_attention.to(old_log_probs.dtype)
        score = -(old_log_probs * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        ranked = sorted(range(n), key=lambda i: (-float(score[i]), i))
        return sorted(ranked[:count])
    raise ValueError(f"Unknown gate: {gate}")


def prepare_teacher_inputs(batch, tokenizer, meta, template_kwargs=None):
    """Add teacher-only tensors; never decode/re-encode student responses."""
    import torch

    if meta.get("exposure", "plain") == "plain":
        return batch
    raw_prompts = batch.non_tensor_batch.get("raw_prompt")
    extras = batch.non_tensor_batch.get("extra_info")
    if raw_prompts is None or extras is None:
        raise ValueError("conditioned OPD requires raw_prompt and extra_info.reference_solution")
    prompt_ids = []
    for raw, extra in zip(raw_prompts, extras):
        solution = extra.get("reference_solution") if isinstance(extra, Mapping) else None
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError("Every selected training sample needs nonempty extra_info.reference_solution")
        if meta.get("exposure") == "partial":
            solution = partial_reference(solution, float(meta.get("exposure_fraction", 0.5)))
        messages = [dict(message) for message in list(raw)]
        if not messages or messages[-1].get("role") != "user" or not isinstance(messages[-1].get("content"), str):
            raise ValueError("MVP expects text-only single-turn chat ending in a user message")
        if any(message.get("role") in {"assistant", "tool"} for message in messages):
            raise ValueError("MVP conditioned OPD does not support previous assistant/tool turns")
        messages[-1]["content"] += (
            "\n\nTeacher-only reference information:\n" + solution
            + "\n\nUse the reference information to assess the student's continuation of the original problem."
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **dict(template_kwargs or {})
        )
        tokens = tokenizer.encode(rendered, add_special_tokens=False)
        maximum = int(meta.get("teacher_max_prompt_length", 4096))
        if len(tokens) > maximum:
            raise ValueError(f"Teacher prompt has {len(tokens)} tokens, exceeds teacher_max_prompt_length={maximum}; shorten data or raise the cap")
        prompt_ids.append(tokens)
    responses = batch.batch["responses"]
    prompt_length = max(map(len, prompt_ids))
    if tokenizer.pad_token_id is None:
        raise ValueError("MVP teacher tokenizer needs pad_token_id")
    prompt_tensor = torch.full((len(prompt_ids), prompt_length), tokenizer.pad_token_id, dtype=responses.dtype, device=responses.device)
    prompt_mask = torch.zeros_like(prompt_tensor)
    for i, tokens in enumerate(prompt_ids):
        prompt_tensor[i, -len(tokens):] = torch.tensor(tokens, dtype=responses.dtype, device=responses.device)
        prompt_mask[i, -len(tokens):] = 1
    # Attention, NOT response_mask: a loss mask can contain holes in other paths.
    response_attention = batch.batch["attention_mask"][:, -responses.shape[-1]:]
    attention = torch.cat((prompt_mask, response_attention), dim=-1)
    batch.batch["ref_input_ids"] = torch.cat((prompt_tensor, responses), dim=-1)
    batch.batch["ref_attention_mask"] = attention
    batch.batch["ref_position_ids"] = torch.cumsum(attention, dim=-1).sub(1).clamp_min(0)
    return batch


def score_selected_teacher(batch, worker, tokenizer, meta, step, template_kwargs=None):
    """Pad the queried subset for distributed dispatch, then scatter only its scores."""
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    old = batch.batch["old_log_probs"]
    response_length = batch.batch["responses"].shape[-1]
    response_attention = batch.batch["attention_mask"][:, -response_length:]
    indices = choose_query_indices(
        old, response_attention, float(meta.get("query_fraction", 1)),
        meta.get("gate", "random"), int(meta.get("seed", 42)) + int(step),
    )
    reference = old.detach().clone()
    counters = {"teacher_queries": 0, "teacher_selected_queries": len(indices), "teacher_scored_tokens": 0, "teacher_response_tokens": 0}
    if indices:
        selected = batch.select_idxs(indices)
        selected = prepare_teacher_inputs(selected, tokenizer, meta, template_kwargs)
        if "ref_input_ids" in selected.batch:
            # G-OPD's dynamic microbatch packer reads the ordinary attention mask.
            # On this teacher-only subset, expose the augmented lengths there too
            # so its token budget does not underestimate full-reference prompts.
            for key in ("input_ids", "attention_mask", "position_ids"):
                selected.batch[key] = selected.batch[f"ref_{key}"]
        selected.meta_info = dict(selected.meta_info)
        selected, pad_size = pad_dataproto_to_divisor(selected, worker.world_size)
        selected.meta_info["global_token_num"] = selected.batch["attention_mask"].sum(-1).tolist()
        counters["teacher_queries"] = len(selected)
        counters["teacher_scored_tokens"] = int(selected.batch.get("ref_attention_mask", selected.batch["attention_mask"]).sum().item())
        counters["teacher_response_tokens"] = int(selected.batch["attention_mask"][:, -response_length:].sum().item())
        result = worker.compute_ref_log_prob(selected)
        result = unpad_dataproto(result, pad_size=pad_size)
        scores = result.batch["ref_log_prob"]
        if tuple(scores.shape) != (len(indices), response_length):
            raise ValueError("Teacher log-probabilities are not aligned with the original student response tokens")
        if not scores.isfinite().all():
            raise ValueError("Teacher returned non-finite token log-probabilities")
        reference[indices] = scores.to(reference.device, dtype=reference.dtype)
    batch.batch["ref_log_prob"] = reference
    return batch, counters


def _jsonable(obj):
    if hasattr(obj, "item"):
        return _jsonable(obj.item())
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


class PhaseState:
    """Driver-side bookkeeping; a deadline is checked only at batch boundaries."""
    def __init__(self, meta):
        self.meta = meta
        self.start_wall = time.time()
        self.start_step = 0
        self.end_step = 0
        self.target_step = 0
        self.checkpoint_path = None
        self.validation_step = None
        self.val_metrics = {}
        self.counters = {"teacher_queries": 0, "teacher_selected_queries": 0, "teacher_scored_tokens": 0, "teacher_response_tokens": 0}

    def begin(self, restored_step):
        self.start_step = self.end_step = int(restored_step)
        self.target_step = self.start_step + int(self.meta.get("phase_steps", 0))

    def should_stop_for_budget(self, now=None):
        deadline = self.meta.get("deadline_unix")
        if deadline is None:
            return False
        now = time.time() if now is None else now
        return now >= float(deadline) - float(self.meta.get("validation_reserve_seconds", 0))

    def add_queries(self, counters):
        for key, count in counters.items():
            self.counters[key] += int(count)

    def validated(self, step, metrics):
        self.validation_step = int(step)
        self.val_metrics = dict(metrics)

    def write(self, status, error=None):
        now = time.time()
        deadline = self.meta.get("deadline_unix")
        metrics = self.val_metrics if self.validation_step == self.end_step else {}
        score = metrics.get("meta_teaching/val_score")
        payload = {
            "schema_version": 1,
            "phase_id": self.meta.get("phase_id"),
            "spec_fingerprint": self.meta.get("spec_fingerprint"),
            "seed": self.meta.get("seed"),
            "validation_step": self.validation_step,
            "source_checkpoint": self.meta.get("source_checkpoint"),
            "phase": self.meta.get("phase"), "arm_id": self.meta.get("arm_id", "unknown"),
            "status": status, "start_step": self.start_step, "end_step": self.end_step,
            "steps_completed": self.end_step - self.start_step,
            "step_unit": "rollout_training_batch; an empty teacher gate skips the optimizer call",
            "checkpoint_path": self.checkpoint_path, "val_score": score, "val_metrics": metrics,
            "elapsed_seconds": now - self.start_wall,
            "teacher_enabled": self.meta.get("phase") == "warmup",
            "deadline_overrun_seconds": max(0.0, now - float(deadline)) if deadline is not None else 0.0,
            "budget_enforcement": "batch_boundary_with_validation_reserve",
            "teacher_counter_scope": "actual_scored_sequences_and_valid_input_tokens_including_dispatch_padding",
            "elapsed_scope": "trainer_construction_through_final_validation_and_checkpoint; use launcher wall time for full job cost",
            "optimizer_resume": "preserve_model_optimizer_training_rng_scheduler; constant_lr",
            "rollout_resume": "torch_rollout_rng_restored; inference_engine_reinitialized_at_phase_boundary; no_bitwise_replay_guarantee",
            **self.counters,
        }
        if error is not None:
            payload["error"] = str(error)
        directory = Path(self.meta["output_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".phase_summary.", suffix=".json", dir=directory)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as output:
                json.dump(_jsonable(payload), output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, directory / "phase_summary.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return payload
