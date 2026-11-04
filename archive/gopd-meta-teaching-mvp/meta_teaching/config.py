"""Validated JSON configuration. File paths are relative to the project root."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from .common import REPOSITORY_ROOT, finite_number, project_path, read_json

DEFAULTS = {
    "schema_version": 1,
    "models": {"student": "", "teacher": ""},
    "data": {"manifest": "data/mvp/manifest.json", "reward": "math_verify"},
    "training": {
        "n_gpus": 8,
        "rollout_tp": 2,
        "train_batch_size": 16,
        "ppo_mini_batch_size": 16,
        "micro_batch_size_per_gpu": 1,
        "rollout_n": 4,
        "max_prompt_length": 1024,
        "max_response_length": 1024,
        "teacher_max_prompt_length": 3072,
        "learning_rate": 1e-6,
        "warmup_steps": 5,
        "rl_steps": 20,
        "walltime_max_steps": 100000,
        "gpu_memory_utilization": 0.45,
        "temperature": 1.0,
        "eval_n": 1,
        "eval_temperature": 0.0,
        "checkpoint_every": 5,
    },
    "experiment": {
        "output_dir": "runs/meta-teaching",
        "seed": 42,
        "rounds": 2,
        "repeats": 1,
        "protocol": "fixed_steps",
        "branch_budget_seconds": 3600,
        "warmup_budget_fraction": 0.25,
        "validation_reserve_seconds": 180,
        "deadline_grace_seconds": 300,
        "budget_tolerance_fraction": 0.1,
        "max_total_gpu_hours": 24,
        "min_rl_steps": 1,
        "cost_penalty_per_gpu_hour": 0.0,
        "promotion": "policy",
        "teacher_mode": "fixed",
        "evaluate_final_test": True,
        "bootstrap_all_actions": True,
        "candidates_per_round": 1,
    },
    "policy": {"ridge": 1.0, "exploration": 0.2},
    "actions": [
        {"id": "plain", "exposure": "plain", "exposure_fraction": 0.0, "query_fraction": 1.0, "gate": "random"},
        {"id": "partial", "exposure": "partial", "exposure_fraction": 0.5, "query_fraction": 0.5, "gate": "uncertainty"},
        {"id": "full", "exposure": "full", "exposure_fraction": 1.0, "query_fraction": 1.0, "gate": "random"},
    ],
}


def _merge(base, override, prefix=""):
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be an object")
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value


def _expand(value):
    if isinstance(value, str):
        value = os.path.expandvars(value)
        if re.search(r"\$\{[^}]+\}", value):
            raise ValueError(f"Unset environment variable in configuration: {value}")
        return value
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_config(config: dict) -> dict:
    if config["schema_version"] != 1:
        raise ValueError("Unsupported config schema_version")
    for role in ("student", "teacher"):
        if not isinstance(config["models"][role], str) or not config["models"][role].strip():
            raise ValueError(f"models.{role} must be a local model path or Hugging Face model ID")
    training = config["training"]
    for key in ("n_gpus", "rollout_tp", "train_batch_size", "ppo_mini_batch_size",
                "micro_batch_size_per_gpu", "rollout_n", "max_prompt_length",
                "max_response_length", "teacher_max_prompt_length", "warmup_steps",
                "rl_steps", "walltime_max_steps", "eval_n", "checkpoint_every"):
        _positive_int(training[key], f"training.{key}")
    if training["rollout_n"] < 2:
        raise ValueError("GRPO requires training.rollout_n >= 2")
    if training["n_gpus"] % training["rollout_tp"]:
        raise ValueError("n_gpus must be divisible by rollout_tp")
    if training["train_batch_size"] % training["n_gpus"]:
        raise ValueError("train_batch_size must be divisible by n_gpus")
    if training["train_batch_size"] % training["ppo_mini_batch_size"]:
        raise ValueError("train_batch_size must be divisible by ppo_mini_batch_size")
    if training["ppo_mini_batch_size"] % training["n_gpus"]:
        raise ValueError("ppo_mini_batch_size must be divisible by n_gpus for OPD n=1")
    local_mini = training["ppo_mini_batch_size"] // training["n_gpus"]
    if local_mini % training["micro_batch_size_per_gpu"]:
        raise ValueError("Per-rank OPD mini-batch must be divisible by micro_batch_size_per_gpu")
    finite_number(training["learning_rate"], "learning_rate", 1e-12)
    finite_number(training["temperature"], "temperature", 1e-12)
    finite_number(training["eval_temperature"], "eval_temperature", 0)
    utilization = finite_number(training["gpu_memory_utilization"], "gpu_memory_utilization", 0.01)
    if utilization >= 1:
        raise ValueError("gpu_memory_utilization must be < 1")
    experiment = config["experiment"]
    for key in ("rounds", "repeats", "min_rl_steps", "candidates_per_round"):
        _positive_int(experiment[key], f"experiment.{key}")
    _positive_int(experiment["seed"], "experiment.seed", 0)
    if experiment["protocol"] not in {"fixed_steps", "fixed_walltime"}:
        raise ValueError("protocol must be fixed_steps or fixed_walltime")
    if experiment["promotion"] not in {"policy", "validation_best"}:
        raise ValueError("promotion must be policy or validation_best (the latter is a search baseline)")
    if experiment["teacher_mode"] not in {"fixed", "previous_student"}:
        raise ValueError("teacher_mode must be fixed or previous_student")
    for key in ("branch_budget_seconds", "max_total_gpu_hours"):
        finite_number(experiment[key], f"experiment.{key}", 1e-12)
    for key in ("validation_reserve_seconds", "deadline_grace_seconds",
                "budget_tolerance_fraction", "cost_penalty_per_gpu_hour"):
        finite_number(experiment[key], f"experiment.{key}", 0)
    fraction = finite_number(experiment["warmup_budget_fraction"], "warmup_budget_fraction")
    if not 0 < fraction < 1:
        raise ValueError("warmup_budget_fraction must lie in (0,1)")
    if experiment["budget_tolerance_fraction"] > 0.5:
        raise ValueError("budget_tolerance_fraction > .5 would make matching meaningless")
    if experiment["protocol"] == "fixed_walltime":
        if experiment["validation_reserve_seconds"] >= fraction * experiment["branch_budget_seconds"]:
            raise ValueError("Reserve must be smaller than the warmup allocation; increase branch budget")
    for name in ("evaluate_final_test", "bootstrap_all_actions"):
        if not isinstance(experiment[name], bool):
            raise ValueError(f"experiment.{name} must be boolean")
    if config["data"]["reward"] not in {"math_verify", "numeric_demo"}:
        raise ValueError("data.reward must be math_verify or numeric_demo")
    if not isinstance(config["actions"], list) or not config["actions"]:
        raise ValueError("actions must be a nonempty list")
    ids = set()
    allowed = {"id", "exposure", "exposure_fraction", "query_fraction", "gate"}
    for action in config["actions"]:
        if not isinstance(action, dict) or set(action) != allowed:
            raise ValueError(f"Every action must contain exactly {sorted(allowed)}")
        name = action["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError(f"Unsafe action id: {name!r}")
        if name == "direct_rl" or name in ids:
            raise ValueError(f"Duplicate or reserved action id: {name}")
        ids.add(name)
        if action["exposure"] not in {"plain", "partial", "full"}:
            raise ValueError("Unknown teacher exposure")
        if action["gate"] not in {"random", "uncertainty"}:
            raise ValueError("Unknown query gate")
        for key in ("exposure_fraction", "query_fraction"):
            number = finite_number(action[key], key, 0)
            if number > 1:
                raise ValueError(f"{key} must be <= 1")
        if action["query_fraction"] == 0:
            raise ValueError("Use direct_rl for no teaching; warmup actions require query_fraction > 0")
        if action["exposure"] == "partial" and not 0 < action["exposure_fraction"] < 1:
            raise ValueError("Partial exposure requires a fraction in (0,1)")
    if experiment["candidates_per_round"] > len(ids):
        raise ValueError("candidates_per_round exceeds number of teaching actions")
    finite_number(config["policy"]["ridge"], "policy.ridge", 1e-12)
    exploration = finite_number(config["policy"]["exploration"], "policy.exploration", 0)
    if exploration > 1:
        raise ValueError("policy.exploration must be <= 1")
    return config


def load_config(path: Path | str, output_dir: str | None = None, repository_root: Path = REPOSITORY_ROOT) -> dict:
    config = copy.deepcopy(DEFAULTS)
    _merge(config, read_json(path))
    config = _expand(config)
    if output_dir:
        config["experiment"]["output_dir"] = output_dir
    config["data"]["manifest"] = str(project_path(config["data"]["manifest"], repository_root))
    config["experiment"]["output_dir"] = str(project_path(config["experiment"]["output_dir"], repository_root))
    for key, value in config["models"].items():
        if value.startswith((".", "/", "~")):
            config["models"][key] = str(project_path(value, repository_root))
    return validate_config(config)


def hydra_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if "${" in value or any(ord(char) < 32 for char in value):
            raise ValueError("Hydra CLI strings must not contain interpolation expressions or control characters")
        # Hydra quotes do not use JSON's general backslash escaping.
        escaped = re.sub(r'(\\+)(?="|$)', lambda match: match.group(0) * 2, value)
        return '"' + escaped.replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    if isinstance(value, dict):
        if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in value):
            raise ValueError("Hydra mapping keys must be simple identifiers")
        return "{" + ",".join(key + ":" + hydra_value(item) for key, item in value.items()) + "}"
    raise ValueError(f"Unsupported Hydra value: {type(value).__name__}")
