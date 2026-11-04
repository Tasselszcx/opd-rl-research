"""Read-only deployment checks. Never installs packages or downloads models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

from .data import validate_manifest

# From the original G-OPD installation script, not an assertion of GPU validation.
RECOMMENDED_CORE = {"torch": "2.6.0", "vllm": "0.8.5.post1"}
REQUIRED_PACKAGES = (
    "torch", "vllm", "transformers", "ray", "hydra-core", "omegaconf", "datasets",
    "pyarrow", "tensordict", "torchdata", "numpy", "accelerate", "flash-attn", "verl",
)


def _check(name: str, status: str, detail: str, **extra) -> dict:
    return {"name": name, "status": status, "detail": detail, **extra}


def _json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def inspect_local_model(path: str | Path) -> dict:
    """Inspect config, weights and tokenizer semantics without loading model code."""
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise ValueError(f"Local model directory does not exist: {directory}")
    config_path, tokenizer_path = directory / "config.json", directory / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise ValueError(f"Model requires config.json and tokenizer.json: {directory}")
    config, tokenizer = _json(config_path), _json(tokenizer_path)
    model = tokenizer.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), (dict, list)):
        raise ValueError(f"tokenizer.json has no supported vocabulary: {directory}")
    vocab = model["vocab"]
    token_ids = list(vocab.values()) if isinstance(vocab, dict) else list(range(len(vocab)))
    added = tokenizer.get("added_tokens", [])
    if not isinstance(added, list) or any(not isinstance(item, dict) for item in added):
        raise ValueError(f"Invalid added_tokens in {directory}")
    for token in added:
        token_id = token.get("id")
        if not isinstance(token_id, int) or not isinstance(token.get("content"), str):
            raise ValueError(f"Invalid added token entry in {directory}")
        token_ids.append(token_id)
    if not token_ids or any(not isinstance(value, int) or value < 0 for value in token_ids):
        raise ValueError(f"Invalid token IDs in {directory}")
    vocab_size = config.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= max(token_ids):
        raise ValueError(f"Model vocab_size does not cover tokenizer IDs in {directory}")
    semantics = {key: tokenizer.get(key) for key in (
        "model", "normalizer", "pre_tokenizer", "post_processor", "decoder"
    )}
    semantics["added_tokens"] = sorted(added, key=lambda item: (item["id"], item["content"]))
    weights = list(directory.glob("*.safetensors")) + list(directory.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError(f"No local safetensors or pytorch_model*.bin weights in {directory}")
    for index_path in directory.glob("*.index.json"):
        weight_map = _json(index_path).get("weight_map")
        if isinstance(weight_map, dict):
            missing = [name for name in set(weight_map.values()) if not (directory / name).is_file()]
            if missing:
                raise ValueError(f"Missing model shards in {directory}: {missing[:3]}")
    tokenizer_config_path = directory / "tokenizer_config.json"
    tokenizer_config = _json(tokenizer_config_path) if tokenizer_config_path.is_file() else {}
    chat_template_present = bool(tokenizer_config.get("chat_template")) or (directory / "chat_template.jinja").is_file()
    if not chat_template_present:
        raise ValueError(f"No local chat template found for chat-formatted data in {directory}")
    semantics["chat_template"] = tokenizer_config.get("chat_template")
    template_path = directory / "chat_template.jinja"
    if template_path.is_file():
        semantics["chat_template"] = template_path.read_text(encoding="utf-8")
    semantics["special_tokens"] = {key: tokenizer_config.get(key) for key in ("bos_token", "eos_token", "pad_token")}
    signature = hashlib.sha256(json.dumps(semantics, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        "path": str(directory.resolve()), "model_type": config.get("model_type"),
        "vocab_size": vocab_size, "tokenizer_sha256": signature,
        "vocab_entries": len(vocab), "added_tokens": len(added),
        "weight_files": len(weights), "chat_template_present": chat_template_present,
    }


def compare_tokenizers(student_path: str | Path, teacher_path: str | Path) -> dict:
    student, teacher = inspect_local_model(student_path), inspect_local_model(teacher_path)
    if student["tokenizer_sha256"] != teacher["tokenizer_sha256"]:
        raise ValueError("Student and teacher tokenizer semantics differ (vocabulary, added tokens, merges or preprocessing)")
    if student["vocab_size"] != teacher["vocab_size"]:
        raise ValueError("Student and teacher model vocab_size differ; this MVP requires the same output vocabulary")
    return {"student": student, "teacher": teacher}


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~")) or Path(value).exists()


def _resolve_model(value: Any, root: Path) -> tuple[str, bool]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model path must be a nonempty string")
    if "${" in value:
        raise ValueError("Unexpanded model path variable; load the config through meta_teaching.config first")
    expanded = os.path.expanduser(value)
    if Path(expanded).is_absolute():
        return expanded, True
    candidate = root / expanded
    if candidate.exists() or _looks_like_local_path(value):
        return str(candidate), True
    # HF IDs cannot be verified locally without resolving/downloading a snapshot.
    return value, False


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in (*REQUIRED_PACKAGES, "math-verify", "packaging"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu_checks(expected: int) -> list[dict]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [_check("gpu", "fail", "nvidia-smi is not available; run deployment preflight on the GPU host")]
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=15, check=True,
        )
        devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.SubprocessError, OSError) as exc:
        return [_check("gpu", "fail", f"nvidia-smi failed: {type(exc).__name__}: {exc}")]
    if len(devices) < expected:
        return [_check("gpu", "fail", f"Expected {expected} GPUs, nvidia-smi reports {len(devices)}", devices=devices)]
    # Query availability only; never create a CUDA tensor, context, model or process group.
    try:
        import torch
        visible_count = torch.cuda.device_count()
        available = torch.cuda.is_available()
        if not available or visible_count < expected:
            return [_check("gpu", "fail", f"PyTorch sees {visible_count} usable GPUs, expected {expected}", devices=devices)]
    except Exception as exc:
        return [_check("gpu", "fail", f"PyTorch CUDA availability check failed: {type(exc).__name__}: {exc}")]
    return [_check("gpu", "pass", f"{visible_count} visible CUDA devices; availability only, no training performed", devices=devices)]


def run_preflight(config: dict, repository_root: Path, require_gpu: bool = False) -> dict:
    root = Path(repository_root).resolve()
    checks: list[dict] = []
    entry = root / "verl" / "verl" / "trainer" / "main_ppo.py"
    checks.append(_check("repository", "pass" if entry.is_file() else "fail", str(entry)))
    models = config.get("models", {})
    model_reports = {}
    for role in ("student", "teacher"):
        value = models.get(role, config.get(f"{role}_model"))
        try:
            model_path, local = _resolve_model(value, root)
            if not local:
                checks.append(_check(f"model.{role}", "fail" if require_gpu else "warn",
                                     f"not_locally_verified: {model_path}; supply a downloaded local model directory"))
                continue
            model_reports[role] = inspect_local_model(model_path)
            checks.append(_check(f"model.{role}", "pass", "Local metadata, tokenizer and weight-file presence checked; weights not loaded",
                                 model=model_reports[role]))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            checks.append(_check(f"model.{role}", "fail", str(exc)))
    if len(model_reports) == 2:
        student, teacher = model_reports["student"], model_reports["teacher"]
        same = student["tokenizer_sha256"] == teacher["tokenizer_sha256"] and student["vocab_size"] == teacher["vocab_size"]
        checks.append(_check("tokenizer_compatibility", "pass" if same else "fail",
                             "Compared vocabulary IDs, added tokens, tokenizer model/merges, normalization and output vocab_size"
                             if same else "Student/teacher tokenizer semantics or output vocab_size differ; not supported by this MVP"))
    else:
        checks.append(_check("tokenizer_compatibility", "fail" if require_gpu else "warn", "not_locally_verified: both local tokenizers are required"))
    data = config.get("data", {})
    manifest_value = data.get("manifest", config.get("manifest_path"))
    manifest = None
    try:
        if not manifest_value:
            raise ValueError("data.manifest is required")
        manifest_path = Path(manifest_value).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        manifest = validate_manifest(manifest_path)
        checks.append(_check("data", "pass", "Verified Parquet hashes, actual schema, identities and cross-split isolation"))
        if manifest.get("demo_only"):
            checks.append(_check("demo_data", "warn", "Synthetic arithmetic fixtures only: not research evidence or a benchmark"))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        checks.append(_check("data", "fail", str(exc)))
    reward = data.get("reward", "math_verify")
    if reward not in ("numeric_demo", "math_verify"):
        checks.append(_check("reward", "fail", f"Unsupported reward verifier: {reward}"))
    elif reward == "numeric_demo":
        checks.append(_check("reward", "warn" if manifest and manifest.get("demo_only") else "fail",
                             "numeric_demo is explicitly selected and permitted only with a demo_only manifest"))
    else:
        checks.append(_check("reward", "pass", "Production scorer explicitly uses upstream math-verify, with no numeric fallback"))
    versions = _package_versions()
    required = list(REQUIRED_PACKAGES) + (["math-verify"] if reward == "math_verify" else [])
    missing = [name for name in required if versions.get(name) is None]
    checks.append(_check("dependencies", "fail" if missing and require_gpu else "warn" if missing else "pass",
                         f"Missing packages: {', '.join(missing)}" if missing else "Required package metadata found; import/ABI compatibility still needs a GPU smoke run"))
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        # Requirements visible in this pinned checkout; this is not a CUDA ABI test.
        supported = {"numpy": "<2", "pyarrow": ">=19", "transformers": ">=4.51", "ray": ">=2.41"}
        for name, spec in supported.items():
            actual = versions.get(name)
            if actual and Version(actual) not in SpecifierSet(spec):
                checks.append(_check(f"constraint.{name}", "fail" if require_gpu else "warn",
                                     f"Installed {actual}, pinned checkout requires {name}{spec}"))
    except (ImportError, ValueError) as exc:
        checks.append(_check("version_constraints", "fail" if require_gpu else "warn",
                             f"Could not verify version constraints: {exc}"))
    for name, expected in RECOMMENDED_CORE.items():
        actual = versions.get(name)
        if actual and actual.split("+")[0] != expected:
            checks.append(_check(f"version.{name}", "warn", f"Installed {actual}; original G-OPD script pins {expected}. Validate this combination before long runs"))
    if versions.get("tensordict"):
        checks.append(_check("version.tensordict", "warn", "Upstream installer pins 0.6.2 but setup.py asks >=0.8,<=0.10,!=0.9; follow the MVP environment pin and smoke-test rather than auto-upgrading"))
    training = config.get("training", {})
    try:
        n_gpus = int(training.get("n_gpus", 8))
        tp = int(training.get("rollout_tp", 2))
        n = int(training.get("rollout_n", 4))
        batch = int(training.get("train_batch_size", 16))
        if min(n_gpus, tp, batch) < 1 or n_gpus % tp or n < 2:
            raise ValueError("Need positive GPU/batch sizes, n_gpus divisible by rollout_tp, and rollout_n >=2 for GRPO")
        checks.append(_check("training_shape", "pass", f"n_gpus={n_gpus}, rollout_tp={tp}, train_batch_size={batch}, rollout_n={n}"))
    except (ValueError, TypeError) as exc:
        n_gpus = 8
        checks.append(_check("training_shape", "fail", str(exc)))
    if require_gpu:
        checks.extend(_gpu_checks(n_gpus))
        if platform.system() != "Linux":
            checks.append(_check("platform", "fail", "GPU deployment profile requires Linux"))
    else:
        checks.append(_check("gpu", "warn", "GPU validation skipped; run python -m meta_teaching preflight --gpu on the deployment host"))
    return {
        "ok": all(check["status"] != "fail" for check in checks),
        "require_gpu": require_gpu, "checks": checks, "versions": versions,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "validation_scope": "read-only metadata/data checks; no model load, download or training",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    # Prefer the common loader so ${VAR} expansion and schema rules are identical.
    from .config import load_config
    config = load_config(args.config)
    report = run_preflight(config, Path(args.repository_root), args.require_gpu)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
