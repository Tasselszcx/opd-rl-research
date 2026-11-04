#!/usr/bin/env bash
# Default invocation is read-only. No environment activation, process cleanup,
# CUDA installation, model downloads or author-supplied installation scripts.
set -euo pipefail
MVP_SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MVP_PYTHON="${MVP_PYTHON:-python3}"
MVP_ENV_MODE="${1:-check}"
if [[ $# -gt 1 ]]; then
  echo "Usage: MVP_PYTHON=/path/to/python bash scripts/mvp_environment.sh [check|check-training|audit-metadata|install-cpu|install-hopper]" >&2
  exit 2
fi
exec "$MVP_PYTHON" - "$MVP_ENV_MODE" "$MVP_SCRIPT_ROOT" <<'PY'
"""Environment management confined to explicit modes; functions are CPU-testable."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.request

FLASH_VERSION = "2.7.4.post1"
CORE = ("torch", "vllm", "transformers", "tokenizers", "tensordict", "torchdata", "ray")


def require_isolated_environment():
    """Check the selected interpreter, not merely a stale activation variable."""
    prefix = Path(sys.prefix).resolve()
    if sys.prefix != sys.base_prefix:
        return
    conda = os.environ.get("CONDA_PREFIX")
    if conda and Path(conda).resolve() == prefix and (prefix / "conda-meta").is_dir():
        return
    raise RuntimeError(
        "Installation requires a dedicated active venv/conda interpreter. "
        "Create one yourself, then set MVP_PYTHON to its Python. "
        "The script does not alter system Python or create/replace environments."
    )


def require_hopper_supported():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("install-hopper requires Linux x86_64; other hosts may use check/install-cpu")
    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        raise RuntimeError("This pinned Hopper profile requires Python 3.10 or 3.11")
    smi = shutil.which("nvidia-smi")
    nvcc = shutil.which("nvcc")
    if not smi or not nvcc:
        raise RuntimeError("install-hopper needs nvidia-smi and a CUDA 12.x nvcc toolchain before changing packages")
    result = subprocess.run(
        [smi, "--query-gpu=compute_cap,name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("No GPU was detected; use an existing compatible environment or a GPU host")
    for row in rows:
        capability, separator, name = row.partition(",")
        if not separator or not re.fullmatch(r"\d+\.\d+", capability.strip()):
            raise RuntimeError(f"Cannot establish GPU compute capability from nvidia-smi: {row}")
        major = int(capability.strip().split(".")[0])
        if major >= 10:
            raise RuntimeError(
                f"Blackwell/newer GPU {name.strip()} (compute capability {capability.strip()}) detected. "
                "Do not install the torch2.6/vLLM0.8.5 Hopper profile on B300/B200. "
                "Reuse a separately validated Blackwell-compatible G-OPD environment; "
                "the orchestration code remains usable."
            )
        if major < 8:
            raise RuntimeError("This FlashAttention 2 profile requires an Ampere/Hopper GPU (8.x/9.x)")
    compiler = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True, timeout=20)
    if not re.search(r"release\s+12\.", compiler.stdout):
        raise RuntimeError("This profile requires a CUDA 12.x nvcc toolchain for FlashAttention compilation")
    return rows


def read_pins(path):
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            pins.update(read_pins(Path(path).parent / line[3:].strip()))
            continue
        match = re.fullmatch(r"([A-Za-z0-9_-]+)(?:\[([^]]+)\])?==([^\s]+)", line)
        if not match:
            raise RuntimeError(f"Unrecognized pinned requirement: {line}")
        name, extras, version = match.groups()
        pins[re.sub(r"[-_.]+", "-", name).lower()] = {
            "version": version, "extras": extras.split(",") if extras else [],
        }
    return pins


def inspect_environment(root):
    pins = read_pins(root / "mvp_requirements_hopper.txt")
    pins["flash-attn"] = {"version": FLASH_VERSION}
    pins["verl"] = {"version": "this checkout (editable)"}
    packages = {}
    for name, pin in pins.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        packages[name] = {"installed": installed, "hopper_profile": pin["version"]}
    report = {
        "mode": "check", "read_only": True, "python": sys.version.split()[0],
        "interpreter": sys.executable, "prefix": sys.prefix,
        "platform": f"{platform.system()} {platform.machine()}", "packages": packages,
        "note": "Version inventory only: no import, install, CUDA allocation or GPU training validation.",
        "blackwell": "B300/B200 require an existing separately validated compatible environment; do not use install-hopper.",
    }
    print(json.dumps(report, indent=2))


def runtime_check(root):
    """Explicit import and availability checks; no model, Ray cluster or CUDA tensors."""
    verl = importlib.import_module("verl")
    expected_module = (root / "verl" / "verl" / "__init__.py").resolve()
    if Path(verl.__file__).resolve() != expected_module:
        raise RuntimeError(
            f"Python is loading a different verl checkout: {verl.__file__}. "
            f"In your selected environment run: python -m pip install --no-deps -e {root / 'verl'}"
        )
    modules = (
        "torch", "tensordict", "torchdata.stateful_dataloader", "transformers", "ray",
        "vllm", "flash_attn", "verl.workers.fsdp_workers",
    )
    for name in modules:
        importlib.import_module(name)
        print(f"Import OK: {name}", flush=True)
    import torch
    import tensordict
    from torchdata.stateful_dataloader import StatefulDataLoader
    batch = tensordict.TensorDict({"ids": torch.tensor([[1, 2], [3, 4]])}, batch_size=[2])
    assert torch.cat([batch[:1], batch[1:]], dim=0)["ids"].equal(batch["ids"])
    loader = StatefulDataLoader(list(range(6)), batch_size=2)
    assert next(iter(loader)).tolist() == [0, 1]
    restored = StatefulDataLoader(list(range(6)), batch_size=2)
    restored.load_state_dict(loader.state_dict())
    assert next(iter(restored)).tolist() == [2, 3]
    expected = int(os.environ.get("MVP_EXPECTED_GPUS", "8"))
    count = torch.cuda.device_count()
    if expected < 1 or not torch.cuda.is_available() or count < expected:
        raise RuntimeError(f"Expected at least {expected} visible usable GPUs; torch reports {count}")
    print(f"CPU TensorDict/dataloader smoke passed; {count} CUDA devices available.")
    print("No CUDA kernels, NCCL collective, generation or optimizer step tested. Run the project GPU smoke next.")


def audit_metadata(root):
    """Check official PyPI direct dependency constraints; not a full lock solver."""
    try:
        from packaging.markers import default_environment
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
    except ImportError as exc:
        raise RuntimeError("audit-metadata needs packaging; install-cpu provides it") from exc
    pins = read_pins(root / "mvp_requirements_hopper.txt")
    pins["flash-attn"] = {"version": FLASH_VERSION, "extras": []}

    def fetch(item):
        name, pin = item
        url = f"https://pypi.org/pypi/{name}/{pin['version']}/json"
        with urllib.request.urlopen(url, timeout=30) as response:
            info = json.load(response)["info"]
        return name, {"url": url, "requires_python": info["requires_python"], "requires_dist": info["requires_dist"] or []}

    information = dict(ThreadPoolExecutor(max_workers=6).map(fetch, pins.items()))
    errors = []
    checked_edges = []
    for python_version in ("3.10", "3.11"):
        environment = default_environment()
        environment.update(python_version=python_version, python_full_version=python_version + ".0",
                           platform_system="Linux", sys_platform="linux", platform_machine="x86_64")
        for name, info in information.items():
            if info["requires_python"] and python_version not in SpecifierSet(info["requires_python"]):
                errors.append(f"{name} excludes Python {python_version}")
            for specification in info["requires_dist"]:
                dependency = Requirement(specification)
                key = re.sub(r"[-_.]+", "-", dependency.name).lower()
                if key not in pins:
                    continue
                if dependency.marker and not any(
                    dependency.marker.evaluate(dict(environment, extra=extra))
                    for extra in [""] + pins[name]["extras"]
                ):
                    continue
                if pins[key]["version"] not in dependency.specifier:
                    errors.append(f"{name} needs {specification}, pinned {key}=={pins[key]['version']}")
                checked_edges.append(f"{name} -> {specification}")
    print(json.dumps({
        "check": "PyPI direct pinned-dependency metadata", "status": "fail" if errors else "pass",
        "target": "Linux x86_64, Python 3.10/3.11", "direct_pins": len(pins),
        "checked_edges": sorted(set(checked_edges)), "errors": sorted(set(errors)),
        "core_sources": {name: information[name] for name in CORE},
        "limits": "Does not solve all transitive dependencies or prove Linux wheel ABI/CUDA/NCCL/training compatibility.",
        "torchdata_compatibility": "https://github.com/meta-pytorch/data/blob/v0.11.0/README.md",
    }, indent=2))
    if errors:
        raise RuntimeError("Pinned dependency metadata is inconsistent")


def install(root, hopper):
    require_isolated_environment()
    if hopper:
        devices = require_hopper_supported()
        print("Installing dedicated Ampere/Hopper profile for: " + "; ".join(devices), flush=True)
    else:
        print("Installing CPU orchestration dependencies only (no torch/vLLM/verl).", flush=True)
    requirement = root / ("mvp_requirements_hopper.txt" if hopper else "mvp_requirements_cpu.txt")
    command = [sys.executable, "-m", "pip", "install", "-r", str(requirement)]
    # Review/resolve first. This does not prove runtime compatibility.
    subprocess.run(command + ["--dry-run"], check=True)
    subprocess.run(command, check=True)
    if hopper:
        import torch
        if torch.__version__.split("+")[0] != "2.6.0" or not (torch.version.cuda or "").startswith("12."):
            raise RuntimeError("Resolved torch build is not the intended torch2.6/CUDA12 build; stop before compiling")
        build_environment = dict(os.environ)
        build_environment["FLASH_ATTENTION_FORCE_BUILD"] = "TRUE"
        build_environment.setdefault("MAX_JOBS", "4")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
             "--no-binary=flash-attn", f"flash-attn=={FLASH_VERSION}"],
            check=True, env=build_environment,
        )
        # Install our checkout, without letting upstream resolve a second stack.
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(root / "verl")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print("Installation complete. Run check-training, then the project GPU smoke; training is not yet validated.")


def main():
    mode, root = sys.argv[1], Path(sys.argv[2]).resolve()
    if sys.version_info < (3, 10):
        raise RuntimeError("The MVP requires Python >=3.10")
    if mode == "check":
        inspect_environment(root)
    elif mode == "check-training":
        inspect_environment(root)
        runtime_check(root)
    elif mode == "audit-metadata":
        audit_metadata(root)
    elif mode in ("install-cpu", "install-hopper"):
        install(root, hopper=mode == "install-hopper")
    elif mode in ("help", "--help", "-h"):
        print("Modes: check (default, read-only), check-training (explicit imports/CUDA availability), "
              "audit-metadata (read-only PyPI queries), install-cpu, install-hopper. "
              "Set MVP_PYTHON to an existing interpreter. Installation requires an isolated environment.")
    else:
        raise RuntimeError(f"Unknown mode {mode!r}; use --help")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, ImportError, subprocess.SubprocessError) as exc:
        print(f"Environment check/install stopped: {exc}", file=sys.stderr)
        sys.exit(1)
PY
