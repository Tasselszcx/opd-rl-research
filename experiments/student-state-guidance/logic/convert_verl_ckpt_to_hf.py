#!/usr/bin/env python
"""Convert all verl FSDP checkpoints in an experiment to HuggingFace format.

Usage:
    python scripts/convert_verl_ckpt_to_hf.py <experiment_ckpt_dir> [--output_base DIR] [--no-cleanup]

`<experiment_ckpt_dir>` is the verl `default_local_dir` for one run, e.g.
`training/checkpoints/<project>/<run>/`. It contains `global_step_*` subfolders.

For every `global_step_X`, the script merges its `actor/` shards and writes the
HuggingFace checkpoint to:

    <output_base>/<project>/<run>/global_step_X/

where `<output_base>` defaults to the input path with the `checkpoints` segment
replaced by `hg_ckpt` (e.g. `training/hg_ckpt/<project>/<run>/global_step_X/`).

After every step has been converted successfully, all `global_step_*` folders
in `<experiment_ckpt_dir>` are deleted except the latest one.

The actual FSDP -> HF merging is delegated to verl's upstream
`legacy_model_merger.py` (vendored as `scripts/verl_legacy_model_merger.py`).
"""

import argparse
import re
import shutil
import sys
import types
from pathlib import Path

GLOBAL_STEP_RE = re.compile(r"^global_step_(\d+)$")


def install_transformers_compat_shim() -> None:
    """verl 0.2.0.dev does not ship `verl.utils.transformers_compat`, which the
    upstream merger imports. Inject a minimal shim before importing it."""
    if "verl.utils.transformers_compat" in sys.modules:
        return
    shim = types.ModuleType("verl.utils.transformers_compat")

    def get_auto_model_for_vision2seq():
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq

    shim.get_auto_model_for_vision2seq = get_auto_model_for_vision2seq
    sys.modules["verl.utils.transformers_compat"] = shim


def find_global_steps(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    steps = []
    for entry in checkpoint_dir.iterdir():
        if not entry.is_dir():
            continue
        m = GLOBAL_STEP_RE.match(entry.name)
        if m:
            steps.append((int(m.group(1)), entry))
    steps.sort(key=lambda x: x[0])
    return steps


def derive_output_base(checkpoint_dir: Path) -> Path:
    """Replace the `checkpoints` path segment with `hg_ckpt`, preserving the
    rest (incl. `<project>/<run>`)."""
    parts = list(checkpoint_dir.resolve().parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index("checkpoints")
    except ValueError:
        sys.exit(
            f"Could not derive output base: '{checkpoint_dir}' does not contain "
            "a 'checkpoints' segment. Please pass --output_base explicitly."
        )
    parts[idx] = "hg_ckpt"
    return Path(*parts)


def cleanup_old_steps(steps: list[tuple[int, Path]]) -> None:
    if len(steps) <= 1:
        print("[cleanup] Only one global_step present; nothing to remove.")
        return
    keep_step, keep_path = steps[-1]
    for step, path in steps[:-1]:
        print(f"[cleanup] Removing global_step_{step}: {path}")
        shutil.rmtree(path)
    print(f"[cleanup] Kept global_step_{keep_step} at {keep_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint_dir", type=Path,
        help="Directory containing global_step_* subfolders "
             "(e.g. training/checkpoints/<project>/<run>/).",
    )
    parser.add_argument(
        "--output_base", type=Path, default=None,
        help="Base output directory. Each step is saved under "
             "<output_base>/global_step_X/. Defaults to the input path with "
             "'checkpoints' replaced by 'hg_ckpt'.",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Keep older global_step_* folders in the source dir instead of deleting them.",
    )
    args = parser.parse_args()

    if not args.checkpoint_dir.is_dir():
        sys.exit(f"Not a directory: {args.checkpoint_dir}")

    steps = find_global_steps(args.checkpoint_dir)
    if not steps:
        sys.exit(f"No global_step_* folders in {args.checkpoint_dir}")

    output_base = args.output_base or derive_output_base(args.checkpoint_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"[main] Found {len(steps)} checkpoints. Output base: {output_base}")

    install_transformers_compat_shim()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verl_legacy_model_merger import FSDPModelMerger, ModelMergerConfig  # noqa: E402

    for step, step_dir in steps:
        actor_dir = step_dir / "actor"
        if not actor_dir.is_dir():
            print(f"[main] Skipping global_step_{step}: no actor/ subdir at {step_dir}")
            continue
        out_dir = output_base / f"global_step_{step}"
        if (out_dir / "config.json").exists() and any(out_dir.glob("*.safetensors")):
            print(f"[main] global_step_{step}: already merged at {out_dir}, skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[main] Converting global_step_{step} -> {out_dir}")
        cfg = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=str(actor_dir),
            hf_model_config_path=str(actor_dir),  # auto-detects ./huggingface subdir
            target_dir=str(out_dir),
        )
        FSDPModelMerger(cfg).merge_and_save()
        print(f"[main] global_step_{step}: merge succeeded")

    print(f"[main] All {len(steps)} checkpoints converted.")

    if args.no_cleanup:
        print("[main] --no-cleanup: leaving older checkpoints in place.")
        return
    cleanup_old_steps(steps)
    print("[main] Done.")


if __name__ == "__main__":
    main()
