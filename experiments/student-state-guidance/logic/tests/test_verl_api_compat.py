"""Check that the trainer's calls into verl match verl's actual signatures.

The trainer subclasses verl's RayPPOTrainer and calls into verl in a dozen
places. When the verl base moves, a changed signature does not fail at import --
it fails minutes into a GPU job, at whatever point that call is first reached.
This walks every such call and compares it against the real signature, which
takes seconds and needs no GPU.

It covers symbols imported from verl and inherited RayPPOTrainer methods. It
does not cover Ray worker-group calls (`self.actor_rollout_wg.*`), which are
dispatched dynamically, nor behavioural differences behind an unchanged
signature.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECKER = Path(__file__).parent / "_verl_api_compat.py"
TRAINER = REPO / "logic" / "training" / "trainers" / "ray_grpo_trainer.py"


def test_trainer_calls_match_verl_signatures():
    if not TRAINER.exists():
        pytest.skip(f"trainer not found at {TRAINER}")
    proc = subprocess.run(
        [sys.executable, str(CHECKER), str(TRAINER)],
        capture_output=True, text=True, timeout=900,
        cwd=str(REPO),
        env={**__import__("os").environ,
             "PYTHONPATH": f"{REPO / 'verl'}:{REPO / 'logic'}"},
    )
    if "cannot resolve" in proc.stdout and "No module named" in proc.stdout:
        pytest.skip("verl not importable in this environment")
    assert proc.returncode == 0, f"verl API mismatch:\n{proc.stdout}\n{proc.stderr[-1500:]}"
