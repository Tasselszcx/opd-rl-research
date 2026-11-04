"""Explicitly synthetic control-plane exercise. No LLM is loaded or trained."""

from __future__ import annotations

import time
from pathlib import Path

from .common import atomic_json, read_json


class SimulatedBackend:
    simulated = True

    def __init__(self, config):
        self.config = config

    def execute(self, spec):
        path = Path(spec.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        record_path = path / "launcher_record.json"
        if record_path.exists():
            saved = read_json(record_path)
            if saved.get("spec_fingerprint") != spec.identity():
                raise ValueError("Simulated phase changed")
            return saved
        before = {"step": 0, "score": 0.2}
        if spec.resume_checkpoint:
            before = read_json(Path(spec.resume_checkpoint) / "SIMULATED_STATE.json")
        score = before["score"]
        if spec.phase == "warmup":
            score += {"plain": 0.01, "partial": 0.005, "full": 0.02}[spec.exposure]
        elif spec.phase == "rl":
            # A fixture with reversed warmup/downstream ranking; not a research claim.
            score += 0.02 + {"direct_rl": 0, "plain": 0.01, "partial": 0.04, "full": 0.005}.get(spec.arm_id, 0)
        score = min(score, 0.95)
        step = before["step"] + spec.steps
        checkpoint = spec.resume_checkpoint
        if spec.phase != "eval":
            checkpoint = str(path / "checkpoints" / f"global_step_{step}")
            atomic_json(Path(checkpoint) / "SIMULATED_STATE.json", {"score": score, "step": step})
        summary = {
            "schema_version": 1, "simulated": True, "phase": spec.phase, "arm_id": spec.arm_id,
            "status": "completed", "start_step": before["step"], "end_step": step,
            "steps_completed": spec.steps, "checkpoint_path": checkpoint, "val_score": score,
            "val_metrics": {"meta_teaching/val_score": score},
            "teacher_enabled": spec.phase == "warmup",
            "teacher_queries": 8 if spec.phase == "warmup" else 0,
            "teacher_scored_tokens": 1024 if spec.phase == "warmup" else 0,
            "deadline_overrun_seconds": 0,
            "warning": "SIMULATED FIXTURE; NOT AN EXPERIMENTAL RESULT",
        }
        atomic_json(path / "phase_summary.json", summary)
        seconds = 1.0 + spec.steps * (0.2 if spec.phase == "warmup" else 0.1)
        now = time.time()
        record = {
            "schema_version": 1, "backend": "simulation", "simulated": True,
            "phase_id": spec.phase_id, "spec_fingerprint": spec.identity(), "status": "completed",
            "started_unix": now, "finished_unix": now, "wall_seconds": seconds,
            "allocated_gpu_seconds": seconds * self.config["training"]["n_gpus"],
            "summary": summary, "cost_scope": "synthetic fixture cost, no GPUs used",
        }
        atomic_json(record_path, record)
        return record
