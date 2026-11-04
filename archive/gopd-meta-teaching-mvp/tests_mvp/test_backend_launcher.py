"""Real local child-process checks of the launcher; fixtures do not use a GPU.

The child only creates JSON/checkpoint placeholders. Its scores are test data,
never evidence of model training, CUDA support, or research performance.
"""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from meta_teaching.backend import PhaseFailure, PhaseSpec, VerlBackend
from meta_teaching.common import atomic_json, read_json
from meta_teaching.config import DEFAULTS
from meta_teaching.runner import ExperimentRunner


CHILD = r'''
import json, pathlib, sys, time
root, phase, arm, mode = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
definition = json.loads(sys.argv[5])
root.mkdir(parents=True, exist_ok=True)
(root / "fixture_started").write_text("started")
if mode == "sleep":
    time.sleep(60)
source = definition["resume_checkpoint"]
start = int(pathlib.Path(source).name.removeprefix("global_step_")) if source else 0
end = start + definition["steps"]
if mode == "valid_budget_early":
    end = start + max(0, definition["steps"] - 1)
checkpoint = root / "checkpoints" / f"global_step_{end}"
if mode == "foreign_checkpoint":
    checkpoint = root.parent / "foreign" / f"global_step_{end}"
if mode == "wrong_checkpoint_step":
    checkpoint = root / "checkpoints" / f"global_step_{end + 1}"
checkpoint.mkdir(parents=True, exist_ok=True)
actor = checkpoint / "actor"
(actor / "huggingface").mkdir(parents=True, exist_ok=True)
(checkpoint / "data.pt").write_text("fixture only")
(actor / "fsdp_config.json").write_text(json.dumps({"world_size": 8}))
(actor / "huggingface" / "config.json").write_text("{}")
for rank in range(8):
    for kind in ("model", "optim", "extra_state"):
        (actor / f"{kind}_world_size_8_rank_{rank}.pt").write_text("fixture only")
    (actor / f"rollout_rng_rank_{rank}.pt").write_text("fixture only")
summary = {
    "schema_version": 1,
    "status": "completed", "phase": phase, "arm_id": arm,
    "phase_id": definition["phase_id"], "spec_fingerprint": definition["spec_fingerprint"],
    "seed": definition["seed"], "source_checkpoint": source,
    "val_score": 0.5, "val_metrics": {"meta_teaching/val_score": 0.5},
    "checkpoint_path": source if phase == "eval" else str(checkpoint),
    "start_step": start, "end_step": end, "validation_step": end,
    "teacher_enabled": phase == "warmup", "teacher_queries": 0,
    "teacher_selected_queries": 0, "teacher_scored_tokens": 0, "teacher_response_tokens": 0,
    "steps_completed": end - start,
}
changes = {
    "wrong_start": {"start_step": start + 1},
    "wrong_end": {"end_step": end + 1},
    "wrong_count": {"steps_completed": definition["steps"] + 1},
    "bool_count": {"steps_completed": True},
    "bool_start": {"start_step": False},
    "negative_start": {"start_step": -1},
    "too_many_steps": {"steps_completed": definition["steps"] + 1, "end_step": end + 1, "validation_step": end + 1},
    "short_completed": {"steps_completed": 0, "end_step": start, "validation_step": start},
    "wrong_validation": {"validation_step": end + 1},
    "wrong_metric": {"val_metrics": {"meta_teaching/val_score": 0.6}},
    "wrong_phase_id": {"phase_id": "different/phase"},
    "wrong_seed": {"seed": definition["seed"] + 1},
    "bool_seed": {"seed": True},
    "wrong_fingerprint": {"spec_fingerprint": "different-spec"},
    "wrong_source": {"source_checkpoint": "/foreign/global_step_0"},
    "negative_queries": {"teacher_queries": -1},
    "bool_queries": {"teacher_queries": False},
    "bool_teacher_enabled": {"teacher_enabled": 0},
    "bool_scored_tokens": {"teacher_scored_tokens": False},
    "negative_response_tokens": {"teacher_response_tokens": -1},
    "teacher_tokens_in_rl": {"teacher_scored_tokens": 1},
    "invalid_selected_queries": {"teacher_selected_queries": "0"},
    "bad_schema": {"schema_version": 2},
    "bool_schema": {"schema_version": True},
    "budget_without_deadline": {"status": "budget_exhausted"},
    "valid_budget_early": {"status": "budget_exhausted"},
    "valid_budget_full": {"status": "budget_exhausted"},
    "eval_updates": {"steps_completed": 1, "end_step": start + 1, "validation_step": start + 1},
    "eval_changed_checkpoint": {"checkpoint_path": str(checkpoint)},
}
summary.update(changes.get(mode, {}))
if mode == "teacher_in_rl":
    summary["teacher_queries"] = 1
if mode == "malformed":
    (root / "phase_summary.json").write_text("{truncated_json")
elif mode == "json_list":
    (root / "phase_summary.json").write_text("[1]")
else:
    (root / "phase_summary.json").write_text(json.dumps(summary))
'''


class BackendLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "verl").mkdir()
        self.config = copy.deepcopy(DEFAULTS)
        self.config["models"] = {"student": "fixture/student", "teacher": "fixture/teacher"}
        self.config["experiment"].update(
            output_dir=str(self.root / "run"), rounds=1, repeats=1,
            validation_reserve_seconds=0, deadline_grace_seconds=0.05,
            max_total_gpu_hours=1, evaluate_final_test=False,
        )
        self.config["actions"] = self.config["actions"][:1]
        manifest = self.root / "data" / "manifest.json"
        atomic_json(manifest, {"splits": {name: {"path": name + ".parquet"} for name in ("train", "meta_val", "test")}})
        self.config["data"]["manifest"] = str(manifest)
        self.backend = VerlBackend(self.config, repository_root=self.root)
        self.spec = PhaseSpec(
            phase_id="round_000/repeat_000/plain/rl", phase="rl", arm_id="plain",
            output_dir=str(self.root / "run" / "phases" / "round_000/repeat_000/plain/rl"),
            student_model="fixture/student", teacher_model=None, resume_checkpoint=None,
            train_path="fixture/train", validation_path="fixture/meta_val", steps=1, seed=42,
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def command(mode="ok"):
        def make(config, spec, **kwargs):
            definition = {"phase_id": spec.phase_id, "seed": spec.seed,
                          "resume_checkpoint": spec.resume_checkpoint,
                          "steps": spec.steps, "spec_fingerprint": spec.identity()}
            return [sys.executable, "-c", CHILD, spec.output_dir, spec.phase, spec.arm_id, mode,
                    json.dumps(definition)]
        return make

    def test_real_child_success_cost_and_completed_reuse(self):
        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command()) as build:
            result = self.backend.execute(self.spec)
            again = self.backend.execute(self.spec)
        self.assertEqual(result, again)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["wall_seconds"], 0)
        self.assertAlmostEqual(result["allocated_gpu_seconds"], result["wall_seconds"] * 8)
        self.assertTrue(Path(result["summary"]["checkpoint_path"]).is_dir())

    def test_completed_fingerprint_rejects_changed_training_definition(self):
        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command()):
            self.backend.execute(self.spec)
            with self.assertRaisesRegex((RuntimeError, ValueError), "changed|differs"):
                self.backend.execute(replace(self.spec, steps=2))

    def test_teacher_use_in_rl_is_rejected_with_accounted_cost(self):
        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command("teacher_in_rl")):
            with self.assertRaises(PhaseFailure) as caught:
                self.backend.execute(self.spec)
        self.assertGreater(caught.exception.record["allocated_gpu_seconds"], 0)
        self.assertEqual(caught.exception.record["status"], "failed")

    @unittest.skipUnless(os.name == "posix", "POSIX process-group termination")
    def test_deadline_terminates_fixture_and_preserves_cost(self):
        spec = replace(self.spec, deadline_unix=time.time() + 0.15)
        started = time.monotonic()
        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command("sleep")):
            with self.assertRaises(PhaseFailure) as caught:
                self.backend.execute(spec)
        record = caught.exception.record
        self.assertEqual(record["status"], "failed")
        self.assertGreater(record["allocated_gpu_seconds"], 0)
        self.assertLess(time.monotonic() - started, 5)
        with self.assertRaises(ProcessLookupError):
            os.kill(record["pid"], 0)

    def test_malformed_summary_still_finalizes_failure_and_cost(self):
        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command("malformed")):
            with self.assertRaises(PhaseFailure):
                self.backend.execute(self.spec)
        record = read_json(Path(self.spec.output_dir) / "launcher_record.json")
        self.assertEqual(record["status"], "failed")
        self.assertGreater(record["allocated_gpu_seconds"], 0)

    def test_invalid_summary_contract_is_rejected_and_charged(self):
        modes = (
            "wrong_start", "wrong_end", "wrong_count", "bool_count", "bool_start",
            "negative_start", "too_many_steps", "short_completed", "wrong_validation", "wrong_metric",
            "wrong_phase_id", "wrong_seed", "bool_seed", "wrong_fingerprint", "wrong_source",
            "negative_queries", "bool_queries", "bool_teacher_enabled", "bool_scored_tokens",
            "negative_response_tokens", "teacher_tokens_in_rl", "invalid_selected_queries",
            "wrong_checkpoint_step", "foreign_checkpoint", "bad_schema", "bool_schema", "json_list",
            "budget_without_deadline",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                spec = replace(self.spec, phase_id=f"contracts/{mode}", output_dir=str(self.root / "contracts" / mode),
                               seed=1 if mode == "bool_seed" else self.spec.seed)
                with patch("meta_teaching.backend.build_phase_command", side_effect=self.command(mode)):
                    with self.assertRaises(PhaseFailure):
                        self.backend.execute(spec)
                record = read_json(Path(spec.output_dir) / "launcher_record.json")
                self.assertEqual(record["status"], "failed")
                self.assertGreater(record["allocated_gpu_seconds"], 0)

    def test_budget_status_requires_deadline_but_allows_partial_or_full_count(self):
        for mode in ("valid_budget_early", "valid_budget_full"):
            with self.subTest(mode=mode):
                spec = replace(self.spec, phase_id=f"budgets/{mode}", steps=2,
                               output_dir=str(self.root / "budgets" / mode), deadline_unix=time.time() + 10)
                with patch("meta_teaching.backend.build_phase_command", side_effect=self.command(mode)):
                    record = self.backend.execute(spec)
                self.assertEqual(record["status"], "completed")
                self.assertEqual(record["summary"]["status"], "budget_exhausted")
                self.assertEqual(record["summary"]["steps_completed"], 1 if mode.endswith("early") else 2)

    def test_eval_rejects_updates_or_checkpoint_changes(self):
        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command()):
            source = self.backend.execute(self.spec)["summary"]["checkpoint_path"]
        for mode in ("ok", "eval_updates", "eval_changed_checkpoint"):
            with self.subTest(mode=mode):
                spec = replace(self.spec, phase_id=f"evaluations/{mode}", phase="eval", steps=0,
                               resume_checkpoint=source, output_dir=str(self.root / "evaluations" / mode))
                with patch("meta_teaching.backend.build_phase_command", side_effect=self.command(mode)):
                    if mode == "ok":
                        record = self.backend.execute(spec)
                        self.assertEqual(record["summary"]["checkpoint_path"], source)
                        self.assertEqual(record["summary"]["steps_completed"], 0)
                    else:
                        with self.assertRaises(PhaseFailure):
                            self.backend.execute(spec)

    def test_completed_reuse_rejects_missing_checkpoint(self):
        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command()):
            result = self.backend.execute(self.spec)
        shutil.rmtree(result["summary"]["checkpoint_path"])
        with self.assertRaises((PhaseFailure, ValueError, RuntimeError)):
            self.backend.execute(self.spec)

    def test_keyboard_interrupt_reaches_experiment_cost_ledger(self):
        runner = ExperimentRunner(self.config, backend=self.backend)
        real_sleep = time.sleep
        interrupted = False

        def interrupt_once(seconds):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            # subprocess.wait also calls time.sleep; allow cleanup to complete.
            real_sleep(seconds)

        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command("sleep")), \
             patch("meta_teaching.backend.time.sleep", side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                runner.run()
        launcher = read_json(runner.output / "phases" / "initial_eval" / "launcher_record.json")
        state = read_json(runner.state_path)
        self.assertGreater(launcher["allocated_gpu_seconds"], 0)
        self.assertTrue(state["cost_events"], "Interrupted work must not disappear from the experiment ledger")
        self.assertAlmostEqual(
            sum(event["allocated_gpu_seconds"] for event in state["cost_events"].values()
                if event["phase_id"] == "initial_eval"),
            launcher["allocated_gpu_seconds"],
        )

    def test_explicit_retry_keeps_both_attempt_costs(self):
        runner = ExperimentRunner(self.config, backend=self.backend)
        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command("malformed")):
            with self.assertRaises(PhaseFailure) as failed:
                runner._phase(self.spec)
        original_cost = failed.exception.record["allocated_gpu_seconds"]
        retry_backend = VerlBackend(self.config, repository_root=self.root, retry_failed=True)
        resumed = ExperimentRunner(self.config, backend=retry_backend, resume=True)
        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command()):
            retried = resumed._phase(self.spec)
        attempts = [event for event in resumed.state["cost_events"].values()
                    if event["phase_id"] == self.spec.phase_id]
        self.assertEqual(len(attempts), 2)
        self.assertAlmostEqual(
            sum(event["allocated_gpu_seconds"] for event in attempts),
            original_cost + retried["allocated_gpu_seconds"],
        )
        self.assertTrue(list(Path(self.spec.output_dir).parent.glob("rl.failed-*")))

    def test_completed_attempt_reconciled_after_controller_state_write_gap(self):
        runner = ExperimentRunner(self.config, backend=self.backend)
        # A real child finishes, but the controller has not accounted the return yet.
        with patch("meta_teaching.backend.build_phase_command", side_effect=self.command()):
            completed = self.backend.execute(self.spec)
        resumed = ExperimentRunner(self.config, backend=self.backend, resume=True)
        self.assertFalse(resumed.state["cost_events"])
        resumed._reconcile_disk_records()
        attempts = [event for event in resumed.state["cost_events"].values()
                    if event["phase_id"] == self.spec.phase_id]
        self.assertEqual(len(attempts), 1)
        self.assertAlmostEqual(attempts[0]["allocated_gpu_seconds"], completed["allocated_gpu_seconds"])
        with patch("meta_teaching.backend.build_phase_command") as build:
            reused = resumed._phase(self.spec)
        build.assert_not_called()
        self.assertEqual(reused["pid"], completed["pid"])
        attempts = [event for event in resumed.state["cost_events"].values()
                    if event["phase_id"] == self.spec.phase_id]
        self.assertEqual(len(attempts), 1, "Reconciliation and phase reuse must not double-charge")


if __name__ == "__main__":
    unittest.main()
