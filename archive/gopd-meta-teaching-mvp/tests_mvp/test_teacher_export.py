"""Exercise export process/recovery with tiny files, never model-merger or GPUs.

Source checkpoint and safetensors files are explicitly byte placeholders. These
tests validate lifecycle and file-presence contracts, not tensor serialization
or the ability to merge a real FSDP model.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from meta_teaching.backend import PhaseFailure, VerlBackend, export_teacher
from meta_teaching.common import atomic_json, read_json
from meta_teaching.config import DEFAULTS
from meta_teaching.runner import ExperimentRunner


EXPORT_CHILD = r'''
import json, pathlib, sys, time
target, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
target.mkdir(parents=True, exist_ok=True)
(target / "FIXTURE_ONLY.txt").write_text("No model weights; test subprocess only.\n")
if mode == "sleep":
    time.sleep(60)
if mode == "fail":
    raise SystemExit(7)
(target / "config.json").write_text(json.dumps({"model_type": "fixture"}))
(target / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"fixture": 0}}}))
for part in (1, 2):
    (target / f"model-{part:05d}-of-00002.safetensors").write_bytes(b"fixture; not actual safetensors")
(target / "model.safetensors.index.json").write_text(json.dumps({
    "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
}))
'''


def make_checkpoint(path, world_size=8):
    actor = path / "actor"
    (actor / "huggingface").mkdir(parents=True)
    (path / "data.pt").write_bytes(b"fixture data state")
    (actor / "fsdp_config.json").write_text(json.dumps({"world_size": world_size}))
    (actor / "huggingface" / "config.json").write_text("{}")
    for rank in range(world_size):
        for name in ("model", "optim", "extra_state"):
            (actor / f"{name}_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"fixture only")
        (actor / f"rollout_rng_rank_{rank}.pt").write_bytes(b"fixture RNG")
    return path


class TeacherExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = make_checkpoint(self.root / "source" / "global_step_7")
        self.target = self.root / "exported"
        self.config = copy.deepcopy(DEFAULTS)
        self.config["models"] = {"student": "fixture/student", "teacher": "fixture/teacher"}
        self.config["experiment"].update(
            output_dir=str(self.root / "run"), teacher_mode="previous_student",
            validation_reserve_seconds=0, max_total_gpu_hours=1,
        )
        manifest = self.root / "data" / "manifest.json"
        atomic_json(manifest, {"splits": {key: {"path": key + ".parquet"}
                                         for key in ("train", "meta_val", "test")}})
        self.config["data"]["manifest"] = str(manifest)
        self.real_popen = subprocess.Popen

    def tearDown(self):
        self.temporary.cleanup()

    def fixture_popen(self, mode="ok"):
        def launch(command, **kwargs):
            self.assertEqual(command[1:4], ["-m", "verl.model_merger", "merge"])
            self.assertEqual(command[command.index("--local_dir") + 1], str(self.source / "actor"))
            target = command[command.index("--target_dir") + 1]
            return self.real_popen([sys.executable, "-c", EXPORT_CHILD, target, mode], **kwargs)
        return launch

    def export(self, **kwargs):
        return export_teacher(self.config, self.source, self.target, "teacher_export_001", 5, **kwargs)

    def runner(self, *, resume=False, retry_failed=False):
        backend = VerlBackend(self.config, retry_failed=retry_failed)
        runner = ExperimentRunner(self.config, backend=backend, resume=resume)
        if not resume:
            runner.state["rounds"] = [{"fixture": "Only the round count is used by export preparation"}]
            runner.state["current_checkpoint"] = str(self.source)
            runner._save()
        return runner

    def test_success_and_completed_reuse_do_not_spawn_twice(self):
        with patch("meta_teaching.backend.subprocess.Popen", side_effect=self.fixture_popen()) as launch:
            result = self.export()
            reused = self.export()
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(result, reused)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["source_checkpoint"], str(self.source))
        self.assertGreater(result["allocated_gpu_seconds"], 0)
        self.assertAlmostEqual(result["allocated_gpu_seconds"], 8 * result["wall_seconds"])
        self.assertEqual(read_json(self.target / "launcher_record.json"), result)

    def test_changed_source_rejected_before_launch(self):
        with patch("meta_teaching.backend.subprocess.Popen", side_effect=self.fixture_popen()):
            self.export()
        changed = make_checkpoint(self.root / "other" / "global_step_8")
        with patch("meta_teaching.backend.subprocess.Popen") as launch:
            with self.assertRaisesRegex(ValueError, "source changed"):
                export_teacher(self.config, changed, self.target, "teacher_export_001", 5)
        launch.assert_not_called()

    def test_failure_is_charged_and_does_not_retry_implicitly(self):
        with patch("meta_teaching.backend.subprocess.Popen", side_effect=self.fixture_popen("fail")) as launch:
            with self.assertRaises(PhaseFailure) as failed:
                self.export()
            with self.assertRaisesRegex(PhaseFailure, "Previous teacher export failed"):
                self.export()
        self.assertEqual(launch.call_count, 1)
        record = failed.exception.record
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["return_code"], 7)
        self.assertGreater(record["allocated_gpu_seconds"], 0)

    @unittest.skipUnless(os.name == "posix", "POSIX process-group termination")
    def test_timeout_terminates_only_fixture_and_preserves_cost(self):
        with patch("meta_teaching.backend.subprocess.Popen", side_effect=self.fixture_popen("sleep")):
            with self.assertRaises(PhaseFailure) as failed:
                export_teacher(self.config, self.source, self.target, "teacher_export_001", 0.15)
        record = failed.exception.record
        self.assertEqual(record["status"], "failed")
        self.assertIn("TimeoutExpired", record["error"])
        self.assertGreater(record["allocated_gpu_seconds"], 0)
        with self.assertRaises(ProcessLookupError):
            os.kill(record["pid"], 0)

    def test_explicit_retry_archives_previous_attempt(self):
        with patch("meta_teaching.backend.subprocess.Popen", side_effect=self.fixture_popen("fail")):
            with self.assertRaises(PhaseFailure) as failed:
                self.export()
        old_record = failed.exception.record
        with patch("meta_teaching.backend.subprocess.Popen", side_effect=self.fixture_popen()):
            result = self.export(retry_failed=True)
        self.assertEqual(result["status"], "completed")
        prior = read_json(self.target / "previous_attempt.json")
        self.assertEqual(prior["record"], old_record)
        archive = Path(prior["archive"])
        self.assertEqual(read_json(archive / "launcher_record.json"), old_record)
        self.assertTrue((archive / "export.log").is_file())
        self.assertNotEqual(old_record["started_unix"], result["started_unix"])

    def test_deleted_exported_shard_rejected_on_completed_reuse(self):
        with patch("meta_teaching.backend.subprocess.Popen", side_effect=self.fixture_popen()):
            self.export()
        (self.target / "model-00002-of-00002.safetensors").unlink()
        with patch("meta_teaching.backend.subprocess.Popen") as launch:
            with self.assertRaisesRegex(ValueError, "missing shard"):
                self.export()
        launch.assert_not_called()

    def test_runner_charges_failed_and_retried_export_attempts(self):
        runner = self.runner()
        with patch("meta_teaching.backend.subprocess.Popen", side_effect=self.fixture_popen("fail")):
            with self.assertRaises(PhaseFailure) as failed:
                runner._prepare_recursive_teacher()
        old_cost = failed.exception.record["allocated_gpu_seconds"]
        attempts = [event for event in runner.state["cost_events"].values()
                    if event["phase_id"] == "teacher_export_001"]
        self.assertEqual(len(attempts), 1)
        self.assertAlmostEqual(attempts[0]["allocated_gpu_seconds"], old_cost)
        resumed = self.runner(resume=True, retry_failed=True)
        with patch("meta_teaching.backend.subprocess.Popen", side_effect=self.fixture_popen()):
            resumed._prepare_recursive_teacher()
        target = Path(resumed.state["current_teacher"])
        self.assertTrue((target / "export_manifest.json").is_file())
        attempts = [event for event in resumed.state["cost_events"].values()
                    if event["phase_id"] == "teacher_export_001"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual({event["status"] for event in attempts}, {"failed", "completed"})
        current = read_json(target / "launcher_record.json")
        self.assertAlmostEqual(sum(event["allocated_gpu_seconds"] for event in attempts),
                               old_cost + current["allocated_gpu_seconds"])

    def test_completed_export_reconciled_after_controller_write_gap(self):
        runner = self.runner()
        target = runner.output / "teachers" / "round_001"
        with patch("meta_teaching.backend.subprocess.Popen", side_effect=self.fixture_popen()):
            exported = export_teacher(self.config, self.source, target, "teacher_export_001", 5)
        self.assertFalse(runner.state["cost_events"])
        resumed = self.runner(resume=True)
        resumed._reconcile_disk_records()
        with patch("meta_teaching.backend.subprocess.Popen") as launch:
            resumed._prepare_recursive_teacher()
        launch.assert_not_called()
        attempts = [event for event in resumed.state["cost_events"].values()
                    if event["phase_id"] == "teacher_export_001"]
        self.assertEqual(len(attempts), 1)
        self.assertAlmostEqual(attempts[0]["allocated_gpu_seconds"], exported["allocated_gpu_seconds"])
        self.assertEqual(resumed.state["current_teacher"], str(target))


if __name__ == "__main__":
    unittest.main()
