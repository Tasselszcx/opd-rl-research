"""CPU-only regression checks for durable training-phase validation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPRODUCTION = Path(__file__).resolve().parents[1]
with mock.patch.object(sys, "path", [str(REPRODUCTION), *sys.path]):
    spec = importlib.util.spec_from_file_location(
        "phase_validation_coordinator", REPRODUCTION / "coordinator.py"
    )
    coordinator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coordinator)


class PhaseValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.patch_root = mock.patch.object(coordinator, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.patch_retry = mock.patch.object(coordinator, "ALLOW_RETRY", False)
        self.patch_retry.start()
        self.addCleanup(self.patch_retry.stop)

    def completed_phase(self, name="opd-001", method="opd", first=1, end=3):
        folder = self.root / "runs" / name
        checkpoint = folder / "checkpoints" / f"global_step_{end}"
        actor = checkpoint / "actor"
        actor.mkdir(parents=True)
        (checkpoint / "data.pt").write_bytes(b"state")
        for rank in range(8):
            for kind in ("model", "optim", "extra_state"):
                (actor / f"{kind}_world_size_8_rank_{rank}.pt").write_bytes(b"state")
        record = {
            "status": "completed",
            "return_code": 0,
            "method": method,
            "end_step": end,
            "first_logged_step": first,
            "last_logged_step": end,
            "training_batches": end - first + 1,
            "n_gpus": 8,
            "checkpoint": str(checkpoint),
            "allocated_gpu_hours": 1.25,
        }
        self.write_record(folder, record)
        self.write_metrics(folder, range(first, end + 1))
        return folder, checkpoint

    @staticmethod
    def write_record(folder, record):
        (folder / "launcher_record.json").write_text(json.dumps(record))

    @staticmethod
    def read_record(folder):
        return json.loads((folder / "launcher_record.json").read_text())

    @staticmethod
    def write_metrics(folder, steps, gradient=1.0):
        rows = [
            {"step": step, "metrics": {"actor/grad_norm": gradient}}
            for step in steps
        ]
        (folder / "checkpoints" / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    def assert_failed_record(self, folder):
        record = self.read_record(folder)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure_kind"], "phase_validation")
        self.assertEqual(record["validation_status"], "failed")
        self.assertTrue(record["validation_error"])
        self.assertEqual(record["allocated_gpu_hours"], 1.25)

    def assert_validation_fails(self, folder, method="opd", end=3, resume=None):
        with self.assertRaises(RuntimeError):
            coordinator.validate_phase(folder, method, end, resume)
        self.assert_failed_record(folder)

    def test_valid_cached_phase_needs_no_gpu_or_launch(self):
        folder, checkpoint = self.completed_phase()
        with mock.patch.object(coordinator, "free_gpus") as free_gpus, \
                mock.patch.object(coordinator.subprocess, "run") as launch:
            self.assertEqual(coordinator.phase("opd", folder.name, 3), checkpoint)
        free_gpus.assert_not_called()
        launch.assert_not_called()
        record = self.read_record(folder)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["validation_status"], "passed")

    def test_cached_nan_is_rejected_and_cannot_be_accepted_on_restart(self):
        folder, _ = self.completed_phase()
        self.write_metrics(folder, range(1, 4), gradient=float("nan"))
        with mock.patch.object(coordinator, "free_gpus") as free_gpus, \
                mock.patch.object(coordinator.subprocess, "run") as launch:
            with self.assertRaises(RuntimeError):
                coordinator.phase("opd", folder.name, 3)
            self.assert_failed_record(folder)
            with self.assertRaises(RuntimeError):
                coordinator.phase("opd", folder.name, 3)
        free_gpus.assert_not_called()
        launch.assert_not_called()
        self.assert_failed_record(folder)

    def test_fresh_phase_is_validated_after_launcher_finishes(self):
        name = "fresh-opd"

        def finish_launcher(*args, **kwargs):
            folder, _ = self.completed_phase(name=name)
            self.write_metrics(folder, range(1, 4), gradient=float("nan"))

        with mock.patch.object(coordinator, "free_gpus"), \
                mock.patch.object(coordinator, "status"), \
                mock.patch.object(coordinator.subprocess, "run",
                                  side_effect=finish_launcher) as launch:
            with self.assertRaises(RuntimeError):
                coordinator.phase("opd", name, 3)
        launch.assert_called_once()
        self.assert_failed_record(self.root / "runs" / name)

    def test_discontinuous_steps_fail_even_when_count_and_endpoints_match(self):
        folder, _ = self.completed_phase()
        self.write_metrics(folder, [1, 1, 3])
        self.assert_validation_fails(folder)

    def test_missing_checkpoint_shard_fails_for_each_state_kind(self):
        for kind in ("model", "optim", "extra_state"):
            with self.subTest(kind=kind):
                folder, checkpoint = self.completed_phase(name=kind)
                (checkpoint / "actor" / f"{kind}_world_size_8_rank_7.pt").unlink()
                self.assert_validation_fails(folder)

    def test_empty_checkpoint_files_fail(self):
        filenames = ["data.pt"] + [
            f"actor/{kind}_world_size_8_rank_7.pt"
            for kind in ("model", "optim", "extra_state")
        ]
        for index, filename in enumerate(filenames):
            with self.subTest(filename=filename):
                folder, checkpoint = self.completed_phase(name=f"empty-{index}")
                (checkpoint / filename).write_bytes(b"")
                self.assert_validation_fails(folder)

    def test_zero_gradient_is_valid(self):
        folder, checkpoint = self.completed_phase()
        self.write_metrics(folder, range(1, 4), gradient=0.0)
        self.assertEqual(coordinator.validate_phase(folder, "opd", 3, None), checkpoint)
        self.assertEqual(self.read_record(folder)["validation_status"], "passed")

    def test_infinite_gradient_is_rejected(self):
        for index, gradient in enumerate((float("inf"), float("-inf"))):
            with self.subTest(gradient=gradient):
                folder, _ = self.completed_phase(name=f"infinite-{index}")
                self.write_metrics(folder, range(1, 4), gradient=gradient)
                self.assert_validation_fails(folder)

    def test_launcher_record_must_match_requested_phase(self):
        changes = {
            "method": "grpo",
            "end_step": 4,
            "first_logged_step": 2,
            "last_logged_step": 2,
            "training_batches": 2,
            "return_code": 1,
            "n_gpus": 0,
        }
        for key, value in changes.items():
            with self.subTest(field=key):
                folder, _ = self.completed_phase(name=key)
                record = self.read_record(folder)
                record[key] = value
                self.write_record(folder, record)
                self.assert_validation_fails(folder)

    def test_checkpoint_path_must_be_this_phase_final_checkpoint(self):
        folder, _ = self.completed_phase()
        _, other_checkpoint = self.completed_phase(name="other-phase")
        record = self.read_record(folder)
        record["checkpoint"] = str(other_checkpoint)
        self.write_record(folder, record)
        self.assert_validation_fails(folder)

    def test_resumed_phase_accepts_only_the_remaining_global_steps(self):
        resume = self.root / "prefix" / "global_step_60"
        folder, checkpoint = self.completed_phase(
            name="opd-then-rl-001", method="opd_then_rl", first=61, end=63
        )
        self.assertEqual(
            coordinator.validate_phase(folder, "opd_then_rl", 63, resume),
            checkpoint,
        )
        self.assertEqual(self.read_record(folder)["validation_status"], "passed")
        self.assert_validation_fails(folder, "opd_then_rl", 63, resume=None)


if __name__ == "__main__":
    unittest.main()
