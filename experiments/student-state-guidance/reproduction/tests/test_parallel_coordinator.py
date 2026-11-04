"""CPU-only checks for two-slot scheduling and preserved failure evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock


REPRODUCTION = Path(__file__).resolve().parents[1]
with mock.patch.object(sys, "path", [str(REPRODUCTION), *sys.path]):
    spec = importlib.util.spec_from_file_location(
        "parallel_test_coordinator", REPRODUCTION / "coordinator_4x2.py"
    )
    coordinator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coordinator)


class ParallelCoordinatorTests(unittest.TestCase):
    @staticmethod
    def lanes():
        return [
            [coordinator.job("grpo", "grpo", 150, "A"),
             coordinator.job("kdrl_anneal", "joint", 150, "A")],
            [coordinator.job("opd", "opd", 150, "B"),
             coordinator.job("opd_then_rl", "sequential", 150, "B")],
        ]

    def test_slot_check_ignores_other_slots_memory(self):
        output = "\n".join(
            f"{index}, {0 if index < 4 else 190000}" for index in range(8)
        )
        with mock.patch.object(coordinator.subprocess, "check_output", return_value=output), \
                mock.patch.object(coordinator.time, "sleep") as sleep:
            coordinator.free_slot((0, 1, 2, 3))
        sleep.assert_not_called()

    def test_parallel_lanes_preserve_each_lanes_order(self):
        barrier = threading.Barrier(2)
        calls = []

        def fake_phase(**phase):
            calls.append(phase["method"])
            if phase["method"] in ("grpo", "opd"):
                barrier.wait(timeout=5)
            return Path("/" + phase["method"])

        with mock.patch.object(coordinator, "phase", side_effect=fake_phase), \
                mock.patch.object(coordinator, "study_status"), \
                mock.patch.object(coordinator, "job_snapshots", return_value=[]):
            result = coordinator.run_lanes("test", self.lanes())
        self.assertEqual(set(result), {"grpo", "opd", "opd_then_rl", "kdrl_anneal"})
        self.assertEqual(set(calls[:2]), {"grpo", "opd"})
        self.assertLess(calls.index("opd"), calls.index("opd_then_rl"))
        self.assertLess(calls.index("grpo"), calls.index("kdrl_anneal"))

    def test_failure_allows_inflight_completion_but_blocks_successors(self):
        started = threading.Event()
        completed = threading.Event()
        calls = []

        def fake_phase(**phase):
            calls.append(phase["method"])
            if phase["method"] == "grpo":
                self.assertTrue(started.wait(5))
                raise ValueError("owned slot failed")
            if phase["method"] == "opd":
                started.set()
                self.assertTrue(phase["cancel_event"].wait(5))
                completed.set()
                return Path("/opd")
            self.fail("A successor must not launch after a failure in either slot")

        with mock.patch.object(coordinator, "phase", side_effect=fake_phase), \
                mock.patch.object(coordinator, "study_status"), \
                mock.patch.object(coordinator, "job_snapshots", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "other in-flight"):
                coordinator.run_lanes("test", self.lanes())
        self.assertTrue(completed.is_set())
        self.assertEqual(set(calls), {"grpo", "opd"})

    def test_eight_gpu_cache_is_rejected_with_failure_and_cost_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            record_path = folder / "launcher_record.json"
            record_path.write_text(json.dumps({
                "n_gpus": 8, "status": "completed", "allocated_gpu_hours": 3,
            }))
            with mock.patch.object(
                coordinator, "validate_phase",
                return_value=folder / "checkpoints/global_step_150",
            ):
                with self.assertRaises(RuntimeError):
                    coordinator.validate_four_gpu_phase(folder, "grpo", 150)
            record = coordinator.read(record_path)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["validation_status"], "failed")
            self.assertEqual(record["allocated_gpu_hours"], 3)

    def test_retry_requires_flag_and_archive_preserves_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "job"
            folder.mkdir()
            (folder / "launcher_record.json").write_text(json.dumps({
                "status": "failed", "allocated_gpu_hours": 3,
            }))
            with mock.patch.object(coordinator, "ALLOW_RETRY", False):
                with self.assertRaises(RuntimeError):
                    coordinator.archive_failed(folder)
            with mock.patch.object(coordinator, "ALLOW_RETRY", True):
                coordinator.archive_failed(folder)
            self.assertFalse(folder.exists())
            archived = list(Path(directory).glob("job.failed-*"))
            self.assertEqual(len(archived), 1)
            record = coordinator.read(archived[0] / "launcher_record.json")
            self.assertEqual(record["allocated_gpu_hours"], 3)

    def test_launcher_receives_exact_assigned_gpu_slot(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(coordinator, "ROOT", Path(directory)), \
                mock.patch.object(coordinator, "free_slot"), \
                mock.patch.object(coordinator, "validate_four_gpu_phase", return_value=Path("/done")), \
                mock.patch.object(coordinator.subprocess, "run") as run:
            coordinator.phase("grpo", "grpo", 150, (4, 5, 6, 7))
            args = run.call_args.args[0]
            self.assertEqual(args[args.index("--gpu-ids") + 1], "4,5,6,7")

    def test_status_includes_prior_ledger_without_merging_run_folders(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(coordinator, "ROOT", Path(directory)), \
                mock.patch.object(coordinator, "costs", return_value=2), \
                mock.patch.object(coordinator, "status") as status:
            (Path(directory) / "prior_cost_ledger.json").write_text(json.dumps({
                "events": [{"allocated_gpu_hours": 3}],
            }))
            coordinator.study_status("test")
            self.assertEqual(status.call_args.kwargs["total_recorded_gpu_hours"], 5)
            self.assertEqual(status.call_args.kwargs["current_schedule_gpu_hours"], 2)
            self.assertEqual(status.call_args.kwargs["prior_schedule_and_setup_gpu_hours"], 3)


if __name__ == "__main__":
    unittest.main()
