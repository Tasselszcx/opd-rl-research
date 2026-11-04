"""Control-plane contract tests with explicitly synthetic, temporary results.

FakeBackend below is only a dependency-injected test double; it never trains a
model, exports real weights, or produces research evidence. TemporaryDirectory
removes all fixtures and simulated score reports at test completion.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from meta_teaching.backend import PhaseFailure
from meta_teaching.common import atomic_json, read_json
from meta_teaching.config import DEFAULTS, validate_config
from meta_teaching.runner import BudgetExhausted, ExperimentRunner


class FakeBackend:
    simulated = True

    def __init__(self, config, *, score=None, duration=None, fail_on=None):
        self.config = config
        self.executions = []
        self.snapshots = []
        self.records = []
        self.score = score or self.default_score
        self.duration = duration or (lambda spec: 1.0 if spec.phase == "eval" else 2.0)
        self.fail_on = fail_on
        self.failed_once = False

    @staticmethod
    def default_score(spec):
        if spec.phase_id == "initial_eval":
            return 0.2
        if spec.phase_id == "final_test":
            return 0.99
        if spec.phase == "warmup":
            return {"plain": 0.8, "partial": 0.7, "full": 0.6}[spec.arm_id]
        return {"direct_rl": 0.4, "plain": 0.45, "partial": 0.55, "full": 0.65}[spec.arm_id]

    def execute(self, spec):
        directory = Path(spec.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / "launcher_record.json"
        if record_path.exists():
            saved = read_json(record_path)
            if saved["spec_fingerprint"] != spec.identity():
                raise ValueError("Changed fixture phase")
            if saved["status"] == "completed":
                return saved
        self.executions.append(copy.deepcopy(spec))
        state_path = Path(self.config["experiment"]["output_dir"]) / "experiment_state.json"
        self.snapshots.append(read_json(state_path))
        wall = self.duration(spec)
        started = 10000.0 + len(self.records) * 100.0
        checkpoint = None
        if spec.phase != "eval":
            checkpoint = str(directory / "checkpoints" / f"global_step_{spec.steps}")
            (Path(checkpoint) / "actor").mkdir(parents=True, exist_ok=True)
            (Path(checkpoint) / "SIMULATED_ONLY.txt").write_text("No model weights. Unit-test fixture only.\n")
        record = {
            "schema_version": 1, "backend": "test_double", "simulated": True,
            "phase_id": spec.phase_id, "spec_fingerprint": spec.identity(), "status": "completed",
            "started_unix": started, "finished_unix": started + wall,
            "wall_seconds": wall, "allocated_gpu_seconds": wall * self.config["training"]["n_gpus"],
            "summary": {"phase": spec.phase, "arm_id": spec.arm_id, "status": "completed",
                        "val_score": self.score(spec), "checkpoint_path": checkpoint,
                        "steps_completed": spec.steps,
                        "teacher_enabled": spec.phase == "warmup",
                        "teacher_queries": 4 if spec.phase == "warmup" else 0},
        }
        if not self.failed_once and self.fail_on and self.fail_on(spec):
            self.failed_once = True
            record.update(status="failed", error="injected test failure")
            self.records.append(copy.deepcopy(record))
            atomic_json(record_path, record)
            raise PhaseFailure("injected test failure", record)
        self.records.append(copy.deepcopy(record))
        atomic_json(record_path, record)
        return record


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, *, rounds=2, repeats=2):
        config = copy.deepcopy(DEFAULTS)
        config["models"] = {"student": "/local/mock/student", "teacher": "/local/mock/teacher"}
        config["data"]["manifest"] = str(self.root / "data" / "manifest.json")
        config["experiment"].update(output_dir=str(self.root / "run"), rounds=rounds, repeats=repeats,
                                     validation_reserve_seconds=0.1, max_total_gpu_hours=100)
        config["training"].update(warmup_steps=3, rl_steps=7)
        validate_config(config)
        return config

    def one_action(self, config):
        config["actions"] = [copy.deepcopy(config["actions"][-1])]
        return config

    def test_two_rounds_pair_source_seed_and_equal_student_steps(self):
        config = self.config()
        backend = FakeBackend(config)
        runner = ExperimentRunner(config, backend=backend)
        report = runner.run()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["simulated"])
        self.assertIn("NO MODEL TRAINING OR RESEARCH RESULT", report["warning"])
        self.assertIn("NOT equal total compute", report["comparison"])
        self.assertEqual(len(runner.state["rounds"]), 2)
        specs = {spec.phase_id: spec for spec in backend.executions}
        for index, plan in enumerate(runner.state["rounds"]):
            for repeat in range(2):
                branches = [branch for branch in plan["branches"].values() if branch["repeat"] == repeat]
                self.assertEqual(len({branch["source_checkpoint"] for branch in branches}), 1)
                self.assertEqual(len({branch["seed"] for branch in branches}), 1)
                self.assertEqual(branches[0]["seed"], config["experiment"]["seed"] + index * 1009 + repeat)
                for branch in branches:
                    self.assertEqual(branch["warmup_steps"] + branch["rl_steps"], 10)
                    phase = specs[branch["branch_id"] + ("/rl" if branch["arm_id"] == "direct_rl" else "/warmup")]
                    self.assertEqual(phase.resume_checkpoint, plan["source_checkpoint"])
                    self.assertEqual(phase.seed, branch["seed"])
                    if branch["arm_id"] == "direct_rl":
                        self.assertEqual(phase.steps, 10)
                    else:
                        self.assertEqual(phase.steps, 3)
                        rl = specs[branch["branch_id"] + "/rl"]
                        self.assertEqual(rl.steps, 7)
                        self.assertIsNone(rl.teacher_model)
                        self.assertIn("/warmup/checkpoints/", rl.resume_checkpoint)
            if index:
                previous = runner.state["rounds"][index - 1]
                promoted = previous["branches"]["0:" + previous["promoted_action"]]
                self.assertEqual(plan["source_checkpoint"], promoted["checkpoint_path"])

    def test_policy_labels_use_post_rl_difference_not_warmup_score(self):
        config = self.config(rounds=1, repeats=1)
        runner = ExperimentRunner(config, backend=FakeBackend(config))
        runner.run()
        feedback = {row["arm_id"]: row["post_rl_incremental_value"] for row in runner.state["rounds"][0]["feedback"]}
        for arm, expected in {"plain": 0.05, "partial": 0.15, "full": 0.25}.items():
            self.assertAlmostEqual(feedback[arm], expected)
            self.assertAlmostEqual(runner.policy.diagnostics()["arms"][arm]["mean_observed_reward"], expected)
        context = runner.state["rounds"][0]["context"]
        self.assertGreater(runner.policy.predict(context, "full"), runner.policy.predict(context, "plain"))
        self.assertEqual(runner.policy.total_updates, 4)

    def test_paired_feedback_subtracts_incremental_measured_cost(self):
        config = self.one_action(self.config(rounds=1, repeats=1))
        config["experiment"]["cost_penalty_per_gpu_hour"] = 9.0
        runner = ExperimentRunner(config, backend=FakeBackend(config))
        runner.run()
        # Candidate = 4 wall seconds * 8 GPUs; baseline = 2 * 8.
        expected = (0.65 - 0.4) - 9.0 * ((32.0 - 16.0) / 3600.0)
        feedback = runner.state["rounds"][0]["feedback"]
        self.assertEqual(len(feedback), 1)
        self.assertAlmostEqual(feedback[0]["post_rl_incremental_value"], expected)

    def test_deployment_committed_before_feedback_and_not_best_seed(self):
        config = self.config(rounds=1, repeats=2)
        def score(spec):
            if spec.phase == "rl" and spec.arm_id == "full":
                return 0.3 if "repeat_000" in spec.phase_id else 0.95
            return FakeBackend.default_score(spec)
        backend = FakeBackend(config, score=score)
        runner = ExperimentRunner(config, backend=backend)
        original_select = runner.policy.select
        with patch.object(runner.policy, "select", side_effect=lambda context, eligible=None:
                          "full" if eligible is None else original_select(context, eligible)):
            runner.run()
        plan = runner.state["rounds"][0]
        self.assertEqual(plan["deployment_action_before_feedback"], "full")
        self.assertEqual(plan["promoted_action"], "full")
        self.assertEqual(runner.state["current_checkpoint"], plan["branches"]["0:full"]["checkpoint_path"])
        self.assertEqual(runner.state["current_score"], 0.3)
        snapshots = [snap for spec, snap in zip(backend.executions, backend.snapshots) if spec.phase != "eval"]
        for snapshot in snapshots:
            pending = snapshot["pending_round"]
            self.assertEqual(pending["deployment_action_before_feedback"], "full")
            self.assertEqual(snapshot["policy"], pending["policy_after_selection"])
            self.assertEqual(sum(arm["count"] for arm in snapshot["policy"]["arms"].values()), 0)

    def test_final_test_is_last_and_never_updates_policy(self):
        config = self.config(rounds=1, repeats=1)
        backend = FakeBackend(config)
        runner = ExperimentRunner(config, backend=backend)
        runner.run()
        test_calls = [spec for spec in backend.executions if spec.validation_path.endswith("/test.parquet")]
        self.assertEqual(len(test_calls), 1)
        self.assertEqual(backend.executions[-1].phase_id, "final_test")
        self.assertEqual(test_calls[0].phase, "eval")
        self.assertIsNone(test_calls[0].teacher_model)
        self.assertEqual(backend.snapshots[-1]["policy"], runner.state["policy"])
        self.assertEqual(runner.state["final_test"]["score"], 0.99)
        self.assertNotEqual(runner.state["current_score"], 0.99)
        self.assertEqual(len(backend.snapshots[-1]["rounds"]), 1)

    def test_completed_resume_does_not_execute_or_bill_twice(self):
        config = self.config(rounds=1, repeats=1)
        backend = FakeBackend(config)
        runner = ExperimentRunner(config, backend=backend)
        first = runner.run()
        events = copy.deepcopy(runner.state["cost_events"])
        count = len(backend.executions)
        resumed = ExperimentRunner(config, backend=backend, resume=True)
        second = resumed.run()
        self.assertEqual(len(backend.executions), count)
        phase_ids = {spec.phase_id for spec in backend.executions}
        self.assertEqual({key: event for key, event in resumed.state["cost_events"].items() if event["phase_id"] in phase_ids},
                         {key: event for key, event in events.items() if event["phase_id"] in phase_ids})
        # Reopening the controller may legitimately add newly measured CPU/idle
        # reservation overhead; existing training attempts must not be rebilled.
        self.assertGreaterEqual(second["total_allocated_gpu_hours"], first["total_allocated_gpu_hours"])

    def test_failure_charged_and_resume_retains_completed_phase(self):
        config = self.one_action(self.config(rounds=1, repeats=1))
        backend = FakeBackend(config, fail_on=lambda spec: spec.phase == "rl" and spec.arm_id == "full")
        runner = ExperimentRunner(config, backend=backend)
        with self.assertRaisesRegex(PhaseFailure, "injected"):
            runner.run()
        self.assertEqual(runner.state["status"], "failed")
        failures = [event for event in runner.state["cost_events"].values() if event["status"] == "failed"]
        self.assertEqual(len(failures), 1)
        self.assertGreater(failures[0]["allocated_gpu_seconds"], 0)
        completed_ids = [record["phase_id"] for record in backend.records if record["status"] == "completed"]
        resumed = ExperimentRunner(config, backend=backend, resume=True)
        resumed.run()
        for phase_id in completed_ids:
            self.assertEqual(sum(spec.phase_id == phase_id for spec in backend.executions), 1)
        failure_id = failures[0]["phase_id"]
        costs = [event for event in resumed.state["cost_events"].values() if event["phase_id"] == failure_id]
        self.assertEqual(len(costs), 2)
        self.assertEqual({event["status"] for event in costs}, {"failed", "completed"})
        self.assertEqual(resumed.state["status"], "completed")

    def test_cached_backend_record_after_controller_crash_is_not_rebilled(self):
        config = self.config(rounds=1, repeats=1)
        backend = FakeBackend(config)
        runner = ExperimentRunner(config, backend=backend)
        spec = runner._spec("initial_eval", "eval", "initial", None, 0, 42)
        record = runner._phase(spec)
        events = copy.deepcopy(runner.state["cost_events"])
        # Controller persisted its ledger but lost the phase-result pointer.
        runner.state["phase_results"].pop("initial_eval")
        runner._save()
        resumed = ExperimentRunner(config, backend=backend, resume=True)
        replay = resumed._phase(resumed._spec("initial_eval", "eval", "initial", None, 0, 42))
        self.assertEqual(replay["started_unix"], record["started_unix"])
        self.assertEqual({key: event for key, event in resumed.state["cost_events"].items() if event["phase_id"] == "initial_eval"},
                         {key: event for key, event in events.items() if event["phase_id"] == "initial_eval"})
        self.assertEqual(len(backend.executions), 1)

    def test_keyboard_interrupt_retains_durable_attempt_cost(self):
        config = self.config(rounds=1, repeats=1)
        class InterruptedBackend(FakeBackend):
            def execute(self, spec):
                record = super().execute(spec)
                record["status"] = "interrupted"
                atomic_json(Path(spec.output_dir) / "launcher_record.json", record)
                raise KeyboardInterrupt()
        backend = InterruptedBackend(config)
        runner = ExperimentRunner(config, backend=backend)
        with self.assertRaises(KeyboardInterrupt):
            runner.run()
        self.assertEqual(runner.state["status"], "interrupted")
        events = [event for event in runner.state["cost_events"].values() if event["phase_id"] == "initial_eval"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "interrupted")
        self.assertEqual(events[0]["allocated_gpu_seconds"], 8.0)
        self.assertEqual(read_json(runner.state_path)["cost_events"], runner.state["cost_events"])

    def test_global_budget_stops_before_next_phase(self):
        config = self.config(rounds=1, repeats=1)
        config["experiment"]["max_total_gpu_hours"] = 8 / 3600
        backend = FakeBackend(config)
        runner = ExperimentRunner(config, backend=backend)
        report = runner.run()
        self.assertEqual(report["status"], "budget_exhausted")
        self.assertEqual([spec.phase_id for spec in backend.executions], ["initial_eval"])
        self.assertEqual(runner.policy.total_updates, 0)
        self.assertIsNone(report["final_test"])

    def test_walltime_global_budget_cannot_shorten_a_matched_branch(self):
        config = self.one_action(self.config(rounds=1, repeats=1))
        config["experiment"].update(protocol="fixed_walltime", branch_budget_seconds=20,
                                     budget_tolerance_fraction=0, max_total_gpu_hours=12 * 8 / 3600)
        validate_config(config)
        backend = FakeBackend(config)
        runner = ExperimentRunner(config, backend=backend)
        report = runner.run()
        self.assertEqual(report["status"], "budget_exhausted")
        self.assertIn("full matched allocation", report["error"])
        self.assertEqual([spec.phase_id for spec in backend.executions], ["initial_eval"])
        self.assertEqual(runner.state["pending_round"]["branches"], {})
        self.assertEqual(runner.policy.total_updates, 0)

    def test_phase_rejects_global_budget_truncation_without_moving_deadline(self):
        config = self.config(rounds=1, repeats=1)
        config["experiment"]["max_total_gpu_hours"] = 10 * 8 / 3600
        backend = FakeBackend(config)
        runner = ExperimentRunner(config, backend=backend)
        requested_deadline = time.time() + 20
        spec = runner._spec("limited/rl", "rl", "direct_rl", None, 10, 42,
                            deadline=requested_deadline)
        with self.assertRaisesRegex(BudgetExhausted, "original wall-time allocation"):
            runner._phase(spec)
        self.assertEqual(spec.deadline_unix, requested_deadline)
        self.assertEqual(backend.executions, [])
        self.assertNotIn(spec.phase_id, runner.state["phase_results"])

    def test_walltime_overrun_excludes_candidate_from_feedback(self):
        config = self.one_action(self.config(rounds=1, repeats=1))
        config["experiment"].update(protocol="fixed_walltime", branch_budget_seconds=20,
                                     budget_tolerance_fraction=0, evaluate_final_test=False)
        validate_config(config)
        backend = FakeBackend(config, duration=lambda spec: 1 if spec.phase == "eval" else
                              2 if spec.phase == "warmup" else 25 if spec.arm_id == "full" else 10)
        runner = ExperimentRunner(config, backend=backend)
        runner.run()
        plan = runner.state["rounds"][0]
        full = plan["branches"]["0:full"]
        self.assertFalse(full["valid_for_policy_update"])
        self.assertIn("branch_budget_overrun", full["exclusion_reasons"])
        self.assertEqual(plan["feedback"], [])
        self.assertEqual(runner.policy.arm_counts["full"], 0)
        self.assertEqual(plan["promoted_action"], "direct_rl")
        self.assertTrue(all(spec.deadline_unix is not None for spec in backend.executions))

    def test_unpromotable_round_does_not_leave_policy_report_ahead_of_checkpoint(self):
        config = self.one_action(self.config(rounds=1, repeats=2))
        config["experiment"].update(protocol="fixed_walltime", branch_budget_seconds=20,
                                     budget_tolerance_fraction=0, evaluate_final_test=False)
        backend = FakeBackend(config, duration=lambda spec: 1 if spec.phase == "eval" else
                              2 if spec.phase == "warmup" else 25 if "repeat_000" in spec.phase_id else 5)
        runner = ExperimentRunner(config, backend=backend)
        report = runner.run()
        self.assertEqual(report["status"], "budget_exhausted")
        # Repeat zero cannot be promoted, but repeat one has valid labels. A
        # failed promotion must not leave uncheckpointed updates in the report.
        self.assertEqual(runner.state["policy"], runner.policy.state_dict())
        self.assertEqual(read_json(runner.state_path)["policy"], runner.policy.state_dict())

    def test_previous_student_teacher_export_is_simulated_and_reused(self):
        config = self.one_action(self.config(rounds=2, repeats=1))
        config["experiment"]["teacher_mode"] = "previous_student"
        backend = FakeBackend(config)
        runner = ExperimentRunner(config, backend=backend)
        with patch("meta_teaching.runner.subprocess.run", side_effect=AssertionError("Simulation must not invoke real model merger")):
            runner.run()
        rounds = runner.state["rounds"]
        export = Path(config["experiment"]["output_dir"]) / "teachers" / "round_001"
        manifest = read_json(export / "export_manifest.json")
        self.assertTrue(manifest["simulated"])
        self.assertEqual(manifest["source_checkpoint"], rounds[1]["source_checkpoint"])
        self.assertEqual(rounds[1]["teacher_model"], str(export))
        first_warmup = next(spec for spec in backend.executions if spec.phase == "warmup" and spec.phase_id.startswith("round_000"))
        next_warmup = next(spec for spec in backend.executions if spec.phase == "warmup" and spec.phase_id.startswith("round_001"))
        self.assertEqual(first_warmup.teacher_model, config["models"]["teacher"])
        self.assertEqual(next_warmup.teacher_model, str(export))
        self.assertIsNone(next(spec for spec in backend.executions if spec.phase == "rl" and spec.phase_id.startswith("round_001")).teacher_model)


if __name__ == "__main__":
    unittest.main()
