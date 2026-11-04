"""Exercise the actual CLI in child processes without loading models or GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(sys.version_info < (3, 10), "MVP requires Python >=3.10")
class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.environment = dict(os.environ)
        self.environment.pop("STUDENT_MODEL", None)
        self.environment.pop("TEACHER_MODEL", None)

    def tearDown(self):
        self.temporary.cleanup()

    def cli(self, *arguments, expected=0):
        result = subprocess.run(
            [sys.executable, "-m", "meta_teaching", *map(str, arguments)],
            cwd=ROOT, env=self.environment, text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_default_simulation_resume_status_without_model_environment(self):
        output = self.directory / "simulated"
        initial = self.cli("simulate", "--output-dir", output)
        self.assertIn("SIMULATION ONLY", initial.stdout)
        printed = json.loads(initial.stdout[initial.stdout.index("{"):])
        self.assertEqual(printed, json.loads((output / "results.json").read_text()),
                         "Printed report must include the final bookkeeping costs written to disk")
        state = json.loads((output / "experiment_state.json").read_text())
        self.assertTrue(state["simulated"])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["rounds"]), 3)
        attempts_before = {str(path): path.read_bytes() for path in output.rglob("launcher_record.json")}
        self.assertTrue(attempts_before)
        status = json.loads(self.cli("status", "--output-dir", output).stdout)
        self.assertTrue(status["simulated"])
        self.assertEqual(status["rounds_completed"], 3)
        resumed = self.cli("simulate", "--output-dir", output, "--resume")
        attempts_after = {str(path): path.read_bytes() for path in output.rglob("launcher_record.json")}
        self.assertEqual(attempts_before, attempts_after, "A completed simulation must not rerun phases")
        report = json.loads((output / "results.json").read_text())
        self.assertEqual(json.loads(resumed.stdout[resumed.stdout.index("{"):]), report)
        self.assertIn("NO MODEL TRAINING", report["warning"])
        self.assertTrue(report["simulated"])

    def test_plan_save_does_not_launch_any_phase(self):
        plan_path = self.directory / "saved-plan.json"
        output = self.directory / "unstarted"
        result = self.cli(
            "plan", "--config", ROOT / "mvp_configs/simulation.json",
            "--output-dir", output, "--save", plan_path,
        )
        self.assertIn("no models loaded or jobs launched", result.stdout)
        plan = json.loads(plan_path.read_text())
        self.assertEqual(len(plan["phases"]), 7)
        self.assertTrue(any(phase["spec"]["phase"] == "warmup" for phase in plan["phases"]))
        self.assertTrue(any(phase["spec"]["arm_id"] == "direct_rl" for phase in plan["phases"]))
        self.assertFalse(list(output.rglob("launcher_record.json")))
        self.assertFalse((output / "experiment_state.json").exists())

    def test_real_config_fails_cleanly_on_unset_model_variables(self):
        output = self.directory / "must-not-launch"
        result = self.cli("plan", "--output-dir", output, expected=1)
        self.assertIn("Unset environment variable", result.stderr)
        self.assertIn("STUDENT_MODEL", result.stderr)
        self.assertFalse(list(output.rglob("launcher_record.json")))

    def test_status_missing_run_reports_error(self):
        result = self.cli("status", "--output-dir", self.directory / "absent", expected=1)
        self.assertIn("ERROR:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
