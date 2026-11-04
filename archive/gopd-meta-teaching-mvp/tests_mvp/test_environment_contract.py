"""Offline deployment guards: never install or touch GPU processes in tests."""
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mvp_environment.sh"


def environment_module():
    embedded = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    module = types.ModuleType("mvp_environment_contract")
    exec(compile(embedded, str(SCRIPT), "exec"), module.__dict__)
    return module


class EnvironmentContractTests(unittest.TestCase):
    def setUp(self):
        self.module = environment_module()

    @unittest.skipIf(sys.version_info < (3, 10), "MVP requires Python >=3.10")
    def test_default_mode_only_reports_versions(self):
        import os
        environment = dict(os.environ, MVP_PYTHON=sys.executable)
        completed = subprocess.run(["bash", str(SCRIPT)], env=environment, capture_output=True, text=True, timeout=30, check=True)
        report = json.loads(completed.stdout)
        self.assertEqual(report["mode"], "check")
        self.assertTrue(report["read_only"])
        self.assertIn("torch", report["packages"])
        self.assertIn("no import, install", report["note"])

    def test_cpu_profile_contains_no_training_stack(self):
        pins = self.module.read_pins(ROOT / "mvp_requirements_cpu.txt")
        self.assertEqual(set(pins), {"pytest", "pyarrow", "hydra-core", "packaging"})

    def test_hopper_direct_pins_resolve_original_tensordict_conflict(self):
        pins = self.module.read_pins(ROOT / "mvp_requirements_hopper.txt")
        expected = {"torch": "2.6.0", "vllm": "0.8.5.post1", "tensordict": "0.8.3", "torchdata": "0.11.0", "transformers": "4.51.3"}
        for name, value in expected.items():
            self.assertEqual(pins[name]["version"], value)
        self.assertNotIn("tensordict==0.6.2", (ROOT / "mvp_requirements_hopper.txt").read_text())

    def test_system_interpreter_install_refused_before_pip(self):
        with patch.object(self.module.sys, "prefix", "/system/python"), \
             patch.object(self.module.sys, "base_prefix", "/system/python"), \
             patch.dict(self.module.os.environ, {}, clear=True), \
             patch.object(self.module.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "dedicated active venv"):
                self.module.install(ROOT, hopper=False)
            run.assert_not_called()

    def test_non_linux_hopper_refused(self):
        with patch.object(self.module.platform, "system", return_value="Darwin"):
            with self.assertRaisesRegex(RuntimeError, "Linux x86_64"):
                self.module.require_hopper_supported()

    def gpu_guard(self, gpu_line):
        with patch.object(self.module.platform, "system", return_value="Linux"), \
             patch.object(self.module.platform, "machine", return_value="x86_64"), \
             patch.object(self.module.sys, "version_info", (3, 10, 14)), \
             patch.object(self.module.shutil, "which", side_effect=lambda name: "/bin/" + name), \
             patch.object(self.module.subprocess, "run", side_effect=[
                 subprocess.CompletedProcess([], 0, gpu_line),
                 subprocess.CompletedProcess([], 0, "Cuda compilation tools, release 12.4, V12.4.131"),
             ]):
            return self.module.require_hopper_supported()

    def test_blackwell_refused_before_install(self):
        with self.assertRaisesRegex(RuntimeError, "Blackwell/newer GPU"):
            self.gpu_guard("10.3, NVIDIA B300\n")

    def test_hopper_accepted_and_unknown_capability_refused(self):
        self.assertEqual(self.gpu_guard("9.0, NVIDIA H800\n"), ["9.0, NVIDIA H800"])
        with self.assertRaisesRegex(RuntimeError, "Cannot establish GPU compute capability"):
            self.gpu_guard("N/A, Unknown GPU\n")


if __name__ == "__main__":
    unittest.main()
