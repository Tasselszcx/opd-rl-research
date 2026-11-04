import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from state_guidance.runtime_paths import enter_driver_directory, enter_task_directory


class RuntimePathTests(unittest.TestCase):
    def test_ray_starts_local_but_validation_stays_in_shared_run(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            code = base / '代码 cache'
            source = code / 'logic/training/train_grpo.py'
            source.parent.mkdir(parents=True)
            source.write_text('')
            run = base / '共享 outputs/run'
            (run / 'checkpoints').mkdir(parents=True)
            config = SimpleNamespace(data={'g3_frozen_task': 'kk'},
                                     trainer=SimpleNamespace(default_local_dir=str(run / 'checkpoints')))
            try:
                os.chdir(run)
                with patch.dict(os.environ, {'REPRO_RAY_ISOLATE': '1'}):
                    enter_driver_directory(config, source)
                self.assertEqual(Path.cwd(), code.resolve())
                enter_task_directory(config)
                output = Path('val_logs') / 'step_105' / 'metrics.json'
                output.parent.mkdir(parents=True)
                output.write_text('{"step": 105}')
                self.assertTrue((run / output).is_file())
                self.assertFalse((code / output).exists())
            finally:
                os.chdir(original)

    def test_non_g3_entry_keeps_its_existing_directory(self):
        config = SimpleNamespace(data={})
        before = Path.cwd()
        with patch.dict(os.environ, {'REPRO_RAY_ISOLATE': '1'}):
            enter_driver_directory(config, '/unused/logic/training/train_grpo.py')
        enter_task_directory(config)
        self.assertEqual(Path.cwd(), before)

    def test_relative_task_outputs_are_rejected_before_chdir(self):
        before = Path.cwd()
        config = SimpleNamespace(data={'g3_frozen_task': 'kk'},
                                 trainer=SimpleNamespace(default_local_dir='relative/checkpoints'))
        with self.assertRaises(ValueError):
            enter_task_directory(config)
        self.assertEqual(Path.cwd(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
