"""Keep Ray startup local while G3 task outputs remain in the shared run folder."""
import os
from pathlib import Path


def enter_driver_directory(config, code_file):
    if os.environ.get('REPRO_RAY_ISOLATE') == '1' and config.data.get('g3_frozen_task'):
        root = Path(code_file).resolve().parents[2]
        assert root.is_dir()
        os.chdir(root)


def enter_task_directory(config):
    if config.data.get('g3_frozen_task'):
        checkpoint = Path(config.trainer.default_local_dir)
        if not checkpoint.is_absolute() or checkpoint.name != 'checkpoints':
            raise ValueError('G3 requires an absolute shared checkpoints directory')
        assert checkpoint.parent.is_dir()
        os.chdir(checkpoint.parent)
