"""Exercise the real trainer loop on CPU with fake model workers.

We compile the unchanged method AST from ray_trainer.py to avoid importing CUDA-
only distributed worker modules on a laptop. DataProto operations and PyTorch
arithmetic are real; generation/parameter updates are explicitly fake.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
from pprint import pprint
import sys
import types
import uuid

import pytest

np = pytest.importorskip("numpy", reason="backend loop tests require optional CPU training dependencies")
torch = pytest.importorskip("torch", reason="backend loop tests require PyTorch")
pytest.importorskip("ray", reason="real DataProto imports Ray (no Ray cluster is started)")
pytest.importorskip("tensordict", reason="real DataProto requires TensorDict")
OmegaConf = pytest.importorskip("omegaconf", reason="trainer loop config requires OmegaConf").OmegaConf

from test_backend_hooks import hooks, config, ROOT


class Progress:
    def __init__(self, **kwargs):
        self.count = 0
    def update(self, count):
        self.count += count
    def close(self):
        pass


@contextmanager
def marked_timer(key, timings, **kwargs):
    try:
        yield
    finally:
        timings[key] = 0.001


class Actor:
    def __init__(self):
        self.updates = 0
        self.generations = 0
    def generate_sequences(self, inputs):
        from verl import DataProto
        self.generations += 1
        n = len(inputs)
        responses = torch.tensor([[7, 8]] * n)
        ids = torch.cat((inputs.batch["input_ids"], responses), -1)
        attention = torch.ones_like(ids)
        return DataProto.from_dict(
            tensors={"responses": responses, "input_ids": ids, "attention_mask": attention,
                     "position_ids": attention.cumsum(-1) - 1},
            meta_info={"timing": {"generation": 0.001}},
        )
    def compute_log_prob(self, batch):
        from verl import DataProto
        shape = batch.batch["responses"].shape
        return DataProto.from_dict(tensors={"old_log_probs": -torch.ones(shape), "entropys": torch.ones(shape)})
    def update_actor(self, batch):
        from verl import DataProto
        self.updates += 1
        self.last_advantages = batch.batch["advantages"].clone()
        return DataProto(meta_info={"metrics": {}})


class Teacher:
    world_size = 1
    def __init__(self, enabled):
        self.enabled = enabled
        self.calls = 0
    def compute_ref_log_prob(self, data):
        from verl import DataProto
        assert self.enabled, "Independent RL must not call teacher"
        self.calls += 1
        return DataProto.from_dict(tensors={"ref_log_prob": -0.5 * torch.ones_like(data.batch["old_log_probs"])})


class Loader:
    sampler = object()
    def __len__(self):
        return 1
    def __iter__(self):
        yield {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
               "position_ids": torch.tensor([[0, 1]]), "data_source": np.array(["test"], dtype=object)}


def make_trainer(tmp_path, monkeypatch, phase="rl", steps=2, deadline=None, query_fraction=1):
    from verl import DataProto
    tracking = types.ModuleType("verl.utils.tracking")
    class Tracking:
        def __init__(self, **kwargs):
            self.logs = []
        def log(self, **kwargs):
            self.logs.append(kwargs)
    tracking.Tracking = Tracking
    monkeypatch.setitem(sys.modules, "verl.utils.tracking", tracking)

    cfg = config(phase, str(tmp_path))
    cfg["meta_teaching"].update(phase_steps=steps, deadline_unix=deadline, query_fraction=query_fraction)
    cfg["trainer"].update(project_name="test", experiment_name="test", logger=["console"], total_epochs=1,
                          balance_batch=False, val_before_train=False, critic_warmup=0,
                          test_freq=-1, save_freq=-1, esi_redundant_time=0, default_local_dir=str(tmp_path / "checkpoints"))
    cfg["global_profiler"] = {"steps": None, "profile_continuous_steps": False}
    cfg["actor_rollout_ref"]["rollout"].update(n=2)
    cfg["actor_rollout_ref"]["actor"].update(loss_agg_mode="token-mean")
    cfg["algorithm"].update(gamma=1, lam=1)
    cfg["reward_model"] = {"launch_reward_fn_async": False}
    cfg = OmegaConf.create(cfg)

    def compute_advantage(data, **kwargs):
        data.batch["advantages"] = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        data.batch["returns"] = torch.zeros_like(data.batch["advantages"])
        return data
    namespace = dict(
        os=os, np=np, torch=torch, uuid=uuid, deepcopy=deepcopy, pprint=pprint, DataProto=DataProto,
        tqdm=Progress, marked_timer=marked_timer,
        score_selected_teacher=hooks.score_selected_teacher,
        AdvantageEstimator=types.SimpleNamespace(REMAX="remax"),
        Role=types.SimpleNamespace(RefPolicy="ref"),
        agg_loss=lambda loss_mat, **kwargs: loss_mat.mean(),
        compute_reward=lambda data, fn: (torch.ones_like(data.batch["responses"], dtype=torch.float32), {}),
        compute_response_mask=lambda data: data.batch["attention_mask"][:, -data.batch["responses"].shape[-1]:],
        compute_advantage=compute_advantage, reduce_metrics=lambda x: x,
        compute_data_metrics=lambda **kwargs: {}, compute_timing_metrics=lambda **kwargs: {},
        compute_throughout_metrics=lambda **kwargs: {}, should_save_ckpt_esi=lambda **kwargs: False,
        AbstractCurriculumSampler=type("Curriculum", (), {}),
    )
    source = ast.parse((ROOT / "verl/verl/trainer/ppo/ray_trainer.py").read_text())
    trainer_node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    names = {"fit", "_fit_impl", "_finish_meta_phase", "_get_gen_batch"}
    methods = [node for node in trainer_node.body if isinstance(node, ast.FunctionDef) and node.name in names]
    cls = ast.ClassDef(name="CPUTrainer", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(ROOT / "verl/verl/trainer/ppo/ray_trainer.py"), "exec"), namespace)
    trainer = namespace["CPUTrainer"]()
    trainer.config = cfg
    trainer.meta_teaching = hooks.PhaseState(cfg.meta_teaching)
    trainer.total_training_steps = steps
    trainer.global_steps = 0
    trainer.actor_rollout_wg = Actor()
    trainer.ref_policy_wg = Teacher(phase == "warmup")
    trainer.use_reference_policy = phase == "warmup"
    trainer.use_base_models = trainer.use_critic = trainer.use_rm = trainer.async_rollout_mode = False
    trainer.val_reward_fn = trainer.reward_fn = object()
    trainer.tokenizer = object()
    trainer.train_dataloader = Loader()
    trainer.train_dataset = object()
    trainer.resource_pool_manager = types.SimpleNamespace(get_n_gpus=lambda: 1)
    trainer._start_profiling = trainer._stop_profiling = lambda *args: None
    trainer.validation_calls = 0
    trainer.saved = []
    def load():
        trainer.global_steps = 17
    def save():
        path = tmp_path / "checkpoints" / f"global_step_{trainer.global_steps}"
        (path / "actor").mkdir(parents=True, exist_ok=True)
        trainer.meta_teaching.checkpoint_path = str(path)
        trainer.saved.append(trainer.global_steps)
    def validate():
        trainer.validation_calls += 1
        metrics = {"meta_teaching/val_score": 0.75}
        trainer.meta_teaching.validated(trainer.global_steps, metrics)
        return metrics
    trainer._load_checkpoint = load
    trainer._save_checkpoint = save
    trainer._validate = validate
    return trainer


@pytest.mark.parametrize("phase", ["warmup", "rl"])
def test_restored_nonzero_step_gets_two_new_updates_and_final_checkpoint(tmp_path, monkeypatch, phase):
    trainer = make_trainer(tmp_path, monkeypatch, phase)
    trainer.fit()
    summary = json.loads((tmp_path / "phase_summary.json").read_text())
    assert trainer.actor_rollout_wg.updates == 2
    assert summary["start_step"] == 17 and summary["end_step"] == 19
    assert summary["steps_completed"] == 2 and summary["status"] == "completed"
    assert summary["checkpoint_path"].endswith("global_step_19")
    assert trainer.saved == [19] and trainer.validation_calls == 1
    assert summary["val_score"] == 0.75
    assert trainer.ref_policy_wg.calls == (2 if phase == "warmup" else 0)
    assert torch.all(trainer.actor_rollout_wg.last_advantages == (0.5 if phase == "warmup" else 0))


def test_budget_boundary_stops_without_generation_and_still_saves_and_validates(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, monkeypatch, deadline=1)
    trainer.fit()
    summary = json.loads((tmp_path / "phase_summary.json").read_text())
    assert trainer.actor_rollout_wg.generations == 0
    assert summary["steps_completed"] == 0 and summary["status"] == "budget_exhausted"
    assert trainer.saved == [17] and trainer.validation_calls == 1
    assert summary["deadline_overrun_seconds"] > 0


def test_zero_step_phase_creates_anchor_and_scores_it(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, monkeypatch, steps=0)
    trainer.fit()
    summary = json.loads((tmp_path / "phase_summary.json").read_text())
    assert trainer.actor_rollout_wg.generations == 0
    assert summary["steps_completed"] == 0 and summary["status"] == "completed"
    assert trainer.saved == [17] and summary["val_score"] == 0.75


def test_empty_warmup_gate_skips_teacher_and_optimizer_momentum_update(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, monkeypatch, phase="warmup", query_fraction=0)
    trainer.fit()
    assert trainer.ref_policy_wg.calls == trainer.actor_rollout_wg.updates == 0
    summary = json.loads((tmp_path / "phase_summary.json").read_text())
    assert summary["teacher_queries"] == 0


def test_evaluation_failure_never_writes_a_completed_label(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, monkeypatch)
    trainer._validate = lambda: (_ for _ in ()).throw(RuntimeError("validation failed"))
    with pytest.raises(RuntimeError, match="validation failed"):
        trainer.fit()
    summary = json.loads((tmp_path / "phase_summary.json").read_text())
    assert summary["status"] == "failed" and summary["val_score"] is None
    assert summary["checkpoint_path"].endswith("global_step_19")
