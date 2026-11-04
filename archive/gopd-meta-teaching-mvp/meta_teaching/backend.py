"""Launch real G-OPD phases as isolated processes; never manufacture metrics."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .common import REPOSITORY_ROOT, atomic_json, fingerprint, read_json
from .config import hydra_value


@dataclass
class PhaseSpec:
    phase_id: str
    phase: str
    arm_id: str
    output_dir: str
    student_model: str
    teacher_model: str | None
    resume_checkpoint: str | None
    train_path: str
    validation_path: str
    steps: int
    seed: int
    exposure: str = "plain"
    exposure_fraction: float = 1.0
    query_fraction: float = 1.0
    gate: str = "random"
    deadline_unix: float | None = None

    def identity(self):
        values = asdict(self)
        values.pop("deadline_unix")
        return fingerprint(values)


def build_phase_command(config: dict, spec: PhaseSpec, python: str = sys.executable,
                        repository_root: Path = REPOSITORY_ROOT) -> list[str]:
    t, exp = config["training"], config["experiment"]
    warmup = spec.phase == "warmup"
    evaluation = spec.phase == "eval"
    if spec.phase not in {"warmup", "rl", "eval"}:
        raise ValueError("Unsupported phase")
    if warmup and not spec.teacher_model:
        raise ValueError("Warmup requires a teacher")
    if not warmup and spec.teacher_model is not None:
        raise ValueError("RL/eval must not receive a teacher model")
    reward_name = "compute_numeric_demo_score" if config["data"]["reward"] == "numeric_demo" else "compute_score"
    values = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.use_kl_in_reward": False,
        "algorithm.use_ref_solution_distillation": False,
        "algorithm.critique_vllm_url": None,
        "data.train_files": [spec.train_path],
        "data.val_files": [spec.validation_path],
        "data.train_batch_size": t["train_batch_size"],
        "data.val_batch_size": t["train_batch_size"],
        "data.max_prompt_length": t["max_prompt_length"],
        "data.max_response_length": t["max_response_length"],
        "data.return_raw_chat": True,
        "data.shuffle": True,
        "data.seed": spec.seed,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.apply_chat_template_kwargs.enable_thinking": False,
        "actor_rollout_ref.model.path": spec.student_model,
        "actor_rollout_ref.model.base_model_path": None,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.lora_rank": 0,
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.optim.lr": t["learning_rate"],
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.optim.weight_decay": 0.0,
        "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages": warmup,
        "actor_rollout_ref.actor.policy_loss.lambda_vals": 1.0,
        "actor_rollout_ref.actor.policy_loss.multi_teacher_distill": False,
        "actor_rollout_ref.actor.use_kl_loss": warmup,
        "actor_rollout_ref.actor.kl_loss_coef": 0.0,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.use_rollout_log_probs": True,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.ppo_mini_batch_size": t["ppo_mini_batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": t["micro_batch_size_per_gpu"],
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.actor.checkpoint.load_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.ref.model.path": spec.teacher_model if warmup else None,
        "actor_rollout_ref.ref.model.base_model_path": None,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": t["micro_batch_size_per_gpu"],
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.seed": spec.seed,
        "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": t["rollout_tp"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": t["gpu_memory_utilization"],
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.max_num_batched_tokens": max(8192, t["max_prompt_length"] + t["max_response_length"]),
        "actor_rollout_ref.rollout.max_model_len": t["max_prompt_length"] + t["max_response_length"],
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": t["micro_batch_size_per_gpu"],
        "actor_rollout_ref.rollout.n": 1 if warmup else t["rollout_n"],
        "actor_rollout_ref.rollout.temperature": t["temperature"],
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.multi_turn.enable": False,
        "actor_rollout_ref.rollout.val_kwargs.n": t["eval_n"],
        "actor_rollout_ref.rollout.val_kwargs.temperature": t["eval_temperature"],
        "actor_rollout_ref.rollout.val_kwargs.do_sample": t["eval_temperature"] > 0,
        "custom_reward_function.path": str(repository_root / "mvp_tools" / "math_reward.py"),
        "custom_reward_function.name": reward_name,
        "reward_model.enable": False,
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": t["n_gpus"],
        "trainer.total_epochs": max(1, spec.steps + 1),
        "trainer.total_training_steps": max(1, spec.steps),
        "trainer.resume_mode": "resume_path" if spec.resume_checkpoint else "disable",
        "trainer.resume_from_path": spec.resume_checkpoint,
        "trainer.del_local_ckpt_after_load": False,
        "trainer.val_only": evaluation,
        "trainer.val_before_train": evaluation,
        "trainer.test_freq": -1,
        "trainer.save_freq": t["checkpoint_every"],
        "trainer.max_actor_ckpt_to_keep": 2,
        "trainer.default_hdfs_dir": None,
        "trainer.default_local_dir": str(Path(spec.output_dir) / "checkpoints"),
        "trainer.project_name": "meta_teaching_mvp",
        "trainer.experiment_name": spec.phase_id.replace("/", "_"),
        "trainer.logger": ["console"],
        "ray_kwargs.ray_init.address": "local",
        "meta_teaching.enabled": True,
        "meta_teaching.phase": spec.phase,
        "meta_teaching.phase_id": spec.phase_id,
        "meta_teaching.spec_fingerprint": spec.identity(),
        "meta_teaching.source_checkpoint": spec.resume_checkpoint,
        "meta_teaching.arm_id": spec.arm_id,
        "meta_teaching.exposure": spec.exposure,
        "meta_teaching.exposure_fraction": spec.exposure_fraction,
        "meta_teaching.query_fraction": spec.query_fraction,
        "meta_teaching.gate": spec.gate,
        "meta_teaching.seed": spec.seed,
        "meta_teaching.phase_steps": spec.steps,
        "meta_teaching.output_dir": spec.output_dir,
        "meta_teaching.teacher_max_prompt_length": t["teacher_max_prompt_length"],
        "meta_teaching.deadline_unix": spec.deadline_unix,
        "meta_teaching.validation_reserve_seconds": exp["validation_reserve_seconds"],
    }
    return [python, "-m", "verl.trainer.main_ppo"] + [
        f"++{key}={hydra_value(value)}" for key, value in values.items()
    ]


def process_start_token(pid: int) -> str | None:
    """Linux PID-start token prevents mistaking a recycled PID for our job."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        return raw[raw.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


def process_is_live(record: dict) -> bool:
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    expected = record.get("process_start_token")
    current = process_start_token(pid)
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        if raw[raw.rfind(")") + 2:].split()[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    return expected is None or current is None or expected == current


def check_checkpoint(path: str | None, n_gpus: int) -> None:
    if not path:
        raise ValueError("Missing checkpoint path")
    root = Path(path)
    required = [root / "data.pt", root / "actor" / "fsdp_config.json",
                root / "actor" / "huggingface" / "config.json"]
    for rank in range(n_gpus):
        required.extend(root / "actor" / f"{kind}_world_size_{n_gpus}_rank_{rank}.pt"
                        for kind in ("model", "optim", "extra_state"))
        required.append(root / "actor" / f"rollout_rng_rank_{rank}.pt")
    missing = [str(item) for item in required if not item.is_file() or item.stat().st_size == 0]
    if missing:
        raise ValueError(f"Checkpoint is incomplete for {n_gpus} ranks: {missing[:4]}")


def checkpoint_step(path: str | None) -> int:
    if path is None:
        return 0
    match = re.fullmatch(r"global_step_(\d+)", Path(path).name)
    if not match:
        raise ValueError(f"Expected a global_step_N checkpoint directory: {path}")
    return int(match.group(1))


def validate_summary(summary, spec: PhaseSpec, n_gpus: int):
    """A completed subprocess is not evidence unless its artifact matches the request."""
    if not isinstance(summary, dict) or type(summary.get("schema_version")) is not int or summary["schema_version"] != 1:
        raise ValueError("Unsupported or malformed phase summary")
    for name, expected in (
        ("phase", spec.phase), ("arm_id", spec.arm_id), ("phase_id", spec.phase_id),
        ("spec_fingerprint", spec.identity()), ("seed", spec.seed),
        ("source_checkpoint", spec.resume_checkpoint),
    ):
        if summary.get(name) != expected:
            raise ValueError(f"Phase summary {name} mismatch")
    if type(summary.get("seed")) is not int:
        raise ValueError("Invalid seed type in phase summary")
    for name in ("start_step", "end_step", "steps_completed", "validation_step",
                 "teacher_queries", "teacher_selected_queries", "teacher_scored_tokens", "teacher_response_tokens"):
        number = summary.get(name)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"Invalid summary integer: {name}")
    start, end, count = (summary[name] for name in ("start_step", "end_step", "steps_completed"))
    if start != checkpoint_step(spec.resume_checkpoint) or end - start != count or count > spec.steps:
        raise ValueError("Restored start/end/update count does not match requested phase")
    if summary["validation_step"] != end:
        raise ValueError("Validation score belongs to a different checkpoint")
    status = summary.get("status")
    if status not in {"completed", "budget_exhausted"}:
        raise ValueError(f"Unusable phase status: {status}")
    if status == "completed" and count != spec.steps:
        raise ValueError("Completed phase did not perform the requested updates")
    if status == "budget_exhausted" and (spec.deadline_unix is None or spec.phase == "eval"):
        raise ValueError("Unexpected budget-exhausted status")
    teacher_enabled = summary.get("teacher_enabled")
    if not isinstance(teacher_enabled, bool) or teacher_enabled != (spec.phase == "warmup"):
        raise ValueError("Teacher allocation does not match phase")
    if spec.phase != "warmup" and any(summary[name] != 0 for name in (
        "teacher_queries", "teacher_selected_queries", "teacher_scored_tokens", "teacher_response_tokens"
    )):
        raise ValueError("Teacher-free phase used a teacher")
    if summary["teacher_selected_queries"] > summary["teacher_queries"]:
        raise ValueError("Selected teacher queries exceed actual dispatched queries")
    if summary["teacher_scored_tokens"] < summary["teacher_response_tokens"]:
        raise ValueError("Teacher token counters are inconsistent")
    score = summary.get("val_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Expected finite final task reward in [0,1]")
    metrics = summary.get("val_metrics")
    if not isinstance(metrics, dict) or metrics.get("meta_teaching/val_score") != score:
        raise ValueError("Final task reward and validation metrics disagree")
    if spec.phase == "eval":
        if count != 0 or summary.get("checkpoint_path") != spec.resume_checkpoint:
            raise ValueError("Evaluation updated the model or changed its checkpoint")
    else:
        expected = (Path(spec.output_dir) / "checkpoints" / f"global_step_{end}").resolve()
        actual = Path(summary.get("checkpoint_path") or "").resolve()
        if actual != expected:
            raise ValueError("Checkpoint path does not belong to this phase's final step")
        check_checkpoint(str(actual), n_gpus)


class PhaseFailure(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


class VerlBackend:
    simulated = False

    def __init__(self, config: dict, repository_root: Path = REPOSITORY_ROOT, retry_failed: bool = False):
        self.config = config
        self.root = repository_root
        self.retry_failed = retry_failed

    def execute(self, spec: PhaseSpec) -> dict:
        directory = Path(spec.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / "launcher_record.json"
        identity = spec.identity()
        if record_path.exists():
            record = read_json(record_path)
            if record.get("spec_fingerprint") != identity:
                raise RuntimeError(f"Phase definition changed: {directory}; use a new run directory")
            if record.get("status") == "completed":
                if spec.phase != "eval":
                    check_checkpoint(record.get("summary", {}).get("checkpoint_path"), self.config["training"]["n_gpus"])
                return record
            if record.get("status") == "running" and process_is_live(record):
                raise RuntimeError(f"Phase process {record['pid']} is still running: {directory}")
            # A failure may already have consumed GPU budget. Never silently rerun.
            if not self.retry_failed:
                raise PhaseFailure(
                    f"Interrupted/failed phase: {directory}. Inspect it, then use --resume --retry-failed "
                    "to repeat this phase from its original source checkpoint; all attempts stay charged.", record)
            archive = directory.with_name(directory.name + f".failed-{time.time_ns()}")
            directory.rename(archive)
            directory.mkdir()
            atomic_json(directory / "previous_attempt.json", {
                "archive": str(archive), "record": record,
                "warning": "This retry starts from the original phase source, not a partial checkpoint.",
            })
        command = build_phase_command(self.config, spec, repository_root=self.root)
        atomic_json(directory / "phase_request.json", {
            "schema_version": 1, "backend": "verl", "spec": asdict(spec), "argv": command,
        })
        started = time.time()
        record = {
            "schema_version": 1, "backend": "verl", "simulated": False,
            "phase_id": spec.phase_id, "spec_fingerprint": identity,
            "status": "running", "started_unix": started, "pid": None,
            "deadline_unix": spec.deadline_unix, "n_gpus": self.config["training"]["n_gpus"],
        }
        atomic_json(record_path, record)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(self.root / "verl"), str(self.root)] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
        )
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["PYTHONHASHSEED"] = str(spec.seed)
        # Each phase owns its Ray cluster. Never connect to / stop somebody else's.
        environment["RAY_ADDRESS"] = "local"
        timeout = False
        process = None
        try:
            with (directory / "train.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, cwd=self.root / "verl", env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"),
                )
                record["pid"] = process.pid
                record["process_start_token"] = process_start_token(process.pid)
                atomic_json(record_path, record)
                hard_deadline = None if spec.deadline_unix is None else (
                    spec.deadline_unix + self.config["experiment"]["deadline_grace_seconds"]
                )
                while process.poll() is None:
                    if hard_deadline is not None and time.time() >= hard_deadline:
                        timeout = True
                        self._terminate(process)
                        break
                    time.sleep(0.25)
                return_code = process.wait()
        except BaseException as error:
            if process is not None and process.poll() is None:
                self._terminate(process)
            record.update(status="interrupted", error=f"{type(error).__name__}: {error}")
            self._finish_record(record, started)
            atomic_json(record_path, record)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise PhaseFailure(str(error), record) from error
        self._finish_record(record, started)
        record["return_code"] = return_code
        # Persist incurred cost before parsing a potentially incomplete child artifact.
        record["status"] = "validating"
        atomic_json(record_path, record)
        summary_path = directory / "phase_summary.json"
        try:
            summary = read_json(summary_path) if summary_path.exists() else None
        except (ValueError, OSError) as error:
            record.update(status="failed", error=f"Unreadable phase summary: {error}")
            atomic_json(record_path, record)
            raise PhaseFailure(record["error"], record) from error
        record["summary"] = summary
        if timeout or return_code != 0 or not summary:
            record["status"] = "failed"
            record["error"] = "deadline exceeded" if timeout else "phase failed or omitted summary"
        else:
            try:
                validate_summary(summary, spec, self.config["training"]["n_gpus"])
                record["status"] = "completed"
            except (ValueError, OSError, TypeError) as error:
                record.update(status="failed", error=str(error))
        atomic_json(record_path, record)
        if record["status"] != "completed":
            raise PhaseFailure(f"{record['error']}; inspect {directory / 'train.log'}", record)
        return record

    def _finish_record(self, record, started):
        record["finished_unix"] = time.time()
        record["wall_seconds"] = max(0.0, record["finished_unix"] - started)
        record["allocated_gpu_seconds"] = record["wall_seconds"] * self.config["training"]["n_gpus"]
        record["cost_scope"] = "reserved GPUs times whole subprocess wall time, including startup/eval/checkpoint/shutdown"
        deadline = record.get("deadline_unix")
        record["deadline_overrun_seconds"] = max(0.0, record["finished_unix"] - deadline) if deadline else 0.0

    @staticmethod
    def _terminate(process):
        if process.poll() is not None:
            return
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        else:
            process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()


def check_export(path: Path):
    if not (path / "config.json").is_file() or not (path / "tokenizer.json").is_file():
        raise ValueError(f"Incomplete exported teacher metadata: {path}")
    weights = list(path.glob("*.safetensors"))
    if not weights or any(item.stat().st_size == 0 for item in weights):
        raise ValueError(f"No complete exported teacher weights: {path}")
    for index in path.glob("*.safetensors.index.json"):
        for shard in set(read_json(index).get("weight_map", {}).values()):
            if not (path / shard).is_file():
                raise ValueError(f"Exported teacher is missing shard {shard}")


def export_teacher(config: dict, source: Path, target: Path, phase_id: str,
                   timeout_seconds: float, retry_failed=False) -> dict:
    """A checkpoint merge is a separately recorded, budgeted process too."""
    identity = fingerprint({"operation": "fsdp_export", "source_checkpoint": str(source)})
    target.mkdir(parents=True, exist_ok=True)
    record_path = target / "launcher_record.json"
    if record_path.exists():
        old = read_json(record_path)
        if old.get("spec_fingerprint") != identity:
            raise ValueError("Teacher-export source changed")
        if old.get("status") == "completed":
            check_export(target)
            return old
        if old.get("status") == "running" and process_is_live(old):
            raise RuntimeError(f"Teacher export PID {old['pid']} is still running")
        if not retry_failed:
            raise PhaseFailure("Previous teacher export failed; inspect logs before --resume --retry-failed", old)
        archive = target.with_name(target.name + f".failed-{time.time_ns()}")
        target.rename(archive)
        target.mkdir()
        atomic_json(target / "previous_attempt.json", {"archive": str(archive), "record": old})
    if timeout_seconds <= 0:
        raise ValueError("No remaining time for teacher export")
    check_checkpoint(str(source), config["training"]["n_gpus"])
    command = [
        sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
        "--local_dir", str(source / "actor"), "--target_dir", str(target),
    ]
    atomic_json(target / "export_request.json", {"argv": command, "source_checkpoint": str(source)})
    started = time.time()
    record = {
        "schema_version": 1, "backend": "verl_model_merger", "simulated": False,
        "phase_id": phase_id, "spec_fingerprint": identity, "status": "running",
        "started_unix": started, "pid": None, "n_gpus": config["training"]["n_gpus"],
        "source_checkpoint": str(source), "deadline_unix": started + timeout_seconds,
    }
    atomic_json(record_path, record)
    process = None
    interrupted = None
    try:
        with (target / "export.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=REPOSITORY_ROOT / "verl", stdout=log, stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            record.update(pid=process.pid, process_start_token=process_start_token(process.pid))
            atomic_json(record_path, record)
            code = process.wait(timeout=timeout_seconds)
        record["return_code"] = code
        if code != 0:
            raise RuntimeError(f"Teacher export exited with code {code}")
        check_export(target)
        record["status"] = "completed"
    except BaseException as error:
        if process is not None and process.poll() is None:
            VerlBackend._terminate(process)
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            interrupted = error
    finally:
        record["finished_unix"] = time.time()
        record["wall_seconds"] = max(0.0, record["finished_unix"] - started)
        record["allocated_gpu_seconds"] = record["wall_seconds"] * config["training"]["n_gpus"]
        record["cost_scope"] = "CPU checkpoint merge while the declared GPU allocation is reserved"
        atomic_json(record_path, record)
    if interrupted:
        raise interrupted
    if record["status"] != "completed":
        raise PhaseFailure(record["error"], record)
    return record
