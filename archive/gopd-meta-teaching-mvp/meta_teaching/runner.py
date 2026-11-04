"""Paired OPD -> teacher-free RL trials and a persistent teaching policy.

The final-test split is used only after all policy updates and promotions.
The default policy promotion is committed before observing this round's trials.
"""

from __future__ import annotations

import csv
import contextlib
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .backend import PhaseFailure, PhaseSpec, VerlBackend, build_phase_command, check_checkpoint, process_is_live, check_export, export_teacher
from .common import REPOSITORY_ROOT, UPSTREAM_COMMIT, atomic_json, fingerprint, read_json, file_digest
from .policy import TeachingPolicy, pair_reward


class BudgetExhausted(RuntimeError):
    pass


def data_paths(config: dict, require_manifest=True) -> dict[str, str]:
    manifest_path = Path(config["data"]["manifest"])
    if not manifest_path.exists():
        if require_manifest:
            raise FileNotFoundError(f"Prepare the data first: {manifest_path}")
        return {name: str(manifest_path.parent / f"{name}.parquet") for name in ("train", "meta_val", "test")}
    manifest = read_json(manifest_path)
    result = {}
    for name in ("train", "meta_val", "test"):
        path = Path(manifest["splits"][name]["path"])
        result[name] = str(path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve())
    if len(set(result.values())) != 3:
        raise ValueError("Train, meta-val and final-test paths must be distinct")
    return result


def initial_state(config: dict, simulated: bool) -> dict:
    actions = ["direct_rl"] + [action["id"] for action in config["actions"]]
    policy = TeachingPolicy(actions, **config["policy"], seed=config["experiment"]["seed"])
    return {
        "schema_version": 1, "upstream_commit": UPSTREAM_COMMIT,
        "config_fingerprint": fingerprint(config), "simulated": simulated,
        "status": "created", "policy": policy.state_dict(), "rounds": [],
        "pending_round": None, "cost_events": {}, "phase_results": {},
        "initial_score": None, "current_score": None, "current_checkpoint": None,
        "current_teacher": config["models"]["teacher"], "final_test": None,
        "comparison": (
            "Equal student-update counts only; NOT equal total compute. Teacher and outer-search costs are measured."
            if config["experiment"]["protocol"] == "fixed_steps"
            else "Equal per-branch wall-time allocations; reject excessive overruns. Global cost includes all searched branches."
        ),
    }


def dataset_snapshot(config: dict, paths: dict[str, str]) -> dict:
    manifest = Path(config["data"]["manifest"])
    return {
        "manifest_sha256": file_digest(manifest) if manifest.is_file() else None,
        "splits": {name: {"path": path, "sha256": file_digest(path) if Path(path).is_file() else None}
                   for name, path in paths.items()},
    }


class ExperimentRunner:
    def __init__(self, config: dict, backend=None, resume=False, retry_failed=False):
        self.config = config
        self.output = Path(config["experiment"]["output_dir"])
        self.state_path = self.output / "experiment_state.json"
        self.backend = backend if backend is not None else VerlBackend(config, retry_failed=retry_failed)
        self.paths = data_paths(config, require_manifest=not self.backend.simulated)
        current_data = dataset_snapshot(config, self.paths)
        self.output.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            if not resume:
                raise ValueError("Run directory already exists. Use --resume or choose a new --output-dir.")
            self.state = read_json(self.state_path)
            if self.state["config_fingerprint"] != fingerprint(config):
                raise ValueError("Configuration differs from saved experiment; use a new run directory")
            if self.state["simulated"] != self.backend.simulated:
                raise ValueError("Simulation and real experiments cannot share a run directory")
            if self.state.get("dataset_snapshot") != current_data:
                raise ValueError("Dataset/manifest contents changed since this experiment started; use a new run directory")
        else:
            if resume:
                raise ValueError("--resume requested but experiment_state.json does not exist")
            self.state = initial_state(config, self.backend.simulated)
            self.state["dataset_snapshot"] = current_data
            atomic_json(self.output / "resolved_config.json", config)
            self._save()
        self.policy = TeachingPolicy.from_state_dict(self.state["policy"])
        self.actions = {action["id"]: action for action in config["actions"]}
        self.control_started = time.time()

    def _verify_data_snapshot(self):
        current = dataset_snapshot(self.config, self.paths)
        if self.state["dataset_snapshot"] != current:
            raise ValueError("Dataset/manifest changed during this experiment; cached feedback must not be mixed")

    def _save(self):
        self.state["updated_unix"] = time.time()
        atomic_json(self.state_path, self.state)

    def _gpu_seconds(self):
        return sum(event["allocated_gpu_seconds"] for event in self.state["cost_events"].values())

    def _remaining_gpu_seconds(self):
        return self.config["experiment"]["max_total_gpu_hours"] * 3600 - self._gpu_seconds()

    def _account(self, record: dict):
        if record.get("started_unix") is None or "allocated_gpu_seconds" not in record:
            return
        key = f"{record['phase_id']}:{record['started_unix']}"
        # Dedup is necessary when resuming a completed phase after a controller crash.
        self.state["cost_events"][key] = {
            "phase_id": record["phase_id"], "status": record["status"],
            "wall_seconds": record["wall_seconds"],
            "allocated_gpu_seconds": record["allocated_gpu_seconds"],
            "started_unix": record["started_unix"],
            "finished_unix": record.get("finished_unix"),
            "simulated": bool(record.get("simulated")),
            "cost_estimated_upper_bound": bool(record.get("cost_estimated_upper_bound")),
            "cost_scope": record.get("cost_scope", "reserved GPU wall time"),
        }

    def _overhead(self, phase_id: str, seconds: float):
        if seconds <= 0:
            return
        started = time.time() - seconds
        self._account({
            "phase_id": phase_id, "status": "completed", "started_unix": started,
            "finished_unix": time.time(), "wall_seconds": seconds,
            "allocated_gpu_seconds": seconds * self.config["training"]["n_gpus"],
            "simulated": self.backend.simulated,
        })

    def _flush_control_cost(self):
        now = time.time()
        self._overhead("controller_bookkeeping", max(0, now - self.control_started))
        self.control_started = time.time()

    def _reconcile_disk_records(self):
        """Charge every attempt, including attempts interrupted before controller save."""
        records = list((self.output / "phases").rglob("launcher_record.json"))
        records.extend((self.output / "teachers").rglob("launcher_record.json"))
        for path in sorted(records):
            record = read_json(path)
            if record.get("status") == "running" and "wall_seconds" not in record:
                if process_is_live(record):
                    raise RuntimeError(f"Previous phase PID {record['pid']} is still running: {path}")
                # A killed controller cannot know exactly when its orphaned child died.
                # Charge a conservative upper bound; never silently erase this attempt.
                finished = time.time()
                elapsed = max(0.0, finished - record["started_unix"])
                record.update(
                    status="interrupted", finished_unix=finished, wall_seconds=elapsed,
                    allocated_gpu_seconds=elapsed * record["n_gpus"],
                    cost_estimated_upper_bound=True,
                    error="Parent/child disappeared without a final record; elapsed time conservatively bounded by recovery time",
                )
                atomic_json(path, record)
            self._account(record)
        self._save()

    def _phase(self, spec: PhaseSpec) -> dict:
        self._verify_data_snapshot()
        existing = self.state["phase_results"].get(spec.phase_id)
        if existing and existing["status"] == "completed":
            if existing.get("spec_fingerprint") != spec.identity():
                raise ValueError(f"Cannot reuse a changed phase: {spec.phase_id}")
            if not self.backend.simulated and spec.phase != "eval":
                check_checkpoint(existing.get("summary", {}).get("checkpoint_path"), self.config["training"]["n_gpus"])
            return existing
        self._flush_control_cost()
        # An interrupted attempt must be charged BEFORE deciding its retry budget.
        launcher_path = Path(spec.output_dir) / "launcher_record.json"
        if launcher_path.exists():
            self._account(read_json(launcher_path))
        n_gpus = self.config["training"]["n_gpus"]
        remaining = self._remaining_gpu_seconds() / n_gpus
        reserve = self.config["experiment"]["validation_reserve_seconds"]
        if remaining <= reserve:
            raise BudgetExhausted("Global GPU-hour budget has no room for a phase and its final evaluation")
        global_deadline = time.time() + remaining
        if spec.deadline_unix is not None and spec.deadline_unix > global_deadline:
            raise BudgetExhausted("Global budget cannot fund this branch's original wall-time allocation")
        spec.deadline_unix = spec.deadline_unix or global_deadline
        try:
            record = self.backend.execute(spec)
        except PhaseFailure as failure:
            self._account(failure.record)
            self.state["phase_results"][spec.phase_id] = failure.record
            self._save()
            raise
        except BaseException:
            if launcher_path.exists():
                self._account(read_json(launcher_path))
                self._save()
            raise
        finally:
            self.control_started = time.time()
        self._account(record)
        try:
            self._verify_data_snapshot()
        except ValueError as error:
            record.update(status="failed", error=str(error))
            self._account(record)
            if launcher_path.exists():
                atomic_json(launcher_path, record)
            self.state["phase_results"][spec.phase_id] = record
            self._save()
            raise
        self.state["phase_results"][spec.phase_id] = record
        self._save()
        return record

    def _spec(self, phase_id, phase, arm_id, checkpoint, steps, seed, action=None,
              teacher=None, validation_split="meta_val", deadline=None):
        action = action or {}
        return PhaseSpec(
            phase_id=phase_id, phase=phase, arm_id=arm_id,
            output_dir=str(self.output / "phases" / phase_id),
            student_model=self.config["models"]["student"],
            teacher_model=teacher if phase == "warmup" else None,
            resume_checkpoint=checkpoint, train_path=self.paths["train"],
            validation_path=self.paths[validation_split], steps=steps, seed=seed,
            exposure=action.get("exposure", "plain"),
            exposure_fraction=action.get("exposure_fraction", 1.0),
            query_fraction=action.get("query_fraction", 1.0),
            gate=action.get("gate", "random"), deadline_unix=deadline,
        )

    def _events_for(self, prefix):
        return [event for event in self.state["cost_events"].values()
                if event["phase_id"].startswith(prefix + "/")]

    def _make_round_plan(self):
        exp = self.config["experiment"]
        round_index = len(self.state["rounds"])
        context = {
            "student_score": self.state["current_score"],
            "remaining_budget_fraction": max(0.0, self._remaining_gpu_seconds()) / (exp["max_total_gpu_hours"] * 3600),
            "round_fraction": round_index / max(1, exp["rounds"] - 1),
        }
        before = self.policy.state_dict()
        deployment_action = self.policy.select(context)
        if round_index == 0 and exp["bootstrap_all_actions"]:
            candidates = list(self.actions)
        else:
            candidates = [] if deployment_action == "direct_rl" else [deployment_action]
            while len(candidates) < exp["candidates_per_round"]:
                eligible = [name for name in self.actions if name not in candidates]
                candidates.append(self.policy.select(context, eligible=eligible))
        order = []
        for repeat in range(exp["repeats"]):
            names = ["direct_rl"] + candidates
            random.Random(exp["seed"] + round_index * 1009 + repeat).shuffle(names)
            order.extend({"arm_id": name, "repeat": repeat} for name in names)
        plan = {
            "index": round_index, "context": context,
            "source_checkpoint": self.state["current_checkpoint"],
            "teacher_model": self.state["current_teacher"],
            "deployment_action_before_feedback": deployment_action,
            "candidates": candidates, "order": order, "branches": {},
            "policy_before_selection": before,
            "policy_after_selection": self.policy.state_dict(),
        }
        self.state["pending_round"] = plan
        self.state["policy"] = self.policy.state_dict()
        self._save()
        return plan

    def _branch(self, plan, arm_id, repeat):
        exp, training = self.config["experiment"], self.config["training"]
        branch_id = f"round_{plan['index']:03d}/repeat_{repeat:03d}/{arm_id}"
        checkpoint = plan["source_checkpoint"]
        seed = exp["seed"] + plan["index"] * 1009 + repeat
        warmup_score = None
        warmup_steps = 0
        walltime = exp["protocol"] == "fixed_walltime"
        if walltime:
            self._flush_control_cost()
            already_spent = sum(event["wall_seconds"] for event in self._events_for(branch_id))
            needed = max(0, exp["branch_budget_seconds"] - already_spent)
            if self._remaining_gpu_seconds() / training["n_gpus"] < needed:
                raise BudgetExhausted(f"Global budget cannot fund the full matched allocation for {branch_id}")
        if arm_id != "direct_rl":
            action = self.actions[arm_id]
            warmup_id = branch_id + "/warmup"
            previous_warmup_seconds = sum(event["wall_seconds"] for event in self._events_for(branch_id)
                                          if event["phase_id"] == warmup_id)
            allocation = exp["branch_budget_seconds"] * exp["warmup_budget_fraction"]
            warmup_deadline = time.time() + allocation - previous_warmup_seconds if walltime else None
            warmup = self._phase(self._spec(
                warmup_id, "warmup", arm_id, checkpoint, training["warmup_steps"], seed,
                action=action, teacher=plan["teacher_model"], deadline=warmup_deadline,
            ))["summary"]
            checkpoint = warmup["checkpoint_path"]
            warmup_score = warmup["val_score"]
            warmup_steps = warmup["steps_completed"]
        previous_seconds = sum(event["wall_seconds"] for event in self._events_for(branch_id))
        rl_deadline = time.time() + exp["branch_budget_seconds"] - previous_seconds if walltime else None
        if walltime and rl_deadline - time.time() <= exp["validation_reserve_seconds"]:
            raise BudgetExhausted(f"No independent-RL time left in branch {branch_id}")
        steps = training["walltime_max_steps"] if walltime else (
            training["rl_steps"] + (training["warmup_steps"] if arm_id == "direct_rl" else 0)
        )
        rl = self._phase(self._spec(
            branch_id + "/rl", "rl", arm_id, checkpoint, steps, seed, deadline=rl_deadline,
        ))["summary"]
        events = self._events_for(branch_id)
        wall_seconds = sum(event["wall_seconds"] for event in events)
        gpu_seconds = sum(event["allocated_gpu_seconds"] for event in events)
        reasons = []
        if rl["steps_completed"] < exp["min_rl_steps"]:
            reasons.append("insufficient_independent_rl_steps")
        if not walltime and warmup_steps + rl["steps_completed"] != training["warmup_steps"] + training["rl_steps"]:
            reasons.append("incomplete_fixed_step_allocation")
        if walltime and wall_seconds > exp["branch_budget_seconds"] * (1 + exp["budget_tolerance_fraction"]):
            reasons.append("branch_budget_overrun")
        return {
            "branch_id": branch_id, "arm_id": arm_id, "repeat": repeat, "seed": seed,
            "source_checkpoint": plan["source_checkpoint"], "teacher_model": plan["teacher_model"],
            "checkpoint_path": rl["checkpoint_path"], "warmup_val_score": warmup_score,
            "post_rl_val_score": rl["val_score"], "warmup_steps": warmup_steps,
            "rl_steps": rl["steps_completed"], "wall_seconds": wall_seconds,
            "allocated_gpu_seconds": gpu_seconds, "valid_for_policy_update": not reasons,
            "exclusion_reasons": reasons,
            "teacher_queries": sum(
                result["summary"].get("teacher_queries", 0)
                for name, result in self.state["phase_results"].items()
                if name.startswith(branch_id + "/") and result.get("summary")
            ),
        }

    def _complete_round(self, plan):
        exp = self.config["experiment"]
        valid_first = {name: plan["branches"][f"0:{name}"]
                       for name in ["direct_rl"] + plan["candidates"]
                       if plan["branches"][f"0:{name}"]["valid_for_policy_update"]}
        if not valid_first:
            raise BudgetExhausted("No valid completed branch available for promotion")
        updated_policy = TeachingPolicy.from_state_dict(self.policy.state_dict())
        feedback = []
        for repeat in range(exp["repeats"]):
            baseline = plan["branches"][f"{repeat}:direct_rl"]
            if not baseline["valid_for_policy_update"]:
                continue
            updated_policy.update(plan["context"], "direct_rl", 0.0)
            for arm_id in plan["candidates"]:
                candidate = plan["branches"][f"{repeat}:{arm_id}"]
                if not candidate["valid_for_policy_update"]:
                    continue
                gain = pair_reward(
                    candidate["post_rl_val_score"], baseline["post_rl_val_score"],
                    candidate["allocated_gpu_seconds"], baseline["allocated_gpu_seconds"],
                    cost_penalty_per_gpu_hour=exp["cost_penalty_per_gpu_hour"],
                )
                updated_policy.update(plan["context"], arm_id, gain)
                feedback.append({"arm_id": arm_id, "repeat": repeat, "post_rl_incremental_value": gain})
        # No picking the luckiest seed. The first predeclared seed is always promoted.
        if exp["promotion"] == "validation_best":
            promoted_id = max(valid_first, key=lambda name: valid_first[name]["post_rl_val_score"])
            selection_scope = "explicit validation-search baseline, not prospective learned-policy deployment"
        else:
            promoted_id = plan["deployment_action_before_feedback"]
            if promoted_id not in valid_first:
                if "direct_rl" not in valid_first:
                    raise BudgetExhausted("Preselected branch and its direct-RL fallback are invalid")
                promoted_id = "direct_rl"
            selection_scope = "action committed before this round's feedback; seed zero fixed in advance"
        promoted = valid_first[promoted_id]
        self.policy = updated_policy
        plan.update(
            feedback=feedback, promoted_action=promoted_id, promotion_scope=selection_scope,
            policy_after_feedback=self.policy.state_dict(), policy_diagnostics=self.policy.diagnostics(),
        )
        self.state["current_checkpoint"] = promoted["checkpoint_path"]
        self.state["current_score"] = promoted["post_rl_val_score"]
        self.state["policy"] = self.policy.state_dict()
        self.state["rounds"].append(plan)
        self.state["pending_round"] = None
        self._flush_control_cost()
        self._save()
        self.policy.save(self.output / "teaching_policy.json")

    def _prepare_recursive_teacher(self):
        if self.config["experiment"]["teacher_mode"] != "previous_student" or not self.state["rounds"]:
            return
        index = len(self.state["rounds"])
        checkpoint = Path(self.state["current_checkpoint"])
        target = self.output / "teachers" / f"round_{index:03d}"
        marker = target / "export_manifest.json"
        if marker.exists():
            if read_json(marker)["source_checkpoint"] != str(checkpoint):
                raise ValueError("Recursive teacher export source changed")
            if not self.backend.simulated:
                check_export(target)
        elif self.backend.simulated:
            target.mkdir(parents=True, exist_ok=True)
            atomic_json(marker, {"source_checkpoint": str(checkpoint), "simulated": True})
        else:
            if self._remaining_gpu_seconds() <= 0:
                raise BudgetExhausted("No budget for teacher checkpoint export")
            self._flush_control_cost()
            try:
                record = export_teacher(
                    self.config, checkpoint, target, f"teacher_export_{index:03d}",
                    timeout_seconds=self._remaining_gpu_seconds() / self.config["training"]["n_gpus"],
                    retry_failed=getattr(self.backend, "retry_failed", False),
                )
                self._account(record)
            finally:
                record_path = target / "launcher_record.json"
                if record_path.exists():
                    self._account(read_json(record_path))
                self._save()
                self.control_started = time.time()
            atomic_json(marker, {"source_checkpoint": str(checkpoint), "simulated": False})
        self.state["current_teacher"] = str(target)
        self._save()

    @contextlib.contextmanager
    def _lock(self):
        if os.name != "posix":
            raise RuntimeError("This runner requires a POSIX host for process and experiment isolation")
        import fcntl
        with (self.output / ".runner.lock").open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another controller is using this run directory") from error
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def run(self):
        with self._lock():
            self._reconcile_disk_records()
            try:
                self._run_locked()
            finally:
                self._flush_control_cost()
                if self.state["status"] == "completed" and self._remaining_gpu_seconds() < 0:
                    self.state.update(status="budget_exhausted",
                                      error="Completed computation exceeded the global reservation cap; inspect the cost ledger")
                self._save()
                self._report()
            return self._report()

    def _run_locked(self):
        self._verify_data_snapshot()
        if self.state["status"] == "completed":
            if not self.backend.simulated:
                check_checkpoint(self.state["current_checkpoint"], self.config["training"]["n_gpus"])
            return self._report()
        self.state["status"] = "running"
        self.state.pop("error", None)
        self._save()
        exp = self.config["experiment"]
        try:
            if self.state["initial_score"] is None:
                initial = self._phase(self._spec(
                    "initial_eval", "eval", "initial", None, 0, exp["seed"]
                ))["summary"]
                self.state["initial_score"] = self.state["current_score"] = initial["val_score"]
                self._save()
            while len(self.state["rounds"]) < exp["rounds"]:
                self._prepare_recursive_teacher()
                plan = self.state["pending_round"] or self._make_round_plan()
                for item in plan["order"]:
                    key = f"{item['repeat']}:{item['arm_id']}"
                    if key not in plan["branches"]:
                        plan["branches"][key] = self._branch(plan, item["arm_id"], item["repeat"])
                        self._save()
                self._complete_round(plan)
            if exp["evaluate_final_test"] and self.state["final_test"] is None:
                final = self._phase(self._spec(
                    "final_test", "eval", "final_selected", self.state["current_checkpoint"],
                    0, exp["seed"] + 99991, validation_split="test",
                ))["summary"]
                self.state["final_test"] = {"score": final["val_score"], "checkpoint_path": self.state["current_checkpoint"]}
            self.state["status"] = "completed"
        except BudgetExhausted as error:
            self.state.update(status="budget_exhausted", error=str(error))
        except BaseException as error:
            self.state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                              error=f"{type(error).__name__}: {error}")
            self._save()
            self._report()
            raise
        self._save()
        return self._report()

    def _report(self):
        report = {
            "schema_version": 1, "status": self.state["status"],
            "simulated": self.state["simulated"],
            "warning": "SIMULATED CONTROL-PLANE TEST; NO MODEL TRAINING OR RESEARCH RESULT" if self.state["simulated"] else None,
            "comparison": self.state["comparison"], "protocol": self.config["experiment"]["protocol"],
            "rounds_completed": len(self.state["rounds"]), "initial_meta_val_score": self.state["initial_score"],
            "selected_meta_val_score": self.state["current_score"],
            "selected_checkpoint": self.state["current_checkpoint"], "final_test": self.state["final_test"],
            "total_allocated_gpu_hours": self._gpu_seconds() / 3600,
            "global_budget_gpu_hours": self.config["experiment"]["max_total_gpu_hours"],
            "global_budget_overrun_gpu_hours": max(0.0, -self._remaining_gpu_seconds()) / 3600,
            "all_branch_costs_included": True, "cost_events": len(self.state["cost_events"]),
            "estimated_upper_bound_cost_events": sum(
                bool(event.get("cost_estimated_upper_bound")) for event in self.state["cost_events"].values()
            ),
            "policy_diagnostics": self.policy.diagnostics(),
            "error": self.state.get("error"),
        }
        atomic_json(self.output / "results.json", report)
        rows = [branch for item in self.state["rounds"] for branch in item["branches"].values()]
        if self.state["pending_round"]:
            rows.extend(self.state["pending_round"]["branches"].values())
        columns = ["branch_id", "arm_id", "repeat", "seed", "warmup_val_score", "post_rl_val_score",
                   "warmup_steps", "rl_steps", "wall_seconds", "allocated_gpu_seconds",
                   "teacher_queries", "valid_for_policy_update", "exclusion_reasons"]
        with (self.output / "branches.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return report


def make_plan(config: dict) -> dict:
    paths = data_paths(config, require_manifest=False)
    output = Path(config["experiment"]["output_dir"])
    phases = []
    direct_steps = config["training"]["warmup_steps"] + config["training"]["rl_steps"]
    for arm in [{"id": "direct_rl"}] + config["actions"]:
        checkpoint = None
        if arm["id"] != "direct_rl":
            spec = PhaseSpec(
                phase_id=f"example/{arm['id']}/warmup", phase="warmup", arm_id=arm["id"],
                output_dir=str(output / "phases" / "example" / arm["id"] / "warmup"),
                student_model=config["models"]["student"], teacher_model=config["models"]["teacher"],
                resume_checkpoint=None, train_path=paths["train"], validation_path=paths["meta_val"],
                steps=config["training"]["warmup_steps"], seed=config["experiment"]["seed"],
                exposure=arm["exposure"], exposure_fraction=arm["exposure_fraction"],
                query_fraction=arm["query_fraction"], gate=arm["gate"],
            )
            phases.append({"spec": asdict(spec), "argv": build_phase_command(config, spec)})
            checkpoint = "<actual checkpoint_path from warmup phase_summary.json>"
        spec = PhaseSpec(
            phase_id=f"example/{arm['id']}/rl", phase="rl", arm_id=arm["id"],
            output_dir=str(output / "phases" / "example" / arm["id"] / "rl"),
            student_model=config["models"]["student"], teacher_model=None,
            resume_checkpoint=checkpoint, train_path=paths["train"], validation_path=paths["meta_val"],
            steps=(config["training"]["walltime_max_steps"]
                   if config["experiment"]["protocol"] == "fixed_walltime"
                   else direct_steps if arm["id"] == "direct_rl" else config["training"]["rl_steps"]),
            seed=config["experiment"]["seed"],
        )
        phases.append({"spec": asdict(spec), "argv": build_phase_command(config, spec)})
    return {
        "schema_version": 1, "upstream_commit": UPSTREAM_COMMIT,
        "note": "Illustrative first-round commands; real checkpoints/deadlines are resolved by run, never paste placeholders.",
        "config": config, "phases": phases,
    }
