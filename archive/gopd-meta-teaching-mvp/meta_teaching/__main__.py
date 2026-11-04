"""Portable planning, preflight and real/simulated experiment entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import REPOSITORY_ROOT, atomic_json, read_json
from .config import load_config
from .runner import ExperimentRunner, make_plan


def parser():
    root = argparse.ArgumentParser(description="G-OPD downstream-RL meta-teaching MVP")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "run", "simulate"):
        command = commands.add_parser(name)
        default = "simulation.json" if name == "simulate" else "smoke_8gpu.json"
        command.add_argument("--config", default=str(REPOSITORY_ROOT / "mvp_configs" / default))
        command.add_argument("--output-dir")
        if name in {"run", "simulate"}:
            command.add_argument("--resume", action="store_true")
        if name == "run":
            command.add_argument("--retry-failed", action="store_true",
                                 help="Repeat a failed phase from its original source; old attempts remain charged")
        if name == "preflight":
            command.add_argument("--gpu", action="store_true", help="Require local GPU/model/training dependencies")
        if name == "plan":
            command.add_argument("--save", help="Write the resolved plan JSON to this path")
    status = commands.add_parser("status")
    status.add_argument("--output-dir", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "status":
            directory = Path(args.output_dir).expanduser().resolve()
            state = read_json(directory / "experiment_state.json")
            print(json.dumps({
                "status": state["status"], "simulated": state["simulated"],
                "rounds_completed": len(state["rounds"]),
                "pending_round": None if state["pending_round"] is None else state["pending_round"]["index"],
                "current_checkpoint": state["current_checkpoint"],
                "allocated_gpu_hours": sum(e["allocated_gpu_seconds"] for e in state["cost_events"].values()) / 3600,
                "error": state.get("error"),
            }, ensure_ascii=False, indent=2))
            return 0
        config = load_config(args.config, args.output_dir)
        if args.command == "plan":
            plan = make_plan(config)
            destination = Path(args.save) if args.save else Path(config["experiment"]["output_dir"]) / "plan.json"
            atomic_json(destination, plan)
            print(f"Plan only; no models loaded or jobs launched. {destination}")
            print(f"{len(plan['phases'])} example phases; protocol={config['experiment']['protocol']}")
            return 0
        if args.command in {"preflight", "run"}:
            from .preflight import run_preflight
            report = run_preflight(config, REPOSITORY_ROOT, require_gpu=args.command == "run" or args.gpu)
            destination = Path(config["experiment"]["output_dir"]) / "preflight.json"
            atomic_json(destination, report)
            for check in report["checks"]:
                print(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
            if not report["ok"]:
                print(f"Preflight failed; no training launched. See {destination}", file=sys.stderr)
                return 2
            if args.command == "preflight":
                return 0
        if args.command == "simulate":
            from .simulation import SimulatedBackend
            print("SIMULATION ONLY: no model training, no real accuracy/cost measurements.")
            backend = SimulatedBackend(config)
        else:
            backend = None
            if args.retry_failed and not args.resume:
                raise ValueError("--retry-failed requires --resume")
        result = ExperimentRunner(
            config, backend=backend, resume=args.resume,
            retry_failed=getattr(args, "retry_failed", False),
        ).run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 3
    except KeyboardInterrupt:
        print("Interrupted. Logs and incurred costs are preserved; use status before resuming.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
