"""Lightweight tests of the actual CLI cleanup function, without starting Ray."""
import ast
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("enabled,already_connected,outcome,expect_shutdown", [
    (True, False, "success", True),
    (True, False, "failure", True),
    (True, False, "sigterm", True),
    (True, True, "success", False),
    (False, False, "success", False),
])
def test_main_closes_only_ray_session_it_created(enabled, already_connected, outcome, expect_shutdown):
    state = {"connected": already_connected, "shutdown": 0}
    original_handler = object()
    handlers = {15: original_handler}
    def shutdown():
        state["shutdown"] += 1
        state["connected"] = False
    def signal_set(signum, handler):
        previous = handlers[signum]
        handlers[signum] = handler
        return previous
    fake_ray = types.SimpleNamespace(is_initialized=lambda: state["connected"], shutdown=shutdown)
    fake_signal = types.SimpleNamespace(SIGTERM=15, getsignal=lambda signum: handlers[signum], signal=signal_set)
    def run(config):
        state["connected"] = True
        if outcome == "failure":
            raise RuntimeError("worker failed")
        if outcome == "sigterm":
            handlers[15](15, None)
    source = ast.parse((ROOT / "verl/verl/trainer/main_ppo.py").read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    function.decorator_list = []
    scope = {"ray": fake_ray, "signal": fake_signal, "meta_teaching_enabled": lambda config: enabled, "run_ppo": run}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "main_ppo.py", "exec"), scope)
    if outcome == "failure":
        with pytest.raises(RuntimeError, match="worker failed"):
            scope["main"]({})
    elif outcome == "sigterm":
        with pytest.raises(SystemExit) as exc:
            scope["main"]({})
        assert exc.value.code == 143
    else:
        scope["main"]({})
    assert state["shutdown"] == int(expect_shutdown)
    assert handlers[15] is original_handler
