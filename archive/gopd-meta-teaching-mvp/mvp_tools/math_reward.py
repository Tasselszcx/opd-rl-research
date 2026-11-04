"""Explicit verifiers for the G-OPD custom_reward_function interface."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import importlib
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Use the upstream math-verify scorer; missing dependencies are fatal."""
    try:
        importlib.import_module("math_verify")
    except ImportError as exc:
        raise RuntimeError(
            "Production math scoring requires math-verify. Install the documented environment; "
            "numeric demo scoring is never selected automatically."
        ) from exc
    from verl.utils.reward_score.math_verify import compute_score as upstream_score
    score = float(upstream_score(model_output=solution_str, ground_truth=str(ground_truth)))
    return {"score": score, "acc": score}


def compute_numeric_demo_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Smoke-test only: accept a simple finite number inside the last boxed answer."""
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", solution_str)
    score = 0.0
    if boxes:
        try:
            prediction, target = Decimal(boxes[-1].strip()), Decimal(str(ground_truth).strip())
            score = float(prediction.is_finite() and target.is_finite() and prediction == target)
        except InvalidOperation:
            pass
    return {"score": score, "acc": score}
