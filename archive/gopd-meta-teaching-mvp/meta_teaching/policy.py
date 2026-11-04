"""Small, dependency-free teaching-action policy for downstream-RL experiments.

Each action has its own ridge-linear model of *observed downstream RL gain*.
This is an action-level future-return estimator, not a PPO value network and
not an update to the teacher language model's weights. The caller must obtain
rewards from completed, comparable downstream-RL branches. This module never
substitutes a warmup score, a prediction, or a synthetic reward for that label.

Context features use the fixed transform x / (1 + abs(x)), plus an intercept.
Missing declared features mean zero; undeclared keys and non-finite numbers
are rejected. Fixed transforms keep stored regression statistics compatible
across rounds. Callers should provide semantically stable features, e.g.
fractions rather than changing the units between rounds.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence


DEFAULT_FEATURE_NAMES = (
    "student_score", "remaining_budget_fraction", "round_fraction"
)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _names(values: Sequence[str], label: str, *, allow_empty: bool) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a list or tuple of unique names")
    names = list(values)
    if not allow_empty and not names:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicate names")
    return names


def _solve_spd(matrix: list[list[float]], target: list[float]) -> list[float]:
    """Solve the small ridge system using a Cholesky factorization."""
    size = len(target)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for col in range(row + 1):
            value = matrix[row][col] - math.fsum(
                lower[row][k] * lower[col][k] for k in range(col)
            )
            if row == col:
                if value <= 0.0 or not math.isfinite(value):
                    raise ValueError("Invalid or numerically singular ridge statistics")
                lower[row][col] = math.sqrt(value)
            else:
                lower[row][col] = value / lower[col][col]
    intermediate = [0.0] * size
    for row in range(size):
        intermediate[row] = (
            target[row]
            - math.fsum(lower[row][col] * intermediate[col] for col in range(row))
        ) / lower[row][row]
    solution = [0.0] * size
    for row in reversed(range(size)):
        solution[row] = (
            intermediate[row]
            - math.fsum(lower[col][row] * solution[col] for col in range(row + 1, size))
        ) / lower[row][row]
    if any(not math.isfinite(value) for value in solution):
        raise ValueError("Ridge statistics produced non-finite coefficients")
    return solution


def pair_reward(
    candidate_score: float,
    baseline_score: float,
    candidate_gpu_seconds: float,
    baseline_gpu_seconds: float,
    cost_penalty_per_gpu_hour: float = 0.0,
) -> float:
    """Downstream score difference minus incremental GPU-hour cost penalty.

    Scores must be terminal *independent RL* evaluation scores measured with
    the same protocol. GPU seconds are GPU count times elapsed seconds for
    the accounted branch, including teacher/warmup/RL work. Shared outer-loop
    and evaluation overhead still belongs in the experiment budget ledger.
    This function cannot establish comparability or provenance of its inputs.
    The penalty has units of score per GPU hour; zero means score-only gain.
    """
    candidate = _finite(candidate_score, "candidate_score")
    baseline = _finite(baseline_score, "baseline_score")
    candidate_cost = _finite(candidate_gpu_seconds, "candidate_gpu_seconds")
    baseline_cost = _finite(baseline_gpu_seconds, "baseline_gpu_seconds")
    penalty = _finite(cost_penalty_per_gpu_hour, "cost_penalty_per_gpu_hour")
    if candidate_cost < 0 or baseline_cost < 0 or penalty < 0:
        raise ValueError("GPU seconds and cost penalty must be nonnegative")
    return _finite(
        candidate - baseline - penalty * ((candidate_cost - baseline_cost) / 3600.0),
        "paired reward",
    )


class TeachingPolicy:
    """Per-action ridge regression with epsilon-greedy action selection.

    Unobserved actions predict zero; ties are broken uniformly at random.
    Exploration samples uniformly from eligible actions. Exploration should
    normally be positive while collecting labels; there is no forced first
    visit to every action, and this estimator does not claim causal validity
    for rewards obtained without controlled downstream branch comparisons.
    """

    def __init__(
        self,
        action_ids: list[str],
        feature_names: list[str] | None = None,
        ridge: float = 1.0,
        exploration: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.action_ids = _names(action_ids, "action_ids", allow_empty=False)
        self.feature_names = _names(
            list(DEFAULT_FEATURE_NAMES) if feature_names is None else feature_names,
            "feature_names", allow_empty=True,
        )
        self.ridge = _finite(ridge, "ridge")
        self.exploration = _finite(exploration, "exploration")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        if not 0 <= self.exploration <= 1:
            raise ValueError("exploration must be between zero and one")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self._rng = random.Random(seed)
        size = len(self.feature_names) + 1
        self._arms = {
            action: {
                "xtx": [
                    [self.ridge if row == col else 0.0 for col in range(size)]
                    for row in range(size)
                ],
                "xty": [0.0] * size,
                "count": 0,
                "reward_sum": 0.0,
            }
            for action in self.action_ids
        }

    def _features(self, context: Mapping[str, float]) -> list[float]:
        if not isinstance(context, Mapping):
            raise ValueError("context must be a mapping")
        unknown = set(context).difference(self.feature_names)
        if unknown:
            raise ValueError(f"Undeclared context features: {sorted(map(str, unknown))}")
        values = [_finite(context.get(name, 0.0), name) for name in self.feature_names]
        return [1.0] + [value / (1.0 + abs(value)) for value in values]

    def _arm(self, action_id: str) -> dict[str, Any]:
        if not isinstance(action_id, str) or action_id not in self._arms:
            raise ValueError(f"Unknown teaching action: {action_id!r}")
        return self._arms[action_id]

    def _prediction(self, features: list[float], action_id: str) -> float:
        arm = self._arm(action_id)
        weights = _solve_spd(arm["xtx"], arm["xty"])
        return _finite(math.fsum(x * w for x, w in zip(features, weights)), "prediction")

    def predict(self, context: dict[str, float], action_id: str) -> float:
        return self._prediction(self._features(context), action_id)

    def select(self, context: dict[str, float], eligible: list[str] | None = None) -> str:
        features = self._features(context)
        choices = self.action_ids if eligible is None else _names(
            eligible, "eligible", allow_empty=False
        )
        for action in choices:
            self._arm(action)
        if self._rng.random() < self.exploration:
            return self._rng.choice(choices)
        predictions = {action: self._prediction(features, action) for action in choices}
        best = max(predictions.values())
        ties = [
            action for action in choices
            if math.isclose(predictions[action], best, rel_tol=1e-12, abs_tol=1e-15)
        ]
        return self._rng.choice(ties)

    def update(self, context: dict[str, float], action_id: str, reward: float) -> None:
        features = self._features(context)
        arm = self._arm(action_id)
        observed = _finite(reward, "reward")
        # Validate all new sufficient statistics before mutating existing state.
        xtx = [
            [_finite(old + x * y, "xtx") for old, y in zip(row, features)]
            for row, x in zip(arm["xtx"], features)
        ]
        xty = [
            _finite(old + x * observed, "xty") for old, x in zip(arm["xty"], features)
        ]
        reward_sum = _finite(arm["reward_sum"] + observed, "reward_sum")
        _solve_spd(xtx, xty)
        arm.update(xtx=xtx, xty=xty, reward_sum=reward_sum, count=arm["count"] + 1)

    @property
    def total_updates(self) -> int:
        return sum(arm["count"] for arm in self._arms.values())

    @property
    def arm_counts(self) -> dict[str, int]:
        return {action: arm["count"] for action, arm in self._arms.items()}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "total_updates": self.total_updates,
            "feature_transform": "x/(1+abs(x)); missing=0; intercept=1",
            "arms": {
                action: {
                    "count": arm["count"],
                    "mean_observed_reward": (
                        arm["reward_sum"] / arm["count"] if arm["count"] else None
                    ),
                    "coefficients": _solve_spd(arm["xtx"], arm["xty"]),
                }
                for action, arm in self._arms.items()
            },
        }

    def state_dict(self) -> dict[str, Any]:
        # JSON round-trip returns an independent, directly serializable snapshot.
        return json.loads(json.dumps({
            "schema_version": 1,
            "algorithm": "per_arm_ridge_epsilon_greedy",
            "feature_transform": "signed_soft_bound_v1",
            "action_ids": self.action_ids,
            "feature_names": self.feature_names,
            "ridge": self.ridge,
            "exploration": self.exploration,
            "rng_state": self._rng.getstate(),
            "arms": self._arms,
        }, allow_nan=False))

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> TeachingPolicy:
        try:
            if state["schema_version"] != 1 or state["algorithm"] != "per_arm_ridge_epsilon_greedy":
                raise ValueError("Unsupported teaching policy schema or algorithm")
            if state["feature_transform"] != "signed_soft_bound_v1":
                raise ValueError("Unsupported context feature transform")
            policy = cls(state["action_ids"], state["feature_names"], state["ridge"], state["exploration"])
            if set(state["arms"]) != set(policy.action_ids):
                raise ValueError("Saved arm names do not match action_ids")
            size = len(policy.feature_names) + 1
            for action in policy.action_ids:
                saved = state["arms"][action]
                count = saved["count"]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("Saved arm count must be a nonnegative integer")
                matrix, vector = saved["xtx"], saved["xty"]
                if len(matrix) != size or any(len(row) != size for row in matrix) or len(vector) != size:
                    raise ValueError("Saved ridge statistics have incorrect dimensions")
                matrix = [[_finite(value, "saved xtx") for value in row] for row in matrix]
                vector = [_finite(value, "saved xty") for value in vector]
                if any(matrix[i][j] != matrix[j][i] for i in range(size) for j in range(size)):
                    raise ValueError("Saved ridge matrix must be symmetric")
                _solve_spd(matrix, vector)
                reward_sum = _finite(saved["reward_sum"], "saved reward_sum")
                if not math.isclose(matrix[0][0], policy.ridge + count, rel_tol=1e-12):
                    raise ValueError("Saved intercept statistics and sample count disagree")
                if not math.isclose(vector[0], reward_sum, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("Saved intercept target and reward sum disagree")
                policy._arms[action] = {
                    "xtx": matrix, "xty": vector, "count": count, "reward_sum": reward_sum,
                }

            def as_tuple(value: Any) -> Any:
                return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value

            policy._rng.setstate(as_tuple(state["rng_state"]))
            return policy
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
            raise ValueError(f"Invalid teaching policy state: {exc}") from exc

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically replace a JSON checkpoint, preserving model and RNG state."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = handle.name
                json.dump(self.state_dict(), handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> TeachingPolicy:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_state_dict(json.load(handle))
