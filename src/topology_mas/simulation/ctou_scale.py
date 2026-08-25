"""Local-law features and correlated Round-0 models for CTOU scaling.

This module contains no graph rollout logic.  It defines the two inputs that
must be validated before an n>10 topology simulation is scientifically useful:
the local response representation and the correlated initial-state generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CTOU_STATES = ("correct", "target", "other", "unparsed")
ROUND_ZERO_STATES = ("correct", "other", "unparsed")
COUNT_COLUMNS = tuple(f"incoming_{state}_count" for state in CTOU_STATES)
LOCAL_LAW_VARIANTS = (
    "proportions",
    "absolute_counts",
    "counts_plus_proportions",
    "proportions_log1p_volume",
    "proportions_saturating_volume_k1",
    "proportions_saturating_volume_k2",
    "proportions_saturating_volume_k4",
)


def _volume(variant: str, degree: np.ndarray) -> np.ndarray:
    if variant == "proportions_log1p_volume":
        return np.log1p(degree)
    prefix = "proportions_saturating_volume_k"
    if variant.startswith(prefix):
        scale = float(variant.removeprefix(prefix))
        return np.divide(degree, degree + scale, out=np.zeros_like(degree), where=True)
    raise ValueError(f"variant has no volume transform: {variant}")


def local_law_feature_names(variant: str, *, maximum_round: int = 3) -> tuple[str, ...]:
    if variant not in LOCAL_LAW_VARIANTS:
        raise ValueError(f"unknown local-law variant: {variant}")
    prefix = tuple(f"previous_{state}" for state in CTOU_STATES) + tuple(
        f"round_{index}" for index in range(1, maximum_round + 1)
    )
    counts = COUNT_COLUMNS
    proportions = tuple(name.replace("_count", "_proportion") for name in counts)
    if variant == "proportions":
        return prefix + proportions
    if variant == "absolute_counts":
        return prefix + counts
    if variant == "counts_plus_proportions":
        return prefix + counts + proportions
    return (
        prefix
        + proportions
        + ("evidence_volume",)
        + tuple(f"{name}_x_volume" for name in proportions)
        + tuple(f"previous_{state}_x_volume" for state in CTOU_STATES)
    )


def ctou_design_matrix(frame: Any, variant: str, *, maximum_round: int = 3) -> np.ndarray:
    """Build a frozen local-law design matrix from a pandas-like frame."""

    if variant not in LOCAL_LAW_VARIANTS:
        raise ValueError(f"unknown local-law variant: {variant}")
    previous_values = np.asarray(frame["previous_attack_state"])
    previous_index = np.full(len(frame), -1, dtype=int)
    for index, state in enumerate(CTOU_STATES):
        previous_index[previous_values == state] = index
    if np.any(previous_index < 0):
        unknown = sorted(set(previous_values[previous_index < 0]))
        raise ValueError(f"unknown previous CTOU states: {unknown}")
    rounds = np.asarray(frame["round_index"], dtype=int)
    if len(rounds) and (rounds.min() < 1 or rounds.max() > maximum_round):
        raise ValueError(f"round index outside 1..{maximum_round}")
    previous = np.eye(len(CTOU_STATES), dtype=np.float64)[previous_index]
    round_one_hot = np.eye(maximum_round, dtype=np.float64)[rounds - 1]
    counts = np.asarray(frame[list(COUNT_COLUMNS)], dtype=np.float64)
    degree = counts.sum(axis=1, keepdims=True)
    proportions = np.divide(counts, degree, out=np.zeros_like(counts), where=degree > 0)
    if variant == "proportions":
        pieces = (previous, round_one_hot, proportions)
    elif variant == "absolute_counts":
        pieces = (previous, round_one_hot, counts)
    elif variant == "counts_plus_proportions":
        pieces = (previous, round_one_hot, counts, proportions)
    else:
        volume = _volume(variant, degree)
        pieces = (
            previous,
            round_one_hot,
            proportions,
            volume,
            proportions * volume,
            previous * volume,
        )
    matrix = np.column_stack(pieces)
    expected = len(local_law_feature_names(variant, maximum_round=maximum_round))
    if matrix.shape != (len(frame), expected):
        raise RuntimeError(
            f"feature matrix has shape {matrix.shape}, expected {(len(frame), expected)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("local-law feature matrix contains a non-finite value")
    return matrix


@dataclass(frozen=True)
class HierarchicalRoundZeroModel:
    """Task-shrunk Dirichlet-multinomial initializer for C/O/U states."""

    global_mean: tuple[float, float, float]
    task_means: dict[str, tuple[float, float, float]]
    concentration: float
    smoothing_strength: float

    def mean_for_task(self, task_id: str) -> np.ndarray:
        return np.asarray(self.task_means.get(str(task_id), self.global_mean), dtype=float)

    def sample_counts(
        self,
        *,
        task_id: str,
        node_count: int,
        draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if node_count < 1 or draws < 1:
            raise ValueError("node_count and draws must be positive")
        mean = self.mean_for_task(task_id)
        theta = rng.dirichlet(self.concentration * mean, size=draws)
        return np.asarray([rng.multinomial(node_count, value) for value in theta], dtype=int)


def extract_round_zero_groups(cases: Any) -> Any:
    """Return one clean Round-0 C/O/U count vector per task and graph run."""

    import pandas as pd

    required = {"task_id", "graph_id", "n", "initial_states"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Round-0 cases are missing columns: {missing}")
    unique = cases[["task_id", "graph_id", "n", "initial_states"]].drop_duplicates(
        subset=["task_id", "graph_id"]
    )
    rows: list[dict[str, object]] = []
    for row in unique.itertuples(index=False):
        states = tuple(int(value) for value in row.initial_states)
        if len(states) != int(row.n):
            raise ValueError("Round-0 vector length differs from n")
        if 1 in states:
            raise ValueError("clean Round-0 vector unexpectedly contains target state")
        rows.append(
            {
                "task_id": str(row.task_id),
                "graph_id": str(row.graph_id),
                "n": int(row.n),
                "correct_count": states.count(0),
                "other_count": states.count(2),
                "unparsed_count": states.count(3),
            }
        )
    result = pd.DataFrame(rows)
    if result.duplicated(["task_id", "graph_id"]).any():
        raise RuntimeError("Round-0 extraction did not produce unique task-graph groups")
    return result


def fit_hierarchical_round_zero(
    groups: Any,
    *,
    smoothing_strength: float = 3.0,
    required_sizes: set[int] | None = None,
) -> HierarchicalRoundZeroModel:
    """Fit task means and shared overdispersion from grouped clean states."""

    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln

    observed_sizes = set(np.asarray(groups["n"], dtype=int))
    if required_sizes is not None and observed_sizes != required_sizes:
        raise ValueError(
            f"Round-0 fit sizes {sorted(observed_sizes)} differ from required "
            f"sizes {sorted(required_sizes)}"
        )
    count_columns = ["correct_count", "other_count", "unparsed_count"]
    counts = np.asarray(groups[count_columns], dtype=float)
    totals = np.asarray(groups["n"], dtype=float)
    if np.any(counts < 0) or not np.allclose(counts.sum(axis=1), totals):
        raise ValueError("invalid Round-0 count rows")
    global_counts = counts.sum(axis=0) + 0.5
    global_mean = global_counts / global_counts.sum()
    task_means: dict[str, tuple[float, float, float]] = {}
    for task_id, frame in groups.groupby("task_id", sort=True):
        task_counts = np.asarray(frame[count_columns], dtype=float).sum(axis=0)
        mean = (task_counts + smoothing_strength * global_mean) / (
            task_counts.sum() + smoothing_strength
        )
        task_means[str(task_id)] = tuple(float(value) for value in mean)

    task_ids = np.asarray(groups["task_id"], dtype=str)
    means = np.vstack([task_means[value] for value in task_ids])
    totals = counts.sum(axis=1)

    def objective(log_concentration: float) -> float:
        concentration = float(np.exp(log_concentration))
        alpha = concentration * means
        log_likelihood = (
            gammaln(concentration)
            - gammaln(concentration + totals)
            + (gammaln(alpha + counts) - gammaln(alpha)).sum(axis=1)
        )
        return float(-log_likelihood.sum())

    fitted = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(5_000.0)),
        method="bounded",
        options={"xatol": 1e-7},
    )
    if not fitted.success:
        raise RuntimeError(f"Round-0 concentration fit failed: {fitted.message}")
    concentration = float(np.exp(fitted.x))
    return HierarchicalRoundZeroModel(
        global_mean=tuple(float(value) for value in global_mean),
        task_means=task_means,
        concentration=concentration,
        smoothing_strength=float(smoothing_strength),
    )
