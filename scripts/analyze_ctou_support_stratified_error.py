"""Relate CTOU range-extrapolation errors to local transition-table support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_ctou_density_extrapolation import (
    attach_losses,
    fit_table_lookup,
    load_cases_and_graphs,
    sparse_mask,
    split_boundaries,
)
from analyze_ctou_recursive_rollout import (
    STATE_INDEX,
    STATES,
    composition_distribution,
    graph_maps,
    query_frame,
)
from analyze_ctou_transition_prediction import COUNT_COLUMNS, TABLE_KEYS
from analyze_node_round_adoption import read_json

DEFAULT_SEED = 20_260_817
SUPPORT_THRESHOLDS = (5, 10, 20)
IDENTITY = ("task_id", "graph_id", "attack_node")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--extrapolation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--table-prior-strength", type=float, default=5.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def transition_count_lookup(
    train: pd.DataFrame,
    *,
    maximum_neighbors: int,
    horizon: int,
) -> np.ndarray:
    """Create a dense lookup containing exact training counts per CTOU cell."""
    query = query_frame(maximum_neighbors, horizon)
    counts = train.groupby(list(TABLE_KEYS)).size().rename("support_count")
    supported = query.merge(
        counts.reset_index(),
        on=list(TABLE_KEYS),
        how="left",
        validate="one_to_one",
    )
    values = supported.support_count.fillna(0).to_numpy(float)
    shape = (
        horizon + 1,
        len(STATES),
        *(maximum_neighbors + 1 for _ in STATES),
    )
    lookup = np.full(shape, np.nan, dtype=np.float32)
    for position, row in enumerate(query.itertuples(index=False)):
        cell = (
            int(row.round_index),
            STATE_INDEX[str(row.previous_attack_state)],
            *(int(getattr(row, column)) for column in COUNT_COLUMNS),
        )
        lookup[cell] = values[position]
    return lookup


def mean_field_rollout_with_support(
    *,
    graph: dict[str, Any],
    initial_states: tuple[int, ...],
    attack_node: int,
    probability_lookup: np.ndarray,
    support_lookup: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Roll out CTOU and track probability-weighted visits to low-support cells."""
    n = int(graph["node_count"])
    horizon = int(graph["max_rounds"])
    readout = int(graph["readout_node"])
    incoming, _ = graph_maps(graph)
    distances = _distances_to_readout(graph)
    marginals = np.eye(len(STATES), dtype=float)[np.asarray(initial_states)]
    metrics = {
        "expected_transition_visits": 0.0,
        "expected_unseen_visits": 0.0,
        "expected_log1p_support_sum": 0.0,
        **{f"expected_support_lt_{value}_visits": 0.0 for value in SUPPORT_THRESHOLDS},
    }
    for round_index in range(1, horizon + 1):
        updated = marginals.copy()
        for node in range(n):
            if round_index + distances[node] > horizon:
                continue
            if node == attack_node:
                updated[node] = np.eye(len(STATES))[STATE_INDEX["target"]]
                continue
            result = np.zeros(len(STATES), dtype=float)
            compositions = composition_distribution(
                [marginals[source] for source in incoming[node]]
            )
            for previous, previous_mass in enumerate(marginals[node]):
                if previous_mass == 0:
                    continue
                for counts, composition_mass in compositions.items():
                    mass = float(previous_mass * composition_mass)
                    cell = (round_index, previous, *counts)
                    support = float(support_lookup[cell])
                    if np.isnan(support):
                        raise RuntimeError(f"missing support lookup cell {cell}")
                    metrics["expected_transition_visits"] += mass
                    metrics["expected_unseen_visits"] += mass * float(support == 0)
                    metrics["expected_log1p_support_sum"] += mass * np.log1p(support)
                    for threshold in SUPPORT_THRESHOLDS:
                        metrics[f"expected_support_lt_{threshold}_visits"] += mass * float(
                            support < threshold
                        )
                    result += mass * probability_lookup[cell]
            updated[node] = result / result.sum()
        marginals = updated
    visits = metrics["expected_transition_visits"]
    metrics["expected_unseen_fraction"] = metrics["expected_unseen_visits"] / visits
    metrics["expected_mean_log1p_support"] = metrics["expected_log1p_support_sum"] / visits
    for threshold in SUPPORT_THRESHOLDS:
        metrics[f"expected_support_lt_{threshold}_fraction"] = (
            metrics[f"expected_support_lt_{threshold}_visits"] / visits
        )
    return marginals[readout], metrics


def _distances_to_readout(graph: dict[str, Any]) -> list[int]:
    n = int(graph["node_count"])
    readout = int(graph["readout_node"])
    _, outgoing = graph_maps(graph)
    distances: list[int] = []
    for source in range(n):
        frontier = [(source, 0)]
        seen = {source}
        found: int | None = None
        while frontier:
            node, distance = frontier.pop(0)
            if node == readout:
                found = distance
                break
            for target in outgoing[node]:
                if target not in seen:
                    seen.add(target)
                    frontier.append((target, distance + 1))
        if found is None:
            raise ValueError(f"node {source} cannot reach readout")
        distances.append(found)
    return distances


def observed_path_support(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Aggregate post-hoc support counts along the realized LLM trace."""
    counts = train.groupby(list(TABLE_KEYS)).size()
    keys = pd.MultiIndex.from_frame(test[list(TABLE_KEYS)])
    evaluated = test[list(IDENTITY)].copy()
    evaluated["support_count"] = counts.reindex(keys, fill_value=0).to_numpy(float)
    evaluated["log1p_support"] = np.log1p(evaluated.support_count)
    evaluated["unseen"] = evaluated.support_count.eq(0)
    for threshold in SUPPORT_THRESHOLDS:
        evaluated[f"support_lt_{threshold}"] = evaluated.support_count.lt(threshold)
    grouped = evaluated.groupby(list(IDENTITY), sort=False)
    result = grouped.agg(
        actual_transition_visits=("support_count", "size"),
        actual_min_support=("support_count", "min"),
        actual_unseen_visits=("unseen", "sum"),
        actual_mean_log1p_support=("log1p_support", "mean"),
    )
    for threshold in SUPPORT_THRESHOLDS:
        result[f"actual_support_lt_{threshold}_visits"] = grouped[f"support_lt_{threshold}"].sum()
    result = result.reset_index()
    result["actual_unseen_fraction"] = result.actual_unseen_visits / result.actual_transition_visits
    for threshold in SUPPORT_THRESHOLDS:
        result[f"actual_support_lt_{threshold}_fraction"] = (
            result[f"actual_support_lt_{threshold}_visits"] / result.actual_transition_visits
        )
    return result


def expected_support_stratum(frame: pd.DataFrame) -> pd.Series:
    fraction = frame.expected_support_lt_20_fraction
    return pd.Series(
        np.select(
            [fraction.le(1e-12), fraction.le(0.05), fraction.le(0.20)],
            ["all_high_support", "low_mass_le_5pct", "low_mass_5_20pct"],
            default="low_mass_gt_20pct",
        ),
        index=frame.index,
    )


def actual_support_stratum(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                frame.actual_support_lt_20_visits.eq(0),
                frame.actual_unseen_visits.eq(0),
                frame.actual_unseen_visits.eq(1),
            ],
            ["all_high_support", "seen_but_low_support", "one_unseen"],
            default="multiple_unseen",
        ),
        index=frame.index,
    )


def execute_range_support(
    *,
    updates: pd.DataFrame,
    cases: pd.DataFrame,
    graphs: dict[str, dict[str, Any]],
    existing_predictions: pd.DataFrame,
    folds: int,
    prior_strength: float,
) -> pd.DataFrame:
    boundaries = split_boundaries(cases)
    update_sparse = sparse_mask(updates, boundaries)
    case_sparse = sparse_mask(cases, boundaries)
    directions = {
        "sparse_to_dense": (update_sparse, ~case_sparse),
        "dense_to_sparse": (~update_sparse, case_sparse),
    }
    maximum_neighbors = int(cases.n.max() - 1)
    horizon = int(cases.horizon.max())
    rows: list[pd.DataFrame] = []
    for direction, (train_region, test_region) in directions.items():
        for scope in ("density_only", "density_task"):
            task_folds: list[int | None] = [None] if scope == "density_only" else list(range(folds))
            for task_fold in task_folds:
                train_mask = train_region.copy()
                test_case_mask = test_region.copy()
                test_update_mask = (
                    ~update_sparse if direction == "sparse_to_dense" else update_sparse
                )
                if task_fold is not None:
                    train_mask &= updates.task_fold.ne(task_fold)
                    test_case_mask &= cases.task_fold.eq(task_fold)
                    test_update_mask &= updates.task_fold.eq(task_fold)
                train = updates[train_mask]
                test_cases = cases[test_case_mask]
                test_updates = updates[test_update_mask]
                probability_lookup = fit_table_lookup(
                    train,
                    maximum_neighbors=maximum_neighbors,
                    horizon=horizon,
                    prior_strength=prior_strength,
                )
                support_lookup = transition_count_lookup(
                    train,
                    maximum_neighbors=maximum_neighbors,
                    horizon=horizon,
                )
                realized = observed_path_support(train, test_updates)
                cache: dict[
                    tuple[str, int, tuple[int, ...]], tuple[np.ndarray, dict[str, float]]
                ] = {}
                prediction_rows: list[dict[str, object]] = []
                for case in test_cases.itertuples(index=False):
                    key = (str(case.graph_id), int(case.attack_node), tuple(case.initial_states))
                    if key not in cache:
                        cache[key] = mean_field_rollout_with_support(
                            graph=graphs[str(case.graph_id)],
                            initial_states=tuple(case.initial_states),
                            attack_node=int(case.attack_node),
                            probability_lookup=probability_lookup,
                            support_lookup=support_lookup,
                        )
                    probability, support = cache[key]
                    prediction_rows.append(
                        {
                            "validation_scope": scope,
                            "direction": direction,
                            "test_task_fold": task_fold,
                            "task_id": case.task_id,
                            "graph_id": case.graph_id,
                            "attack_node": case.attack_node,
                            "n": case.n,
                            "m": case.m,
                            "actual_state": case.actual_state,
                            "actual_state_index": case.actual_state_index,
                            "actual_target": case.actual_target,
                            "actual_correct": case.actual_correct,
                            **{
                                f"p_{state}": float(probability[index])
                                for index, state in enumerate(STATES)
                            },
                            **support,
                        }
                    )
                result = pd.DataFrame(prediction_rows).merge(
                    realized,
                    on=list(IDENTITY),
                    how="left",
                    validate="one_to_one",
                )
                reference = existing_predictions[
                    existing_predictions.validation_scope.eq(scope)
                    & existing_predictions.direction.eq(direction)
                ]
                if task_fold is not None:
                    reference = reference[reference.test_task_fold.eq(task_fold)]
                check = result.merge(
                    reference[[*IDENTITY, *[f"p_{state}" for state in STATES]]],
                    on=list(IDENTITY),
                    suffixes=("_new", "_reference"),
                    validate="one_to_one",
                )
                maximum_gap = max(
                    (check[f"p_{state}_new"] - check[f"p_{state}_reference"]).abs().max()
                    for state in STATES
                )
                if maximum_gap > 1e-6:
                    raise RuntimeError(f"rollout mismatch {direction}/{scope}: {maximum_gap}")
                rows.append(result)
                print(
                    f"support {direction} {scope} fold={task_fold}: "
                    f"train={len(train)} endpoints={len(result)} cache={len(cache)}",
                    flush=True,
                )
    combined = pd.concat(rows, ignore_index=True)
    combined["expected_support_stratum"] = expected_support_stratum(combined)
    combined["actual_support_stratum"] = actual_support_stratum(combined)
    return attach_losses(combined)


def stratified_summary(
    frame: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    metrics = (
        "multiclass_brier",
        "target_brier",
        "correct_brier",
        "absolute_target_error",
        "absolute_correct_error",
    )
    frame = frame.copy()
    frame["absolute_target_error"] = (frame.p_target - frame.actual_target).abs()
    frame["absolute_correct_error"] = (frame.p_correct - frame.actual_correct).abs()
    rows: list[dict[str, object]] = []
    for source, stratum_column in (
        ("expected_rollout", "expected_support_stratum"),
        ("observed_trace", "actual_support_stratum"),
    ):
        keys = ["validation_scope", "direction", "n", stratum_column]
        task = frame.groupby([*keys, "task_id"], sort=False)[list(metrics)].mean().reset_index()
        for values, group in task.groupby(keys, sort=False):
            matrix = group[list(metrics)].to_numpy(float)
            sample = rng.integers(0, len(matrix), size=(replicates, len(matrix)))
            bootstrap = matrix[sample].mean(axis=1)
            for index, metric in enumerate(metrics):
                low, high = np.quantile(bootstrap[:, index], [0.025, 0.975])
                rows.append(
                    {
                        "support_source": source,
                        **dict(zip(keys, values, strict=True)),
                        "support_stratum": values[-1],
                        "metric": metric,
                        "estimate": matrix[:, index].mean(),
                        "ci95_low": low,
                        "ci95_high": high,
                        "tasks": len(matrix),
                        "endpoints": len(
                            frame[
                                frame.validation_scope.eq(values[0])
                                & frame.direction.eq(values[1])
                                & frame.n.eq(values[2])
                                & frame[stratum_column].eq(values[3])
                            ]
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    return result.drop(
        columns=["expected_support_stratum", "actual_support_stratum"], errors="ignore"
    )


def continuous_associations(frame: pd.DataFrame) -> pd.DataFrame:
    evaluated = frame.copy()
    evaluated["absolute_target_error"] = (evaluated.p_target - evaluated.actual_target).abs()
    evaluated["absolute_correct_error"] = (evaluated.p_correct - evaluated.actual_correct).abs()
    support_metrics = (
        "expected_unseen_fraction",
        "expected_support_lt_5_fraction",
        "expected_support_lt_10_fraction",
        "expected_support_lt_20_fraction",
        "expected_mean_log1p_support",
        "actual_unseen_fraction",
        "actual_support_lt_5_fraction",
        "actual_support_lt_10_fraction",
        "actual_support_lt_20_fraction",
        "actual_mean_log1p_support",
    )
    error_metrics = (
        "multiclass_brier",
        "target_brier",
        "correct_brier",
        "absolute_target_error",
        "absolute_correct_error",
    )
    rows: list[dict[str, object]] = []
    keys = ["validation_scope", "direction", "n"]
    for values, group in evaluated.groupby(keys, sort=False):
        for support in support_metrics:
            for error in error_metrics:
                rows.append(
                    {
                        **dict(zip(keys, values, strict=True)),
                        "support_metric": support,
                        "error_metric": error,
                        "spearman": group[support].corr(group[error], method="spearman"),
                        "endpoints": len(group),
                    }
                )
    return pd.DataFrame(rows)


def pooled_within_group_rank_correlation(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    left: str,
    right: str,
) -> float:
    """Correlate group-centered within-group ranks.

    This is a descriptive fixed-stratum rank association. It removes all
    between-stratum variation before pooling but is not a causal estimator.
    """
    selected = frame[[*group_columns, left, right]].dropna().copy()
    if selected.empty:
        return float("nan")
    left_rank = selected.groupby(group_columns, sort=False)[left].rank(method="average")
    right_rank = selected.groupby(group_columns, sort=False)[right].rank(method="average")
    left_centered = left_rank - left_rank.groupby(
        [selected[column] for column in group_columns], sort=False
    ).transform("mean")
    right_centered = right_rank - right_rank.groupby(
        [selected[column] for column in group_columns], sort=False
    ).transform("mean")
    if left_centered.std() == 0 or right_centered.std() == 0:
        return float("nan")
    return float(left_centered.corr(right_centered, method="pearson"))


def conditional_associations(frame: pd.DataFrame) -> pd.DataFrame:
    evaluated = frame.copy()
    evaluated["absolute_target_error"] = (evaluated.p_target - evaluated.actual_target).abs()
    evaluated["absolute_correct_error"] = (
        evaluated.p_correct - evaluated.actual_correct
    ).abs()
    support_metrics = (
        "expected_unseen_fraction",
        "expected_support_lt_5_fraction",
        "expected_support_lt_10_fraction",
        "expected_support_lt_20_fraction",
        "actual_unseen_fraction",
        "actual_support_lt_5_fraction",
        "actual_support_lt_10_fraction",
        "actual_support_lt_20_fraction",
    )
    error_metrics = (
        "multiclass_brier",
        "target_brier",
        "correct_brier",
        "absolute_target_error",
        "absolute_correct_error",
    )
    conditioning = {
        "within_task_density": ["task_id", "m"],
        "within_task_graph": ["task_id", "graph_id"],
    }
    rows: list[dict[str, object]] = []
    keys = ["validation_scope", "direction", "n"]
    for values, group in evaluated.groupby(keys, sort=False):
        for condition_name, columns in conditioning.items():
            for support in support_metrics:
                for error in error_metrics:
                    rows.append(
                        {
                            **dict(zip(keys, values, strict=True)),
                            "conditioning": condition_name,
                            "support_metric": support,
                            "error_metric": error,
                            "pooled_within_rank_correlation": (
                                pooled_within_group_rank_correlation(
                                    group,
                                    group_columns=columns,
                                    left=support,
                                    right=error,
                                )
                            ),
                            "endpoints": len(group),
                        }
                    )
    return pd.DataFrame(rows)


def primary_task_conditional_summary(
    frame: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Summarize primary conditional associations across independent tasks."""
    evaluated = frame[frame.validation_scope.eq("density_task")].copy()
    evaluated["absolute_target_error"] = (evaluated.p_target - evaluated.actual_target).abs()
    evaluated["absolute_correct_error"] = (
        evaluated.p_correct - evaluated.actual_correct
    ).abs()
    support = "expected_support_lt_20_fraction"
    errors = (
        "multiclass_brier",
        "target_brier",
        "correct_brier",
        "absolute_target_error",
        "absolute_correct_error",
    )
    conditioning = {
        "within_task_density": ["m"],
        "within_task_graph": ["graph_id"],
    }
    rows: list[dict[str, object]] = []
    for (direction, n), group in evaluated.groupby(["direction", "n"], sort=False):
        for condition_name, group_columns in conditioning.items():
            for error in errors:
                task_correlations: list[float] = []
                for _, task in group.groupby("task_id", sort=False):
                    value = pooled_within_group_rank_correlation(
                        task,
                        group_columns=group_columns,
                        left=support,
                        right=error,
                    )
                    if np.isfinite(value):
                        task_correlations.append(value)
                values = np.asarray(task_correlations, dtype=float)
                sample = rng.integers(0, len(values), size=(replicates, len(values)))
                bootstrap = values[sample].mean(axis=1)
                low, high = np.quantile(bootstrap, [0.025, 0.975])
                rows.append(
                    {
                        "validation_scope": "density_task",
                        "direction": direction,
                        "n": int(n),
                        "conditioning": condition_name,
                        "support_metric": support,
                        "error_metric": error,
                        "mean_task_correlation": values.mean(),
                        "median_task_correlation": np.median(values),
                        "positive_task_fraction": (values > 0).mean(),
                        "ci95_low": low,
                        "ci95_high": high,
                        "tasks": len(values),
                    }
                )
    return pd.DataFrame(rows)


def graph_residual_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    support_columns = [
        "expected_unseen_fraction",
        "expected_support_lt_5_fraction",
        "expected_support_lt_10_fraction",
        "expected_support_lt_20_fraction",
        "expected_mean_log1p_support",
        "actual_unseen_fraction",
        "actual_support_lt_5_fraction",
        "actual_support_lt_10_fraction",
        "actual_support_lt_20_fraction",
        "actual_mean_log1p_support",
    ]
    keys = ["validation_scope", "direction", "graph_id", "n", "m"]
    graph = (
        frame.groupby(keys, sort=False)
        .agg(
            endpoints=("actual_correct", "size"),
            observed_target=("actual_target", "mean"),
            predicted_target=("p_target", "mean"),
            observed_correct=("actual_correct", "mean"),
            predicted_correct=("p_correct", "mean"),
            **{column: (column, "mean") for column in support_columns},
        )
        .reset_index()
    )
    for outcome in ("target", "correct"):
        graph[f"signed_{outcome}_residual"] = (
            graph[f"predicted_{outcome}"] - graph[f"observed_{outcome}"]
        )
        graph[f"absolute_{outcome}_residual"] = graph[f"signed_{outcome}_residual"].abs()
    rows: list[dict[str, object]] = []
    for values, group in graph.groupby(["validation_scope", "direction", "n"], sort=False):
        for support in support_columns:
            for outcome in ("target", "correct"):
                for residual_type in ("signed", "absolute"):
                    residual = f"{residual_type}_{outcome}_residual"
                    rows.append(
                        {
                            "validation_scope": values[0],
                            "direction": values[1],
                            "n": values[2],
                            "support_metric": support,
                            "residual_metric": residual,
                            "spearman": group[support].corr(group[residual], method="spearman"),
                            "graphs": len(group),
                        }
                    )
    return graph, pd.DataFrame(rows)


def conditional_graph_associations(graph: pd.DataFrame) -> pd.DataFrame:
    support_columns = (
        "expected_unseen_fraction",
        "expected_support_lt_5_fraction",
        "expected_support_lt_10_fraction",
        "expected_support_lt_20_fraction",
        "actual_unseen_fraction",
        "actual_support_lt_5_fraction",
        "actual_support_lt_10_fraction",
        "actual_support_lt_20_fraction",
    )
    residuals = (
        "signed_target_residual",
        "absolute_target_residual",
        "signed_correct_residual",
        "absolute_correct_residual",
    )
    rows: list[dict[str, object]] = []
    keys = ["validation_scope", "direction", "n"]
    for values, group in graph.groupby(keys, sort=False):
        for support in support_columns:
            for residual in residuals:
                rows.append(
                    {
                        **dict(zip(keys, values, strict=True)),
                        "conditioning": "within_density",
                        "support_metric": support,
                        "residual_metric": residual,
                        "pooled_within_rank_correlation": (
                            pooled_within_group_rank_correlation(
                                group,
                                group_columns=["m"],
                                left=support,
                                right=residual,
                            )
                        ),
                        "graphs": len(group),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    update_cache = args.extrapolation_dir / "normalized_transition_updates.pkl"
    update_audit_path = args.extrapolation_dir / "normalized_update_audit.json"
    if not update_cache.exists() or not update_audit_path.exists():
        raise FileNotFoundError("density extrapolation transition cache is required")
    updates = pd.read_pickle(update_cache)
    update_audit = read_json(update_audit_path)
    cases, graphs, case_audit = load_cases_and_graphs(args.reference_dir)
    if not update_audit["passed"] or not case_audit["passed"]:
        raise RuntimeError("input integrity audit failed")
    existing = pd.read_csv(args.extrapolation_dir / "endpoint_predictions.csv")
    existing = existing[
        existing.experiment.eq("range_extrapolation") & existing.model.eq("ctou_table")
    ].copy()
    endpoint = execute_range_support(
        updates=updates,
        cases=cases,
        graphs=graphs,
        existing_predictions=existing,
        folds=args.folds,
        prior_strength=args.table_prior_strength,
    )
    rng = np.random.default_rng(args.seed)
    strata = stratified_summary(endpoint, replicates=args.bootstrap_replicates, rng=rng)
    associations = continuous_associations(endpoint)
    conditional = conditional_associations(endpoint)
    task_conditional = primary_task_conditional_summary(
        endpoint,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )
    graph, graph_associations = graph_residual_analysis(endpoint)
    graph_conditional = conditional_graph_associations(graph)
    endpoint.to_csv(args.output_dir / "endpoint_support_predictions.csv", index=False)
    strata.to_csv(args.output_dir / "endpoint_support_strata.csv", index=False)
    associations.to_csv(args.output_dir / "endpoint_support_associations.csv", index=False)
    conditional.to_csv(
        args.output_dir / "endpoint_support_conditional_associations.csv",
        index=False,
    )
    task_conditional.to_csv(
        args.output_dir / "primary_task_conditional_summary.csv",
        index=False,
    )
    graph.to_csv(args.output_dir / "graph_support_residuals.csv", index=False)
    graph_associations.to_csv(args.output_dir / "graph_support_associations.csv", index=False)
    graph_conditional.to_csv(
        args.output_dir / "graph_support_conditional_associations.csv",
        index=False,
    )
    manifest = {
        "analysis_version": "ctou-support-stratified-error-v1",
        "run_root": str(args.run_root.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "extrapolation_dir": str(args.extrapolation_dir.resolve()),
        "support_thresholds": list(SUPPORT_THRESHOLDS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "endpoints": len(endpoint),
        "information_boundary": {
            "expected_rollout_support": "available from model rollout without Round-1+ trace",
            "observed_trace_support": "post-hoc diagnostic using realized Round-1+ trace",
        },
        "claim_limits": [
            "support associations are descriptive and do not establish causal error origin",
            "observed-trace support is outcome-dependent and unavailable at prediction time",
            "the threshold 20 is a reporting convention; continuous and 5/10 "
            "sensitivity metrics are retained",
            "results are conditional on the current model, task set, graph sample, "
            "and attack protocol",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
