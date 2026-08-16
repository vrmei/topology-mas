"""Evaluate a frozen n=5/8 CTOU law on held-out n=6/7 systems.

The primary recursive evaluator observes the test run's categorical Round-0 state
vector, graph, readout, attack position, and fixed schedule. It never observes test
Round-1+ states or compositions while predicting. Realized test transitions are read
only afterward for one-step and support/error diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_ctou_clean_utility import load_clean_data
from analyze_ctou_density_extrapolation import fit_table_lookup
from analyze_ctou_recursive_rollout import (
    EPSILON,
    STATE_INDEX,
    STATES,
    composition_distribution,
    distances_to_readout,
    graph_maps,
    load_rollout_cases,
    mean_field_rollout,
)
from analyze_ctou_support_stratified_error import (
    SUPPORT_THRESHOLDS,
    transition_count_lookup,
)
from analyze_ctou_transition_prediction import (
    COUNT_COLUMNS,
    TABLE_KEYS,
    design_matrix,
    load_updates,
    static_predictions,
    table_predictions,
)
from analyze_node_round_adoption import read_json
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

TRAIN_SIZES = (5, 8)
TEST_SIZES = (6, 7)
MODELS = ("persistence", "degroot_equal", "ctou_table")
ONE_STEP_MODELS = (*MODELS, "ctou_logit")
DEFAULT_SEED = 20_260_816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run-root", type=Path, required=True)
    parser.add_argument("--test-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--table-prior-strength", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    def json_default(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"cannot JSON serialize {type(item).__name__}")

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_pickle(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_or_cache_inputs(
    *,
    run_root: Path,
    cache_root: Path,
    label: str,
    folds: int,
) -> dict[str, Any]:
    cache_path = cache_root / f"{label}_normalized.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    status = read_json(run_root / "orchestrator_status.json")
    if status.get("status") != "completed":
        raise RuntimeError(f"{label} run is not complete")
    clean_cases, clean_updates, clean_graphs, clean_audit = load_clean_data(run_root, status, folds)
    existing_update_cache = (
        run_root / "ctou-density-extrapolation-v1/normalized_transition_updates.pkl"
    )
    existing_case_cache = run_root / "ctou-recursive-rollout-v1/normalized_rollout_cases.pkl"
    if existing_update_cache.exists() and existing_case_cache.exists():
        with existing_update_cache.open("rb") as handle:
            attack_updates = pickle.load(handle)
        with existing_case_cache.open("rb") as handle:
            attack_cases = pickle.load(handle)
        attack_graphs = clean_graphs
        required_update_columns = {
            "task_id",
            "graph_id",
            "attack_node",
            "receiver_node",
            "round_index",
            "task_fold",
            "n",
            "m",
            "current_state_index",
            "current_attack_state",
            "previous_attack_state",
            *COUNT_COLUMNS,
        }
        required_case_columns = {
            "task_id",
            "graph_id",
            "attack_node",
            "task_fold",
            "n",
            "m",
            "initial_states",
            "actual_state",
            "actual_state_index",
            "actual_target",
            "actual_correct",
        }
        update_errors = sorted(required_update_columns - set(attack_updates.columns))
        case_errors = sorted(required_case_columns - set(attack_cases.columns))
        unknown_graphs = sorted(set(attack_cases.graph_id) - set(attack_graphs))
        attack_update_audit = {
            "passed": not update_errors,
            "errors": [f"missing cached update column: {x}" for x in update_errors],
            "source": str(existing_update_cache),
            "rows": len(attack_updates),
        }
        attack_case_audit = {
            "passed": not case_errors and not unknown_graphs,
            "errors": [
                *[f"missing cached case column: {x}" for x in case_errors],
                *[f"cached case references unknown graph: {x}" for x in unknown_graphs[:20]],
            ],
            "source": str(existing_case_cache),
            "rows": len(attack_cases),
        }
    else:
        attack_updates, attack_update_audit = load_updates(run_root, folds)
        attack_cases, attack_graphs, attack_case_audit = load_rollout_cases(run_root, status, folds)
    clean_updates = clean_updates.copy()
    clean_updates["attack_node"] = -1
    readout_by_graph = {
        graph_id: int(graph["readout_node"]) for graph_id, graph in clean_graphs.items()
    }
    clean_updates["receiver_scope"] = np.where(
        clean_updates.receiver_node.eq(clean_updates.graph_id.map(readout_by_graph)),
        "readout",
        "internal",
    )
    value = {
        "status": status,
        "attack_updates": attack_updates,
        "attack_cases": attack_cases,
        "attack_graphs": attack_graphs,
        "clean_updates": clean_updates,
        "clean_cases": clean_cases,
        "clean_graphs": clean_graphs,
        "audits": {
            "attack_updates": attack_update_audit,
            "attack_cases": attack_case_audit,
            "clean": clean_audit,
        },
    }
    temporary = cache_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache_path)
    return value


def validate_boundary(train: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    for label, bundle in (("train", train), ("test", test)):
        for name, audit in bundle["audits"].items():
            if not audit.get("passed"):
                errors.append(f"{label} {name} integrity failed")
    train_sizes = set(train["attack_updates"].n.unique())
    test_sizes = set(test["attack_updates"].n.unique())
    if train_sizes != set(TRAIN_SIZES):
        errors.append(f"unexpected train sizes: {sorted(train_sizes)}")
    if test_sizes != set(TEST_SIZES):
        errors.append(f"unexpected test sizes: {sorted(test_sizes)}")
    train_tasks = set(train["attack_updates"].task_id)
    test_tasks = set(test["attack_updates"].task_id)
    if train_tasks != test_tasks:
        errors.append("train/test task collections differ")
    train_graphs = set(train["attack_updates"].graph_id)
    test_graphs = set(test["attack_updates"].graph_id)
    graph_overlap = train_graphs & test_graphs
    if graph_overlap:
        errors.append(f"cross-size graph overlap: {len(graph_overlap)}")
    audit = {
        "passed": not errors,
        "errors": errors,
        "train_sizes": sorted(train_sizes),
        "test_sizes": sorted(test_sizes),
        "train_tasks": len(train_tasks),
        "test_tasks": len(test_tasks),
        "train_graphs": len(train_graphs),
        "test_graphs": len(test_graphs),
        "graph_overlap": len(graph_overlap),
    }
    if errors:
        raise RuntimeError("; ".join(errors))
    return audit


def aligned_logit_predictions(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=0)
    model.fit(design_matrix(train), train.current_state_index.to_numpy(int))
    raw = model.predict_proba(design_matrix(test))
    aligned = np.zeros((len(test), len(STATES)), dtype=np.float32)
    aligned[:, model.classes_.astype(int)] = raw
    return aligned


def support_counts(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    exact = train.groupby(list(TABLE_KEYS)).size()
    composition = train.groupby(list(COUNT_COLUMNS)).size()
    result = test[
        [
            "task_id",
            "graph_id",
            "attack_node",
            "n",
            "m",
            "round_index",
            "receiver_scope",
        ]
    ].copy()
    exact_keys = pd.MultiIndex.from_frame(test[list(TABLE_KEYS)])
    composition_keys = pd.MultiIndex.from_frame(test[list(COUNT_COLUMNS)])
    result["exact_support"] = exact.reindex(exact_keys, fill_value=0).to_numpy(int)
    result["composition_support"] = composition.reindex(composition_keys, fill_value=0).to_numpy(
        int
    )
    result["exact_seen"] = result.exact_support.gt(0)
    result["composition_seen"] = result.composition_support.gt(0)
    result["exact_log1p_support"] = np.log1p(result.exact_support)
    for threshold in SUPPORT_THRESHOLDS:
        result[f"exact_support_lt_{threshold}"] = result.exact_support.lt(threshold)
    return result


def cross_size_one_step(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    condition: str,
    folds: int,
    prior_strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = static_predictions(test)
    predictions["ctou_table"] = np.full((len(test), len(STATES)), np.nan, dtype=np.float32)
    predictions["ctou_logit"] = np.full((len(test), len(STATES)), np.nan, dtype=np.float32)
    support_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for task_fold in range(folds):
        train_fold = train[train.task_fold.ne(task_fold)]
        mask = test.task_fold.eq(task_fold).to_numpy()
        test_fold = test.loc[mask]
        if test_fold.empty:
            continue
        train_tasks = set(train_fold.task_id)
        test_tasks = set(test_fold.task_id)
        train_graphs = set(train_fold.graph_id)
        test_graphs = set(test_fold.graph_id)
        audit_rows.append(
            {
                "condition": condition,
                "task_fold": task_fold,
                "train_rows": len(train_fold),
                "test_rows": len(test_fold),
                "train_tasks": len(train_tasks),
                "test_tasks": len(test_tasks),
                "task_overlap": len(train_tasks & test_tasks),
                "train_graphs": len(train_graphs),
                "test_graphs": len(test_graphs),
                "graph_overlap": len(train_graphs & test_graphs),
                "train_sizes": ",".join(map(str, sorted(train_fold.n.unique()))),
                "test_sizes": ",".join(map(str, sorted(test_fold.n.unique()))),
            }
        )
        predictions["ctou_table"][mask] = table_predictions(train_fold, test_fold, prior_strength)
        predictions["ctou_logit"][mask] = aligned_logit_predictions(train_fold, test_fold)
        part = support_counts(train_fold, test_fold)
        part["condition"] = condition
        part["task_fold"] = task_fold
        support_parts.append(part)
    for model in ("ctou_table", "ctou_logit"):
        if np.isnan(predictions[model]).any():
            raise RuntimeError(f"incomplete one-step {condition} predictions for {model}")
    labels = test.current_state_index.to_numpy(int)
    rows: list[pd.DataFrame] = []
    for model, probability in predictions.items():
        clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
        one_hot = np.eye(len(STATES))[labels]
        part = test[
            [
                "task_id",
                "graph_id",
                "n",
                "m",
                "round_index",
                "receiver_scope",
            ]
        ].copy()
        part["condition"] = condition
        part["model"] = model
        part["actual_state"] = test.current_attack_state.to_numpy()
        part["multiclass_brier"] = ((probability - one_hot) ** 2).sum(axis=1)
        part["multiclass_log_loss"] = -np.log(clipped[np.arange(len(test)), labels])
        part["classification_error"] = (probability.argmax(axis=1) != labels).astype(np.int8)
        for index, state in enumerate(STATES):
            part[f"p_{state}"] = probability[:, index]
        rows.append(part)
    return (
        pd.concat(rows, ignore_index=True),
        pd.concat(support_parts, ignore_index=True),
        pd.DataFrame(audit_rows),
    )


def mean_field_trajectory_with_support(
    *,
    graph: dict[str, Any],
    initial_states: tuple[int, ...],
    attack_node: int | None,
    probability_lookup: np.ndarray,
    support_lookup: np.ndarray,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    n = int(graph["node_count"])
    horizon = int(graph["max_rounds"])
    readout = int(graph["readout_node"])
    incoming, _ = graph_maps(graph)
    distances = distances_to_readout(graph)
    marginals = np.eye(len(STATES), dtype=float)[np.asarray(initial_states)]
    trajectory = [marginals.copy()]
    support_rows: list[dict[str, object]] = []
    for round_index in range(1, horizon + 1):
        updated = marginals.copy()
        for node in range(n):
            if round_index + distances[node] > horizon:
                continue
            if attack_node is not None and node == attack_node:
                updated[node] = np.eye(len(STATES))[STATE_INDEX["target"]]
                continue
            result = np.zeros(len(STATES), dtype=float)
            metrics = Counter()
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
                        raise RuntimeError(f"missing support cell {cell}")
                    metrics["visits"] += mass
                    metrics["unseen"] += mass * float(support == 0)
                    metrics["log1p_support"] += mass * np.log1p(support)
                    for threshold in SUPPORT_THRESHOLDS:
                        metrics[f"support_lt_{threshold}"] += mass * float(support < threshold)
                    result += mass * probability_lookup[cell]
            updated[node] = result / result.sum()
            visits = float(metrics["visits"])
            row: dict[str, object] = {
                "round_index": round_index,
                "receiver_node": node,
                "receiver_scope": "readout" if node == readout else "internal",
                "expected_transition_visits": visits,
                "expected_unseen_fraction": metrics["unseen"] / visits,
                "expected_mean_log1p_support": metrics["log1p_support"] / visits,
            }
            for threshold in SUPPORT_THRESHOLDS:
                row[f"expected_support_lt_{threshold}_fraction"] = (
                    metrics[f"support_lt_{threshold}"] / visits
                )
            support_rows.append(row)
        marginals = updated
        trajectory.append(marginals.copy())
    return trajectory, support_rows


def update_lookup(frame: pd.DataFrame) -> dict[tuple[object, ...], Any]:
    result: dict[tuple[object, ...], Any] = {}
    for row in frame.itertuples(index=False):
        key = (
            str(row.task_id),
            str(row.graph_id),
            int(row.attack_node),
            int(row.receiver_node),
            int(row.round_index),
        )
        if key in result:
            raise RuntimeError(f"duplicate update key: {key}")
        result[key] = row
    return result


def rollout_fold(
    *,
    train_updates: pd.DataFrame,
    test_updates: pd.DataFrame,
    cases: pd.DataFrame,
    graphs: dict[str, dict[str, Any]],
    condition: str,
    task_fold: int,
    maximum_neighbors: int,
    horizon: int,
    prior_strength: float,
) -> dict[str, pd.DataFrame]:
    probability_lookup = fit_table_lookup(
        train_updates,
        maximum_neighbors=maximum_neighbors,
        horizon=horizon,
        prior_strength=prior_strength,
    )
    support_lookup = transition_count_lookup(
        train_updates, maximum_neighbors=maximum_neighbors, horizon=horizon
    )
    actual_updates = update_lookup(test_updates)
    endpoint_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    for case in cases.itertuples(index=False):
        graph = graphs[str(case.graph_id)]
        attack_node = int(case.attack_node) if condition == "attack" else None
        trajectory, support = mean_field_trajectory_with_support(
            graph=graph,
            initial_states=tuple(case.initial_states),
            attack_node=attack_node,
            probability_lookup=probability_lookup,
            support_lookup=support_lookup,
        )
        base = {
            "condition": condition,
            "task_id": str(case.task_id),
            "graph_id": str(case.graph_id),
            "attack_node": int(case.attack_node),
            "n": int(case.n),
            "m": int(case.m),
            "rho": int(case.m) / ((int(case.n) - 1) ** 2),
            "task_fold": task_fold,
            "actual_state": str(case.actual_state),
            "actual_state_index": int(case.actual_state_index),
            "actual_target": int(case.actual_target),
            "actual_correct": int(case.actual_correct),
        }
        if condition == "clean":
            base["round0_correct"] = int(case.round0_correct)
        readout = int(graph["readout_node"])
        model_probabilities = {
            "ctou_table": trajectory[-1][readout],
            "persistence": mean_field_rollout(
                graph=graph,
                initial_states=tuple(case.initial_states),
                attack_node=attack_node,
                model="persistence",
                lookup=None,
            ),
            "degroot_equal": mean_field_rollout(
                graph=graph,
                initial_states=tuple(case.initial_states),
                attack_node=attack_node,
                model="degroot_equal",
                lookup=None,
            ),
        }
        for model, probability in model_probabilities.items():
            endpoint_rows.append(
                {
                    **base,
                    "model": model,
                    **{
                        f"p_{state}": float(probability[index])
                        for index, state in enumerate(STATES)
                    },
                }
            )
        for row in support:
            support_rows.append({**base, **row})

        incoming, _ = graph_maps(graph)
        stored_attack = int(case.attack_node)
        for round_index in range(1, int(case.horizon) + 1):
            for receiver in range(int(case.n)):
                key = (
                    str(case.task_id),
                    str(case.graph_id),
                    stored_attack,
                    receiver,
                    round_index,
                )
                actual = actual_updates.get(key)
                if actual is None:
                    continue
                probability = trajectory[round_index][receiver]
                label = int(actual.current_state_index)
                one_hot = np.eye(len(STATES))[label]
                expected_counts = np.asarray(
                    [
                        trajectory[round_index - 1][incoming[receiver], state].sum()
                        if incoming[receiver]
                        else 0.0
                        for state in range(len(STATES))
                    ]
                )
                observed_counts = np.asarray(
                    [float(getattr(actual, column)) for column in COUNT_COLUMNS]
                )
                degree = int(observed_counts.sum())
                round_rows.append(
                    {
                        **base,
                        "round_index": round_index,
                        "receiver_node": receiver,
                        "receiver_scope": str(actual.receiver_scope),
                        "actual_state": str(actual.current_attack_state),
                        "actual_correct": int(str(actual.current_attack_state) == "correct"),
                        "actual_target": int(str(actual.current_attack_state) == "target"),
                        **{
                            f"p_{state}": float(probability[index])
                            for index, state in enumerate(STATES)
                        },
                        "state_brier": float(((probability - one_hot) ** 2).sum()),
                        "state_log_loss": float(-np.log(max(probability[label], EPSILON))),
                        "state_error": int(probability.argmax() != label),
                        "composition_count_mae": float(
                            np.abs(expected_counts - observed_counts).mean()
                        ),
                        "composition_tv": float(
                            np.abs(expected_counts - observed_counts).sum() / (2 * max(degree, 1))
                        ),
                    }
                )
    return {
        "endpoints": pd.DataFrame(endpoint_rows),
        "support": pd.DataFrame(support_rows),
        "rounds": pd.DataFrame(round_rows),
    }


def recursive_transfer(
    *,
    train_updates: pd.DataFrame,
    test_updates: pd.DataFrame,
    cases: pd.DataFrame,
    graphs: dict[str, dict[str, Any]],
    condition: str,
    folds: int,
    prior_strength: float,
    checkpoint_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    paths: list[Path] = []
    maximum_neighbors = max(TEST_SIZES) - 1
    horizon = int(cases.horizon.max())
    for task_fold in range(folds):
        path = checkpoint_root / f"{condition}_task_fold_{task_fold}.pkl"
        paths.append(path)
        train_fold = train_updates[train_updates.task_fold.ne(task_fold)]
        test_fold = test_updates[test_updates.task_fold.eq(task_fold)]
        test_cases = cases[cases.task_fold.eq(task_fold)]
        train_tasks = set(train_fold.task_id)
        test_tasks = set(test_cases.task_id)
        audit_rows.append(
            {
                "condition": condition,
                "task_fold": task_fold,
                "train_updates": len(train_fold),
                "test_updates": len(test_fold),
                "test_cases": len(test_cases),
                "train_tasks": len(train_tasks),
                "test_tasks": len(test_tasks),
                "task_overlap": len(train_tasks & test_tasks),
                "graph_overlap": len(set(train_fold.graph_id) & set(test_cases.graph_id)),
                "train_sizes": ",".join(map(str, sorted(train_fold.n.unique()))),
                "test_sizes": ",".join(map(str, sorted(test_cases.n.unique()))),
            }
        )
        if path.exists():
            continue
        result = rollout_fold(
            train_updates=train_fold,
            test_updates=test_fold,
            cases=test_cases,
            graphs=graphs,
            condition=condition,
            task_fold=task_fold,
            maximum_neighbors=maximum_neighbors,
            horizon=horizon,
            prior_strength=prior_strength,
        )
        temporary = path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        del result
        gc.collect()
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing recursive checkpoints: {missing}")
    values = []
    for path in paths:
        with path.open("rb") as handle:
            values.append(pickle.load(handle))
    return (
        pd.concat([x["endpoints"] for x in values], ignore_index=True),
        pd.concat([x["support"] for x in values], ignore_index=True),
        pd.concat([x["rounds"] for x in values], ignore_index=True),
        pd.DataFrame(audit_rows),
    )


def attach_endpoint_losses(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    probability = result[[f"p_{state}" for state in STATES]].to_numpy(float)
    labels = result.actual_state_index.to_numpy(int)
    one_hot = np.eye(len(STATES))[labels]
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    result["multiclass_brier"] = ((probability - one_hot) ** 2).sum(axis=1)
    result["multiclass_log_loss"] = -np.log(clipped[np.arange(len(result)), labels])
    for outcome in ("correct", "target"):
        observed = result[f"actual_{outcome}"].to_numpy(float)
        predicted = result[f"p_{outcome}"].to_numpy(float)
        result[f"{outcome}_absolute_error"] = np.abs(predicted - observed)
        result[f"{outcome}_brier"] = (predicted - observed) ** 2
    return result


def aggregate_one_step(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(
        ["condition", "model", "n", "round_index", "receiver_scope"],
        as_index=False,
    ).agg(
        updates=("actual_state", "size"),
        tasks=("task_id", "nunique"),
        multiclass_brier=("multiclass_brier", "mean"),
        multiclass_log_loss=("multiclass_log_loss", "mean"),
        classification_error=("classification_error", "mean"),
    )


def aggregate_realized_support(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "updates": ("exact_support", "size"),
        "exact_cell_coverage": ("exact_seen", "mean"),
        "composition_coverage": ("composition_seen", "mean"),
        "mean_log1p_exact_support": ("exact_log1p_support", "mean"),
    }
    for threshold in SUPPORT_THRESHOLDS:
        aggregations[f"exact_support_lt_{threshold}_mass"] = (
            f"exact_support_lt_{threshold}",
            "mean",
        )
    return frame.groupby(
        ["condition", "n", "m", "round_index", "receiver_scope"],
        as_index=False,
    ).agg(**aggregations)


def aggregate_expected_support(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "receiver_updates": ("expected_transition_visits", "size"),
        "expected_transition_visits": ("expected_transition_visits", "sum"),
        "expected_unseen_mass": ("expected_unseen_fraction", "mean"),
        "expected_mean_log1p_support": ("expected_mean_log1p_support", "mean"),
    }
    for threshold in SUPPORT_THRESHOLDS:
        aggregations[f"expected_support_lt_{threshold}_mass"] = (
            f"expected_support_lt_{threshold}_fraction",
            "mean",
        )
    return frame.groupby(
        ["condition", "n", "m", "round_index", "receiver_scope"],
        as_index=False,
    ).agg(**aggregations)


def aggregate_rounds(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["condition", "n", "round_index", "receiver_scope"], as_index=False)
        .agg(
            updates=("actual_state", "size"),
            tasks=("task_id", "nunique"),
            state_brier=("state_brier", "mean"),
            state_log_loss=("state_log_loss", "mean"),
            state_error=("state_error", "mean"),
            observed_correct=("actual_correct", "mean"),
            predicted_correct=("p_correct", "mean"),
            observed_target=("actual_target", "mean"),
            predicted_target=("p_target", "mean"),
            composition_count_mae=("composition_count_mae", "mean"),
            composition_tv=("composition_tv", "mean"),
        )
        .assign(
            correct_probability_error=lambda x: (x.predicted_correct - x.observed_correct).abs(),
            target_probability_error=lambda x: (x.predicted_target - x.observed_target).abs(),
        )
    )


def graph_and_curve_outputs(
    endpoints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = endpoints.groupby(
        ["condition", "model", "graph_id", "n", "m", "rho"], as_index=False
    ).agg(
        cases=("actual_correct", "size"),
        observed_correct=("actual_correct", "mean"),
        predicted_correct=("p_correct", "mean"),
        observed_target=("actual_target", "mean"),
        predicted_target=("p_target", "mean"),
        observed_round0_correct=("round0_correct", "mean"),
    )
    clean = grouped[grouped.condition.eq("clean")].set_index(["model", "graph_id", "n", "m", "rho"])
    attack = grouped[grouped.condition.eq("attack")].set_index(
        ["model", "graph_id", "n", "m", "rho"]
    )
    common = clean.index.intersection(attack.index)
    utility_robustness = pd.DataFrame(
        {
            "model": [x[0] for x in common],
            "graph_id": [x[1] for x in common],
            "n": [x[2] for x in common],
            "m": [x[3] for x in common],
            "rho": [x[4] for x in common],
            "observed_utility": clean.loc[common, "observed_correct"].to_numpy(),
            "predicted_utility": clean.loc[common, "predicted_correct"].to_numpy(),
            "observed_robustness": attack.loc[common, "observed_correct"].to_numpy(),
            "predicted_robustness": attack.loc[common, "predicted_correct"].to_numpy(),
            "observed_target_risk": attack.loc[common, "observed_target"].to_numpy(),
            "predicted_target_risk": attack.loc[common, "predicted_target"].to_numpy(),
            "observed_u0": clean.loc[common, "observed_round0_correct"].to_numpy(float),
        }
    )
    utility_robustness["predicted_u0"] = utility_robustness.observed_u0
    utility_robustness["observed_delta_utility"] = (
        utility_robustness.observed_utility - utility_robustness.observed_u0
    )
    utility_robustness["predicted_delta_utility"] = (
        utility_robustness.predicted_utility - utility_robustness.predicted_u0
    )
    utility_robustness["observed_attack_loss"] = (
        utility_robustness.observed_utility - utility_robustness.observed_robustness
    )
    utility_robustness["predicted_attack_loss"] = (
        utility_robustness.predicted_utility - utility_robustness.predicted_robustness
    )
    curves = utility_robustness.groupby(["model", "n", "m", "rho"], as_index=False).agg(
        graphs=("graph_id", "size"),
        observed_utility=("observed_utility", "mean"),
        predicted_utility=("predicted_utility", "mean"),
        observed_robustness=("observed_robustness", "mean"),
        predicted_robustness=("predicted_robustness", "mean"),
        observed_target_risk=("observed_target_risk", "mean"),
        predicted_target_risk=("predicted_target_risk", "mean"),
        observed_attack_loss=("observed_attack_loss", "mean"),
        predicted_attack_loss=("predicted_attack_loss", "mean"),
        observed_u0=("observed_u0", "mean"),
        predicted_u0=("predicted_u0", "mean"),
        observed_delta_utility=("observed_delta_utility", "mean"),
        predicted_delta_utility=("predicted_delta_utility", "mean"),
    )
    metric_rows: list[dict[str, object]] = []
    quantities = (
        "utility",
        "robustness",
        "target_risk",
        "attack_loss",
        "u0",
        "delta_utility",
    )
    for (model, n), group in utility_robustness.groupby(["model", "n"]):
        for quantity in quantities:
            observed = group[f"observed_{quantity}"].to_numpy(float)
            predicted = group[f"predicted_{quantity}"].to_numpy(float)
            metric_rows.append(
                {
                    "level": "graph",
                    "model": model,
                    "n": int(n),
                    "quantity": quantity,
                    "units": len(group),
                    "mae": float(np.abs(predicted - observed).mean()),
                    "spearman": float(spearmanr(observed, predicted).statistic),
                }
            )
    for (model, n), group in curves.groupby(["model", "n"]):
        for quantity in quantities:
            observed = group[f"observed_{quantity}"].to_numpy(float)
            predicted = group[f"predicted_{quantity}"].to_numpy(float)
            metric_rows.append(
                {
                    "level": "m_curve",
                    "model": model,
                    "n": int(n),
                    "quantity": quantity,
                    "units": len(group),
                    "mae": float(np.abs(predicted - observed).mean()),
                    "spearman": float(spearmanr(observed, predicted).statistic),
                }
            )
    return utility_robustness, curves, pd.DataFrame(metric_rows)


def observed_all_size_curves(
    train: dict[str, Any], test: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for source, bundle in (("old_n5_n8", train), ("new_n6_n7", test)):
        for condition in ("clean", "attack"):
            frame = bundle[f"{condition}_cases"].copy()
            frame["condition"] = condition
            frame["source_run"] = source
            rows.append(frame)
    cases = pd.concat(rows, ignore_index=True)
    graph = cases.groupby(["source_run", "condition", "graph_id", "n", "m"], as_index=False).agg(
        cases=("actual_correct", "size"),
        observed_correct=("actual_correct", "mean"),
        observed_target=("actual_target", "mean"),
        observed_u0=("round0_correct", "mean"),
    )
    graph["rho"] = graph.m / ((graph.n - 1) ** 2)
    clean = graph[graph.condition.eq("clean")].set_index(
        ["source_run", "graph_id", "n", "m", "rho"]
    )
    attack = graph[graph.condition.eq("attack")].set_index(
        ["source_run", "graph_id", "n", "m", "rho"]
    )
    common = clean.index.intersection(attack.index)
    combined = pd.DataFrame(
        {
            "source_run": [x[0] for x in common],
            "graph_id": [x[1] for x in common],
            "n": [x[2] for x in common],
            "m": [x[3] for x in common],
            "rho": [x[4] for x in common],
            "utility": clean.loc[common, "observed_correct"].to_numpy(float),
            "u0": clean.loc[common, "observed_u0"].to_numpy(float),
            "robustness": attack.loc[common, "observed_correct"].to_numpy(float),
            "target_risk": attack.loc[common, "observed_target"].to_numpy(float),
        }
    )
    combined["delta_utility"] = combined.utility - combined.u0
    combined["attack_loss"] = combined.utility - combined.robustness
    curves = combined.groupby(["source_run", "n", "m", "rho"], as_index=False).agg(
        graphs=("graph_id", "size"),
        utility=("utility", "mean"),
        utility_std=("utility", "std"),
        u0=("u0", "mean"),
        delta_utility=("delta_utility", "mean"),
        robustness=("robustness", "mean"),
        robustness_std=("robustness", "std"),
        target_risk=("target_risk", "mean"),
        attack_loss=("attack_loss", "mean"),
    )
    return combined, curves


def same_cell_scale_stability(
    train: dict[str, Any], test: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for bundle in (train, test):
        for condition in ("clean", "attack"):
            frame = bundle[f"{condition}_updates"].copy()
            frame["condition"] = condition
            parts.append(frame)
    updates = pd.concat(parts, ignore_index=True)
    keys = ["condition", "n", *TABLE_KEYS]
    counts = (
        updates.groupby([*keys, "current_attack_state"], as_index=False, observed=True)
        .size()
        .rename(columns={"size": "outcome_count"})
    )
    totals = (
        counts.groupby(keys, as_index=False, observed=True)
        .outcome_count.sum()
        .rename(columns={"outcome_count": "cell_support"})
    )
    cells = counts.merge(totals, on=keys, validate="many_to_one")
    cells["probability"] = cells.outcome_count / cells.cell_support
    pivot = cells.pivot_table(
        index=["condition", *TABLE_KEYS],
        columns=["n", "current_attack_state"],
        values="probability",
        fill_value=0.0,
    )
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_product([(5, 6, 7, 8), STATES], names=pivot.columns.names),
        fill_value=0.0,
    )
    support = totals.pivot_table(
        index=["condition", *TABLE_KEYS],
        columns="n",
        values="cell_support",
        fill_value=0,
    )
    pair_rows: list[dict[str, object]] = []
    for condition in ("clean", "attack"):
        condition_index = support.index[support.index.get_level_values("condition") == condition]
        for left in (5, 6, 7, 8):
            for right in range(left + 1, 9):
                if right not in (5, 6, 7, 8):
                    continue
                shared = condition_index[
                    support.loc[condition_index, left].gt(0)
                    & support.loc[condition_index, right].gt(0)
                ]
                for minimum_support in (1, 5, 20, 100):
                    eligible = shared[
                        (support.loc[shared, left] >= minimum_support)
                        & (support.loc[shared, right] >= minimum_support)
                    ]
                    if len(eligible) == 0:
                        continue
                    left_probability = pivot.loc[
                        eligible, [(left, state) for state in STATES]
                    ].to_numpy(float)
                    right_probability = pivot.loc[
                        eligible, [(right, state) for state in STATES]
                    ].to_numpy(float)
                    tv = 0.5 * np.abs(left_probability - right_probability).sum(axis=1)
                    weights = np.minimum(
                        support.loc[eligible, left].to_numpy(float),
                        support.loc[eligible, right].to_numpy(float),
                    )
                    pair_rows.append(
                        {
                            "condition": condition,
                            "n_left": left,
                            "n_right": right,
                            "minimum_support_each": minimum_support,
                            "shared_cells": len(eligible),
                            "weighted_mean_tv": float(np.average(tv, weights=weights)),
                            "median_tv": float(np.median(tv)),
                            "p90_tv": float(np.quantile(tv, 0.9)),
                        }
                    )
    return cells, pd.DataFrame(pair_rows)


def support_error_association(
    expected_support: pd.DataFrame, endpoints: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["condition", "task_id", "graph_id", "attack_node", "n", "m"]
    support_columns = [
        "expected_unseen_fraction",
        "expected_mean_log1p_support",
        *[f"expected_support_lt_{threshold}_fraction" for threshold in SUPPORT_THRESHOLDS],
    ]
    by_case = expected_support.groupby(keys, as_index=False)[support_columns].mean()
    ctou = endpoints[endpoints.model.eq("ctou_table")][
        [*keys, "correct_absolute_error", "target_absolute_error"]
    ]
    merged = by_case.merge(ctou, on=keys, how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for (condition, n), group in merged.groupby(["condition", "n"]):
        for support_metric in support_columns:
            for error_metric in ("correct_absolute_error", "target_absolute_error"):
                statistic = spearmanr(
                    group[support_metric].to_numpy(float),
                    group[error_metric].to_numpy(float),
                ).statistic
                rows.append(
                    {
                        "condition": condition,
                        "n": int(n),
                        "support_metric": support_metric,
                        "error_metric": error_metric,
                        "cases": len(group),
                        "spearman": float(statistic),
                    }
                )
    return merged, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("folds must be at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("stage=load_normalized_inputs", flush=True)
    cache_root = args.output_dir / "normalized-cache"
    cache_root.mkdir(exist_ok=True)
    train = load_or_cache_inputs(
        run_root=args.train_run_root,
        cache_root=cache_root,
        label="train_n5_n8",
        folds=args.folds,
    )
    test = load_or_cache_inputs(
        run_root=args.test_run_root,
        cache_root=cache_root,
        label="test_n6_n7",
        folds=args.folds,
    )
    boundary_audit = validate_boundary(train, test)
    atomic_json(args.output_dir / "boundary_audit.json", boundary_audit)
    print("stage=boundary_audit_complete", flush=True)

    one_step_checkpoint = args.output_dir / "one_step_stage.pkl"
    if one_step_checkpoint.exists():
        with one_step_checkpoint.open("rb") as handle:
            one_step, realized_support, one_step_audit = pickle.load(handle)
        print("stage=one_step_loaded", flush=True)
    else:
        one_step_parts = []
        realized_support_parts = []
        one_step_audits = []
        for condition in ("attack", "clean"):
            print(f"stage=one_step_{condition}", flush=True)
            one_step_part, support, audit = cross_size_one_step(
                train=train[f"{condition}_updates"],
                test=test[f"{condition}_updates"],
                condition=condition,
                folds=args.folds,
                prior_strength=args.table_prior_strength,
            )
            one_step_parts.append(one_step_part)
            realized_support_parts.append(support)
            one_step_audits.append(audit)
        one_step = pd.concat(one_step_parts, ignore_index=True)
        realized_support = pd.concat(realized_support_parts, ignore_index=True)
        one_step_audit = pd.concat(one_step_audits, ignore_index=True)
        atomic_pickle(one_step_checkpoint, (one_step, realized_support, one_step_audit))
        print("stage=one_step_complete", flush=True)
    if (one_step_audit.task_overlap != 0).any() or (one_step_audit.graph_overlap != 0).any():
        raise RuntimeError("one-step transfer leakage detected")

    recursive_outputs = []
    recursive_supports = []
    recursive_rounds = []
    recursive_audits = []
    for condition in ("attack", "clean"):
        print(f"stage=recursive_{condition}", flush=True)
        endpoints, support, rounds, audit = recursive_transfer(
            train_updates=train[f"{condition}_updates"],
            test_updates=test[f"{condition}_updates"],
            cases=test[f"{condition}_cases"],
            graphs=test[f"{condition}_graphs"],
            condition=condition,
            folds=args.folds,
            prior_strength=args.table_prior_strength,
            checkpoint_root=args.output_dir / "recursive-fold-checkpoints",
        )
        recursive_outputs.append(endpoints)
        recursive_supports.append(support)
        recursive_rounds.append(rounds)
        recursive_audits.append(audit)
    recursive_audit = pd.concat(recursive_audits, ignore_index=True)
    if (recursive_audit.task_overlap != 0).any() or (recursive_audit.graph_overlap != 0).any():
        raise RuntimeError("recursive transfer leakage detected")
    endpoints = attach_endpoint_losses(pd.concat(recursive_outputs, ignore_index=True))
    expected_support = pd.concat(recursive_supports, ignore_index=True)
    round_predictions = pd.concat(recursive_rounds, ignore_index=True)
    print("stage=recursive_complete", flush=True)

    one_step_summary = aggregate_one_step(one_step)
    realized_support_summary = aggregate_realized_support(realized_support)
    round_summary = aggregate_rounds(round_predictions)
    expected_support_summary = aggregate_expected_support(expected_support)
    utility_robustness, curves, graph_metrics = graph_and_curve_outputs(endpoints)
    print("stage=primary_aggregates_complete", flush=True)
    support_error_rows, support_error_metrics = support_error_association(
        expected_support, endpoints
    )

    one_step.to_csv(args.output_dir / "one_step_predictions.csv.gz", index=False)
    one_step_summary.to_csv(args.output_dir / "one_step_summary.csv", index=False)
    one_step_audit.to_csv(args.output_dir / "one_step_fold_audit.csv", index=False)
    realized_support.to_csv(args.output_dir / "realized_support_rows.csv.gz", index=False)
    realized_support_summary.to_csv(args.output_dir / "realized_support_summary.csv", index=False)
    endpoints.to_csv(args.output_dir / "endpoint_predictions.csv.gz", index=False)
    expected_support.to_csv(args.output_dir / "recursive_expected_support.csv.gz", index=False)
    expected_support_summary.to_csv(
        args.output_dir / "recursive_expected_support_summary.csv", index=False
    )
    round_predictions.to_csv(args.output_dir / "round_predictions.csv.gz", index=False)
    round_summary.to_csv(args.output_dir / "round_error_summary.csv", index=False)
    recursive_audit.to_csv(args.output_dir / "recursive_fold_audit.csv", index=False)
    utility_robustness.to_csv(args.output_dir / "graph_utility_robustness.csv", index=False)
    curves.to_csv(args.output_dir / "density_curves.csv", index=False)
    graph_metrics.to_csv(args.output_dir / "graph_and_curve_metrics.csv", index=False)
    support_error_rows.to_csv(args.output_dir / "support_error_rows.csv.gz", index=False)
    support_error_metrics.to_csv(args.output_dir / "support_error_metrics.csv", index=False)
    print("stage=primary_csv_outputs_complete", flush=True)

    del (
        one_step,
        realized_support,
        endpoints,
        expected_support,
        round_predictions,
        recursive_outputs,
        recursive_supports,
        recursive_rounds,
        support_error_rows,
    )
    gc.collect()

    all_size_graphs, all_size_curves = observed_all_size_curves(train, test)
    all_size_graphs.to_csv(args.output_dir / "observed_all_size_graphs.csv", index=False)
    all_size_curves.to_csv(args.output_dir / "observed_all_size_density_curves.csv", index=False)
    print("stage=all_size_curves_complete", flush=True)
    del all_size_graphs, all_size_curves
    gc.collect()

    cell_probabilities, cell_scale_metrics = same_cell_scale_stability(train, test)
    cell_probabilities.to_csv(args.output_dir / "same_cell_scale_probabilities.csv.gz", index=False)
    cell_scale_metrics.to_csv(args.output_dir / "same_cell_scale_metrics.csv", index=False)
    print("stage=same_cell_complete", flush=True)

    manifest = {
        "analysis_version": "ctou-scale-transfer-v1",
        "train_run_root": str(args.train_run_root.resolve()),
        "test_run_root": str(args.test_run_root.resolve()),
        "train_sizes": list(TRAIN_SIZES),
        "test_sizes": list(TEST_SIZES),
        "folds": args.folds,
        "table_prior_strength": args.table_prior_strength,
        "primary_model": "ctou_table",
        "primary_rollout": "factorized_mean_field",
        "initialization": "observed categorical Round-0 state vector",
        "task_holdout": (
            "for test task fold t, fit only n=5/8 updates whose task fold is not t; "
            "all n=6/7 graphs remain unseen"
        ),
        "information_boundary": (
            "recursive prediction sees graph, schedule, readout, attack node, and true "
            "Round-0 C/T/O/U states; true Round-1+ states/compositions are diagnostic only"
        ),
        "boundary_audit": boundary_audit,
        "claim_limits": [
            "results are conditional on Llama-3.1-8B, GSM8K, T=3, and the persistent target attack",
            "true Round-0 states are observed, so this is not a topology-only evaluator",
            "one-step evaluation uses realized post-treatment composition and is diagnostic",
            (
                "mean-field discards residual joint dependence; prior work found a "
                "small in-support gap"
            ),
            "support association does not establish that support causes prediction error",
            "successful n=6/7 transfer supports only the tested n=5..8 range",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(manifest_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
