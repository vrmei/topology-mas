"""Evaluate recursively generated target provenance in CTOU particle rollout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_ctou_recursive_rollout import (
    aggregate_curves,
    aggregate_graphs,
    attach_losses,
    fit_fold_lookups,
    graph_maps,
    sample_categories,
)
from analyze_ctou_transition_prediction import STATE_INDEX, STATES, stable_fold
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression


P_STATES = ("correct", "target_natural", "target_attack", "other", "unparsed")
P_STATE_INDEX = {state: index for index, state in enumerate(P_STATES)}
P_COUNT_COLUMNS = (
    "incoming_correct_count",
    "direct_target_count",
    "relayed_target_count",
    "natural_target_count",
    "incoming_other_count",
    "incoming_unparsed_count",
)
P_MODELS = ("provenance_table", "provenance_logit")
BASELINE_MODELS = ("ctou_table", "ctou_logit")
DEFAULT_SEED = 20_260_815
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance-updates", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--particles", type=int, default=2_048)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--table-prior-strength", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def stable_seed(*parts: object, seed: int) -> int:
    text = "\x1f".join(str(part) for part in (*parts, seed))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def load_provenance_updates(path: Path, folds: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {
        "previous_provenance_state",
        "next_provenance_state",
        *P_COUNT_COLUMNS,
        "task_id",
        "graph_id",
        "receiver_scope",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"provenance updates are missing columns: {missing}")
    frame["graph_fold"] = frame.graph_id.map(lambda value: stable_fold(str(value), folds))
    frame["task_fold"] = frame.task_id.map(lambda value: stable_fold(str(value), folds))
    frame["current_provenance_index"] = frame.next_provenance_state.map(P_STATE_INDEX)
    frame["previous_attack_state"] = frame.previous_state
    frame["current_attack_state"] = frame.next_state
    frame["current_state_index"] = frame.next_state.map(STATE_INDEX)
    if frame.current_provenance_index.isna().any() or frame.current_state_index.isna().any():
        raise ValueError("unknown state in provenance updates")
    target_sum = frame[["direct_target_count", "relayed_target_count", "natural_target_count"]].sum(axis=1)
    if not target_sum.equals(frame.incoming_target_count):
        raise ValueError("target provenance counts do not reconstruct total target count")
    duplicate_keys = int(
        frame.duplicated(
            ["task_id", "graph_id", "attack_node", "receiver_node", "round_index"]
        ).sum()
    )
    audit = {
        "passed": duplicate_keys == 0,
        "updates": len(frame),
        "tasks": int(frame.task_id.nunique()),
        "graphs": int(frame.graph_id.nunique()),
        "duplicate_keys": duplicate_keys,
    }
    return frame, audit


def count_compositions(dimensions: int, maximum: int) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], remaining_dimensions: int, remaining: int) -> None:
        if remaining_dimensions == 1:
            rows.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            visit((*prefix, value), remaining_dimensions - 1, remaining - value)

    for total in range(maximum + 1):
        visit((), dimensions, total)
    return rows


def provenance_query(maximum_neighbors: int, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    compositions = count_compositions(len(P_COUNT_COLUMNS), maximum_neighbors)
    for round_index, previous, counts in itertools.product(
        range(1, horizon + 1), P_STATES, compositions
    ):
        row = {"round_index": round_index, "previous_provenance_state": previous}
        row.update(dict(zip(P_COUNT_COLUMNS, counts, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def provenance_design_matrix(frame: pd.DataFrame) -> np.ndarray:
    previous = frame.previous_provenance_state.map(P_STATE_INDEX).to_numpy(int)
    rounds = frame.round_index.to_numpy(int)
    previous_one_hot = np.eye(len(P_STATES))[previous]
    round_one_hot = np.eye(4)[rounds]
    counts = frame[list(P_COUNT_COLUMNS)].to_numpy(float)
    total = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    return np.column_stack((previous_one_hot, round_one_hot, counts, fractions))


def provenance_table_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prior_strength: float,
) -> np.ndarray:
    table_keys = ("previous_provenance_state", "round_index", *P_COUNT_COLUMNS)
    base_counts = (
        train.groupby(["previous_provenance_state", "round_index", "next_provenance_state"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=P_STATES, fill_value=0)
    )
    cell_counts = (
        train.groupby([*table_keys, "next_provenance_state"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=P_STATES, fill_value=0)
    )
    base_lookup = {key: value.to_numpy(float) for key, value in base_counts.iterrows()}
    cell_lookup = {key: value.to_numpy(float) for key, value in cell_counts.iterrows()}
    global_counts = train.next_provenance_state.value_counts().reindex(P_STATES, fill_value=0)
    global_prior = (global_counts.to_numpy(float) + 1) / (global_counts.sum() + len(P_STATES))
    output = np.zeros((len(test), len(P_STATES)), dtype=np.float32)
    for index, row in enumerate(test.itertuples(index=False)):
        base_key = (str(row.previous_provenance_state), int(row.round_index))
        base = base_lookup.get(base_key)
        prior = global_prior if base is None else (base + 1) / (base.sum() + len(P_STATES))
        cell_key = (
            str(row.previous_provenance_state),
            int(row.round_index),
            *(int(getattr(row, column)) for column in P_COUNT_COLUMNS),
        )
        cell = cell_lookup.get(cell_key)
        output[index] = prior if cell is None else (cell + prior_strength * prior) / (
            cell.sum() + prior_strength
        )
    return output


def dense_provenance_lookup(
    query: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    horizon: int,
    maximum_neighbors: int,
) -> np.ndarray:
    lookup = np.full(
        (
            horizon + 1,
            len(P_STATES),
            *(maximum_neighbors + 1 for _ in P_COUNT_COLUMNS),
            len(P_STATES),
        ),
        np.nan,
        dtype=np.float32,
    )
    for position, row in enumerate(query.itertuples(index=False)):
        lookup[
            (
                int(row.round_index),
                P_STATE_INDEX[str(row.previous_provenance_state)],
                *(int(getattr(row, column)) for column in P_COUNT_COLUMNS),
            )
        ] = probabilities[position]
    return lookup


def fit_provenance_lookups(
    train: pd.DataFrame,
    *,
    maximum_neighbors: int,
    horizon: int,
    prior_strength: float,
) -> dict[str, np.ndarray]:
    query = provenance_query(maximum_neighbors, horizon)
    table = provenance_table_predictions(train, query, prior_strength)
    logistic = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=0)
    logistic.fit(provenance_design_matrix(train), train.current_provenance_index.to_numpy(int))
    raw = logistic.predict_proba(provenance_design_matrix(query))
    aligned = np.zeros((len(query), len(P_STATES)), dtype=np.float32)
    aligned[:, logistic.classes_.astype(int)] = raw
    return {
        "provenance_table": dense_provenance_lookup(
            query, table, horizon=horizon, maximum_neighbors=maximum_neighbors
        ),
        "provenance_logit": dense_provenance_lookup(
            query, aligned, horizon=horizon, maximum_neighbors=maximum_neighbors
        ),
    }


def collapse_provenance(probability: np.ndarray) -> np.ndarray:
    output = np.zeros((*probability.shape[:-1], len(STATES)), dtype=float)
    output[..., STATE_INDEX["correct"]] = probability[..., P_STATE_INDEX["correct"]]
    output[..., STATE_INDEX["target"]] = (
        probability[..., P_STATE_INDEX["target_natural"]]
        + probability[..., P_STATE_INDEX["target_attack"]]
    )
    output[..., STATE_INDEX["other"]] = probability[..., P_STATE_INDEX["other"]]
    output[..., STATE_INDEX["unparsed"]] = probability[..., P_STATE_INDEX["unparsed"]]
    return output


def initialize_provenance_states(
    initial_states: tuple[int, ...], attack_node: int
) -> tuple[int, ...]:
    output: list[int] = []
    for node, state in enumerate(initial_states):
        name = STATES[int(state)]
        if node == attack_node:
            output.append(P_STATE_INDEX["target_attack"])
        elif name == "target":
            output.append(P_STATE_INDEX["target_natural"])
        else:
            output.append(P_STATE_INDEX[name])
    return tuple(output)


def distances_to_readout(graph: dict[str, Any]) -> list[int]:
    _, outgoing = graph_maps(graph)
    readout = int(graph["readout_node"])
    result = []
    for source in range(len(outgoing)):
        queue = [(source, 0)]
        seen = {source}
        found = None
        for node, distance in queue:
            if node == readout:
                found = distance
                break
            for target in outgoing[node]:
                if target not in seen:
                    seen.add(target)
                    queue.append((target, distance + 1))
        if found is None:
            raise ValueError(f"node {source} cannot reach readout")
        result.append(found)
    return result


def provenance_particle_rollout(
    *,
    graph: dict[str, Any],
    initial_states: tuple[int, ...],
    attack_node: int,
    lookup: np.ndarray,
    particles: int,
    seed: int,
) -> np.ndarray:
    n = int(graph["node_count"])
    horizon = int(graph["max_rounds"])
    readout = int(graph["readout_node"])
    incoming, _ = graph_maps(graph)
    distances = distances_to_readout(graph)
    initial = initialize_provenance_states(initial_states, attack_node)
    states = np.tile(np.asarray(initial, dtype=np.int8), (particles, 1))
    rng = np.random.default_rng(seed)
    for round_index in range(1, horizon + 1):
        updated = states.copy()
        for node in range(n):
            if round_index + distances[node] > horizon:
                continue
            if node == attack_node:
                updated[:, node] = P_STATE_INDEX["target_attack"]
                continue
            sources = incoming[node]
            source_states = states[:, sources] if sources else np.empty((particles, 0), dtype=np.int8)
            direct = (
                (source_states == P_STATE_INDEX["target_attack"])
                & (np.asarray(sources)[None, :] == attack_node)
                if sources
                else np.zeros((particles, 0), dtype=bool)
            )
            relayed = (
                (source_states == P_STATE_INDEX["target_attack"])
                & (np.asarray(sources)[None, :] != attack_node)
                if sources
                else np.zeros((particles, 0), dtype=bool)
            )
            counts = np.column_stack(
                (
                    (source_states == P_STATE_INDEX["correct"]).sum(axis=1),
                    direct.sum(axis=1),
                    relayed.sum(axis=1),
                    (source_states == P_STATE_INDEX["target_natural"]).sum(axis=1),
                    (source_states == P_STATE_INDEX["other"]).sum(axis=1),
                    (source_states == P_STATE_INDEX["unparsed"]).sum(axis=1),
                )
            )
            probability = lookup[
                round_index,
                states[:, node],
                counts[:, 0],
                counts[:, 1],
                counts[:, 2],
                counts[:, 3],
                counts[:, 4],
                counts[:, 5],
            ]
            if np.isnan(probability).any():
                raise RuntimeError("missing provenance transition probability")
            updated[:, node] = sample_categories(probability, rng.random(particles))
        states = updated
    extended = np.bincount(states[:, readout], minlength=len(P_STATES)) / particles
    return collapse_provenance(extended)


def prepare_cases(
    baseline_dir: Path,
    updates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    cases = pd.read_pickle(baseline_dir / "normalized_rollout_cases.pkl")
    graphs = json.loads((baseline_dir / "normalized_graphs.json").read_text(encoding="utf-8"))
    final = (
        updates.loc[updates.receiver_scope.eq("readout")]
        .sort_values("round_index")
        .groupby(["task_id", "graph_id", "attack_node"], as_index=False)
        .tail(1)[["task_id", "graph_id", "attack_node", "next_provenance_state"]]
    )
    cases = cases.merge(
        final,
        on=["task_id", "graph_id", "attack_node"],
        how="left",
        validate="one_to_one",
    )
    if cases.next_provenance_state.isna().any():
        raise ValueError("missing final provenance state")
    cases["actual_provenance_index"] = cases.next_provenance_state.map(P_STATE_INDEX)
    return cases, graphs


def lookup_predictions(
    frame: pd.DataFrame,
    lookup: np.ndarray,
    *,
    provenance: bool,
) -> np.ndarray:
    if provenance:
        previous = frame.previous_provenance_state.map(P_STATE_INDEX).to_numpy(int)
        counts = frame[list(P_COUNT_COLUMNS)].to_numpy(int)
    else:
        previous = frame.previous_attack_state.map(STATE_INDEX).to_numpy(int)
        counts = frame[
            [
                "incoming_correct_count",
                "incoming_target_count",
                "incoming_other_count",
                "incoming_unparsed_count",
            ]
        ].to_numpy(int)
    indices: tuple[np.ndarray, ...] = (
        frame.round_index.to_numpy(int),
        previous,
        *(counts[:, index] for index in range(counts.shape[1])),
    )
    return lookup[indices]


def loss_columns(probability: np.ndarray, labels: np.ndarray, prefix: str = "") -> dict[str, np.ndarray]:
    clipped = np.clip(probability, EPSILON, 1 - EPSILON)
    one_hot = np.eye(probability.shape[1])[labels]
    return {
        f"{prefix}multiclass_brier": ((probability - one_hot) ** 2).sum(axis=1),
        f"{prefix}multiclass_log_loss": -np.log(clipped[np.arange(len(labels)), labels]),
    }


def execute_crossed_folds(
    *,
    cases: pd.DataFrame,
    updates: pd.DataFrame,
    graphs: dict[str, dict[str, Any]],
    folds: int,
    particles: int,
    prior_strength: float,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    endpoint_paths: list[Path] = []
    one_step_paths: list[Path] = []
    audit_rows: list[dict[str, Any]] = []
    maximum_neighbors = int(cases.n.max() - 1)
    horizon = int(cases.horizon.max())
    for graph_fold in range(folds):
        for task_fold in range(folds):
            stem = f"graph-{graph_fold}_task-{task_fold}"
            endpoint_path = checkpoint_dir / f"{stem}-endpoint.csv"
            one_step_path = checkpoint_dir / f"{stem}-one-step.csv"
            endpoint_paths.append(endpoint_path)
            one_step_paths.append(one_step_path)
            train = updates[updates.graph_fold.ne(graph_fold) & updates.task_fold.ne(task_fold)]
            test_updates = updates[
                updates.graph_fold.eq(graph_fold) & updates.task_fold.eq(task_fold)
            ]
            test_cases = cases[cases.graph_fold.eq(graph_fold) & cases.task_fold.eq(task_fold)]
            if test_cases.empty:
                continue
            audit_rows.append(
                {
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "train_updates": len(train),
                    "test_updates": len(test_updates),
                    "test_cases": len(test_cases),
                    "graph_overlap": len(set(train.graph_id) & set(test_cases.graph_id)),
                    "task_overlap": len(set(train.task_id) & set(test_cases.task_id)),
                }
            )
            if endpoint_path.exists() and one_step_path.exists():
                continue
            print(
                f"fold graph={graph_fold} task={task_fold} train={len(train)} "
                f"updates={len(test_updates)} cases={len(test_cases)}",
                flush=True,
            )
            p_lookups = fit_provenance_lookups(
                train,
                maximum_neighbors=maximum_neighbors,
                horizon=horizon,
                prior_strength=prior_strength,
            )
            base_lookups = fit_fold_lookups(
                train,
                maximum_neighbors=maximum_neighbors,
                horizon=horizon,
                table_prior_strength=prior_strength,
            )

            task_metrics: list[pd.DataFrame] = []
            for model in (*BASELINE_MODELS, *P_MODELS):
                provenance = model in P_MODELS
                probability = lookup_predictions(
                    test_updates,
                    p_lookups[model] if provenance else base_lookups[model],
                    provenance=provenance,
                )
                collapsed = collapse_provenance(probability) if provenance else probability
                metrics = pd.DataFrame(
                    {
                        "task_id": test_updates.task_id.to_numpy(),
                        "model": model,
                        **loss_columns(
                            collapsed,
                            test_updates.current_state_index.to_numpy(int),
                        ),
                    }
                )
                target_probability = collapsed[:, STATE_INDEX["target"]]
                target_label = test_updates.next_state.eq("target").to_numpy(float)
                metrics["target_brier"] = (target_probability - target_label) ** 2
                metrics["target_log_loss"] = -(
                    target_label * np.log(np.clip(target_probability, EPSILON, 1 - EPSILON))
                    + (1 - target_label)
                    * np.log(np.clip(1 - target_probability, EPSILON, 1 - EPSILON))
                )
                if provenance:
                    extended = loss_columns(
                        probability,
                        test_updates.current_provenance_index.to_numpy(int),
                        prefix="extended_",
                    )
                    for column, values in extended.items():
                        metrics[column] = values
                task_metrics.append(metrics)
            one_step = pd.concat(task_metrics, ignore_index=True)
            one_step = (
                one_step.groupby(["model", "task_id"], as_index=False)
                .agg(
                    rows=("multiclass_brier", "size"),
                    multiclass_brier=("multiclass_brier", "mean"),
                    multiclass_log_loss=("multiclass_log_loss", "mean"),
                    target_brier=("target_brier", "mean"),
                    target_log_loss=("target_log_loss", "mean"),
                    extended_multiclass_brier=("extended_multiclass_brier", "mean"),
                    extended_multiclass_log_loss=("extended_multiclass_log_loss", "mean"),
                )
            )
            one_step.to_csv(one_step_path.with_suffix(".csv.tmp"), index=False)
            one_step_path.with_suffix(".csv.tmp").replace(one_step_path)

            unique = test_cases.drop_duplicates(
                ["graph_id", "task_fold", "attack_node", "initial_states"]
            )
            cache: dict[tuple[Any, ...], np.ndarray] = {}
            for case in unique.itertuples(index=False):
                graph = graphs[case.graph_id]
                rollout_seed = stable_seed(
                    case.graph_id,
                    case.task_fold,
                    case.attack_node,
                    case.initial_states,
                    seed=seed,
                )
                for model in P_MODELS:
                    cache[(model, case.graph_id, case.attack_node, case.initial_states)] = (
                        provenance_particle_rollout(
                            graph=graph,
                            initial_states=case.initial_states,
                            attack_node=case.attack_node,
                            lookup=p_lookups[model],
                            particles=particles,
                            seed=rollout_seed,
                        )
                    )
            rows: list[dict[str, Any]] = []
            for case in test_cases.itertuples(index=False):
                for model in P_MODELS:
                    probability = cache[(model, case.graph_id, case.attack_node, case.initial_states)]
                    rows.append(
                        {
                            "stratum": case.stratum,
                            "task_id": case.task_id,
                            "graph_id": case.graph_id,
                            "attack_node": case.attack_node,
                            "n": case.n,
                            "m": case.m,
                            "graph_fold": graph_fold,
                            "task_fold": task_fold,
                            "actual_state": case.actual_state,
                            "actual_state_index": case.actual_state_index,
                            "actual_target": case.actual_target,
                            "actual_correct": case.actual_correct,
                            "model": model,
                            "rollout_mode": "particle",
                            **{
                                f"p_{state}": float(probability[index])
                                for index, state in enumerate(STATES)
                            },
                        }
                    )
            pd.DataFrame(rows).to_csv(endpoint_path.with_suffix(".csv.tmp"), index=False)
            endpoint_path.with_suffix(".csv.tmp").replace(endpoint_path)
            del train, test_updates, test_cases, p_lookups, base_lookups, cache, rows, task_metrics
            gc.collect()
    endpoints = pd.concat((pd.read_csv(path) for path in endpoint_paths), ignore_index=True)
    one_step = pd.concat((pd.read_csv(path) for path in one_step_paths), ignore_index=True)
    return endpoints, one_step, pd.DataFrame(audit_rows)


def bootstrap_task_metrics(
    task_metrics: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "multiclass_brier",
        "multiclass_log_loss",
        "target_brier",
        "target_log_loss",
        "extended_multiclass_brier",
        "extended_multiclass_log_loss",
    ]
    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    for model, group in task_metrics.groupby("model"):
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            samples = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
            summaries.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": values.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "tasks": len(values),
                }
            )
    comparisons: list[dict[str, Any]] = []
    for candidate, reference in zip(P_MODELS, BASELINE_MODELS, strict=True):
        left = task_metrics[task_metrics.model.eq(candidate)].set_index("task_id")
        right = task_metrics[task_metrics.model.eq(reference)].set_index("task_id")
        common = sorted(set(left.index) & set(right.index))
        for metric in metric_columns[:4]:
            differences = left.loc[common, metric].to_numpy(float) - right.loc[common, metric].to_numpy(float)
            samples = differences[
                rng.integers(0, len(differences), size=(replicates, len(differences)))
            ].mean(axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
            comparisons.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "loss_difference": differences.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "negative_favors_candidate": True,
                    "tasks": len(common),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def slope_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, n), group in curves.groupby(["model", "n"]):
        for outcome in ("target", "correct"):
            observed = np.polyfit(group.m, group[f"observed_{outcome}"], 1)[0]
            predicted = np.polyfit(group.m, group[f"predicted_{outcome}"], 1)[0]
            rows.append(
                {
                    "model": model,
                    "n": n,
                    "outcome": outcome,
                    "m_levels": len(group),
                    "observed_slope_per_edge": observed,
                    "predicted_slope_per_edge": predicted,
                    "slope_error": predicted - observed,
                    "absolute_slope_error": abs(predicted - observed),
                    "sign_agreement": bool(np.sign(predicted) == np.sign(observed)),
                    "curve_spearman": spearmanr(
                        group[f"observed_{outcome}"], group[f"predicted_{outcome}"]
                    ).statistic,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.particles < 256:
        raise ValueError("particles must be at least 256")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    updates, update_audit = load_provenance_updates(args.provenance_updates, args.folds)
    cases, graphs = prepare_cases(args.baseline_dir, updates)
    endpoints, one_step, folds = execute_crossed_folds(
        cases=cases,
        updates=updates,
        graphs=graphs,
        folds=args.folds,
        particles=args.particles,
        prior_strength=args.table_prior_strength,
        seed=args.seed,
        checkpoint_dir=args.output_dir / "fold-checkpoints",
    )
    if (folds.graph_overlap != 0).any() or (folds.task_overlap != 0).any():
        raise RuntimeError("crossed holdout leakage detected")
    baseline = pd.read_csv(args.baseline_dir / "endpoint_predictions.csv")
    baseline = baseline[
        baseline.model.isin(BASELINE_MODELS) & baseline.rollout_mode.eq("particle")
    ]
    combined = pd.concat([baseline, endpoints], ignore_index=True)
    scored = attach_losses(combined)
    one_step_summary, one_step_comparisons = bootstrap_task_metrics(
        one_step,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    endpoint_task = (
        scored.groupby(["model", "task_id"], as_index=False)
        .agg(
            rows=("target_brier", "size"),
            multiclass_brier=("multiclass_brier", "mean"),
            multiclass_log_loss=("multiclass_log_loss", "mean"),
            target_brier=("target_brier", "mean"),
            target_log_loss=("target_log_loss", "mean"),
        )
    )
    endpoint_summary, endpoint_comparisons = bootstrap_task_metrics(
        endpoint_task.assign(
            extended_multiclass_brier=np.nan,
            extended_multiclass_log_loss=np.nan,
        ),
        replicates=args.bootstrap_replicates,
        seed=args.seed + 1,
    )
    curves, curve_metrics = aggregate_curves(scored)
    graph_predictions, graph_metrics = aggregate_graphs(
        scored,
        replicates=args.bootstrap_replicates,
        rng=np.random.default_rng(args.seed),
    )
    slopes = slope_metrics(curves)

    folds.to_csv(args.output_dir / "fold_audit.csv", index=False)
    endpoints.to_csv(args.output_dir / "provenance_endpoint_predictions.csv", index=False)
    one_step.to_csv(args.output_dir / "one_step_task_metrics.csv", index=False)
    one_step_summary.to_csv(args.output_dir / "one_step_loss_summary.csv", index=False)
    one_step_comparisons.to_csv(args.output_dir / "one_step_paired_comparisons.csv", index=False)
    endpoint_summary.to_csv(args.output_dir / "endpoint_loss_summary.csv", index=False)
    endpoint_comparisons.to_csv(args.output_dir / "endpoint_paired_comparisons.csv", index=False)
    curves.to_csv(args.output_dir / "m_curve_predictions.csv", index=False)
    curve_metrics.to_csv(args.output_dir / "m_curve_metrics.csv", index=False)
    slopes.to_csv(args.output_dir / "m_curve_slopes.csv", index=False)
    graph_predictions.to_csv(args.output_dir / "graph_endpoint_predictions.csv", index=False)
    graph_metrics.to_csv(args.output_dir / "graph_endpoint_metrics.csv", index=False)
    manifest = {
        "analysis_version": "provenance-recursive-rollout-v1",
        "states": list(P_STATES),
        "incoming_counts": list(P_COUNT_COLUMNS),
        "models": list(P_MODELS),
        "baseline_models": list(BASELINE_MODELS),
        "particles": args.particles,
        "folds": args.folds,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "information_boundary": (
            "observed Round-0 states, graph, attacker, and schedule only; all Round-1+ "
            "states and target provenance are generated recursively within each particle"
        ),
        "update_audit": update_audit,
        "cases": len(cases),
        "tasks": int(cases.task_id.nunique()),
        "graphs": int(cases.graph_id.nunique()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
