"""Evaluate CTOU recursive rollouts outside the fitted density support."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_ctou_recursive_rollout import (
    EPSILON,
    STATES,
    aggregate_graphs,
    dense_lookup,
    mean_field_rollout,
    query_frame,
)
from analyze_ctou_transition_prediction import (
    COUNT_COLUMNS,
    TABLE_KEYS,
    load_updates,
    table_predictions,
)
from analyze_node_round_adoption import read_json

MODELS = ("persistence", "degroot_equal", "ctou_table")
DEFAULT_SEED = 20_260_816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--table-prior-strength", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_cases_and_graphs(
    reference_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, object]]:
    cases = pd.read_pickle(reference_dir / "normalized_rollout_cases.pkl")
    graphs = json.loads((reference_dir / "normalized_graphs.json").read_text(encoding="utf-8"))
    audit = read_json(reference_dir / "normalized_case_audit.json")
    return cases, graphs, audit


def load_reference_predictions(reference_dir: Path) -> pd.DataFrame:
    columns = [
        "task_id",
        "graph_id",
        "attack_node",
        "actual_state",
        "actual_state_index",
        "actual_target",
        "actual_correct",
        "model",
        "rollout_mode",
        *[f"p_{state}" for state in STATES],
    ]
    frame = pd.read_csv(reference_dir / "endpoint_predictions.csv", usecols=columns)
    selected = frame[
        frame.rollout_mode.eq("mean_field")
        & frame.model.isin(("persistence", "degroot_equal", "ctou_table"))
    ].copy()
    duplicates = selected.duplicated(["task_id", "graph_id", "attack_node", "model"]).sum()
    if duplicates:
        raise RuntimeError(f"duplicate in-support reference predictions: {duplicates}")
    return selected


def fit_table_lookup(
    train: pd.DataFrame,
    *,
    maximum_neighbors: int,
    horizon: int,
    prior_strength: float,
) -> np.ndarray:
    query = query_frame(maximum_neighbors, horizon)
    probabilities = table_predictions(train, query, prior_strength)
    return dense_lookup(query, probabilities, horizon, maximum_neighbors)


def split_boundaries(cases: pd.DataFrame) -> dict[int, int]:
    boundaries: dict[int, int] = {}
    for n, group in cases.groupby("n"):
        levels = sorted(group.m.unique())
        boundaries[int(n)] = int(levels[(len(levels) - 1) // 2])
    return boundaries


def sparse_mask(frame: pd.DataFrame, boundaries: dict[int, int]) -> pd.Series:
    threshold = frame.n.map(boundaries)
    return frame.m.le(threshold)


def support_rows_for_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    """Measure whether test transition cells occurred in the training split.

    This diagnostic is descriptive only: it examines the saved test traces after
    endpoint prediction and is never exposed to the rollout model.
    """
    exact_train = pd.MultiIndex.from_frame(train[list(TABLE_KEYS)].drop_duplicates())
    composition_train = pd.MultiIndex.from_frame(
        train[list(COUNT_COLUMNS)].drop_duplicates()
    )
    evaluated = test.copy()
    evaluated["exact_transition_cell_seen"] = pd.MultiIndex.from_frame(
        evaluated[list(TABLE_KEYS)]
    ).isin(exact_train)
    evaluated["composition_cell_seen"] = pd.MultiIndex.from_frame(
        evaluated[list(COUNT_COLUMNS)]
    ).isin(composition_train)
    evaluated["incoming_total"] = evaluated[list(COUNT_COLUMNS)].sum(axis=1)
    rows: list[dict[str, object]] = []
    for (n, m), group in evaluated.groupby(["n", "m"], sort=True):
        rows.append(
            {
                **metadata,
                "n": int(n),
                "m": int(m),
                "test_updates": len(group),
                "exact_transition_cell_coverage": group.exact_transition_cell_seen.mean(),
                "composition_cell_coverage": group.composition_cell_seen.mean(),
                "mean_incoming_total": group.incoming_total.mean(),
                "p95_incoming_total": group.incoming_total.quantile(0.95),
                "max_incoming_total": group.incoming_total.max(),
                "train_distinct_transition_cells": len(exact_train),
                "train_distinct_composition_cells": len(composition_train),
            }
        )
    return rows


def composition_support_diagnostics(
    updates: pd.DataFrame,
    *,
    boundaries: dict[int, int],
    folds: int,
) -> pd.DataFrame:
    """Audit local-composition support for every extrapolation split."""
    rows: list[dict[str, object]] = []
    for n, m in sorted(updates[["n", "m"]].drop_duplicates().itertuples(index=False)):
        held_level = updates.n.eq(n) & updates.m.eq(m)
        for scope in ("density_only", "density_task"):
            task_folds: list[int | None] = [None] if scope == "density_only" else list(
                range(folds)
            )
            for task_fold in task_folds:
                train_mask = ~held_level
                test_mask = held_level.copy()
                if task_fold is not None:
                    train_mask &= updates.task_fold.ne(task_fold)
                    test_mask &= updates.task_fold.eq(task_fold)
                rows.extend(
                    support_rows_for_split(
                        updates[train_mask],
                        updates[test_mask],
                        metadata={
                            "experiment": "leave_level_out",
                            "validation_scope": scope,
                            "direction": "held_level",
                            "test_task_fold": task_fold,
                        },
                    )
                )

    sparse = sparse_mask(updates, boundaries)
    directions = {
        "sparse_to_dense": (sparse, ~sparse),
        "dense_to_sparse": (~sparse, sparse),
    }
    for direction, (train_region, test_region) in directions.items():
        for scope in ("density_only", "density_task"):
            task_folds = [None] if scope == "density_only" else list(range(folds))
            for task_fold in task_folds:
                train_mask = train_region.copy()
                test_mask = test_region.copy()
                if task_fold is not None:
                    train_mask &= updates.task_fold.ne(task_fold)
                    test_mask &= updates.task_fold.eq(task_fold)
                rows.extend(
                    support_rows_for_split(
                        updates[train_mask],
                        updates[test_mask],
                        metadata={
                            "experiment": "range_extrapolation",
                            "validation_scope": scope,
                            "direction": direction,
                            "test_task_fold": task_fold,
                        },
                    )
                )
    return pd.DataFrame(rows)


def reference_maps(reference: pd.DataFrame) -> dict[str, dict[tuple[str, str, int], np.ndarray]]:
    result: dict[str, dict[tuple[str, str, int], np.ndarray]] = {}
    for model in MODELS:
        selected = reference[reference.model.eq(model)]
        result[model] = {
            (str(row.task_id), str(row.graph_id), int(row.attack_node)): np.asarray(
                [getattr(row, f"p_{state}") for state in STATES], dtype=float
            )
            for row in selected.itertuples(index=False)
        }
    return result


def write_group_checkpoint(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    graphs: dict[str, dict[str, Any]],
    references: dict[str, dict[tuple[str, str, int], np.ndarray]],
    checkpoint: Path,
    metadata: dict[str, object],
    maximum_neighbors: int,
    horizon: int,
    prior_strength: float,
) -> dict[str, object]:
    train_graphs = set(train.graph_id)
    test_graphs = set(test.graph_id)
    train_tasks = set(train.task_id)
    test_tasks = set(test.task_id)
    audit = {
        **metadata,
        "train_updates": len(train),
        "test_cases": len(test),
        "train_graphs": len(train_graphs),
        "test_graphs": len(test_graphs),
        "train_tasks": len(train_tasks),
        "test_tasks": len(test_tasks),
        "graph_overlap": len(train_graphs & test_graphs),
        "task_overlap": len(train_tasks & test_tasks),
    }
    # pandas writes an empty, column-free frame as a single newline. Require a
    # real CSV header before treating a checkpoint as resumable.
    if checkpoint.exists() and checkpoint.stat().st_size > 1:
        audit["resumed"] = True
        return audit
    if checkpoint.exists():
        checkpoint.unlink()
    lookup = fit_table_lookup(
        train,
        maximum_neighbors=maximum_neighbors,
        horizon=horizon,
        prior_strength=prior_strength,
    )
    unique = test.drop_duplicates(["graph_id", "attack_node", "initial_states"])
    cache: dict[tuple[str, int, tuple[int, ...]], np.ndarray] = {}
    for case in unique.itertuples(index=False):
        key = (str(case.graph_id), int(case.attack_node), tuple(case.initial_states))
        cache[key] = mean_field_rollout(
            graph=graphs[str(case.graph_id)],
            initial_states=tuple(case.initial_states),
            attack_node=int(case.attack_node),
            model="ctou_table",
            lookup=lookup,
        )
    rows: list[dict[str, object]] = []
    for case in test.itertuples(index=False):
        case_key = (str(case.task_id), str(case.graph_id), int(case.attack_node))
        rollout_key = (str(case.graph_id), int(case.attack_node), tuple(case.initial_states))
        base = {
            **metadata,
            "stratum": case.stratum,
            "task_id": case.task_id,
            "graph_id": case.graph_id,
            "attack_node": case.attack_node,
            "n": case.n,
            "m": case.m,
            "actual_state": case.actual_state,
            "actual_state_index": case.actual_state_index,
            "actual_target": case.actual_target,
            "actual_correct": case.actual_correct,
            "rollout_mode": "mean_field",
        }
        for model in MODELS:
            probability = (
                cache[rollout_key] if model == "ctou_table" else references[model][case_key]
            )
            rows.append(
                {
                    **base,
                    "model": model,
                    **{
                        f"p_{state}": float(probability[index])
                        for index, state in enumerate(STATES)
                    },
                }
            )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(checkpoint)
    audit["resumed"] = False
    audit["checkpoint_rows"] = len(rows)
    del lookup, cache, rows, unique
    gc.collect()
    return audit


def execute_extrapolation(
    *,
    updates: pd.DataFrame,
    cases: pd.DataFrame,
    graphs: dict[str, dict[str, Any]],
    reference: pd.DataFrame,
    output_dir: Path,
    folds: int,
    prior_strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    checkpoints = output_dir / "fold-checkpoints"
    references = reference_maps(reference)
    maximum_neighbors = int(cases.n.max() - 1)
    horizon = int(cases.horizon.max())
    boundaries = split_boundaries(cases)
    audit_rows: list[dict[str, object]] = []
    expected: list[Path] = []

    for n, m in sorted(cases[["n", "m"]].drop_duplicates().itertuples(index=False)):
        held_level = updates.n.eq(n) & updates.m.eq(m)
        test_level = cases.n.eq(n) & cases.m.eq(m)
        for scope in ("density_only", "density_task"):
            task_folds: list[int | None] = [None] if scope == "density_only" else list(range(folds))
            for task_fold in task_folds:
                train_mask = ~held_level
                # Each task fold must start from the full held-level mask. An
                # in-place intersection would otherwise contaminate later folds.
                test_mask = test_level.copy()
                if task_fold is not None:
                    train_mask &= updates.task_fold.ne(task_fold)
                    test_mask &= cases.task_fold.eq(task_fold)
                train = updates[train_mask]
                test = cases[test_mask]
                checkpoint = checkpoints / (f"leave-level_n-{n}_m-{m}_{scope}_task-{task_fold}.csv")
                expected.append(checkpoint)
                metadata = {
                    "experiment": "leave_level_out",
                    "validation_scope": scope,
                    "direction": "held_level",
                    "held_n": int(n),
                    "held_m": int(m),
                    "test_task_fold": task_fold,
                }
                print(
                    f"leave level n={n} m={m} scope={scope} task_fold={task_fold} "
                    f"train={len(train)} test={len(test)}",
                    flush=True,
                )
                audit_rows.append(
                    write_group_checkpoint(
                        train=train,
                        test=test,
                        graphs=graphs,
                        references=references,
                        checkpoint=checkpoint,
                        metadata=metadata,
                        maximum_neighbors=maximum_neighbors,
                        horizon=horizon,
                        prior_strength=prior_strength,
                    )
                )

    update_sparse = sparse_mask(updates, boundaries)
    case_sparse = sparse_mask(cases, boundaries)
    directions = {
        "sparse_to_dense": (update_sparse, ~case_sparse),
        "dense_to_sparse": (~update_sparse, case_sparse),
    }
    for direction, (train_region, test_region) in directions.items():
        for scope in ("density_only", "density_task"):
            task_folds = [None] if scope == "density_only" else list(range(folds))
            for task_fold in task_folds:
                train_mask = train_region.copy()
                test_mask = test_region.copy()
                if task_fold is not None:
                    train_mask &= updates.task_fold.ne(task_fold)
                    test_mask &= cases.task_fold.eq(task_fold)
                train = updates[train_mask]
                test = cases[test_mask]
                checkpoint = checkpoints / f"range_{direction}_{scope}_task-{task_fold}.csv"
                expected.append(checkpoint)
                metadata = {
                    "experiment": "range_extrapolation",
                    "validation_scope": scope,
                    "direction": direction,
                    "held_n": None,
                    "held_m": None,
                    "test_task_fold": task_fold,
                }
                print(
                    f"range {direction} scope={scope} task_fold={task_fold} "
                    f"train={len(train)} test={len(test)}",
                    flush=True,
                )
                audit_rows.append(
                    write_group_checkpoint(
                        train=train,
                        test=test,
                        graphs=graphs,
                        references=references,
                        checkpoint=checkpoint,
                        metadata=metadata,
                        maximum_neighbors=maximum_neighbors,
                        horizon=horizon,
                        prior_strength=prior_strength,
                    )
                )

    missing = [path for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"missing extrapolation checkpoints: {missing}")
    predictions = pd.concat((pd.read_csv(path) for path in expected), ignore_index=True)
    return predictions, pd.DataFrame(audit_rows), boundaries


def attach_losses(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    probability = result[[f"p_{state}" for state in STATES]].to_numpy(float)
    labels = result.actual_state_index.to_numpy(int)
    one_hot = np.eye(len(STATES))[labels]
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    result["multiclass_brier"] = ((probability - one_hot) ** 2).sum(axis=1)
    result["multiclass_log_loss"] = -np.log(clipped[np.arange(len(result)), labels])
    for outcome in ("target", "correct"):
        observed = result[f"actual_{outcome}"].to_numpy(float)
        predicted = result[f"p_{outcome}"].to_numpy(float)
        clipped_binary = np.clip(predicted, EPSILON, 1.0 - EPSILON)
        result[f"{outcome}_brier"] = (predicted - observed) ** 2
        result[f"{outcome}_log_loss"] = -(
            observed * np.log(clipped_binary) + (1.0 - observed) * np.log(1.0 - clipped_binary)
        )
    return result


def task_bootstrap_summary(
    frame: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    metrics = (
        "multiclass_brier",
        "multiclass_log_loss",
        "target_brier",
        "target_log_loss",
        "correct_brier",
        "correct_log_loss",
    )
    keys = ["experiment", "validation_scope", "direction", "model"]
    task = frame.groupby([*keys, "task_id"], sort=False)[list(metrics)].mean().reset_index()
    rows: list[dict[str, object]] = []
    for group_keys, group in task.groupby(keys, sort=False, dropna=False):
        values = group[list(metrics)].to_numpy(float)
        sample = rng.integers(0, len(values), size=(replicates, len(values)))
        bootstrap = values[sample].mean(axis=1)
        for index, metric in enumerate(metrics):
            low, high = np.quantile(bootstrap[:, index], [0.025, 0.975])
            rows.append(
                {
                    **dict(zip(keys, group_keys, strict=True)),
                    "metric": metric,
                    "estimate": values[:, index].mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "tasks": len(values),
                }
            )
    return pd.DataFrame(rows)


def in_domain_loss_differences(
    scored: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    in_domain = reference[reference.model.eq("ctou_table")].copy()
    in_domain = attach_losses(in_domain)
    metrics = (
        "multiclass_brier",
        "multiclass_log_loss",
        "target_brier",
        "target_log_loss",
        "correct_brier",
        "correct_log_loss",
    )
    identity = ["task_id", "graph_id", "attack_node"]
    reference_columns = [*identity, *metrics]
    merged = scored[scored.model.eq("ctou_table")].merge(
        in_domain[reference_columns],
        on=identity,
        suffixes=("_ood", "_in_domain"),
        validate="many_to_one",
    )
    keys = ["experiment", "validation_scope", "direction"]
    rows: list[dict[str, object]] = []
    for group_keys, group in merged.groupby(keys, sort=False, dropna=False):
        for metric in metrics:
            group = group.copy()
            group["difference"] = group[f"{metric}_ood"] - group[f"{metric}_in_domain"]
            task = group.groupby("task_id").difference.mean().to_numpy(float)
            sample = rng.integers(0, len(task), size=(replicates, len(task)))
            bootstrap = task[sample].mean(axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            rows.append(
                {
                    **dict(zip(keys, group_keys, strict=True)),
                    "metric": metric,
                    "loss_difference_ood_minus_in_domain": task.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "tasks": len(task),
                }
            )
    return pd.DataFrame(rows)


def aggregate_curves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["experiment", "validation_scope", "direction", "model", "n", "m"]
    curves = (
        frame.groupby(keys, sort=False, dropna=False)
        .agg(
            cases=("actual_target", "size"),
            observed_target=("actual_target", "mean"),
            predicted_target=("p_target", "mean"),
            observed_correct=("actual_correct", "mean"),
            predicted_correct=("p_correct", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    group_keys = ["experiment", "validation_scope", "direction", "model", "n"]
    for values, group in curves.groupby(group_keys, sort=False, dropna=False):
        for outcome in ("target", "correct"):
            observed = group[f"observed_{outcome}"].to_numpy(float)
            predicted = group[f"predicted_{outcome}"].to_numpy(float)
            correlation = pd.Series(observed).corr(pd.Series(predicted), method="spearman")
            rows.append(
                {
                    **dict(zip(group_keys, values, strict=True)),
                    "outcome": outcome,
                    "m_levels": len(group),
                    "curve_mae": np.abs(predicted - observed).mean(),
                    "curve_spearman": correlation,
                }
            )
    return curves, pd.DataFrame(rows)


def graph_metrics_by_protocol(
    frame: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    keys = ["experiment", "validation_scope", "direction"]
    for values, group in frame.groupby(keys, sort=False, dropna=False):
        graph_prediction, graph_metric = aggregate_graphs(
            group,
            replicates=replicates,
            rng=rng,
        )
        metadata = dict(zip(keys, values, strict=True))
        for key, value in metadata.items():
            graph_prediction[key] = value
            graph_metric[key] = value
        predictions.append(graph_prediction)
        metrics.append(graph_metric)
    return pd.concat(predictions, ignore_index=True), pd.concat(metrics, ignore_index=True)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    updates_cache = args.output_dir / "normalized_transition_updates.pkl"
    update_audit_cache = args.output_dir / "normalized_update_audit.json"
    if updates_cache.exists() and update_audit_cache.exists():
        updates = pd.read_pickle(updates_cache)
        update_audit = read_json(update_audit_cache)
        print("loaded normalized transition updates from checkpoint", flush=True)
    else:
        updates, update_audit = load_updates(args.run_root, args.folds)
        updates_temporary = updates_cache.with_suffix(".pkl.tmp")
        audit_temporary = update_audit_cache.with_suffix(".json.tmp")
        updates.to_pickle(updates_temporary)
        audit_temporary.write_text(
            json.dumps(update_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        updates_temporary.replace(updates_cache)
        audit_temporary.replace(update_audit_cache)
    cases, graphs, case_audit = load_cases_and_graphs(args.reference_dir)
    reference = load_reference_predictions(args.reference_dir)
    if not update_audit["passed"] or not case_audit["passed"]:
        raise RuntimeError("input integrity audit failed")
    predictions, audit, boundaries = execute_extrapolation(
        updates=updates,
        cases=cases,
        graphs=graphs,
        reference=reference,
        output_dir=args.output_dir,
        folds=args.folds,
        prior_strength=args.table_prior_strength,
    )
    if (audit.graph_overlap != 0).any():
        raise RuntimeError("density extrapolation graph leakage detected")
    strict = audit.validation_scope.eq("density_task")
    if (audit.loc[strict, "task_overlap"] != 0).any():
        raise RuntimeError("density+task extrapolation leakage detected")
    scored = attach_losses(predictions)
    rng = np.random.default_rng(args.seed)
    loss_summary = task_bootstrap_summary(
        scored,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )
    loss_gap = in_domain_loss_differences(
        scored,
        reference,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )
    curves, curve_metrics = aggregate_curves(scored)
    graph_predictions, graph_metrics = graph_metrics_by_protocol(
        scored,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )
    support = composition_support_diagnostics(
        updates,
        boundaries=boundaries,
        folds=args.folds,
    )
    audit.to_csv(args.output_dir / "split_audit.csv", index=False)
    scored.to_csv(args.output_dir / "endpoint_predictions.csv", index=False)
    loss_summary.to_csv(args.output_dir / "endpoint_loss_summary.csv", index=False)
    loss_gap.to_csv(args.output_dir / "loss_gap_vs_in_domain.csv", index=False)
    curves.to_csv(args.output_dir / "m_curve_predictions.csv", index=False)
    curve_metrics.to_csv(args.output_dir / "m_curve_metrics.csv", index=False)
    graph_predictions.to_csv(args.output_dir / "graph_endpoint_predictions.csv", index=False)
    graph_metrics.to_csv(args.output_dir / "graph_endpoint_metrics.csv", index=False)
    support.to_csv(args.output_dir / "composition_support.csv", index=False)
    manifest = {
        "analysis_version": "ctou-density-extrapolation-v1",
        "run_root": str(args.run_root.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "models": list(MODELS),
        "rollout_mode": "mean_field",
        "folds": args.folds,
        "table_prior_strength": args.table_prior_strength,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "split_boundaries": boundaries,
        "composition_support_diagnostic": (
            "post-hoc exact TABLE_KEYS and count-only support coverage; never exposed to rollout"
        ),
        "update_integrity": update_audit,
        "case_integrity": case_audit,
        "information_boundary": (
            "observed Round-0 categorical states plus graph, attack node, and schedule; "
            "no observed Round-1+ state, composition, answer, or text"
        ),
        "claim_limits": [
            "density-only validation intentionally permits the same tasks at other densities",
            "density+task validation excludes the held task fold",
            "graphs are unseen because each graph belongs to one density level or range",
            "true Round-0 states remain observed inputs",
            "mean-field is an approximation validated in the prior in-support pilot",
            "results remain conditional on one model, dataset, and attack protocol",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
