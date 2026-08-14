"""Evaluate CTOU recursive rollout without observing test-task Round-0 states."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_ctou_recursive_rollout import (
    STATE_INDEX,
    STATES,
    aggregate_curves,
    aggregate_graphs,
    attach_losses,
    bootstrap_task_summary,
    fit_fold_lookups,
    load_rollout_cases,
    particle_rollout_from_particles,
    stable_seed,
)
from analyze_ctou_transition_prediction import load_updates
from analyze_node_round_adoption import read_json
from scipy.stats import rankdata, spearmanr

INITIALIZATIONS = ("oracle", "iid_empirical", "correlated_empirical")
DEFAULT_SEED = 20_260_816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--particles", type=int, default=2_048)
    parser.add_argument("--split-half-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--table-prior-strength", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def finite_row_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized row-wise Spearman correlation with tie-aware ranks."""

    left_rank = rankdata(left, axis=1)
    right_rank = rankdata(right, axis=1)
    left_centered = left_rank - left_rank.mean(axis=1, keepdims=True)
    right_centered = right_rank - right_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        (left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1)
    )
    return np.divide(
        (left_centered * right_centered).sum(axis=1),
        denominator,
        out=np.full(len(left), np.nan, dtype=float),
        where=denominator > 0,
    )


def split_half_noise_ceiling(
    cases: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate graph-ranking stability under repeated task split halves."""

    rng = np.random.default_rng(seed)
    task_graph = (
        cases.groupby(["n", "task_id", "graph_id"], sort=False)
        .agg(target=("actual_target", "mean"), correct=("actual_correct", "mean"))
        .reset_index()
    )
    draws: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for n, group in task_graph.groupby("n", sort=True):
        tasks = np.asarray(sorted(group.task_id.unique()))
        if len(tasks) % 2:
            raise ValueError("split-half analysis requires an even number of tasks")
        graphs = np.asarray(sorted(group.graph_id.unique()))
        half = len(tasks) // 2
        for outcome in ("target", "correct"):
            matrix = (
                group.pivot(index="task_id", columns="graph_id", values=outcome)
                .reindex(index=tasks, columns=graphs)
                .to_numpy(float)
            )
            if np.isnan(matrix).any():
                raise ValueError(f"incomplete task x graph matrix for n={n}, {outcome}")
            first = np.empty((replicates, len(graphs)), dtype=float)
            second = np.empty_like(first)
            for index in range(replicates):
                order = rng.permutation(len(tasks))
                first[index] = matrix[order[:half]].mean(axis=0)
                second[index] = matrix[order[half:]].mean(axis=0)
            correlation = finite_row_spearman(first, second)
            valid = correlation[np.isfinite(correlation)]
            corrected = np.divide(
                2.0 * valid,
                1.0 + valid,
                out=np.full_like(valid, np.nan),
                where=np.abs(1.0 + valid) > 1e-12,
            )
            draws.append(
                pd.DataFrame(
                    {
                        "n": int(n),
                        "outcome": outcome,
                        "replicate": np.arange(len(valid)),
                        "split_half_spearman": valid,
                        "spearman_brown": corrected,
                    }
                )
            )
            summaries.append(
                {
                    "n": int(n),
                    "outcome": outcome,
                    "tasks": len(tasks),
                    "graphs": len(graphs),
                    "valid_replicates": len(valid),
                    "split_half_mean": float(valid.mean()),
                    "split_half_median": float(np.median(valid)),
                    "split_half_ci95_low": float(np.quantile(valid, 0.025)),
                    "split_half_ci95_high": float(np.quantile(valid, 0.975)),
                    "spearman_brown_median": float(np.nanmedian(corrected)),
                    "spearman_brown_ci95_low": float(np.nanquantile(corrected, 0.025)),
                    "spearman_brown_ci95_high": float(np.nanquantile(corrected, 0.975)),
                }
            )
    return pd.concat(draws, ignore_index=True), pd.DataFrame(summaries)


def benign_state_pool(
    train_cases: pd.DataFrame,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an IID marginal and complete benign state vectors for one node count."""

    vectors: list[tuple[int, ...]] = []
    for case in train_cases[train_cases.n.eq(n)].itertuples(index=False):
        vector = tuple(
            int(state)
            for node, state in enumerate(case.initial_states)
            if node != int(case.attack_node)
        )
        if len(vector) != n - 1:
            raise ValueError("malformed benign Round-0 vector")
        vectors.append(vector)
    if not vectors:
        raise ValueError(f"empty initialization pool for n={n}")
    pool = np.asarray(vectors, dtype=np.int8)
    counts = np.bincount(pool.ravel(), minlength=len(STATES)).astype(float)
    return counts / counts.sum(), pool


def draw_initial_particles(
    *,
    n: int,
    attack_node: int,
    mode: str,
    particles: int,
    iid_probability: np.ndarray,
    correlated_pool: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw topology-agnostic Round-0 particles from a training-only prior."""

    benign_nodes = np.asarray([node for node in range(n) if node != attack_node])
    result = np.full((particles, n), STATE_INDEX["target"], dtype=np.int8)
    if mode == "iid_empirical":
        cumulative = iid_probability.cumsum()
        cumulative[-1] = 1.0
        uniforms = rng.random((particles, n - 1))
        sampled = (uniforms[..., None] > cumulative).sum(axis=2).astype(np.int8)
    elif mode == "correlated_empirical":
        selected = correlated_pool[rng.integers(0, len(correlated_pool), size=particles)]
        permutations = np.argsort(rng.random((particles, n - 1)), axis=1)
        sampled = np.take_along_axis(selected, permutations, axis=1)
    else:
        raise ValueError(f"unknown empirical initialization: {mode}")
    result[:, benign_nodes] = sampled
    return result


def oracle_predictions(oracle_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(oracle_dir / "endpoint_predictions.csv")
    selected = frame[frame.model.eq("ctou_logit") & frame.rollout_mode.eq("particle")].copy()
    selected["initialization"] = "oracle"
    return selected


def empirical_predictions(
    cases: pd.DataFrame,
    updates: pd.DataFrame,
    graphs: dict[str, dict[str, Any]],
    *,
    folds: int,
    particles: int,
    table_prior_strength: float,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    audits: list[dict[str, object]] = []
    maximum_neighbors = int(cases.n.max() - 1)
    horizon = int(cases.horizon.max())
    for graph_fold in range(folds):
        for task_fold in range(folds):
            path = checkpoint_dir / f"graph-{graph_fold}_task-{task_fold}.csv"
            paths.append(path)
            train_updates = updates[
                updates.graph_fold.ne(graph_fold) & updates.task_fold.ne(task_fold)
            ]
            train_cases = cases[
                cases.graph_fold.ne(graph_fold) & cases.task_fold.ne(task_fold)
            ]
            test = cases[cases.graph_fold.eq(graph_fold) & cases.task_fold.eq(task_fold)]
            if test.empty:
                continue
            audits.append(
                {
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "train_update_rows": len(train_updates),
                    "train_initializations": len(train_cases),
                    "test_cases": len(test),
                    "graph_overlap": len(set(train_cases.graph_id) & set(test.graph_id)),
                    "task_overlap": len(set(train_cases.task_id) & set(test.task_id)),
                }
            )
            if path.exists():
                print(f"resume graph={graph_fold} task={task_fold}", flush=True)
                continue
            lookup = fit_fold_lookups(
                train_updates,
                maximum_neighbors=maximum_neighbors,
                horizon=horizon,
                table_prior_strength=table_prior_strength,
            )["ctou_logit"]
            priors = {
                n: benign_state_pool(train_cases, int(n)) for n in sorted(test.n.unique())
            }
            for n, (probability, pool) in priors.items():
                audits[-1].update(
                    {
                        f"n{n}_pool_vectors": len(pool),
                        **{
                            f"n{n}_p_{state}": float(probability[index])
                            for index, state in enumerate(STATES)
                        },
                    }
                )
            unique = test.drop_duplicates(["graph_id", "attack_node"])
            cache: dict[tuple[str, str, int], np.ndarray] = {}
            for case in unique.itertuples(index=False):
                graph = graphs[case.graph_id]
                probability, pool = priors[int(case.n)]
                for mode in ("iid_empirical", "correlated_empirical"):
                    case_seed = stable_seed(
                        case.graph_id,
                        case.attack_node,
                        graph_fold,
                        task_fold,
                        mode,
                        seed=seed,
                    )
                    initial = draw_initial_particles(
                        n=int(case.n),
                        attack_node=int(case.attack_node),
                        mode=mode,
                        particles=particles,
                        iid_probability=probability,
                        correlated_pool=pool,
                        rng=np.random.default_rng(case_seed),
                    )
                    cache[(mode, case.graph_id, int(case.attack_node))] = (
                        particle_rollout_from_particles(
                            graph=graph,
                            initial_particles=initial,
                            attack_node=int(case.attack_node),
                            model="ctou_logit",
                            lookup=lookup,
                            seed=case_seed,
                        )
                    )
            rows: list[dict[str, object]] = []
            for case in test.itertuples(index=False):
                base = {
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
                    "model": "ctou_logit",
                    "rollout_mode": "particle",
                }
                for mode in ("iid_empirical", "correlated_empirical"):
                    probability = cache[(mode, case.graph_id, int(case.attack_node))]
                    rows.append(
                        {
                            **base,
                            "initialization": mode,
                            **{
                                f"p_{state}": float(probability[index])
                                for index, state in enumerate(STATES)
                            },
                        }
                    )
            temporary = path.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(path)
            print(
                f"fold graph={graph_fold} task={task_fold}: "
                f"unique={len(unique)} rows={len(rows)}",
                flush=True,
            )
            del lookup, priors, cache, rows, train_updates, train_cases, test, unique
            gc.collect()
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing checkpoints: {missing}")
    return (
        pd.concat((pd.read_csv(path) for path in paths), ignore_index=True),
        pd.DataFrame(audits),
    )


def initialization_metrics(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    scored_parts = []
    for initialization, group in frame.groupby("initialization", sort=False):
        part = attach_losses(group)
        part["model"] = initialization
        part["rollout_mode"] = "particle"
        scored_parts.append(part)
    scored = pd.concat(scored_parts, ignore_index=True)
    rng = np.random.default_rng(seed)
    task = bootstrap_task_summary(scored, bootstrap_replicates, rng).rename(
        columns={"model": "initialization"}
    )
    curves, curve_metrics = aggregate_curves(scored)
    graphs, graph_metrics = aggregate_graphs(
        scored, replicates=bootstrap_replicates, rng=rng
    )
    for output in (curves, curve_metrics, graphs, graph_metrics):
        output.rename(columns={"model": "initialization"}, inplace=True)
    return {
        "scored": scored,
        "task": task,
        "curves": curves,
        "curve_metrics": curve_metrics,
        "graphs": graphs,
        "graph_metrics": graph_metrics,
        "within_m_graph_metrics": within_m_graph_metrics(graphs),
    }


def within_m_graph_metrics(graphs: pd.DataFrame) -> pd.DataFrame:
    """Measure topology ordering after removing between-density rank variation."""

    rows: list[dict[str, object]] = []
    for (initialization, n), group in graphs.groupby(["initialization", "n"], sort=True):
        for outcome in ("target", "correct"):
            observed_centered: list[float] = []
            predicted_centered: list[float] = []
            level_correlations: list[float] = []
            for _, level in group.groupby("m", sort=True):
                if len(level) < 3:
                    continue
                observed = rankdata(level[f"observed_{outcome}"].to_numpy(float))
                predicted = rankdata(level[f"predicted_{outcome}"].to_numpy(float))
                observed_centered.extend(observed - observed.mean())
                predicted_centered.extend(predicted - predicted.mean())
                level_correlations.append(float(spearmanr(observed, predicted).statistic))
            rows.append(
                {
                    "initialization": initialization,
                    "n": int(n),
                    "outcome": outcome,
                    "density_levels": len(level_correlations),
                    "pooled_within_m_rank_correlation": float(
                        spearmanr(observed_centered, predicted_centered).statistic
                    ),
                    "median_level_spearman": float(np.nanmedian(level_correlations)),
                    "mean_level_spearman": float(np.nanmean(level_correlations)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.particles < 256:
        raise ValueError("particles must be at least 256")
    if args.split_half_replicates < 1_000 or args.bootstrap_replicates < 1_000:
        raise ValueError("replicate counts must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = read_json(args.run_root / "orchestrator_status.json")
    updates, update_audit = load_updates(args.run_root, args.folds)
    cases, graphs, case_audit = load_rollout_cases(args.run_root, status, args.folds)
    if not update_audit["passed"] or not case_audit["passed"]:
        raise RuntimeError("input integrity audit failed")

    ceiling_draws, ceiling_summary = split_half_noise_ceiling(
        cases,
        replicates=args.split_half_replicates,
        seed=args.seed,
    )
    ceiling_draws.to_csv(args.output_dir / "split_half_draws.csv", index=False)
    ceiling_summary.to_csv(args.output_dir / "split_half_summary.csv", index=False)

    empirical, fold_audit = empirical_predictions(
        cases,
        updates,
        graphs,
        folds=args.folds,
        particles=args.particles,
        table_prior_strength=args.table_prior_strength,
        seed=args.seed,
        checkpoint_dir=args.output_dir / "fold-checkpoints",
    )
    if (fold_audit.graph_overlap != 0).any() or (fold_audit.task_overlap != 0).any():
        raise RuntimeError("crossed holdout leakage detected")
    oracle = oracle_predictions(args.oracle_dir)
    key_columns = ["task_id", "graph_id", "attack_node"]
    expected = cases[key_columns].drop_duplicates()
    for name, frame in (("oracle", oracle), ("empirical", empirical)):
        actual = frame[key_columns].drop_duplicates()
        if len(actual) != len(expected) or len(expected.merge(actual)) != len(expected):
            raise RuntimeError(f"{name} predictions do not match rollout cases")
    combined = pd.concat([oracle, empirical], ignore_index=True, sort=False)
    outputs = initialization_metrics(
        combined,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    fold_audit.to_csv(args.output_dir / "fold_initialization_audit.csv", index=False)
    outputs["scored"].to_csv(args.output_dir / "endpoint_predictions.csv", index=False)
    outputs["task"].to_csv(args.output_dir / "endpoint_loss_summary.csv", index=False)
    outputs["curves"].to_csv(args.output_dir / "m_curve_predictions.csv", index=False)
    outputs["curve_metrics"].to_csv(args.output_dir / "m_curve_metrics.csv", index=False)
    outputs["graphs"].to_csv(args.output_dir / "graph_endpoint_predictions.csv", index=False)
    outputs["graph_metrics"].to_csv(args.output_dir / "graph_endpoint_metrics.csv", index=False)
    outputs["within_m_graph_metrics"].to_csv(
        args.output_dir / "within_m_graph_metrics.csv", index=False
    )
    manifest = {
        "analysis_version": "ctou-round-zero-free-v1",
        "run_root": str(args.run_root.resolve()),
        "oracle_dir": str(args.oracle_dir.resolve()),
        "states": list(STATES),
        "initializations": list(INITIALIZATIONS),
        "transition_model": "ctou_logit",
        "rollout_mode": "joint_particle",
        "particles": args.particles,
        "folds": args.folds,
        "split_half_replicates": args.split_half_replicates,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "information_boundary": {
            "oracle": "observed categorical Round-0 state vector for each test run",
            "iid_empirical": (
                "node-count-matched training-only benign marginal; independent nodes"
            ),
            "correlated_empirical": (
                "node-count-matched training-only complete benign vectors, randomly remapped"
            ),
        },
        "update_integrity": update_audit,
        "case_integrity": case_audit,
        "claim_limits": [
            (
                "split-half stability is an empirical reliability reference, not a "
                "mathematical upper bound"
            ),
            (
                "empirical priors know the model/dataset regime and node count but not "
                "test task text or states"
            ),
            (
                "the correlated prior preserves vector-level dependence but is not a "
                "learned task model"
            ),
            (
                "all results remain conditional on one model, GSM8K, sampled graphs, "
                "and the existing attack protocol"
            ),
            (
                "successful graph ranking does not establish topology-only transfer "
                "to another task distribution"
            ),
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
