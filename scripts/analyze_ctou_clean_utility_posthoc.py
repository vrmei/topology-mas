#!/usr/bin/env python3
"""Task-bootstrap trend summaries for the clean/attack CTOU experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def primary_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame.law.eq("pooled_balanced")
        & frame.initialization.eq("correlated_empirical")
        & frame.rollout_mode.eq("particle")
    ].copy()


def task_level_curves(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame[frame.condition.eq("clean")].copy()
    attack = frame[frame.condition.eq("attack")].copy()
    clean = (
        clean.groupby(["task_id", "graph_id", "n", "m"], sort=False)
        .agg(utility=("actual_correct", "mean"), u0=("actual_round0_correct", "mean"))
        .reset_index()
    )
    attack = (
        attack.groupby(["task_id", "graph_id", "n", "m"], sort=False)
        .agg(robustness=("actual_correct", "mean"))
        .reset_index()
    )
    merged = clean.merge(attack, on=["task_id", "graph_id", "n", "m"], validate="one_to_one")
    merged["delta_u"] = merged.utility - merged.u0
    return merged


def slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - x.mean()
    return float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def trend_bootstrap(task_graph: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    outcomes = ["u0", "utility", "delta_u", "robustness"]
    for n, selected in task_graph.groupby("n", sort=True):
        tasks = np.array(sorted(selected.task_id.unique()))
        levels = np.array(sorted(selected.m.unique()), dtype=float)
        task_level = (
            selected.groupby(["task_id", "m"], sort=True)[outcomes]
            .mean()
            .reindex(pd.MultiIndex.from_product([tasks, levels], names=["task_id", "m"]))
        )
        if task_level.isna().any().any():
            raise RuntimeError(f"incomplete task-by-m table for n={n}")
        values = task_level.to_numpy(float).reshape(len(tasks), len(levels), len(outcomes))
        draws = rng.integers(0, len(tasks), size=(replicates, len(tasks)))
        means = values[draws].mean(axis=1)
        for outcome_index, outcome in enumerate(outcomes):
            observed_curve = values[:, :, outcome_index].mean(axis=0)
            observed_slope = slope(levels, observed_curve)
            observed_rho = float(spearmanr(levels, observed_curve).statistic)
            boot_slopes = np.array(
                [slope(levels, curve) for curve in means[:, :, outcome_index]], dtype=float
            )
            boot_rhos = np.array(
                [spearmanr(levels, curve).statistic for curve in means[:, :, outcome_index]],
                dtype=float,
            )
            slope_low, slope_high = interval(boot_slopes)
            rho_low, rho_high = interval(boot_rhos)
            rows.extend(
                [
                    {
                        "n": int(n),
                        "outcome": outcome,
                        "metric": "slope_per_edge",
                        "estimate": observed_slope,
                        "ci95_low": slope_low,
                        "ci95_high": slope_high,
                        "tasks": len(tasks),
                        "m_levels": len(levels),
                    },
                    {
                        "n": int(n),
                        "outcome": outcome,
                        "metric": "curve_spearman",
                        "estimate": observed_rho,
                        "ci95_low": rho_low,
                        "ci95_high": rho_high,
                        "tasks": len(tasks),
                        "m_levels": len(levels),
                    },
                ]
            )
    return pd.DataFrame(rows)


def graph_association_bootstrap(
    task_graph: pd.DataFrame, *, replicates: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for n, selected in task_graph.groupby("n", sort=True):
        tasks = np.array(sorted(selected.task_id.unique()))
        graphs = np.array(sorted(selected.graph_id.unique()))
        table = selected.set_index(["task_id", "graph_id"])[["utility", "robustness"]].reindex(
            pd.MultiIndex.from_product([tasks, graphs], names=["task_id", "graph_id"])
        )
        if table.isna().any().any():
            raise RuntimeError(f"incomplete task-by-graph table for n={n}")
        values = table.to_numpy(float).reshape(len(tasks), len(graphs), 2)
        observed = values.mean(axis=0)
        estimate = float(spearmanr(observed[:, 0], observed[:, 1]).statistic)
        draws = rng.integers(0, len(tasks), size=(replicates, len(tasks)))
        means = values[draws].mean(axis=1)
        correlations = np.empty(replicates, dtype=float)
        for index, graph_means in enumerate(means):
            correlations[index] = np.corrcoef(
                rankdata(graph_means[:, 0]), rankdata(graph_means[:, 1])
            )[0, 1]
        low, high = interval(correlations)
        rows.append(
            {
                "n": int(n),
                "metric": "graph_spearman_utility_robustness",
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "tasks": len(tasks),
                "graphs": len(graphs),
            }
        )
    return pd.DataFrame(rows)


def overall_bootstrap(task_graph: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    outcomes = ["u0", "utility", "delta_u", "robustness"]
    rows: list[dict[str, object]] = []
    for n, selected in task_graph.groupby("n", sort=True):
        task_means = selected.groupby("task_id", sort=True)[outcomes].mean()
        values = task_means.to_numpy(float)
        draws = rng.integers(0, len(values), size=(replicates, len(values)))
        boot = values[draws].mean(axis=1)
        for index, outcome in enumerate(outcomes):
            low, high = interval(boot[:, index])
            rows.append(
                {
                    "n": int(n),
                    "outcome": outcome,
                    "estimate": values[:, index].mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "tasks": len(values),
                }
            )
    return pd.DataFrame(rows)


def curve_point_bootstrap(task_graph: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    outcomes = ["u0", "utility", "delta_u", "robustness"]
    rows: list[dict[str, object]] = []
    for (n, m), selected in task_graph.groupby(["n", "m"], sort=True):
        task_means = selected.groupby("task_id", sort=True)[outcomes].mean()
        values = task_means.to_numpy(float)
        draws = rng.integers(0, len(values), size=(replicates, len(values)))
        boot = values[draws].mean(axis=1)
        for index, outcome in enumerate(outcomes):
            low, high = interval(boot[:, index])
            rows.append(
                {
                    "n": int(n),
                    "m": int(m),
                    "outcome": outcome,
                    "estimate": values[:, index].mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "tasks": len(values),
                    "graphs": selected.graph_id.nunique(),
                }
            )
    return pd.DataFrame(rows)


def add_pareto_flags(graphs: pd.DataFrame) -> pd.DataFrame:
    result = graphs.copy()
    for prefix in ("observed", "predicted"):
        result[f"{prefix}_pareto"] = False
        for _, selected in result.groupby("n", sort=True):
            utility = selected[f"{prefix}_utility"].to_numpy(float)
            robustness = selected[f"{prefix}_robustness"].to_numpy(float)
            dominated = np.zeros(len(selected), dtype=bool)
            for index in range(len(selected)):
                at_least_as_good = (utility >= utility[index]) & (robustness >= robustness[index])
                strictly_better = (utility > utility[index]) | (robustness > robustness[index])
                dominated[index] = bool(np.any(at_least_as_good & strictly_better))
            result.loc[selected.index, f"{prefix}_pareto"] = ~dominated
    return result


def main() -> None:
    args = parse_args()
    if args.replicates < 1_000:
        raise ValueError("replicates must be at least 1000")
    predictions = pd.read_csv(args.input_dir / "endpoint_predictions.csv", low_memory=False)
    task_graph = task_level_curves(primary_predictions(predictions))
    task_graph.to_csv(args.input_dir / "utility_robustness_task_graph.csv", index=False)
    trend_bootstrap(task_graph, replicates=args.replicates, seed=args.seed).to_csv(
        args.input_dir / "utility_robustness_trend_bootstrap.csv", index=False
    )
    graph_association_bootstrap(task_graph, replicates=args.replicates, seed=args.seed + 1).to_csv(
        args.input_dir / "utility_robustness_association_bootstrap.csv", index=False
    )
    overall_bootstrap(task_graph, replicates=args.replicates, seed=args.seed + 2).to_csv(
        args.input_dir / "utility_robustness_overall_bootstrap.csv", index=False
    )
    curve_point_bootstrap(task_graph, replicates=args.replicates, seed=args.seed + 3).to_csv(
        args.input_dir / "utility_robustness_curve_task_bootstrap.csv", index=False
    )
    graph_values = pd.read_csv(args.input_dir / "utility_robustness_graphs.csv")
    add_pareto_flags(graph_values).to_csv(
        args.input_dir / "utility_robustness_pareto.csv", index=False
    )


if __name__ == "__main__":
    main()
