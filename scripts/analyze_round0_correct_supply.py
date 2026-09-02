#!/usr/bin/env python3
"""Relate clean readout correction to observed Round-0 correct-answer supply."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

STATE_CORRECT = 0
ANALYSIS_VERSION = "round0-correct-supply-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-cases-pkl", type=Path, required=True)
    parser.add_argument("--graphs-json", type=Path, required=True)
    parser.add_argument("--clean-task-graph-csv", type=Path, required=True)
    parser.add_argument("--difficulty-task-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def shortest_distances_to_readout(graph: dict[str, object]) -> tuple[int, ...]:
    n = int(graph["node_count"])
    readout = int(graph["readout_node"])
    reverse: list[list[int]] = [[] for _ in range(n)]
    for edge in graph["edges"]:  # type: ignore[union-attr]
        reverse[int(edge["target"])].append(int(edge["source"]))
    distances = [-1] * n
    distances[readout] = 0
    queue = deque([readout])
    while queue:
        node = queue.popleft()
        for predecessor in reverse[node]:
            if distances[predecessor] < 0:
                distances[predecessor] = distances[node] + 1
                queue.append(predecessor)
    return tuple(distances)


def build_endpoint_frame(
    clean_cases: pd.DataFrame,
    clean_endpoints: pd.DataFrame,
    graphs: dict[str, dict[str, object]],
) -> pd.DataFrame:
    required = {"task_id", "graph_id", "n", "m", "utility", "u0"}
    missing = required - set(clean_endpoints.columns)
    if missing:
        raise ValueError(f"clean endpoints missing columns: {sorted(missing)}")
    case_required = {
        "task_id",
        "graph_id",
        "n",
        "m",
        "readout_node",
        "horizon",
        "initial_states",
    }
    case_missing = case_required - set(clean_cases.columns)
    if case_missing:
        raise ValueError(f"clean cases missing columns: {sorted(case_missing)}")
    endpoint_columns = ["task_id", "graph_id", "n", "m", "utility", "u0"]
    frame = clean_cases[list(case_required)].merge(
        clean_endpoints[endpoint_columns],
        on=["task_id", "graph_id", "n", "m"],
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(clean_cases) or len(frame) != len(clean_endpoints):
        raise ValueError("clean cases and clean endpoints do not match one-to-one")
    rows: list[dict[str, object]] = []
    for case in frame.itertuples(index=False):
        graph = graphs[str(case.graph_id)]
        distances = shortest_distances_to_readout(graph)
        if any(distance < 0 or distance > int(case.horizon) for distance in distances):
            raise ValueError(f"graph violates horizon reachability: {case.graph_id}")
        states = tuple(case.initial_states)
        observed_u0 = int(states[int(case.readout_node)] == STATE_CORRECT)
        if observed_u0 != int(case.u0):
            raise ValueError(f"Round-0 state disagrees with endpoint table: {case.task_id}")
        peers = [node for node in range(int(case.n)) if node != int(case.readout_node)]
        correct_distances = [
            distances[node] for node in peers if states[node] == STATE_CORRECT
        ]
        row = case._asdict()
        row.update(
            {
                "round0_readout_correct": observed_u0,
                "final_readout_correct": int(case.utility),
                "k_correct_total": len(correct_distances),
                "k_correct_within_h1": sum(x <= 1 for x in correct_distances),
                "k_correct_within_h2": sum(x <= 2 for x in correct_distances),
                "k_correct_within_h3": sum(x <= 3 for x in correct_distances),
                "earliest_correct_arrival": (
                    min(correct_distances) if correct_distances else 0
                ),
                "has_correct_peer": int(bool(correct_distances)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def clustered_bootstrap_rates(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    outcome_column: str = "final_readout_correct",
    rate_name: str = "correction_rate",
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Task-cluster bootstrap conditional rates and retain cell support."""

    support = (
        frame.groupby(group_columns, sort=True, dropna=False)
        .agg(
            cases=(outcome_column, "size"),
            tasks=("task_id", "nunique"),
            graphs=("graph_id", "nunique"),
        )
        .reset_index()
    )
    rates = (
        frame.groupby(group_columns, sort=True, dropna=False)[outcome_column]
        .mean()
        .rename(rate_name)
        .reset_index()
    )
    observed = support.merge(rates, on=group_columns, validate="one_to_one")
    tasks = frame.task_id.unique()
    rng = np.random.default_rng(seed)
    keys = list(observed[group_columns].itertuples(index=False, name=None))
    task_index = {task: index for index, task in enumerate(tasks)}
    key_index = {key: index for index, key in enumerate(keys)}
    numerators = np.zeros((len(keys), len(tasks)), dtype=float)
    denominators = np.zeros_like(numerators)
    for row in frame.itertuples(index=False):
        key = tuple(getattr(row, column) for column in group_columns)
        group_index = key_index[key]
        column_index = task_index[row.task_id]
        numerators[group_index, column_index] += float(getattr(row, outcome_column))
        denominators[group_index, column_index] += 1.0
    weights = rng.multinomial(
        len(tasks), np.full(len(tasks), 1.0 / len(tasks)), size=samples
    )
    draw_numerators = weights @ numerators.T
    draw_denominators = weights @ denominators.T
    draws = np.full(draw_numerators.shape, np.nan, dtype=float)
    np.divide(
        draw_numerators,
        draw_denominators,
        out=draws,
        where=draw_denominators != 0,
    )
    intervals = []
    for index, key in enumerate(keys):
        values = draws[:, index]
        values = values[np.isfinite(values)]
        low, high = np.quantile(values, [0.025, 0.975])
        intervals.append((*key, low, high, len(values)))
    interval_frame = pd.DataFrame(
        intervals,
        columns=[*group_columns, "ci95_low", "ci95_high", "valid_bootstrap_draws"],
    )
    return observed.merge(interval_frame, on=group_columns, validate="one_to_one")


def plot_supply(curve: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)
    colors = {5: "#4C78A8", 8: "#F58518"}
    for axis, n in zip(axes, (5, 8), strict=True):
        group = curve.loc[curve.n.eq(n)].sort_values("k_correct_total")
        x = group.k_correct_total.to_numpy(float)
        y = group.correction_rate.to_numpy(float)
        axis.plot(x, y, marker="o", color=colors[n], linewidth=2)
        axis.fill_between(
            x,
            group.ci95_low.to_numpy(float),
            group.ci95_high.to_numpy(float),
            color=colors[n],
            alpha=0.18,
        )
        for row in group.itertuples(index=False):
            axis.annotate(
                f"N={row.cases}",
                (row.k_correct_total, row.correction_rate),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_title(f"n={n}")
        axis.set_xlabel("Correct peer answers at Round 0")
        axis.set_ylim(-0.04, 1.04)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("P(final correct | readout initially wrong)")
    fig.suptitle("Clean readout correction rises with observed correct-answer supply")
    fig.savefig(output / "correct_supply_curve.png", dpi=220)
    plt.close(fig)


def within_group_correlations(
    frame: pd.DataFrame,
    *,
    strata: list[str],
    predictor: str,
    outcome_column: str = "final_readout_correct",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute descriptive within-stratum rank associations.

    Constant-predictor or constant-outcome strata are excluded because their
    Spearman correlation is undefined; support is reported explicitly.
    """

    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(strata, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        if group[predictor].nunique() < 2 or group[outcome_column].nunique() < 2:
            continue
        correlation = float(
            spearmanr(group[predictor], group[outcome_column]).statistic
        )
        if not np.isfinite(correlation):
            continue
        rows.append(
            {
                **dict(zip(strata, key_tuple, strict=True)),
                "cases": len(group),
                "spearman": correlation,
            }
        )
    detail = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    if not detail.empty:
        for n, group in detail.groupby("n", sort=True):
            positive = int(group.spearman.gt(0).sum())
            negative = int(group.spearman.lt(0).sum())
            nonzero = positive + negative
            summaries.append(
                {
                    "n": int(n),
                    "eligible_strata": len(group),
                    "positive": positive,
                    "negative": negative,
                    "zero": int(group.spearman.eq(0).sum()),
                    "mean_spearman": float(group.spearman.mean()),
                    "median_spearman": float(group.spearman.median()),
                    "positive_sign_test_p_one_sided": (
                        float(binomtest(positive, nonzero, 0.5, alternative="greater").pvalue)
                        if nonzero
                        else float("nan")
                    ),
                }
            )
    return detail, pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_cases = pd.read_pickle(args.clean_cases_pkl)
    graphs = json.loads(args.graphs_json.read_text(encoding="utf-8"))
    clean_endpoints = pd.read_csv(args.clean_task_graph_csv)
    endpoints = build_endpoint_frame(clean_cases, clean_endpoints, graphs)
    initially_wrong = endpoints.loc[endpoints.round0_readout_correct.eq(0)].copy()
    if initially_wrong.empty:
        raise ValueError("no initially wrong readout cases")

    supply = clustered_bootstrap_rates(
        initially_wrong,
        group_columns=["n", "k_correct_total"],
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    distance = clustered_bootstrap_rates(
        initially_wrong,
        group_columns=["n", "k_correct_total", "earliest_correct_arrival"],
        samples=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    within_h = clustered_bootstrap_rates(
        initially_wrong,
        group_columns=[
            "n",
            "k_correct_within_h1",
            "k_correct_within_h2",
            "k_correct_within_h3",
        ],
        samples=args.bootstrap_samples,
        seed=args.seed + 2,
    )
    within_task, within_task_summary = within_group_correlations(
        initially_wrong,
        strata=["n", "task_id"],
        predictor="k_correct_total",
    )
    arrival_input = initially_wrong.loc[initially_wrong.k_correct_total.gt(0)]
    within_task_k_arrival, within_task_k_arrival_summary = within_group_correlations(
        arrival_input,
        strata=["n", "task_id", "k_correct_total"],
        predictor="earliest_correct_arrival",
    )
    difficulty_supply = pd.DataFrame()
    if args.difficulty_task_csv is not None:
        bands = pd.read_csv(args.difficulty_task_csv)
        if "threshold_scheme" in bands:
            bands = bands.loc[bands.threshold_scheme.eq("primary_aime_aligned")]
        bands = bands[["evaluation_n", "task_id", "difficulty_band"]].drop_duplicates()
        linked = initially_wrong.merge(
            bands,
            left_on=["n", "task_id"],
            right_on=["evaluation_n", "task_id"],
            validate="many_to_one",
        )
        difficulty_supply = (
            linked.groupby(["n", "difficulty_band"], sort=True)
            .agg(
                cases=("task_id", "size"),
                tasks=("task_id", "nunique"),
                mean_k_correct=("k_correct_total", "mean"),
                probability_any_correct_peer=("has_correct_peer", "mean"),
                correction_rate=("final_readout_correct", "mean"),
            )
            .reset_index()
        )

    endpoints.to_pickle(args.output_dir / "endpoint_round0_supply.pkl")
    endpoints.drop(columns=["initial_states"]).to_csv(
        args.output_dir / "endpoint_round0_supply.csv.gz", index=False
    )
    supply.to_csv(args.output_dir / "correct_supply_curve.csv", index=False)
    distance.to_csv(args.output_dir / "correct_supply_by_distance.csv", index=False)
    within_h.to_csv(args.output_dir / "correct_supply_by_horizon.csv", index=False)
    within_task.to_csv(
        args.output_dir / "within_task_supply_correlations.csv", index=False
    )
    within_task_summary.to_csv(
        args.output_dir / "within_task_supply_summary.csv", index=False
    )
    within_task_k_arrival.to_csv(
        args.output_dir / "within_task_k_arrival_correlations.csv", index=False
    )
    within_task_k_arrival_summary.to_csv(
        args.output_dir / "within_task_k_arrival_summary.csv", index=False
    )
    if not difficulty_supply.empty:
        difficulty_supply.to_csv(
            args.output_dir / "difficulty_supply_link.csv", index=False
        )
    plot_supply(supply, args.output_dir)

    audit = {
        "analysis_version": ANALYSIS_VERSION,
        "estimand": (
            "P(final readout correct | Round-0 readout incorrect, "
            "observed correct-peer supply)"
        ),
        "source_clean_cases": len(clean_cases),
        "clean_task_graph_runs": len(endpoints),
        "initially_wrong_readout_runs": len(initially_wrong),
        "tasks": int(endpoints.task_id.nunique()),
        "graphs": int(endpoints.graph_id.nunique()),
        "sizes": sorted(int(value) for value in endpoints.n.unique()),
        "round0_endpoint_match": bool(
            endpoints.round0_readout_correct.eq(endpoints.u0.astype(int)).all()
        ),
        "all_nodes_reachable_within_horizon": True,
        "reachable_h3_equals_total_correct_supply": bool(
            endpoints.k_correct_within_h3.eq(endpoints.k_correct_total).all()
        ),
        "bootstrap": {
            "unit": "task",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "state_note": (
            "Round-0 readout incorrect includes target, other, and unparsed states; "
            "in a clean run they are all simply non-correct outcomes."
        ),
        "claim_limit": (
            "This observational conditional analysis establishes association, not the "
            "causal effect of injecting an additional correct peer message."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "audit": audit,
                "correct_supply_curve": supply.to_dict(orient="records"),
                "within_task_supply_summary": within_task_summary.to_dict(
                    orient="records"
                ),
                "within_task_k_arrival_summary": within_task_k_arrival_summary.to_dict(
                    orient="records"
                ),
                "difficulty_supply_link": difficulty_supply.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
