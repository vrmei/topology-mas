#!/usr/bin/env python3
"""Relate clean readout corruption to observed Round-0 erroneous peer supply."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from analyze_round0_correct_supply import (
    clustered_bootstrap_rates,
    shortest_distances_to_readout,
    within_group_correlations,
)

STATE_CORRECT = 0
STATE_TARGET = 1
STATE_OTHER = 2
STATE_UNPARSED = 3
ANALYSIS_VERSION = "round0-error-supply-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-supply-pkl", type=Path, required=True)
    parser.add_argument("--graphs-json", type=Path, required=True)
    parser.add_argument("--difficulty-task-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def add_error_supply(
    endpoints: pd.DataFrame,
    graphs: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in endpoints.itertuples(index=False):
        states = tuple(case.initial_states)
        readout = int(case.readout_node)
        peers = [node for node in range(int(case.n)) if node != readout]
        distances = shortest_distances_to_readout(graphs[str(case.graph_id)])
        noncorrect_distances = [
            distances[node] for node in peers if states[node] != STATE_CORRECT
        ]
        parsed_wrong_distances = [
            distances[node]
            for node in peers
            if states[node] in (STATE_TARGET, STATE_OTHER)
        ]
        row = case._asdict()
        row.update(
            {
                "final_readout_corrupted": int(not bool(case.final_readout_correct)),
                "k_noncorrect_total": len(noncorrect_distances),
                "k_parsed_wrong_total": len(parsed_wrong_distances),
                "k_unparsed_total": sum(states[node] == STATE_UNPARSED for node in peers),
                "k_noncorrect_within_h1": sum(x <= 1 for x in noncorrect_distances),
                "k_noncorrect_within_h2": sum(x <= 2 for x in noncorrect_distances),
                "k_noncorrect_within_h3": sum(x <= 3 for x in noncorrect_distances),
                "earliest_noncorrect_arrival": (
                    min(noncorrect_distances) if noncorrect_distances else 0
                ),
                "has_noncorrect_peer": int(bool(noncorrect_distances)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_error_supply(curve: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)
    colors = {5: "#4C78A8", 8: "#F58518"}
    for axis, n in zip(axes, (5, 8), strict=True):
        group = curve.loc[curve.n.eq(n)].sort_values("k_noncorrect_total")
        x = group.k_noncorrect_total.to_numpy(float)
        y = group.corruption_rate.to_numpy(float)
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
                (row.k_noncorrect_total, row.corruption_rate),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_title(f"n={n}")
        axis.set_xlabel("Non-correct peer answers at Round 0")
        axis.set_ylim(-0.04, 1.04)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("P(final non-correct | readout initially correct)")
    fig.suptitle("Clean readout corruption versus erroneous peer supply")
    fig.savefig(output / "error_supply_curve.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoints = pd.read_pickle(args.endpoint_supply_pkl)
    graphs = json.loads(args.graphs_json.read_text(encoding="utf-8"))
    enriched = add_error_supply(endpoints, graphs)
    initially_correct = enriched.loc[enriched.round0_readout_correct.eq(1)].copy()
    if initially_correct.empty:
        raise ValueError("no initially correct readout cases")

    primary = clustered_bootstrap_rates(
        initially_correct,
        group_columns=["n", "k_noncorrect_total"],
        outcome_column="final_readout_corrupted",
        rate_name="corruption_rate",
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    parsed_wrong = clustered_bootstrap_rates(
        initially_correct,
        group_columns=["n", "k_parsed_wrong_total"],
        outcome_column="final_readout_corrupted",
        rate_name="corruption_rate",
        samples=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    distance = clustered_bootstrap_rates(
        initially_correct,
        group_columns=["n", "k_noncorrect_total", "earliest_noncorrect_arrival"],
        outcome_column="final_readout_corrupted",
        rate_name="corruption_rate",
        samples=args.bootstrap_samples,
        seed=args.seed + 2,
    )
    within_task, within_task_summary = within_group_correlations(
        initially_correct,
        strata=["n", "task_id"],
        predictor="k_noncorrect_total",
        outcome_column="final_readout_corrupted",
    )
    arrival_input = initially_correct.loc[initially_correct.k_noncorrect_total.gt(0)]
    within_task_k_arrival, within_task_k_arrival_summary = within_group_correlations(
        arrival_input,
        strata=["n", "task_id", "k_noncorrect_total"],
        predictor="earliest_noncorrect_arrival",
        outcome_column="final_readout_corrupted",
    )

    difficulty_link = pd.DataFrame()
    if args.difficulty_task_csv is not None:
        bands = pd.read_csv(args.difficulty_task_csv)
        if "threshold_scheme" in bands:
            bands = bands.loc[bands.threshold_scheme.eq("primary_aime_aligned")]
        bands = bands[["evaluation_n", "task_id", "difficulty_band"]].drop_duplicates()
        linked = initially_correct.merge(
            bands,
            left_on=["n", "task_id"],
            right_on=["evaluation_n", "task_id"],
            validate="many_to_one",
        )
        difficulty_link = (
            linked.groupby(["n", "difficulty_band"], sort=True)
            .agg(
                cases=("task_id", "size"),
                tasks=("task_id", "nunique"),
                mean_k_noncorrect=("k_noncorrect_total", "mean"),
                probability_any_noncorrect_peer=("has_noncorrect_peer", "mean"),
                corruption_rate=("final_readout_corrupted", "mean"),
            )
            .reset_index()
        )

    enriched.to_pickle(args.output_dir / "endpoint_round0_error_supply.pkl")
    enriched.drop(columns=["initial_states"]).to_csv(
        args.output_dir / "endpoint_round0_error_supply.csv.gz", index=False
    )
    primary.to_csv(args.output_dir / "error_supply_curve.csv", index=False)
    parsed_wrong.to_csv(args.output_dir / "parsed_wrong_supply_curve.csv", index=False)
    distance.to_csv(args.output_dir / "error_supply_by_distance.csv", index=False)
    within_task.to_csv(
        args.output_dir / "within_task_error_supply_correlations.csv", index=False
    )
    within_task_summary.to_csv(
        args.output_dir / "within_task_error_supply_summary.csv", index=False
    )
    within_task_k_arrival.to_csv(
        args.output_dir / "within_task_k_error_arrival_correlations.csv", index=False
    )
    within_task_k_arrival_summary.to_csv(
        args.output_dir / "within_task_k_error_arrival_summary.csv", index=False
    )
    if not difficulty_link.empty:
        difficulty_link.to_csv(args.output_dir / "difficulty_error_supply_link.csv", index=False)
    plot_error_supply(primary, args.output_dir)

    audit = {
        "analysis_version": ANALYSIS_VERSION,
        "estimand": (
            "P(final readout non-correct | Round-0 readout correct, "
            "observed non-correct-peer supply)"
        ),
        "source_clean_runs": len(enriched),
        "initially_correct_readout_runs": len(initially_correct),
        "tasks": int(enriched.task_id.nunique()),
        "graphs": int(enriched.graph_id.nunique()),
        "sizes": sorted(int(value) for value in enriched.n.unique()),
        "noncorrect_definition": "target + other + unparsed",
        "parsed_wrong_definition": "target + other; excludes unparsed",
        "reachable_h3_equals_total_noncorrect_supply": bool(
            enriched.k_noncorrect_within_h3.eq(enriched.k_noncorrect_total).all()
        ),
        "bootstrap": {
            "unit": "task",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "claim_limit": (
            "This observational conditional analysis establishes association, not the "
            "causal effect of injecting an additional erroneous peer message."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "audit": audit,
                "error_supply_curve": primary.to_dict(orient="records"),
                "parsed_wrong_supply_curve": parsed_wrong.to_dict(orient="records"),
                "within_task_error_supply_summary": within_task_summary.to_dict(
                    orient="records"
                ),
                "within_task_k_error_arrival_summary": (
                    within_task_k_arrival_summary.to_dict(orient="records")
                ),
                "difficulty_error_supply_link": difficulty_link.to_dict(
                    orient="records"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
