"""Plot the CTOU joint residual co-movement pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.analysis_dir / "comovement_summary.csv")
    slopes = pd.read_csv(args.analysis_dir / "within_event_topology_slopes.csv")

    models = ["ctou_table", "ctou_logit", "provenance_table", "provenance_logit"]
    colors = {
        "ctou_table": "#666666",
        "ctou_logit": "#AAAAAA",
        "provenance_table": "#0072B2",
        "provenance_logit": "#D55E00",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))

    selected = summary[
        summary.metric.eq("residual_product")
        & summary.residual_variant.eq("task_cell_adjusted")
        & summary.subset.eq("same_cell")
        & summary.pair_scope.eq("all")
    ]
    width = 0.18
    x = np.arange(2)
    for index, model in enumerate(models):
        frame = selected[selected.model.eq(model)].set_index("outcome").loc[["correct", "target"]]
        position = x + (index - 1.5) * width
        estimate = frame.estimate.to_numpy(float)
        lower = frame.task_ci95_low.to_numpy(float)
        upper = frame.task_ci95_high.to_numpy(float)
        axes[0].bar(position, estimate, width, color=colors[model], label=model)
        axes[0].errorbar(
            position,
            estimate,
            yerr=np.vstack((estimate - lower, upper - estimate)),
            fmt="none",
            ecolor="black",
            capsize=2,
            linewidth=0.8,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, ["Correct", "Target"])
    axes[0].set_ylabel("Mean residual product")
    axes[0].set_title("Residual co-movement within exact local cells")
    axes[0].grid(axis="y", alpha=0.25)

    selected = slopes[
        slopes.residual_variant.eq("task_cell_adjusted")
        & slopes.feature.eq("causal_jaccard")
        & slopes.pair_scope.eq("all")
        & slopes.n_group.astype(str).eq("all")
        & slopes.model.isin(["ctou_table", "provenance_table"])
    ]
    positions = []
    labels = []
    cursor = 0
    for subset in ("all", "same_cell"):
        for outcome in ("correct", "target"):
            for model in ("ctou_table", "provenance_table"):
                row = selected[
                    selected.subset.eq(subset)
                    & selected.outcome.eq(outcome)
                    & selected.model.eq(model)
                ].iloc[0]
                axes[1].errorbar(
                    cursor,
                    row.slope,
                    yerr=[[row.slope - row.task_ci95_low], [row.task_ci95_high - row.slope]],
                    fmt="o",
                    color=colors[model],
                    capsize=3,
                )
                positions.append(cursor)
                labels.append(f"{subset}\n{outcome}\n{model.replace('_table', '')}")
                cursor += 1
            cursor += 0.5
        cursor += 0.5
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions, labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Within-event slope")
    axes[1].set_title("Causal-overlap association before/after exact-cell matching")
    axes[1].grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(args.output_dir / "joint_residual_summary.png", dpi=200)


if __name__ == "__main__":
    main()
