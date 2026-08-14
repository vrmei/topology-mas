#!/usr/bin/env python3
"""Plot the clean/attack CTOU analysis without refitting any model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAW_LABELS = {
    "clean_specific": "Clean-only",
    "attack_specific": "Attack-only",
    "pooled_balanced": "Balanced pooled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def plot_local_law(summary: pd.DataFrame, output: Path) -> None:
    frame = summary[
        summary.support_scope.eq("shared_cell") & summary.metric.eq("multiclass_brier")
    ].copy()
    conditions = ["clean", "attack"]
    laws = ["clean_specific", "attack_specific", "pooled_balanced"]
    colors = ["#277DA1", "#F94144", "#43AA8B"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for ax, condition in zip(axes, conditions, strict=True):
        selected = frame[frame.condition.eq(condition)].set_index("law").loc[laws]
        x = np.arange(len(laws))
        estimate = selected.estimate.to_numpy(float)
        lower = estimate - selected.ci95_low.to_numpy(float)
        upper = selected.ci95_high.to_numpy(float) - estimate
        ax.bar(x, estimate, color=colors, width=0.66)
        ax.errorbar(x, estimate, yerr=np.vstack([lower, upper]), fmt="none", color="black")
        ax.set_xticks(x, [LAW_LABELS[law] for law in laws], rotation=18, ha="right")
        ax.set_title(f"Test condition: {condition}")
        ax.set_ylabel("Multiclass Brier loss")
    fig.suptitle("Local transition-law transfer on exactly shared CTOU cells")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def primary_graphs(graph: pd.DataFrame) -> pd.DataFrame:
    return graph[
        graph.law.eq("pooled_balanced")
        & graph.initialization.eq("correlated_empirical")
        & graph.rollout_mode.eq("particle")
    ].copy()


def plot_endpoint_calibration(graph: pd.DataFrame, output: Path) -> None:
    frame = primary_graphs(graph)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.8), sharex=True, sharey=True)
    for ax, ((condition, n), selected) in zip(
        axes.ravel(), frame.groupby(["condition", "n"], sort=True), strict=True
    ):
        ax.scatter(
            selected.observed_correct,
            selected.predicted_correct,
            c=selected.m,
            cmap="viridis",
            s=28,
            alpha=0.8,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
        ax.set_title(f"{condition}, n={n}")
        ax.set_xlabel("Observed endpoint accuracy")
        ax.set_ylabel("Predicted endpoint accuracy")
        ax.set_xlim(0.70, 0.96)
        ax.set_ylim(0.70, 0.96)
    fig.suptitle("Cross-held-out graph calibration: unified CTOU surrogate")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def curve_with_graph_variation(graphs: pd.DataFrame) -> pd.DataFrame:
    return (
        graphs.groupby(["n", "m"], sort=True)
        .agg(
            graphs=("graph_id", "size"),
            observed_u0=("observed_u0", "mean"),
            observed_utility=("observed_utility", "mean"),
            predicted_utility=("predicted_utility", "mean"),
            utility_sd=("observed_utility", "std"),
            observed_robustness=("observed_robustness", "mean"),
            predicted_robustness=("predicted_robustness", "mean"),
            robustness_sd=("observed_robustness", "std"),
        )
        .reset_index()
    )


def plot_utility_robustness_curves(curves: pd.DataFrame, output: Path) -> None:
    ns = sorted(curves.n.unique())
    fig, axes = plt.subplots(len(ns), 1, figsize=(8.8, 3.5 * len(ns)), squeeze=False)
    for ax, n in zip(axes.ravel(), ns, strict=True):
        selected = curves[curves.n.eq(n)].sort_values("m")
        x = selected.m.to_numpy(float)
        ax.errorbar(
            x,
            selected.observed_utility,
            yerr=selected.utility_sd.fillna(0),
            marker="o",
            color="#277DA1",
            label="Observed utility (mean ± graph SD)",
        )
        ax.plot(x, selected.predicted_utility, "--", color="#277DA1", label="Predicted utility")
        ax.errorbar(
            x,
            selected.observed_robustness,
            yerr=selected.robustness_sd.fillna(0),
            marker="s",
            color="#F94144",
            label="Observed robustness (mean ± graph SD)",
        )
        ax.plot(
            x,
            selected.predicted_robustness,
            "--",
            color="#F94144",
            label="Predicted robustness",
        )
        ax.set_title(f"n={n}")
        ax.set_xlabel("Directed edge count m")
        ax.set_ylabel("Endpoint accuracy")
        ax.set_ylim(0, 1)
        ax.legend(ncol=2, fontsize=8)
    fig.suptitle("Clean utility and attack robustness over edge density")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_utility_robustness_scatter(graphs: pd.DataFrame, output: Path) -> None:
    ns = sorted(graphs.n.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(5.0 * len(ns), 4.2), squeeze=False)
    for ax, n in zip(axes.ravel(), ns, strict=True):
        selected = graphs[graphs.n.eq(n)]
        scatter = ax.scatter(
            selected.observed_utility,
            selected.observed_robustness,
            c=selected.m,
            cmap="viridis",
            s=34,
            alpha=0.8,
        )
        ax.set_title(f"n={n}")
        ax.set_xlabel("Observed clean utility")
        ax.set_ylabel("Observed attack robustness")
        fig.colorbar(scatter, ax=ax, label="m")
    fig.suptitle("Graph-level utility–robustness landscape")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    summary = pd.read_csv(args.input_dir / "one_step_loss_summary.csv")
    graph = pd.read_csv(args.input_dir / "graph_endpoint_predictions.csv")
    utility_graphs = pd.read_csv(args.input_dir / "utility_robustness_graphs.csv")
    curves = curve_with_graph_variation(utility_graphs)
    curves.to_csv(args.input_dir / "utility_robustness_curve_graph_variation.csv", index=False)
    plot_local_law(summary, args.output_dir / "local_law_shared_cell_brier.png")
    plot_endpoint_calibration(graph, args.output_dir / "endpoint_graph_calibration.png")
    plot_utility_robustness_curves(curves, args.output_dir / "utility_robustness_curves.png")
    plot_utility_robustness_scatter(
        utility_graphs, args.output_dir / "utility_robustness_graph_scatter.png"
    )


if __name__ == "__main__":
    main()
