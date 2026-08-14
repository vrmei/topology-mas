"""Plot CTOU support-stratified extrapolation errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STRATA = ("low_mass_le_5pct", "low_mass_5_20pct", "low_mass_gt_20pct")
STRATA_LABELS = ("<=5%", "5-20%", ">20%")
COLORS = {"correct_brier": "#0072B2", "target_brier": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plot_strata(summary: pd.DataFrame, output: Path) -> None:
    selected = summary[
        summary.validation_scope.eq("density_task")
        & summary.support_source.eq("expected_rollout")
        & summary.metric.eq("correct_brier")
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    for row, direction in enumerate(("sparse_to_dense", "dense_to_sparse")):
        for column, n in enumerate((5, 8)):
            axis = axes[row, column]
            panel = selected[selected.direction.eq(direction) & selected.n.eq(n)].set_index(
                "support_stratum"
            )
            panel = panel.reindex(STRATA)
            axis.errorbar(
                range(len(panel)),
                panel.estimate,
                yerr=[panel.estimate - panel.ci95_low, panel.ci95_high - panel.estimate],
                color=COLORS["correct_brier"],
                marker="o",
                capsize=4,
                linewidth=2,
            )
            axis.set_xticks(range(len(panel)), STRATA_LABELS)
            axis.set_xlabel("Expected mass on cells with training count <20")
            axis.set_ylabel("Correct Brier")
            axis.set_title(f"{direction.replace('_', ' ')}, n={n}")
            axis.grid(alpha=0.25)
    figure.suptitle("Endpoint error stratified by prediction-time support burden", fontsize=14)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_task_conditional(summary: pd.DataFrame, output: Path) -> None:
    selected = summary[summary.error_metric.isin(("correct_brier", "target_brier"))]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    conditions = ("within_task_density", "within_task_graph")
    condition_labels = ("Within task+density", "Within task+graph")
    for row, direction in enumerate(("sparse_to_dense", "dense_to_sparse")):
        for column, n in enumerate((5, 8)):
            axis = axes[row, column]
            panel = selected[selected.direction.eq(direction) & selected.n.eq(n)]
            for offset, metric in ((-0.07, "correct_brier"), (0.07, "target_brier")):
                curve = panel[panel.error_metric.eq(metric)].set_index("conditioning").reindex(
                    conditions
                )
                x = [index + offset for index in range(len(conditions))]
                axis.errorbar(
                    x,
                    curve.mean_task_correlation,
                    yerr=[
                        curve.mean_task_correlation - curve.ci95_low,
                        curve.ci95_high - curve.mean_task_correlation,
                    ],
                    color=COLORS[metric],
                    marker="o",
                    capsize=4,
                    linestyle="none",
                    label=metric.replace("_", " "),
                )
            axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
            axis.set_xticks(range(len(conditions)), condition_labels, rotation=10)
            axis.set_ylabel("Mean per-task rank association")
            axis.set_title(f"{direction.replace('_', ' ')}, n={n}")
            axis.grid(alpha=0.2)
    axes[0, 1].legend(frameon=False)
    figure.suptitle("Support-error association after controlling task and structure", fontsize=14)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_graph_residuals(graph: pd.DataFrame, output: Path) -> None:
    selected = graph[graph.validation_scope.eq("density_task")]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    for row, direction in enumerate(("sparse_to_dense", "dense_to_sparse")):
        for column, n in enumerate((5, 8)):
            axis = axes[row, column]
            panel = selected[selected.direction.eq(direction) & selected.n.eq(n)]
            scatter = axis.scatter(
                panel.expected_support_lt_20_fraction,
                panel.signed_correct_residual,
                c=panel.m,
                cmap="viridis",
                s=42,
                alpha=0.8,
            )
            axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
            axis.set_xlabel("Graph-average low-support mass")
            axis.set_ylabel("Predicted - observed correct rate")
            axis.set_title(f"{direction.replace('_', ' ')}, n={n}")
            axis.grid(alpha=0.2)
            figure.colorbar(scatter, ax=axis, label="m")
    figure.suptitle("Support burden and signed graph-level calibration error", fontsize=14)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    strata = pd.read_csv(args.input_dir / "endpoint_support_strata.csv")
    task = pd.read_csv(args.input_dir / "primary_task_conditional_summary.csv")
    graph = pd.read_csv(args.input_dir / "graph_support_residuals.csv")
    plot_strata(strata, args.output_dir / "support_stratified_correct_brier.png")
    plot_task_conditional(task, args.output_dir / "support_conditional_association.png")
    plot_graph_residuals(graph, args.output_dir / "support_graph_residuals.png")


if __name__ == "__main__":
    main()
