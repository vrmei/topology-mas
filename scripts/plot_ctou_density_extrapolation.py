"""Plot CTOU leave-density-out and range-extrapolation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "observed": "#111111",
    "ctou_table": "#0072B2",
    "degroot_equal": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plot_leave_level_curves(curves: pd.DataFrame, output: Path) -> None:
    selected = curves[
        curves.experiment.eq("leave_level_out")
        & curves.validation_scope.eq("density_task")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for axis, n in zip(axes, (5, 8), strict=True):
        panel = selected[selected.n.eq(n)]
        observed = panel[["m", "observed_correct"]].drop_duplicates().sort_values("m")
        axis.plot(
            observed.m,
            observed.observed_correct,
            color=COLORS["observed"],
            marker="o",
            linewidth=2.4,
            label="Observed LLM",
        )
        for model, label in (("ctou_table", "CTOU"), ("degroot_equal", "DeGroot")):
            curve = panel[panel.model.eq(model)].sort_values("m")
            axis.plot(
                curve.m,
                curve.predicted_correct,
                color=COLORS[model],
                marker=".",
                linewidth=1.8,
                label=label,
            )
        axis.set_title(f"n={n}")
        axis.set_xlabel("Held-out directed edge count m")
        axis.set_ylabel("Attack accuracy")
        axis.set_ylim(0.4, 0.95)
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.suptitle("Leave-one-density-level-out recursive prediction", fontsize=14)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_range_curves(curves: pd.DataFrame, output: Path) -> None:
    selected = curves[
        curves.experiment.eq("range_extrapolation")
        & curves.validation_scope.eq("density_task")
        & curves.model.eq("ctou_table")
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, direction in enumerate(("sparse_to_dense", "dense_to_sparse")):
        for column, n in enumerate((5, 8)):
            axis = axes[row, column]
            panel = selected[selected.direction.eq(direction) & selected.n.eq(n)].sort_values("m")
            axis.plot(
                panel.m,
                panel.observed_correct,
                color=COLORS["observed"],
                marker="o",
                linewidth=2.4,
                label="Observed LLM",
            )
            axis.plot(
                panel.m,
                panel.predicted_correct,
                color=COLORS["ctou_table"],
                marker=".",
                linewidth=1.8,
                label="CTOU",
            )
            label = (
                "Sparse train to dense test"
                if direction == "sparse_to_dense"
                else "Dense train to sparse test"
            )
            axis.set_title(f"{label}, n={n}")
            axis.set_xlabel("Directed edge count m")
            axis.set_ylabel("Attack accuracy")
            axis.set_ylim(0.68, 0.92)
            axis.grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)
    figure.suptitle("Out-of-range CTOU extrapolation", fontsize=14)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_range_graph_scatter(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output: Path,
) -> None:
    selected = predictions[
        predictions.experiment.eq("range_extrapolation")
        & predictions.validation_scope.eq("density_task")
        & predictions.model.eq("ctou_table")
    ]
    metric_selected = metrics[
        metrics.experiment.eq("range_extrapolation")
        & metrics.validation_scope.eq("density_task")
        & metrics.model.eq("ctou_table")
        & metrics.outcome.eq("correct")
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for row, direction in enumerate(("sparse_to_dense", "dense_to_sparse")):
        for column, n in enumerate((5, 8)):
            axis = axes[row, column]
            panel = selected[selected.direction.eq(direction) & selected.n.eq(n)]
            axis.scatter(
                panel.observed_correct,
                panel.predicted_correct,
                color=COLORS["ctou_table"],
                alpha=0.75,
                s=30,
            )
            lower = min(panel.observed_correct.min(), panel.predicted_correct.min()) - 0.02
            upper = max(panel.observed_correct.max(), panel.predicted_correct.max()) + 0.02
            axis.plot([lower, upper], [lower, upper], linestyle="--", color="#777777")
            metric = metric_selected[
                metric_selected.direction.eq(direction) & metric_selected.n.eq(n)
            ].iloc[0]
            axis.text(
                0.04,
                0.96,
                f"MAE={metric.graph_mae:.3f}\nrho={metric.graph_spearman:.2f}",
                transform=axis.transAxes,
                va="top",
            )
            label = "Sparse to dense" if direction == "sparse_to_dense" else "Dense to sparse"
            axis.set_title(f"{label}, n={n}")
            axis.set_xlabel("Observed graph attack accuracy")
            axis.set_ylabel("Predicted graph attack accuracy")
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.grid(alpha=0.2)
    figure.suptitle("Out-of-range topology ranking", fontsize=14)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_composition_support(support: pd.DataFrame, output: Path) -> None:
    selected = support[
        support.experiment.eq("range_extrapolation")
        & support.validation_scope.eq("density_task")
    ].copy()
    weighted_rows: list[dict[str, float | int | str]] = []
    for (direction, n, m), group in selected.groupby(["direction", "n", "m"]):
        weights = group.test_updates
        weighted_rows.append(
            {
                "direction": direction,
                "n": int(n),
                "m": int(m),
                "exact": (group.exact_transition_cell_coverage * weights).sum() / weights.sum(),
                "composition": (group.composition_cell_coverage * weights).sum() / weights.sum(),
            }
        )
    weighted = pd.DataFrame(weighted_rows)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, direction in enumerate(("sparse_to_dense", "dense_to_sparse")):
        for column, n in enumerate((5, 8)):
            axis = axes[row, column]
            panel = weighted[weighted.direction.eq(direction) & weighted.n.eq(n)].sort_values("m")
            axis.plot(panel.m, panel.exact, marker="o", label="Exact transition cell")
            axis.plot(panel.m, panel.composition, marker="s", label="Composition only")
            axis.set_ylim(0, 1.03)
            axis.set_xlabel("Test edge count m")
            axis.set_ylabel("Training-support coverage")
            axis.set_title(f"{direction.replace('_', ' ')}, n={n}")
            axis.grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)
    figure.suptitle("Local state support under density-range shift", fontsize=14)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(args.input_dir / "m_curve_predictions.csv")
    graph_predictions = pd.read_csv(args.input_dir / "graph_endpoint_predictions.csv")
    graph_metrics = pd.read_csv(args.input_dir / "graph_endpoint_metrics.csv")
    plot_leave_level_curves(curves, args.output_dir / "leave_level_correct_curves.png")
    plot_range_curves(curves, args.output_dir / "range_correct_curves.png")
    plot_range_graph_scatter(
        graph_predictions,
        graph_metrics,
        args.output_dir / "range_graph_scatter.png",
    )
    support_path = args.input_dir / "composition_support.csv"
    if support_path.exists():
        plot_composition_support(
            pd.read_csv(support_path),
            args.output_dir / "composition_support.png",
        )


if __name__ == "__main__":
    main()
