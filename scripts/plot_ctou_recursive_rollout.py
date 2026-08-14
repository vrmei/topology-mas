"""Plot recursive C/T/O/U endpoint predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODELS = ("degroot_equal", "ctou_table", "ctou_logit")
LABELS = {
    "degroot_equal": "DeGroot",
    "ctou_table": "CTOU table",
    "ctou_logit": "CTOU logistic",
}
COLORS = {
    "degroot_equal": "#D55E00",
    "ctou_table": "#0072B2",
    "ctou_logit": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def plot_m_curves(curves: pd.DataFrame, output: Path) -> None:
    selected = curves[curves.rollout_mode.eq("particle")]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, outcome in enumerate(("target", "correct")):
        for column, n in enumerate((5, 8)):
            axis = axes[row, column]
            panel = selected[selected.n.eq(n)]
            observed = panel[["m", f"observed_{outcome}"]].drop_duplicates().sort_values("m")
            axis.plot(
                observed.m,
                observed[f"observed_{outcome}"],
                color="#111111",
                marker="o",
                linewidth=2.4,
                label="Observed LLM",
                zorder=5,
            )
            for model in MODELS:
                curve = panel[panel.model.eq(model)].sort_values("m")
                axis.plot(
                    curve.m,
                    curve[f"predicted_{outcome}"],
                    color=COLORS[model],
                    marker=".",
                    linewidth=1.8,
                    label=LABELS[model],
                )
            title = "Target risk" if outcome == "target" else "Attack accuracy"
            axis.set_title(f"{title}, n={n}")
            axis.set_xlabel("Directed edge count m")
            axis.set_ylabel("Endpoint rate")
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=9)
    figure.suptitle("Recursive rollout from observed Round-0 states", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_graph_scatter(predictions: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    selected = predictions[predictions.rollout_mode.eq("particle") & predictions.model.isin(MODELS)]
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for row, outcome in enumerate(("target", "correct")):
        for column, model in enumerate(MODELS):
            axis = axes[row, column]
            panel = selected[selected.model.eq(model)]
            for n, marker in ((5, "o"), (8, "^")):
                group = panel[panel.n.eq(n)]
                axis.scatter(
                    group[f"observed_{outcome}"],
                    group[f"predicted_{outcome}"],
                    alpha=0.7,
                    s=25,
                    marker=marker,
                    label=f"n={n}",
                )
            axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
            metric = metrics[
                metrics.model.eq(model)
                & metrics.rollout_mode.eq("particle")
                & metrics.outcome.eq(outcome)
            ]
            annotation = "\n".join(
                f"n={int(item.n)}: MAE={item.graph_mae:.3f}, rho={item.graph_spearman:.2f}"
                for item in metric.itertuples(index=False)
            )
            axis.text(0.03, 0.97, annotation, transform=axis.transAxes, va="top", fontsize=8)
            axis.set_title(f"{LABELS[model]} - {outcome}")
            axis.set_xlabel("Observed graph endpoint rate")
            axis.set_ylabel("Predicted graph endpoint rate")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
    axes[0, 2].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("Topology-level endpoint prediction", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_rollout_gap(predictions: pd.DataFrame, output: Path) -> None:
    selected = predictions[predictions.model.isin(("ctou_table", "ctou_logit"))]
    wide = selected.pivot(
        index=["model", "graph_id", "n", "m"],
        columns="rollout_mode",
        values=["predicted_target", "predicted_correct"],
    ).reset_index()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for axis, outcome in zip(axes, ("target", "correct"), strict=True):
        for model in ("ctou_table", "ctou_logit"):
            panel = wide[wide["model"].eq(model)]
            x = panel[(f"predicted_{outcome}", "particle")]
            y = panel[(f"predicted_{outcome}", "mean_field")]
            axis.scatter(x, y, s=20, alpha=0.65, label=LABELS[model])
        axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        axis.set_title(outcome)
        axis.set_xlabel("Joint particle rollout")
        axis.set_ylabel("Factorized mean-field rollout")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=9)
    figure.suptitle("Effect of the node-independence approximation", fontsize=14)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(args.input_dir / "m_curve_predictions.csv")
    graphs = pd.read_csv(args.input_dir / "graph_endpoint_predictions.csv")
    metrics = pd.read_csv(args.input_dir / "graph_endpoint_metrics.csv")
    plot_m_curves(curves, output / "recursive_m_curves.png")
    plot_graph_scatter(graphs, metrics, output / "recursive_graph_scatter.png")
    plot_rollout_gap(graphs, output / "recursive_rollout_mode_gap.png")


if __name__ == "__main__":
    main()
