"""Plot split-half reliability and Round-0-free CTOU results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "oracle": "#1f4e79",
    "iid_empirical": "#e07a5f",
    "correlated_empirical": "#3d9970",
}
LABELS = {
    "oracle": "Observed Round 0",
    "iid_empirical": "IID empirical prior",
    "correlated_empirical": "Correlated empirical prior",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plot_split_half(analysis: Path, output: Path) -> None:
    frame = pd.read_csv(analysis / "split_half_draws.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for axis, outcome in zip(axes, ("correct", "target"), strict=True):
        selected = frame[frame.outcome.eq(outcome)]
        values = [
            selected[selected.n.eq(n)].split_half_spearman.to_numpy(float)
            for n in (5, 8)
        ]
        violin = axis.violinplot(values, positions=[0, 1], showmedians=True, widths=0.75)
        for body in violin["bodies"]:
            body.set_facecolor("#6c8ebf")
            body.set_alpha(0.65)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xticks([0, 1], ["n=5", "n=8"])
        axis.set_title(f"Final {outcome} rate")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("25-vs-25 task split graph Spearman")
    figure.suptitle("Finite-task graph-ranking stability")
    figure.tight_layout()
    figure.savefig(output / "split_half_graph_stability.png", dpi=220)
    plt.close(figure)


def plot_graph_scatter(analysis: Path, output: Path) -> None:
    frame = pd.read_csv(analysis / "graph_endpoint_predictions.csv")
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharex="row", sharey="row")
    for row, outcome in enumerate(("correct", "target")):
        for column, initialization in enumerate(COLORS):
            axis = axes[row, column]
            selected = frame[frame.initialization.eq(initialization)]
            for n, marker in ((5, "o"), (8, "^")):
                group = selected[selected.n.eq(n)]
                axis.scatter(
                    group[f"observed_{outcome}"],
                    group[f"predicted_{outcome}"],
                    s=24,
                    alpha=0.72,
                    marker=marker,
                    color=COLORS[initialization],
                    label=f"n={n}",
                )
            low = min(
                selected[f"observed_{outcome}"].min(),
                selected[f"predicted_{outcome}"].min(),
            )
            high = max(
                selected[f"observed_{outcome}"].max(),
                selected[f"predicted_{outcome}"].max(),
            )
            axis.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=0.8)
            axis.set_title(LABELS[initialization])
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel(f"Predicted final {outcome} rate")
            if row == 1:
                axis.set_xlabel(f"Observed final {outcome} rate")
            if row == 0 and column == 2:
                axis.legend(frameon=False)
    figure.suptitle("Graph-level recursive CTOU prediction by initialization source")
    figure.tight_layout()
    figure.savefig(output / "round_zero_free_graph_scatter.png", dpi=220)
    plt.close(figure)


def plot_m_curves(analysis: Path, output: Path) -> None:
    frame = pd.read_csv(analysis / "m_curve_predictions.csv")
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for row, outcome in enumerate(("correct", "target")):
        for column, n in enumerate((5, 8)):
            axis = axes[row, column]
            selected = frame[frame.n.eq(n)]
            observed = (
                selected.groupby("m", sort=True)[f"observed_{outcome}"].first().reset_index()
            )
            axis.plot(
                observed.m,
                observed[f"observed_{outcome}"],
                color="#222222",
                linewidth=2.0,
                marker="o",
                label="Observed",
            )
            for initialization in COLORS:
                group = selected[selected.initialization.eq(initialization)].sort_values("m")
                axis.plot(
                    group.m,
                    group[f"predicted_{outcome}"],
                    color=COLORS[initialization],
                    linewidth=1.5,
                    marker=".",
                    label=LABELS[initialization],
                )
            axis.set_title(f"n={n}, final {outcome}")
            axis.set_xlabel("Directed edges (m)")
            axis.set_ylabel("Endpoint rate")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Density response under observed and empirical Round-0 initialization")
    figure.tight_layout()
    figure.savefig(output / "round_zero_free_m_curves.png", dpi=220)
    plt.close(figure)


def plot_graph_spearman(analysis: Path, output: Path) -> None:
    frame = pd.read_csv(analysis / "graph_endpoint_metrics.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    x = np.arange(len(COLORS))
    width = 0.34
    for axis, outcome in zip(axes, ("correct", "target"), strict=True):
        selected = frame[frame.outcome.eq(outcome)].set_index(["initialization", "n"])
        for offset, n in ((-width / 2, 5), (width / 2, 8)):
            values = [selected.loc[(name, n), "graph_spearman"] for name in COLORS]
            axis.bar(x + offset, values, width=width, label=f"n={n}")
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xticks(x, ["Observed\nR0", "IID\nprior", "Correlated\nprior"])
        axis.set_title(f"Final {outcome} rate")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Graph-level Spearman")
    axes[1].legend(frameon=False)
    figure.suptitle("Topology ranking retained without observed Round 0")
    figure.tight_layout()
    figure.savefig(output / "round_zero_free_graph_spearman.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_split_half(args.analysis_dir, args.output_dir)
    plot_graph_scatter(args.analysis_dir, args.output_dir)
    plot_m_curves(args.analysis_dir, args.output_dir)
    plot_graph_spearman(args.analysis_dir, args.output_dir)


if __name__ == "__main__":
    main()
