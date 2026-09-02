"""Render the primary 2026 AIME full-rationale clean-MAS result figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def percent_axis(axis: plt.Axes) -> None:
    axis.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edge = pd.read_csv(args.analysis_dir / "edge_level_metrics.csv")
    graph = pd.read_csv(args.analysis_dir / "graph_metrics.csv")
    difficulty = pd.read_csv(args.analysis_dir / "difficulty_band_metrics.csv")

    # Density-level response curves. The complete graph is shown separately
    # because it has no graph-axis replication.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = edge.edge_count.to_numpy()
    for metric, label, color in [
        ("initial_utility", "Round 0", "#7A869A"),
        ("final_utility", "Round 3", "#006D77"),
    ]:
        y = edge[metric].to_numpy()
        low = edge[f"{metric}_hierarchical_bootstrap_95_low"].to_numpy()
        high = edge[f"{metric}_hierarchical_bootstrap_95_high"].to_numpy()
        axes[0].errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            marker="o",
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )
    axes[0].axvline(16, linestyle="--", linewidth=1, color="#C9CDD4")
    axes[0].annotate("unique complete graph", (16, 0.39), xytext=(12.6, 0.29),
                     arrowprops={"arrowstyle": "->", "color": "#7A869A"},
                     fontsize=9, color="#59636E")
    axes[0].set(title="Utility by edge count", xlabel="Directed edges (m)", ylabel="Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_ylim(0.15, 0.92)
    axes[0].legend(frameon=False)
    percent_axis(axes[0])

    y = edge.paired_delta.to_numpy()
    low = edge.paired_delta_hierarchical_bootstrap_95_low.to_numpy()
    high = edge.paired_delta_hierarchical_bootstrap_95_high.to_numpy()
    colors = ["#D98E04" if value < 16 else "#B8BDC7" for value in x]
    axes[1].bar(x, y, width=2.6, color=colors)
    axes[1].errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="none",
                     ecolor="#30343B", capsize=4, linewidth=1.5)
    axes[1].axhline(0, linewidth=1, color="#30343B")
    axes[1].set(title="Paired communication gain", xlabel="Directed edges (m)", ylabel="Round 3 − Round 0")
    axes[1].set_xticks(x)
    axes[1].set_ylim(-0.02, 0.52)
    percent_axis(axes[1])
    fig.savefig(args.output_dir / "density_utility_full_rationale.png", dpi=220)
    plt.close(fig)

    # Difficulty-specific gains, using the externally frozen bands.
    order = ["floor", "informative", "ceiling"]
    d = difficulty.set_index("difficulty_band").loc[order]
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    positions = np.arange(len(order))
    y = d.paired_delta.to_numpy()
    low = d.paired_delta_task_bootstrap_95_low.to_numpy()
    high = d.paired_delta_task_bootstrap_95_high.to_numpy()
    axis.bar(positions, y, color=["#A8B0BA", "#006D77", "#A8B0BA"], width=0.62)
    axis.errorbar(positions, y, yerr=np.vstack([y - low, high - y]), fmt="none",
                  ecolor="#30343B", capsize=5, linewidth=1.6)
    axis.set_xticks(positions, ["Floor (9)", "Intermediate (12)", "Ceiling (9)"])
    axis.set(title="Communication gain peaks at intermediate difficulty",
             ylabel="Round 3 − Round 0")
    axis.set_ylim(0, 0.40)
    percent_axis(axis)
    fig.savefig(args.output_dir / "difficulty_gain_full_rationale.png", dpi=220)
    plt.close(fig)

    # Every point is one independently executed graph over all 30 tasks.
    fig, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    cmap = {4: "#4C78A8", 8: "#F58518", 12: "#54A24B", 16: "#B8BDC7"}
    for m, group in graph.groupby("edge_count"):
        axis.scatter(group.initial_utility, group.final_utility, s=70,
                     color=cmap[int(m)], label=f"m={int(m)}", alpha=0.9,
                     edgecolor="white", linewidth=0.8)
    axis.plot([0.3, 0.65], [0.3, 0.65], linestyle="--", color="#7A869A", linewidth=1)
    axis.set(xlabel="Round-0 utility", ylabel="Round-3 utility",
             title="Graph-level outcomes (30 tasks per graph)")
    axis.set_xlim(0.32, 0.64)
    axis.set_ylim(0.56, 0.80)
    percent_axis(axis)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    axis.legend(frameon=False, ncol=2)
    fig.savefig(args.output_dir / "graph_outcomes_full_rationale.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
