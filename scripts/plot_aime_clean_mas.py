#!/usr/bin/env python3
"""Plot the five-graph AIME clean-MAS utility analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args()


def errorbar(ax, frame, metric: str, label: str, color: str, marker: str) -> None:
    x = frame.edge_count.to_numpy(dtype=float)
    y = frame[metric].to_numpy(dtype=float)
    low = frame[f"{metric}_hierarchical_bootstrap_95_low"].to_numpy(dtype=float)
    high = frame[f"{metric}_hierarchical_bootstrap_95_high"].to_numpy(dtype=float)
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - low, high - y]),
        color=color,
        marker=marker,
        linewidth=2,
        capsize=4,
        label=label,
    )


def main() -> None:
    args = parse_args()
    analysis = args.analysis_dir
    edge = pd.read_csv(analysis / "edge_level_metrics.csv")
    graph = pd.read_csv(analysis / "graph_metrics.csv")
    density_band = pd.read_csv(analysis / "density_difficulty_metrics.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    errorbar(axes[0], edge, "initial_utility", "Round-0 readout", "#6b7280", "o")
    errorbar(axes[0], edge, "final_utility", "Final readout", "#2563eb", "s")
    axes[0].set_title("Endpoint utility")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.2, 1.02)
    axes[0].legend(frameon=False)

    errorbar(axes[1], edge, "paired_delta", "Mean paired gain", "#059669", "o")
    rng = np.random.default_rng(20260901)
    axes[1].scatter(
        graph.edge_count + rng.uniform(-0.18, 0.18, size=len(graph)),
        graph.paired_delta,
        color="#111827",
        alpha=0.65,
        s=25,
        label="Individual graph",
        zorder=3,
    )
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_title("Communication gain")
    axes[1].set_ylabel(r"$U_H-U_0$")
    axes[1].set_ylim(-0.05, 0.42)
    axes[1].legend(frameon=False)

    for metric, label, color, marker in (
        ("correct_preservation_C_to_C", "Correct preservation", "#7c3aed", "o"),
        ("other_error_correction_O_to_C", "Parsed-error correction", "#dc2626", "s"),
    ):
        errorbar(axes[2], edge, metric, label, color, marker)
    axes[2].set_title("Readout transition mechanism")
    axes[2].set_ylabel("Conditional probability")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.set_xlabel("Directed edge count m")
        ax.set_xticks(edge.edge_count)
    fig.suptitle("2026 AIME clean MAS: n=5, H=3, Qwen3-4B", fontsize=14)
    fig.text(
        0.5,
        -0.025,
        "Intervals resample tasks and graphs; m=16 is the unique complete graph and has task-only uncertainty.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.savefig(analysis / "density_utility_transitions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    colors = {"floor": "#dc2626", "informative": "#2563eb", "ceiling": "#059669"}
    labels = {"floor": "Floor", "informative": "Intermediate", "ceiling": "Ceiling"}
    for band in ("floor", "informative", "ceiling"):
        selected = density_band.loc[density_band.difficulty_band == band].sort_values(
            "edge_count"
        )
        x = selected.edge_count.to_numpy(dtype=float)
        y = selected.paired_delta.to_numpy(dtype=float)
        low = selected.paired_delta_hierarchical_bootstrap_95_low.to_numpy(dtype=float)
        high = selected.paired_delta_hierarchical_bootstrap_95_high.to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            marker="o",
            linewidth=2,
            capsize=4,
            color=colors[band],
            label=labels[band],
        )
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(sorted(density_band.edge_count.unique()))
    ax.set_xlabel("Directed edge count m")
    ax.set_ylabel(r"Paired communication gain $U_H-U_0$")
    ax.set_title("Communication gain by frozen task-difficulty band")
    ax.legend(frameon=False)
    fig.savefig(analysis / "density_difficulty_gain.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
