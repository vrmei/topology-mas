#!/usr/bin/env python3
"""Render the primary CTOU cross-scale transfer diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {5: "#8C6BB1", 6: "#2B6CB0", 7: "#D97706", 8: "#2F855A"}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
        }
    )


def plot_observed_density(input_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(input_dir / "observed_all_size_density_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharex=True)
    for n, group in frame.groupby("n", sort=True):
        group = group.sort_values("rho")
        color = COLORS[int(n)]
        axes[0].plot(group["rho"], group["utility"], "o-", color=color, label=f"n={n}")
        axes[1].plot(group["rho"], group["robustness"], "o-", color=color, label=f"n={n}")
    axes[0].set_title("Observed clean utility")
    axes[1].set_title("Observed attack robustness")
    for axis in axes:
        axis.set_xlabel("Normalized density ρ")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0.72, 0.91)
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("Density does not fully collapse the n=5–8 response curves")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_density_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_transfer_curves(input_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(input_dir / "density_curves.csv")
    frame = frame[frame["model"].eq("ctou_table")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7), sharex="col")
    for column, n in enumerate((6, 7)):
        group = frame[frame["n"].eq(n)].sort_values("rho")
        for row, quantity in enumerate(("utility", "robustness")):
            axis = axes[row, column]
            axis.plot(
                group["rho"],
                group[f"observed_{quantity}"],
                "o-",
                color="#111827",
                label="Observed",
            )
            axis.plot(
                group["rho"],
                group[f"predicted_{quantity}"],
                "s--",
                color=COLORS[n],
                label="Frozen CTOU",
            )
            axis.set_title(f"n={n}: {quantity}")
            axis.set_ylabel("Accuracy")
            if row == 1:
                axis.set_xlabel("Normalized density ρ")
            axis.legend(frameon=False)
    fig.suptitle("Frozen n={5,8} CTOU transferred to unseen n={6,7} graphs")
    fig.tight_layout()
    fig.savefig(output_dir / "ctou_transfer_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_round_error(input_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(input_dir / "round_error_summary.csv")
    frame = frame[frame["condition"].eq("attack") & frame["receiver_scope"].eq("readout")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharex=True)
    for n, group in frame.groupby("n", sort=True):
        group = group.sort_values("round_index")
        axes[0].plot(
            group["round_index"],
            group["correct_probability_error"],
            "o-",
            color=COLORS[int(n)],
            label=f"n={n}",
        )
        axes[1].plot(
            group["round_index"],
            group["target_probability_error"],
            "o-",
            color=COLORS[int(n)],
            label=f"n={n}",
        )
    axes[0].set_title("Readout correct-probability error")
    axes[1].set_title("Readout target-probability error")
    for axis in axes:
        axis.set_xlabel("Round")
        axis.set_ylabel("Absolute error")
        axis.set_xticks(sorted(frame["round_index"].unique()))
        axis.legend(frameon=False)
    fig.suptitle("Recursive error accumulates gradually rather than breaking at n=7")
    fig.tight_layout()
    fig.savefig(output_dir / "round_error_accumulation.png", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    plot_observed_density(args.input_dir, output_dir)
    plot_transfer_curves(args.input_dir, output_dir)
    plot_round_error(args.input_dir, output_dir)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
