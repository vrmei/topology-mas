"""Aggregate the frozen CTOU mean-field versus particle validation grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIZES = (5, 10, 15, 20, 30, 40, 50)
METRICS = ("utility", "robustness", "target_risk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_predictions(root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    missing: list[int] = []
    for n in SIZES:
        path = root / f"n{n}" / "particle_predictions.csv.gz"
        if not path.exists():
            missing.append(n)
            continue
        frame = pd.read_csv(path)
        if set(frame.n.unique()) != {n}:
            raise ValueError(f"unexpected n values in {path}: {frame.n.unique()}")
        parts.append(frame)
    if missing:
        raise FileNotFoundError(f"missing particle predictions for n={missing}")
    return pd.concat(parts, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells: list[dict[str, object]] = []
    for keys, frame in predictions.groupby(["version", "n", "density"], sort=True):
        version, n, density = keys
        row: dict[str, object] = {
            "version": version,
            "n": int(n),
            "density": float(density),
            "tasks": int(len(frame)),
        }
        for metric in METRICS:
            difference = (
                frame[f"meanfield_{metric}"] - frame[f"particle_{metric}"]
            )
            row[f"{metric}_task_mae"] = float(np.mean(np.abs(difference)))
            row[f"{metric}_aggregate_bias"] = float(difference.mean())
            row[f"meanfield_{metric}"] = float(frame[f"meanfield_{metric}"].mean())
            row[f"particle_{metric}"] = float(frame[f"particle_{metric}"].mean())
        cells.append(row)
    cell_frame = pd.DataFrame(cells)

    sizes: list[dict[str, object]] = []
    for n, frame in predictions.groupby("n", sort=True):
        row: dict[str, object] = {"n": int(n), "rows": int(len(frame))}
        cell_subset = cell_frame.loc[cell_frame.n.eq(n)]
        for metric in METRICS:
            difference = (
                frame[f"meanfield_{metric}"] - frame[f"particle_{metric}"]
            )
            row[f"{metric}_task_mae"] = float(np.mean(np.abs(difference)))
            row[f"{metric}_mean_bias"] = float(difference.mean())
            row[f"max_{metric}_aggregate_error"] = float(
                cell_subset[f"{metric}_aggregate_bias"].abs().max()
            )
        row["passed"] = bool(
            row["utility_task_mae"] <= 0.03
            and row["robustness_task_mae"] <= 0.03
            and row["max_utility_aggregate_error"] <= 0.05
            and row["max_robustness_aggregate_error"] <= 0.05
        )
        sizes.append(row)
    return cell_frame, pd.DataFrame(sizes)


def overall_gate(
    predictions: pd.DataFrame, cells: pd.DataFrame, by_size: pd.DataFrame
) -> dict[str, object]:
    gate: dict[str, object] = {
        "sizes": list(SIZES),
        "cells": int(len(cells)),
        "rows": int(len(predictions)),
        "failed_individual_sizes": by_size.loc[~by_size.passed, "n"].astype(int).tolist(),
    }
    for metric in ("utility", "robustness"):
        difference = (
            predictions[f"meanfield_{metric}"] - predictions[f"particle_{metric}"]
        )
        gate[f"{metric}_task_mae"] = float(np.mean(np.abs(difference)))
        gate[f"{metric}_mean_bias"] = float(difference.mean())
        gate[f"max_{metric}_aggregate_error"] = float(
            cells[f"{metric}_aggregate_bias"].abs().max()
        )
    gate["passed"] = bool(
        gate["utility_task_mae"] <= 0.03
        and gate["robustness_task_mae"] <= 0.03
        and gate["max_utility_aggregate_error"] <= 0.05
        and gate["max_robustness_aggregate_error"] <= 0.05
    )
    return gate


def plot_errors(cells: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, metric in enumerate(("utility", "robustness")):
        for version, marker in (
            ("strict_n5", "o"),
            ("calibrated_n5_n6_n7_n8_n10", "s"),
        ):
            subset = cells.loc[cells.version.eq(version)]
            axes[0, column].scatter(
                subset.n,
                subset[f"{metric}_task_mae"],
                c=subset.density,
                cmap="viridis",
                marker=marker,
                alpha=0.8,
                label=version,
            )
            axes[1, column].scatter(
                subset.n,
                subset[f"{metric}_aggregate_bias"],
                c=subset.density,
                cmap="viridis",
                marker=marker,
                alpha=0.8,
            )
        axes[0, column].axhline(0.03, color="tab:red", linestyle="--", linewidth=1)
        axes[0, column].set_title(f"{metric}: task MAE")
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        axes[1, column].set_title(f"{metric}: mean-field minus particle")
        axes[1, column].set_xlabel("system size n")
    axes[0, 0].set_ylabel("task-level absolute error")
    axes[1, 0].set_ylabel("aggregate probability-point bias")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Frozen particle validation; color is normalized excess density")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(args.particle_root)
    cells, by_size = summarize(predictions)
    gate = overall_gate(predictions, cells, by_size)
    cells.to_csv(args.output_dir / "particle_cell_summary.csv", index=False)
    by_size.to_csv(args.output_dir / "particle_size_summary.csv", index=False)
    (args.output_dir / "particle_overall_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_errors(cells, args.output_dir / "particle_validation_errors.png")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
