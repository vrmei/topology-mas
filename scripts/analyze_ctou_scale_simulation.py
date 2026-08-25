"""Validate and summarize the CTOU n=5--50 model-based simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRICS = ("utility", "robustness", "attack_penalty", "delta_utility", "target_risk")
VERSIONS = ("strict_n5", "calibrated_n5_n6_n7_n8_n10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--envelope-csv", type=Path)
    return parser.parse_args()


def validate(curves: pd.DataFrame) -> dict[str, object]:
    required = {
        "version",
        "n",
        "m",
        "delta_realized",
        "average_degree",
        "graphs",
        *METRICS,
    }
    missing = sorted(required - set(curves.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    failures: list[str] = []
    if set(curves.version) != set(VERSIONS):
        failures.append(f"unexpected versions: {sorted(curves.version.unique())}")
    if set(curves.n.unique()) != set(range(5, 51)):
        failures.append("n coverage is not exactly 5..50")
    for column in ("utility", "robustness", "target_risk", "u0"):
        if not curves[column].between(0.0, 1.0).all():
            failures.append(f"{column} leaves [0,1]")
    duplicate = curves.duplicated(["version", "n", "m"]).sum()
    if duplicate:
        failures.append(f"duplicate version/n/m rows: {duplicate}")
    expected_graphs = np.where(
        curves.m.to_numpy() == (curves.n.to_numpy() - 1) ** 2,
        1,
        10,
    )
    graph_mismatch = int(np.sum(curves.graphs.to_numpy() != expected_graphs))
    if graph_mismatch:
        failures.append(f"unexpected graph counts in {graph_mismatch} rows")
    return {
        "passed": not failures,
        "failures": failures,
        "rows": int(len(curves)),
        "n_min": int(curves.n.min()),
        "n_max": int(curves.n.max()),
        "graph_count_mismatches": graph_mismatch,
    }


def interpolate_by_axis(
    curves: pd.DataFrame,
    *,
    axis: str,
    targets: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (version, n), group in curves.groupby(["version", "n"], sort=True):
        ordered = group.sort_values(axis)
        x = ordered[axis].to_numpy(float)
        for target in targets:
            if target < x.min() or target > x.max():
                continue
            row: dict[str, object] = {"version": version, "n": n, axis: target}
            for metric in METRICS:
                row[metric] = float(np.interp(target, x, ordered[metric]))
            rows.append(row)
    return pd.DataFrame(rows)


def curve_diagnostics(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (version, n), group in curves.groupby(["version", "n"], sort=True):
        ordered = group.sort_values("delta_realized")
        rho, _ = spearmanr(ordered.utility, ordered.robustness)
        du = np.diff(ordered.utility.to_numpy(float))
        dr = np.diff(ordered.robustness.to_numpy(float))
        rows.append(
            {
                "version": version,
                "n": n,
                "levels": len(ordered),
                "utility_robustness_spearman_across_density": float(rho),
                "utility_negative_steps": int(np.sum(du < 0)),
                "robustness_negative_steps": int(np.sum(dr < 0)),
                "utility_range": float(ordered.utility.max() - ordered.utility.min()),
                "robustness_range": float(
                    ordered.robustness.max() - ordered.robustness.min()
                ),
                "max_attack_penalty": float(ordered.attack_penalty.max()),
            }
        )
    return pd.DataFrame(rows)


def plot_selected_n(curves: pd.DataFrame, metric: str, output: Path) -> None:
    selected_n = (5, 10, 20, 30, 40, 50)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for axis, version in zip(axes, VERSIONS, strict=True):
        subset = curves.loc[curves.version.eq(version)]
        for n in selected_n:
            group = subset.loc[subset.n.eq(n)].sort_values("delta_realized")
            axis.plot(group.delta_realized, group[metric], label=f"n={n}")
        axis.set_title(version)
        axis.set_xlabel("normalized excess density delta")
    axes[0].set_ylabel(f"model-based {metric}")
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle(f"{metric} by density (n>10 is extrapolation)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_heatmap(curves: pd.DataFrame, metric: str, output: Path) -> None:
    density_grid = np.linspace(0.0, 1.0, 101)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    values: list[np.ndarray] = []
    for version in VERSIONS:
        matrix = []
        for n in range(5, 51):
            group = curves.loc[curves.version.eq(version) & curves.n.eq(n)].sort_values(
                "delta_realized"
            )
            matrix.append(np.interp(density_grid, group.delta_realized, group[metric]))
        values.append(np.asarray(matrix))
    vmin = min(float(value.min()) for value in values)
    vmax = max(float(value.max()) for value in values)
    for axis, version, value in zip(axes, VERSIONS, values, strict=True):
        image = axis.imshow(
            value,
            origin="lower",
            aspect="auto",
            extent=(0, 1, 5, 50),
            vmin=vmin,
            vmax=vmax,
            cmap="viridis" if metric != "attack_penalty" else "magma",
        )
        axis.set_title(version)
        axis.set_xlabel("normalized excess density delta")
    axes[0].set_ylabel("system size n")
    fig.colorbar(image, ax=axes, label=metric)
    fig.suptitle(f"{metric}: model-based simulation, not real n>10 measurement")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_version_gap(curves: pd.DataFrame, output: Path) -> None:
    keys = ["n", "m", "delta_realized", "average_degree"]
    strict = curves.loc[curves.version.eq(VERSIONS[0]), keys + list(METRICS)]
    calibrated = curves.loc[curves.version.eq(VERSIONS[1]), keys + list(METRICS)]
    merged = strict.merge(calibrated, on=keys, suffixes=("_strict", "_calibrated"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for axis, metric in zip(axes, ("utility", "robustness"), strict=True):
        difference = merged[f"{metric}_calibrated"] - merged[f"{metric}_strict"]
        scatter = axis.scatter(
            merged.n,
            difference,
            c=merged.delta_realized,
            s=12,
            cmap="viridis",
            alpha=0.75,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"calibrated - strict: {metric}")
        axis.set_xlabel("n")
    axes[0].set_ylabel("probability-point difference")
    fig.colorbar(scatter, ax=axes, label="normalized excess density delta")
    fig.suptitle("Training-boundary sensitivity of the extrapolation")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fixed_axis(
    summary: pd.DataFrame,
    *,
    axis_column: str,
    selected: tuple[float, ...],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")
    for column, version in enumerate(VERSIONS):
        subset = summary.loc[summary.version.eq(version)]
        for value in selected:
            group = subset.loc[np.isclose(subset[axis_column], value)].sort_values("n")
            if group.empty:
                continue
            label = f"{axis_column}={value:g}"
            axes[0, column].plot(group.n, group.utility, label=label)
            axes[1, column].plot(group.n, group.robustness, label=label)
        axes[0, column].set_title(version)
        axes[1, column].set_xlabel("system size n")
    axes[0, 0].set_ylabel("model-based Utility")
    axes[1, 0].set_ylabel("model-based Robustness")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"Fixed {axis_column} comparisons (n>10 is extrapolation)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_model_envelope(envelope: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey="row")
    colors = {"strict_n5": "tab:blue", "calibrated_n5_n6_n7_n8_n10": "tab:orange"}
    for row, metric in enumerate(("utility", "robustness")):
        for column, density in enumerate((0.0, 0.5, 1.0)):
            axis = axes[row, column]
            for version in VERSIONS:
                subset = envelope.loc[
                    envelope.version.eq(version) & np.isclose(envelope.density, density)
                ]
                grouped = subset.groupby("n")[metric]
                low = grouped.min()
                high = grouped.max()
                primary = subset.loc[
                    subset.variant.eq("proportions_saturating_volume_k2")
                ].set_index("n")[metric]
                axis.fill_between(
                    low.index,
                    low.to_numpy(),
                    high.to_numpy(),
                    color=colors[version],
                    alpha=0.15,
                )
                axis.plot(
                    primary.index,
                    primary.to_numpy(),
                    color=colors[version],
                    label=version if row == 0 and column == 0 else None,
                )
            axis.set_title(f"{metric}, delta={density:g}")
            if row == 1:
                axis.set_xlabel("system size n")
    axes[0, 0].set_ylabel("model-based Utility")
    axes[1, 0].set_ylabel("model-based Robustness")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Local-law extrapolation envelope; band is model uncertainty only")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(args.run_dir / "simulated_curves" / "primary_curves.csv")
    audit = validate(curves)
    (args.output_dir / "simulation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["passed"]:
        raise RuntimeError(f"simulation audit failed: {audit['failures']}")

    density = interpolate_by_axis(
        curves,
        axis="delta_realized",
        targets=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    )
    degree = interpolate_by_axis(
        curves,
        axis="average_degree",
        targets=[1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0],
    )
    diagnostics = curve_diagnostics(curves)
    density.to_csv(args.output_dir / "fixed_density_summary.csv", index=False)
    degree.to_csv(args.output_dir / "fixed_average_degree_summary.csv", index=False)
    diagnostics.to_csv(args.output_dir / "curve_diagnostics.csv", index=False)

    for metric, filename in (
        ("utility", "U_by_n_m.csv"),
        ("robustness", "R_by_n_m.csv"),
        ("attack_penalty", "attack_penalty_by_n_m.csv"),
        ("delta_utility", "deltaU_by_n_m.csv"),
    ):
        curves[
            ["version", "n", "m", "delta_realized", "average_degree", metric]
        ].to_csv(args.output_dir / filename, index=False)

    plot_selected_n(curves, "utility", args.output_dir / "selected_n_U_curves.png")
    plot_selected_n(curves, "robustness", args.output_dir / "selected_n_R_curves.png")
    for metric, filename in (
        ("utility", "U_heatmap.png"),
        ("robustness", "R_heatmap.png"),
        ("attack_penalty", "attack_penalty_heatmap.png"),
    ):
        plot_heatmap(curves, metric, args.output_dir / filename)
    plot_fixed_axis(
        density,
        axis_column="delta_realized",
        selected=(0.0, 0.5, 1.0),
        output=args.output_dir / "fixed_density_vs_n.png",
    )
    plot_fixed_axis(
        degree,
        axis_column="average_degree",
        selected=(2.0, 4.0, 8.0, 16.0),
        output=args.output_dir / "fixed_degree_vs_n.png",
    )
    plot_version_gap(curves, args.output_dir / "strict_calibrated_gap.png")
    if args.envelope_csv is not None:
        plot_model_envelope(
            pd.read_csv(args.envelope_csv),
            args.output_dir / "extrapolation_uncertainty.png",
        )


if __name__ == "__main__":
    main()
