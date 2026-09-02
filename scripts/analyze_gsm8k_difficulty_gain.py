#!/usr/bin/env python3
"""Cross-fit GSM8K task difficulty and estimate clean communication gain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_VERSION = "gsm8k-cross-n-difficulty-gain-v1"
PRIMARY_FLOOR_MAX = 0.10
PRIMARY_CEILING_MIN = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-graph-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def band_for_rate(
    rate: pd.Series,
    *,
    floor_max: float,
    ceiling_min: float,
) -> pd.Series:
    return pd.Series(
        np.where(
            rate <= floor_max,
            "floor",
            np.where(rate >= ceiling_min, "ceiling", "intermediate"),
        ),
        index=rate.index,
    )


def task_sufficient_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["task_id", "difficulty", "difficulty_band"], sort=False)
    rows = []
    for keys, group in grouped:
        initial = group.u0.to_numpy(dtype=float)
        final = group.utility.to_numpy(dtype=float)
        rows.append(
            {
                "task_id": keys[0],
                "difficulty": float(keys[1]),
                "difficulty_band": keys[2],
                "runs": len(group),
                "sum_u0": float(initial.sum()),
                "sum_final": float(final.sum()),
                "sum_delta": float((final - initial).sum()),
                "initial_correct": int(initial.sum()),
                "initial_wrong": int((1.0 - initial).sum()),
                "preserved": int(((initial == 1.0) & (final == 1.0)).sum()),
                "corrected": int(((initial == 0.0) & (final == 1.0)).sum()),
                "corrupted": int(((initial == 1.0) & (final == 0.0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize(stats: pd.DataFrame) -> dict[str, float | int]:
    runs = stats.runs.sum()
    initial_correct = stats.initial_correct.sum()
    initial_wrong = stats.initial_wrong.sum()
    return {
        "tasks": int(stats.task_id.nunique()),
        "runs": int(runs),
        "mean_calibration_difficulty": float(stats.difficulty.mean()),
        "u0": float(stats.sum_u0.sum() / runs),
        "utility": float(stats.sum_final.sum() / runs),
        "delta_u": float(stats.sum_delta.sum() / runs),
        "correct_preservation_C0_to_C3": (
            float(stats.preserved.sum() / initial_correct)
            if initial_correct
            else float("nan")
        ),
        "wrong_correction_notC0_to_C3": (
            float(stats.corrected.sum() / initial_wrong)
            if initial_wrong
            else float("nan")
        ),
        "correct_corruption_C0_to_notC3": (
            float(stats.corrupted.sum() / initial_correct)
            if initial_correct
            else float("nan")
        ),
        "corrected_runs": int(stats.corrected.sum()),
        "corrupted_runs": int(stats.corrupted.sum()),
    }


def bootstrap_summary(
    stats: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(stats), np.full(len(stats), 1.0 / len(stats)), size=samples
    )

    def weighted(column: str) -> np.ndarray:
        return weights @ stats[column].to_numpy(dtype=float)

    def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.full(numerator.shape, np.nan, dtype=float)
        return np.divide(numerator, denominator, out=result, where=denominator != 0)

    runs = weighted("runs")
    initial_correct = weighted("initial_correct")
    initial_wrong = weighted("initial_wrong")
    draws = {
        "u0": safe_divide(weighted("sum_u0"), runs),
        "utility": safe_divide(weighted("sum_final"), runs),
        "delta_u": safe_divide(weighted("sum_delta"), runs),
        "correct_preservation_C0_to_C3": safe_divide(
            weighted("preserved"), initial_correct
        ),
        "wrong_correction_notC0_to_C3": safe_divide(
            weighted("corrected"), initial_wrong
        ),
        "correct_corruption_C0_to_notC3": safe_divide(
            weighted("corrupted"), initial_correct
        ),
    }
    return {
        metric: tuple(np.quantile(values[np.isfinite(values)], [0.025, 0.975]))
        for metric, values in draws.items()
    }


def bootstrap_difference(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    metric: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    first_weights = rng.multinomial(
        len(first), np.full(len(first), 1.0 / len(first)), size=samples
    )
    second_weights = rng.multinomial(
        len(second), np.full(len(second), 1.0 / len(second)), size=samples
    )

    def metric_draws(stats: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        def weighted(column: str) -> np.ndarray:
            return weights @ stats[column].to_numpy(dtype=float)

        def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
            result = np.full(numerator.shape, np.nan, dtype=float)
            return np.divide(numerator, denominator, out=result, where=denominator != 0)

        if metric == "delta_u":
            return ratio(weighted("sum_delta"), weighted("runs"))
        if metric == "wrong_correction_notC0_to_C3":
            return ratio(weighted("corrected"), weighted("initial_wrong"))
        if metric == "correct_corruption_C0_to_notC3":
            return ratio(weighted("corrupted"), weighted("initial_correct"))
        raise ValueError(f"unsupported difference metric {metric}")

    draws = metric_draws(first, first_weights) - metric_draws(
        second, second_weights
    )
    draws = draws[np.isfinite(draws)]
    return tuple(np.quantile(draws, [0.025, 0.975]))


def cross_n_frame(
    frame: pd.DataFrame,
    *,
    evaluation_n: int,
    calibration_n: int,
    floor_max: float,
    ceiling_min: float,
) -> pd.DataFrame:
    calibration = (
        frame.loc[frame.n == calibration_n]
        .groupby("task_id", sort=False)
        .u0.mean()
        .rename("difficulty")
    )
    evaluation = frame.loc[frame.n == evaluation_n].copy()
    evaluation = evaluation.join(calibration, on="task_id", validate="many_to_one")
    if evaluation.difficulty.isna().any():
        raise ValueError("calibration difficulty is missing for evaluation tasks")
    evaluation["difficulty_band"] = band_for_rate(
        evaluation.difficulty,
        floor_max=floor_max,
        ceiling_min=ceiling_min,
    )
    evaluation["evaluation_n"] = evaluation_n
    evaluation["calibration_n"] = calibration_n
    return evaluation


def analyze_thresholds(
    frame: pd.DataFrame,
    *,
    floor_max: float,
    ceiling_min: float,
    samples: int,
    seed: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_frames = []
    summaries = []
    differences = []
    for offset, (evaluation_n, calibration_n) in enumerate(((5, 8), (8, 5))):
        crossed = cross_n_frame(
            frame,
            evaluation_n=evaluation_n,
            calibration_n=calibration_n,
            floor_max=floor_max,
            ceiling_min=ceiling_min,
        )
        stats = task_sufficient_statistics(crossed)
        stats.insert(0, "threshold_scheme", label)
        stats.insert(1, "evaluation_n", evaluation_n)
        stats.insert(2, "calibration_n", calibration_n)
        task_frames.append(stats)
        groups = {
            band: group.reset_index(drop=True)
            for band, group in stats.groupby("difficulty_band", sort=False)
        }
        for band_index, band in enumerate(("floor", "intermediate", "ceiling")):
            group = groups[band]
            result = summarize(group)
            intervals = bootstrap_summary(
                group,
                samples=samples,
                seed=seed + offset * 100 + band_index,
            )
            row = {
                "threshold_scheme": label,
                "evaluation_n": evaluation_n,
                "calibration_n": calibration_n,
                "difficulty_band": band,
                **result,
            }
            for metric, (low, high) in intervals.items():
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
            summaries.append(row)
        for comparison_index, other in enumerate(("floor", "ceiling")):
            for metric in (
                "delta_u",
                "wrong_correction_notC0_to_C3",
                "correct_corruption_C0_to_notC3",
            ):
                observed = float(summarize(groups["intermediate"])[metric]) - float(
                    summarize(groups[other])[metric]
                )
                low, high = bootstrap_difference(
                    groups["intermediate"],
                    groups[other],
                    metric=metric,
                    samples=samples,
                    seed=seed + 1000 + offset * 100 + comparison_index * 10,
                )
                differences.append(
                    {
                        "threshold_scheme": label,
                        "evaluation_n": evaluation_n,
                        "calibration_n": calibration_n,
                        "first_band": "intermediate",
                        "second_band": other,
                        "metric": metric,
                        "observed_difference": observed,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return (
        pd.concat(task_frames, ignore_index=True),
        pd.DataFrame(summaries),
        pd.DataFrame(differences),
    )


def plot_primary(summary: pd.DataFrame, output: Path) -> None:
    selected = summary.loc[summary.threshold_scheme == "primary_aime_aligned"].copy()
    order = ["floor", "intermediate", "ceiling"]
    colors = {5: "#4C78A8", 8: "#F58518"}
    fig, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    x = np.arange(3, dtype=float)
    width = 0.34
    for index, n in enumerate((5, 8)):
        group = selected.loc[selected.evaluation_n == n].set_index("difficulty_band").loc[order]
        values = group.delta_u.to_numpy()
        low = group.delta_u_ci95_low.to_numpy()
        high = group.delta_u_ci95_high.to_numpy()
        positions = x + (index - 0.5) * width
        axis.bar(positions, values, width=width, color=colors[n], label=f"evaluate n={n}")
        axis.errorbar(
            positions,
            values,
            yerr=np.vstack([values - low, high - values]),
            fmt="none",
            ecolor="#30343B",
            capsize=4,
            linewidth=1.4,
        )
    axis.axhline(0, color="#30343B", linewidth=1)
    axis.set_xticks(x, ["Floor", "Intermediate", "Ceiling"])
    axis.set_ylabel("Final utility − Round-0 utility")
    axis.set_title("Llama/GSM8K clean communication gain by cross-fitted difficulty")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    axis.legend(frameon=False)
    fig.savefig(output / "cross_n_difficulty_gain.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.task_graph_csv)
    required = {"task_id", "graph_id", "n", "u0", "utility", "delta_u"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if set(frame.n.unique()) != {5, 8}:
        raise ValueError("cross-n analysis requires exactly n=5 and n=8")
    if frame.duplicated(["task_id", "graph_id"]).any():
        raise ValueError("task-graph rows are not unique")
    if not np.allclose(frame.utility - frame.u0, frame.delta_u):
        raise ValueError("delta_u is inconsistent with utility-u0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyses = []
    summaries = []
    differences = []
    for offset, (label, floor_max, ceiling_min) in enumerate(
        (
            ("primary_aime_aligned", PRIMARY_FLOOR_MAX, PRIMARY_CEILING_MIN),
            ("sensitivity_wide_extremes", 0.20, 0.80),
        )
    ):
        task_frame, summary, difference = analyze_thresholds(
            frame,
            floor_max=floor_max,
            ceiling_min=ceiling_min,
            samples=args.bootstrap_samples,
            seed=args.seed + offset * 10_000,
            label=label,
        )
        analyses.append(task_frame)
        summaries.append(summary)
        differences.append(difference)

    task_output = pd.concat(analyses, ignore_index=True)
    summary_output = pd.concat(summaries, ignore_index=True)
    difference_output = pd.concat(differences, ignore_index=True)
    task_output.to_csv(args.output_dir / "task_cross_n_metrics.csv", index=False)
    summary_output.to_csv(args.output_dir / "difficulty_band_metrics.csv", index=False)
    difference_output.to_csv(
        args.output_dir / "difficulty_band_differences.csv", index=False
    )

    primary_task_bands = task_output.loc[
        task_output.threshold_scheme == "primary_aime_aligned",
        ["evaluation_n", "task_id", "difficulty_band"],
    ].drop_duplicates()
    primary_rows = frame.merge(
        primary_task_bands,
        left_on=["n", "task_id"],
        right_on=["evaluation_n", "task_id"],
        validate="many_to_one",
    )
    by_m = (
        primary_rows.groupby(["n", "m", "difficulty_band"], sort=True)
        .agg(
            graphs=("graph_id", "nunique"),
            tasks=("task_id", "nunique"),
            runs=("task_id", "size"),
            u0=("u0", "mean"),
            utility=("utility", "mean"),
            delta_u=("delta_u", "mean"),
        )
        .reset_index()
    )
    by_m.to_csv(args.output_dir / "difficulty_by_m.csv", index=False)

    peak_counts = {}
    for n, group in by_m.groupby("n"):
        pivot = group.pivot(index="m", columns="difficulty_band", values="delta_u")
        middle_is_largest = (pivot.intermediate > pivot.floor) & (
            pivot.intermediate > pivot.ceiling
        )
        peak_counts[str(int(n))] = {
            "m_levels": int(len(pivot)),
            "intermediate_gain_largest_levels": int(middle_is_largest.sum()),
        }

    n_rates = frame.groupby(["n", "task_id"], sort=False).u0.mean().unstack("n")
    audit = {
        "analysis_version": ANALYSIS_VERSION,
        "source": str(args.task_graph_csv),
        "rows": len(frame),
        "tasks": int(frame.task_id.nunique()),
        "graphs": int(frame.graph_id.nunique()),
        "graphs_by_n": {
            str(n): int(frame.loc[frame.n == n].graph_id.nunique()) for n in (5, 8)
        },
        "primary_thresholds": {
            "floor_max": PRIMARY_FLOOR_MAX,
            "ceiling_min": PRIMARY_CEILING_MIN,
        },
        "difficulty_calibration": (
            "For evaluation n=5, difficulty uses only n=8 Round-0 runs; "
            "for evaluation n=8, difficulty uses only n=5 Round-0 runs."
        ),
        "round_zero_rate_pearson_n5_n8": float(n_rates[5].corr(n_rates[8])),
        "round_zero_rate_spearman_n5_n8": float(
            n_rates[5].corr(n_rates[8], method="spearman")
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "task within frozen cross-n difficulty band",
        "within_m_peak_counts": peak_counts,
        "claim_limits": [
            "The analysis is exploratory and was requested after the AIME result.",
            "Only four tasks fall in the primary floor band.",
            "Difficulty and gain are based on the same model/task collection but disjoint n-specific runs.",
            "A raw middle-band gain can partly reflect finite headroom at the ceiling and lack of correct evidence at the floor.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "audit": audit,
                "primary_band_metrics": summary_output.loc[
                    summary_output.threshold_scheme == "primary_aime_aligned"
                ].to_dict(orient="records"),
                "primary_band_differences": difference_output.loc[
                    difference_output.threshold_scheme == "primary_aime_aligned"
                ].to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_primary(summary_output, args.output_dir)


if __name__ == "__main__":
    main()
