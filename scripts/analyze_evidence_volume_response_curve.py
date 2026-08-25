#!/usr/bin/env python3
"""Analyze the frozen evidence-volume response curve and token control."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from topology_mas.experiments.evidence_volume import read_jsonl
from topology_mas.experiments.evidence_volume_curve import EXPERIMENT_VERSION

BOOTSTRAPS = 10_000
SEED = 20_260_825
LOW_SUPPORT = {"c80_t20": 15, "c67_t33": 9, "c50_t50": 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    return parser.parse_args()


def clustered_interval(
    values: pd.Series,
    *,
    bootstraps: int,
    seed: int,
) -> tuple[float, float, float]:
    array = values.dropna().to_numpy(float)
    if len(array) < 2:
        return float(np.mean(array)), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(bootstraps, len(array)))
    draws = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def cell_summary(frame: pd.DataFrame, *, bootstraps: int) -> pd.DataFrame:
    keys = [
        "request_kind",
        "scenario",
        "previous_mode",
        "ratio_id",
        "incoming_degree",
        "token_match_condition",
    ]
    rows: list[dict[str, object]] = []
    for index, (values, group) in enumerate(frame.groupby(keys, dropna=False, sort=True)):
        task_rate = group.groupby("task_id", as_index=False).agg(
            primary=("is_primary_outcome", "mean"),
            target=("is_target", "mean"),
            correct=("is_correct", "mean"),
            other=("is_other", "mean"),
            unparsed=("is_unparsed", "mean"),
        )
        parsed_primary = (
            group.loc[~group.is_unparsed.astype(bool)]
            .groupby("task_id")
            .is_primary_outcome.mean()
        )
        task_rate["primary_parsed"] = task_rate.task_id.map(parsed_primary)
        row = dict(zip(keys, values, strict=True))
        for metric in (
            "primary",
            "primary_parsed",
            "target",
            "correct",
            "other",
            "unparsed",
        ):
            mean, low, high = clustered_interval(
                task_rate[metric], bootstraps=bootstraps, seed=SEED + index
            )
            row[f"{metric}_rate"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        row.update(
            tasks=int(group.task_id.nunique()),
            requests=int(len(group)),
            mean_input_tokens=float(group.input_tokens.mean()),
            mean_output_tokens=float(group.output_tokens.mean()),
            mean_latency_ms=float(group.latency_ms.mean()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def degree_contrasts(frame: pd.DataFrame, *, bootstraps: int) -> pd.DataFrame:
    curve = frame.loc[frame.request_kind.eq("response_curve")].copy()
    task = (
        curve.groupby(
            ["scenario", "previous_mode", "ratio_id", "task_id", "incoming_degree"],
            as_index=False,
        )
        .is_primary_outcome.mean()
    )
    rows: list[dict[str, object]] = []
    group_keys = ["scenario", "previous_mode", "ratio_id"]
    for group_index, (keys, group) in enumerate(task.groupby(group_keys, sort=True)):
        degrees = sorted(group.incoming_degree.unique())
        pairs = list(zip(degrees, degrees[1:], strict=False))
        if 30 in degrees and max(degrees) != 30:
            pairs.append((30, max(degrees)))
        for pair_index, (low_degree, high_degree) in enumerate(pairs):
            wide = group.loc[
                group.incoming_degree.isin([low_degree, high_degree])
            ].pivot(index="task_id", columns="incoming_degree", values="is_primary_outcome")
            difference = wide[high_degree] - wide[low_degree]
            mean, low, high = clustered_interval(
                difference,
                bootstraps=bootstraps,
                seed=SEED + group_index * 100 + pair_index,
            )
            rows.append(
                {
                    **dict(zip(group_keys, keys, strict=True)),
                    "degree_low": low_degree,
                    "degree_high": high_degree,
                    "contrast": "high_tail" if low_degree == 30 else "adjacent",
                    "effect": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "tasks": len(difference),
                }
            )

    attack = task.loc[task.scenario.eq("attack_adoption")]
    for mode_index, (previous_mode, group) in enumerate(
        attack.groupby("previous_mode", sort=True)
    ):
        task_differences: list[pd.Series] = []
        for _ratio, ratio_group in group.groupby("ratio_id", sort=True):
            maximum = int(ratio_group.incoming_degree.max())
            wide = ratio_group.loc[
                ratio_group.incoming_degree.isin([30, maximum])
            ].pivot(index="task_id", columns="incoming_degree", values="is_primary_outcome")
            task_differences.append(wide[maximum] - wide[30])
        pooled = pd.concat(task_differences, axis=1).mean(axis=1)
        mean, low, high = clustered_interval(
            pooled,
            bootstraps=bootstraps,
            seed=SEED + 10_000 + mode_index,
        )
        rows.append(
            {
                "scenario": "attack_adoption",
                "previous_mode": previous_mode,
                "ratio_id": "pooled_ratios",
                "degree_low": 30,
                "degree_high": "ratio_specific_max",
                "contrast": "pooled_high_tail",
                "effect": mean,
                "ci_low": low,
                "ci_high": high,
                "tasks": len(pooled),
            }
        )
    return pd.DataFrame(rows)


def token_matched_contrast(frame: pd.DataFrame, *, bootstraps: int) -> pd.DataFrame:
    selected = frame.loc[frame.request_kind.eq("token_matched")].copy()
    task = (
        selected.groupby(["task_id", "token_match_condition"], as_index=False)
        .is_target.mean()
        .pivot(index="task_id", columns="token_match_condition", values="is_target")
    )
    difference = task["eight_short"] - task["four_long"]
    mean, low, high = clustered_interval(
        difference, bootstraps=bootstraps, seed=SEED + 20_000
    )
    prompt_tokens = selected.groupby("token_match_condition").input_tokens.mean()
    peer_tokens = selected.groupby("token_match_condition").peer_message_tokens.mean()
    return pd.DataFrame(
        [
            {
                "estimand": "eight_short_minus_four_long_target_selection",
                "effect": mean,
                "ci_low": low,
                "ci_high": high,
                "tasks": len(difference),
                "four_long_mean_input_tokens": prompt_tokens["four_long"],
                "eight_short_mean_input_tokens": prompt_tokens["eight_short"],
                "four_long_mean_peer_tokens": peer_tokens["four_long"],
                "eight_short_mean_peer_tokens": peer_tokens["eight_short"],
            }
        ]
    )


def transform_functions() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    result: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "ratio_only": lambda degree: np.zeros_like(degree, dtype=float),
        "raw_degree": lambda degree: degree.astype(float),
        "log1p_degree": lambda degree: np.log1p(degree.astype(float)),
    }
    for k in (1, 2, 4, 8, 16):
        result[f"bounded_k{k}"] = (
            lambda degree, k=k: degree.astype(float) / (degree.astype(float) + k)
        )
    return result


def fit_probability(x: np.ndarray, y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    if len(np.unique(y)) < 2 or np.allclose(x, x[0]):
        probability = (float(y.sum()) + 0.5) / (len(y) + 1.0)
        return np.full(len(test_x), probability)
    model = LogisticRegression(C=100.0, solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(x[:, None], y)
    return model.predict_proba(test_x[:, None])[:, 1]


def out_of_range_link_evaluation(frame: pd.DataFrame) -> pd.DataFrame:
    attack = frame.loc[
        frame.request_kind.eq("response_curve")
        & frame.scenario.eq("attack_adoption")
    ].copy()
    rows: list[dict[str, object]] = []
    for keys, group in attack.groupby(["previous_mode", "ratio_id"], sort=True):
        previous_mode, ratio_id = keys
        cutoff = LOW_SUPPORT[str(ratio_id)]
        train = group.loc[group.incoming_degree.le(cutoff)]
        test = group.loc[group.incoming_degree.gt(cutoff)]
        y_train = train.is_target.to_numpy(int)
        y_test = test.is_target.to_numpy(int)
        for name, transform in transform_functions().items():
            prediction = fit_probability(
                transform(train.incoming_degree.to_numpy(float)),
                y_train,
                transform(test.incoming_degree.to_numpy(float)),
            )
            prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
            rows.append(
                {
                    "previous_mode": previous_mode,
                    "ratio_id": ratio_id,
                    "low_support_max_degree": cutoff,
                    "model": name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "test_log_loss": log_loss(y_test, prediction, labels=[0, 1]),
                    "test_brier": float(np.mean((y_test - prediction) ** 2)),
                    "observed_high_rate": float(y_test.mean()),
                    "predicted_high_rate": float(prediction.mean()),
                }
            )
    return pd.DataFrame(rows)


def classify_high_tail(contrasts: pd.DataFrame) -> dict[str, str]:
    output: dict[str, str] = {}
    selected = contrasts.loc[contrasts.contrast.eq("pooled_high_tail")]
    for row in selected.itertuples(index=False):
        if row.ci_low >= -0.05 and row.ci_high <= 0.05:
            label = "fast_saturation"
        elif row.ci_low > 0.05:
            label = "continued_high_volume_response"
        elif row.effect > 0:
            label = "diminishing_but_unresolved"
        else:
            label = "non_monotone_or_inconclusive"
        output[str(row.previous_mode)] = label
    return output


def plot_curves(cells: pd.DataFrame, output: Path) -> None:
    attack = cells.loc[
        cells.request_kind.eq("response_curve")
        & cells.scenario.eq("attack_adoption")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, mode in zip(axes, ("include", "omit"), strict=True):
        subset = attack.loc[attack.previous_mode.eq(mode)]
        for ratio, group in subset.groupby("ratio_id", sort=True):
            group = group.sort_values("incoming_degree")
            axis.plot(group.incoming_degree, group.target_rate, marker="o", label=ratio)
            axis.fill_between(
                group.incoming_degree,
                group.target_ci_low,
                group.target_ci_high,
                alpha=0.15,
            )
        axis.set_title("with previous C" if mode == "include" else "no previous answer")
        axis.set_xlabel("distinct peer-message count")
        axis.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("target-selection probability")
    axes[1].legend()
    fig.suptitle("Attack-side evidence-volume response curves")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_benign(cells: pd.DataFrame, output: Path) -> None:
    benign = cells.loc[
        cells.request_kind.eq("response_curve")
        & cells.scenario.eq("benign_correction")
    ]
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for ratio, group in benign.groupby("ratio_id", sort=True):
        group = group.sort_values("incoming_degree")
        axis.plot(group.incoming_degree, group.correct_rate, marker="o", label=ratio)
        axis.fill_between(
            group.incoming_degree,
            group.correct_ci_low,
            group.correct_ci_high,
            alpha=0.15,
        )
    axis.set_xlabel("distinct peer-message count")
    axis.set_ylabel("correct-selection probability")
    axis.set_ylim(-0.02, 1.02)
    axis.legend()
    axis.set_title("Benign correction with previous O")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
    if manifest["experiment_version"] != EXPERIMENT_VERSION:
        raise ValueError("prepared experiment version mismatch")
    if status["status"] != "completed" or status["failed"]:
        raise ValueError(f"run is not complete: {status}")
    frame = pd.DataFrame(read_jsonl(args.run_dir / "outcomes.jsonl"))
    if len(frame) != int(manifest["expected_requests"]):
        raise ValueError("outcome count differs from the frozen plan")
    if frame.request_id.duplicated().any():
        raise ValueError("duplicate request IDs")

    cells = cell_summary(frame, bootstraps=args.bootstraps)
    contrasts = degree_contrasts(frame, bootstraps=args.bootstraps)
    token_control = token_matched_contrast(frame, bootstraps=args.bootstraps)
    links = out_of_range_link_evaluation(frame)
    cells.to_csv(args.output_dir / "response_curve_cells.csv", index=False)
    contrasts.to_csv(args.output_dir / "degree_contrasts.csv", index=False)
    token_control.to_csv(args.output_dir / "token_matched_contrast.csv", index=False)
    links.to_csv(args.output_dir / "out_of_range_link_evaluation.csv", index=False)
    plot_curves(cells, args.output_dir / "attack_response_curves.png")
    plot_benign(cells, args.output_dir / "benign_response_curves.png")

    token_row = token_control.iloc[0]
    if token_row.ci_low >= -0.05 and token_row.ci_high <= 0.05:
        token_label = "practical_equivalence"
    elif token_row.ci_low > 0 or token_row.ci_high < 0:
        token_label = "directional_message_count_effect"
    else:
        token_label = "inconclusive"
    result = {
        "analysis_version": "evidence-volume-response-curve-analysis-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "requests": len(frame),
        "tasks": int(frame.task_id.nunique()),
        "failed": int(status["failed"]),
        "high_tail_classification": classify_high_tail(contrasts),
        "token_matched_classification": token_label,
        "claim_limits": [
            "one-step receiver response, not a complete MAS endpoint",
            "token matching does not match semantics or argument quality",
            "the fixed 40 tasks were selected for stimulus support",
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
