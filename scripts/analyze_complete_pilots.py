"""Analyze every complete fixed-T=3 and graph-depth pilot stratum."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FIXED_KEYS = (
    "n5_m4",
    "n5_m8",
    "n5_m12",
    "n5_m16",
    "n8_m7",
    "n8_m10",
    "n8_m14",
    "n8_m21",
)
COMMON_OLD100_KEYS = tuple(key for key in FIXED_KEYS if key != "n8_m10")
METRICS = (
    "utility",
    "r_mean",
    "d_mean",
    "induced_target_rate",
    "propagation_count",
    "clean_correction_rate",
    "clean_corruption_rate",
    "clean_model_calls",
    "attack_model_calls",
    "clean_input_tokens",
    "attack_input_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed100-root", type=Path, required=True)
    parser.add_argument("--fixed500-root", type=Path, required=True)
    parser.add_argument("--fixed500-revised-root", type=Path, required=True)
    parser.add_argument("--graph-depth500-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_808)
    return parser.parse_args()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
    temporary.replace(path)


def parse_key(key: str) -> tuple[int, int]:
    n_text, m_text = key.split("_")
    return int(n_text[1:]), int(m_text[1:])


def fixed500_analysis(args: argparse.Namespace, key: str) -> Path:
    root = args.fixed500_revised_root if key == "n8_m10" else args.fixed500_root
    return root / "strata" / key / "analysis-v1"


def fixed500_batch(args: argparse.Namespace, key: str) -> Path:
    root = args.fixed500_revised_root if key == "n8_m10" else args.fixed500_root
    return root / "strata" / key / "batch"


def load_frames(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_json(path / "run_metrics.jsonl", lines=True)
    pairs = pd.read_json(path / "paired_attacks.jsonl", lines=True)
    graphs = pd.read_csv(path / "graph_metrics.csv")
    return runs, pairs, graphs


def task_metrics(path: Path, task_ids: set[str] | None = None) -> pd.DataFrame:
    runs, pairs, _ = load_frames(path)
    if task_ids is not None:
        runs = runs[runs.task_id.isin(task_ids)]
        pairs = pairs[pairs.task_id.isin(task_ids)]
    clean = runs[runs.condition == "clean"]
    attack = runs[runs.condition == "attack"]
    result = pd.DataFrame(index=sorted(runs.task_id.unique()))
    result.index.name = "task_id"
    result["utility"] = clean.groupby("task_id").final_correct.mean()
    result["r_mean"] = attack.groupby("task_id").final_correct.mean()
    result["d_mean"] = result.utility - result.r_mean
    result["induced_target_rate"] = pairs.groupby("task_id").induced_readout_target.mean()
    result["propagation_count"] = pairs.groupby(
        "task_id"
    ).max_induced_nonattacker_count.mean()
    result["clean_correction_rate"] = clean.assign(
        value=(~clean.readout_round_zero_correct) & clean.final_correct
    ).groupby("task_id").value.mean()
    result["clean_corruption_rate"] = clean.assign(
        value=clean.readout_round_zero_correct & (~clean.final_correct)
    ).groupby("task_id").value.mean()
    result["clean_model_calls"] = clean.groupby("task_id").model_calls.mean()
    result["attack_model_calls"] = attack.groupby("task_id").model_calls.mean()
    result["clean_input_tokens"] = clean.groupby("task_id").input_tokens.mean()
    result["attack_input_tokens"] = attack.groupby("task_id").input_tokens.mean()
    if result.isna().any().any():
        raise ValueError(f"incomplete task table: {path}")
    return result


def bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, replicates: int
) -> tuple[float, float, float]:
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def summarize(
    regime: str,
    key: str,
    frame: pd.DataFrame,
    rng: np.random.Generator,
    replicates: int,
) -> list[dict[str, Any]]:
    n, m = parse_key(key)
    rows = []
    for metric in METRICS:
        point, low, high = bootstrap_mean(
            frame[metric].to_numpy(dtype=float), rng, replicates
        )
        rows.append(
            {
                "regime": regime,
                "stratum": key,
                "n": n,
                "m": m,
                "metric": metric,
                "estimate": point,
                "ci95_low": low,
                "ci95_high": high,
                "tasks": len(frame),
                "ci_scope": "task bootstrap conditional on selected graphs",
            }
        )
    return rows


def paired_comparison(
    comparison: str,
    key: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    rng: np.random.Generator,
    replicates: int,
) -> list[dict[str, Any]]:
    if not left.index.equals(right.index):
        raise ValueError(f"task pairing differs: {comparison}, {key}")
    n, m = parse_key(key)
    rows = []
    for metric in METRICS:
        values = left[metric].to_numpy(dtype=float) - right[metric].to_numpy(dtype=float)
        point, low, high = bootstrap_mean(values, rng, replicates)
        rows.append(
            {
                "comparison": comparison,
                "stratum": key,
                "n": n,
                "m": m,
                "metric": metric,
                "estimate": point,
                "ci95_low": low,
                "ci95_high": high,
                "ci_excludes_zero": low > 0 or high < 0,
                "tasks": len(left),
            }
        )
    return rows


def attack_transition_metrics(path: Path) -> dict[str, float]:
    _, pairs, _ = load_frames(path)
    clean_correct = pairs[pairs.clean_correct]
    clean_wrong = pairs[~pairs.clean_correct]
    return {
        "correct_to_any_error_given_clean_correct": float(
            (~clean_correct.attack_correct).mean()
        ),
        "correct_to_target_given_clean_correct": float(
            clean_correct.correct_to_target_flip.mean()
        ),
        "correct_to_other_error_given_clean_correct": float(
            ((~clean_correct.attack_correct) & (~clean_correct.attack_final_matches_target)).mean()
        ),
        "wrong_to_correct_given_clean_wrong": float(
            clean_wrong.clean_error_corrected_under_attack.mean()
        ),
    }


def old100_compatibility(args: argparse.Namespace, key: str) -> dict[str, Any]:
    old_root = args.fixed100_root / "strata" / key
    new_root = fixed500_analysis(args, key)
    old_manifest = json.loads((old_root / "batch" / "manifest.json").read_text())
    new_manifest = json.loads(
        (fixed500_batch(args, key) / "manifest.json").read_text()
    )
    old_runs = pd.read_json(old_root / "analysis-v1" / "run_metrics.jsonl", lines=True)
    new_runs = pd.read_json(new_root / "run_metrics.jsonl", lines=True)
    new_runs = new_runs[new_runs.task_id.isin(old_manifest["task_ids"])]
    keys = [
        "task_id",
        "graph_id",
        "condition",
        "attack_node",
        "experiment_seed",
        "assignment_seed",
    ]
    merged = old_runs.merge(
        new_runs, on=keys, suffixes=("_old100", "_new500"), validate="one_to_one"
    )
    return {
        "stratum": key,
        "same_task_prefix": old_manifest["task_ids"] == new_manifest["task_ids"][:100],
        "same_graph_ids": old_manifest["graph_ids"] == new_manifest["graph_ids"],
        "old_runner_version": old_manifest["runner_version"],
        "new_runner_version": new_manifest["runner_version"],
        "paired_cells": len(merged),
        "final_answer_difference_rate": float(
            (
                merged.final_parsed_answer_old100.fillna("<NA>")
                != merged.final_parsed_answer_new500.fillna("<NA>")
            ).mean()
        ),
        "correctness_difference_rate": float(
            (merged.final_correct_old100 != merged.final_correct_new500).mean()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    graph_rows = []
    transition_rows = []

    for key in FIXED_KEYS:
        fixed_path = fixed500_analysis(args, key)
        depth_path = args.graph_depth500_root / "strata" / key / "analysis-v1"
        frames[("fixed_t3", key)] = task_metrics(fixed_path)
        frames[("graph_depth", key)] = task_metrics(depth_path)
        for regime, path in (("fixed_t3", fixed_path), ("graph_depth", depth_path)):
            summary_rows.extend(
                summarize(
                    regime,
                    key,
                    frames[(regime, key)],
                    rng,
                    args.bootstrap_replicates,
                )
            )
            _, _, graphs = load_frames(path)
            graphs.insert(0, "regime", regime)
            graphs.insert(1, "stratum", key)
            graph_rows.append(graphs)
            transition_rows.append(
                {"regime": regime, "stratum": key, **attack_transition_metrics(path)}
            )

    horizon_rows = []
    for key in FIXED_KEYS:
        horizon_rows.extend(
            paired_comparison(
                "graph_depth_minus_fixed_t3",
                key,
                frames[("graph_depth", key)],
                frames[("fixed_t3", key)],
                rng,
                args.bootstrap_replicates,
            )
        )

    adjacent_rows = []
    for regime in ("fixed_t3", "graph_depth"):
        for n in (5, 8):
            keys = sorted(
                (key for key in FIXED_KEYS if parse_key(key)[0] == n),
                key=lambda key: parse_key(key)[1],
            )
            for low_key, high_key in zip(keys, keys[1:], strict=False):
                adjacent_rows.extend(
                    paired_comparison(
                        f"{regime}:{high_key}_minus_{low_key}",
                        high_key,
                        frames[(regime, high_key)],
                        frames[(regime, low_key)],
                        rng,
                        args.bootstrap_replicates,
                    )
                )

    sensitivity_rows = []
    for key in FIXED_KEYS:
        manifest = json.loads((fixed500_batch(args, key) / "manifest.json").read_text())
        first100 = set(manifest["task_ids"][:100])
        first = task_metrics(fixed500_analysis(args, key), first100)
        full = frames[("fixed_t3", key)]
        for metric in METRICS:
            sensitivity_rows.append(
                {
                    "stratum": key,
                    "metric": metric,
                    "first100_estimate": float(first[metric].mean()),
                    "full500_estimate": float(full[metric].mean()),
                    "full_minus_first100": float(full[metric].mean() - first[metric].mean()),
                }
            )

    graphs = pd.concat(graph_rows, ignore_index=True)
    correlation_rows = []
    for regime, regime_frame in graphs.groupby("regime"):
        for n, frame in regime_frame.groupby("node_count"):
            for outcome in ("utility", "r_mean", "r_worst", "d_mean"):
                statistic = spearmanr(
                    frame.mean_max_induced_nonattacker_count, frame[outcome]
                ).statistic
                correlation_rows.append(
                    {
                        "regime": regime,
                        "node_count": n,
                        "x": "mean_max_induced_nonattacker_count",
                        "y": outcome,
                        "spearman": float(statistic),
                        "graphs": len(frame),
                        "scope": "descriptive selected-graph correlation",
                    }
                )

    compatibility = [old100_compatibility(args, key) for key in COMMON_OLD100_KEYS]
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "regime_stratum_estimates.csv", index=False
    )
    pd.DataFrame(horizon_rows).to_csv(
        args.output_dir / "horizon_effects_by_stratum.csv", index=False
    )
    pd.DataFrame(adjacent_rows).to_csv(
        args.output_dir / "adjacent_density_contrasts.csv", index=False
    )
    pd.DataFrame(sensitivity_rows).to_csv(
        args.output_dir / "sample_size_sensitivity.csv", index=False
    )
    pd.DataFrame(transition_rows).to_csv(
        args.output_dir / "attack_transition_metrics.csv", index=False
    )
    graphs.to_csv(args.output_dir / "graph_metrics_all.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(
        args.output_dir / "graph_correlations.csv", index=False
    )
    pd.DataFrame(compatibility).to_csv(
        args.output_dir / "old100_protocol_compatibility.csv", index=False
    )
    manifest = {
        "analysis_version": "complete-pilots-v1",
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "fixed_strata": list(FIXED_KEYS),
        "fixed_tasks": 500,
        "graph_depth_tasks": 500,
        "claim_limits": [
            "task-bootstrap intervals are conditional on the selected graph sets",
            "five graphs per non-complete stratum do not identify topology-population effects",
            "one model and one assignment/experiment seed",
            "graph-depth horizon jointly changes benign updates, compute, and attack exposure",
            (
                "the historical 100-task run used a different runner and is not a pure "
                "sample-size control"
            ),
        ],
    }
    atomic_text(
        args.output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
