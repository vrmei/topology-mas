"""Measure task-sampling stability of topology and vulnerable-node rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANALYSIS_VERSION = "topology-ranking-stability-v1"
DEFAULT_SPLITS = 1_000
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_807
GRAPH_OUTCOMES = ("clean_utility", "mean_attack_accuracy", "worst_node_attack_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-replicates", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_attack_table(run_root: Path, status: dict[str, Any]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for descriptor in status["strata"]:
        path = run_root / "strata" / str(descriptor["key"]) / "analysis-v1"
        frame = pd.DataFrame(read_jsonl(path / "paired_attacks.jsonl"))
        frame["stratum"] = str(descriptor["key"])
        frame["n"] = int(descriptor["n"])
        frame["m"] = int(descriptor["m"])
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def metric_tables(attacks: pd.DataFrame, task_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = attacks.loc[attacks["task_id"].isin(task_ids)]
    clean = selected[
        ["stratum", "task_id", "graph_id", "clean_correct"]
    ].drop_duplicates()
    clean_graph = (
        clean.groupby(["stratum", "graph_id"], sort=False)
        .agg(clean_utility=("clean_correct", "mean"))
        .reset_index()
    )
    mean_attack = (
        selected.groupby(["stratum", "graph_id"], sort=False)
        .agg(mean_attack_accuracy=("attack_correct", "mean"))
        .reset_index()
    )
    node = (
        selected.groupby(["stratum", "graph_id", "attack_node"], sort=False)
        .agg(
            attack_accuracy=("attack_correct", "mean"),
            induced_target_rate=("induced_readout_target", "mean"),
            paired_accuracy_drop=("paired_accuracy_drop", "mean"),
        )
        .reset_index()
    )
    worst = (
        node.groupby(["stratum", "graph_id"], sort=False)
        .agg(worst_node_attack_accuracy=("attack_accuracy", "min"))
        .reset_index()
    )
    graph = clean_graph.merge(mean_attack, on=["stratum", "graph_id"]).merge(
        worst, on=["stratum", "graph_id"]
    )
    return graph, node


def fractional_top_overlap(first: pd.Series, second: pd.Series) -> float:
    first_best = set(first.index[np.isclose(first, first.max())])
    second_best = set(second.index[np.isclose(second, second.max())])
    return len(first_best & second_best) / max(len(first_best), len(second_best))


def split_half_stability(
    attacks: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_ids = np.array(sorted(attacks["task_id"].unique()))
    if len(task_ids) % 2:
        raise ValueError("split-half analysis requires an even task count")
    graph_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        shuffled = rng.permutation(task_ids)
        first_ids = set(shuffled[: len(shuffled) // 2])
        second_ids = set(shuffled[len(shuffled) // 2 :])
        first_graph, first_node = metric_tables(attacks, first_ids)
        second_graph, second_node = metric_tables(attacks, second_ids)
        graph_pair = first_graph.merge(
            second_graph,
            on=["stratum", "graph_id"],
            suffixes=("_first", "_second"),
            validate="one_to_one",
        )
        for outcome in GRAPH_OUTCOMES:
            for stratum, frame in graph_pair.groupby("stratum", sort=True):
                if len(frame) < 2:
                    continue
                first = frame[f"{outcome}_first"].to_numpy(dtype=float)
                second = frame[f"{outcome}_second"].to_numpy(dtype=float)
                correlation = (
                    float(spearmanr(first, second).statistic)
                    if np.ptp(first) > 0 and np.ptp(second) > 0
                    else np.nan
                )
                graph_rows.append(
                    {
                        "replicate": replicate,
                        "stratum": stratum,
                        "outcome": outcome,
                        "graph_count": len(frame),
                        "spearman": correlation,
                    }
                )
        node_pair = first_node.merge(
            second_node,
            on=["stratum", "graph_id", "attack_node"],
            suffixes=("_first", "_second"),
            validate="one_to_one",
        )
        for (stratum, graph_id), frame in node_pair.groupby(
            ["stratum", "graph_id"], sort=True
        ):
            first = frame.set_index("attack_node")["induced_target_rate_first"]
            second = frame.set_index("attack_node")["induced_target_rate_second"]
            correlation = (
                float(spearmanr(first, second).statistic)
                if np.ptp(first) > 0 and np.ptp(second) > 0
                else np.nan
            )
            node_rows.append(
                {
                    "replicate": replicate,
                    "stratum": stratum,
                    "graph_id": graph_id,
                    "node_count": len(frame),
                    "induced_target_spearman": correlation,
                    "top_vulnerable_overlap": fractional_top_overlap(first, second),
                }
            )
    return pd.DataFrame(graph_rows), pd.DataFrame(node_rows)


def summarize_stability(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, sort=True)
        .agg(
            observations=("replicate", "size"),
            valid_correlations=("spearman", "count"),
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            q025_spearman=("spearman", lambda values: values.quantile(0.025)),
            q975_spearman=("spearman", lambda values: values.quantile(0.975)),
        )
        .reset_index()
    )


def pairwise_reversal_table(
    attacks: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    clean = attacks[
        ["stratum", "task_id", "graph_id", "clean_correct"]
    ].drop_duplicates()
    attack_task = (
        attacks.groupby(["stratum", "task_id", "graph_id"], sort=False)
        .agg(mean_attack_accuracy=("attack_correct", "mean"))
        .reset_index()
    )
    sources = {
        "clean_utility": clean.rename(columns={"clean_correct": "value"}),
        "mean_attack_accuracy": attack_task.rename(
            columns={"mean_attack_accuracy": "value"}
        ),
    }
    rows: list[dict[str, Any]] = []
    for outcome, source in sources.items():
        for stratum, frame in source.groupby("stratum", sort=True):
            pivot = frame.pivot(index="task_id", columns="graph_id", values="value")
            graph_ids = sorted(pivot.columns)
            if len(graph_ids) < 2:
                continue
            values = pivot[graph_ids].to_numpy(dtype=float)
            sample_indices = rng.integers(0, len(values), size=(replicates, len(values)))
            sampled_means = values[sample_indices].mean(axis=1)
            full_means = values.mean(axis=0)
            for first_index, second_index in combinations(range(len(graph_ids)), 2):
                difference = full_means[first_index] - full_means[second_index]
                draws = sampled_means[:, first_index] - sampled_means[:, second_index]
                if np.isclose(difference, 0.0):
                    reversal_probability = float(np.mean(~np.isclose(draws, 0.0)))
                else:
                    reversal_probability = float(np.mean(np.sign(draws) != np.sign(difference)))
                rows.append(
                    {
                        "stratum": stratum,
                        "outcome": outcome,
                        "graph_a": graph_ids[first_index],
                        "graph_b": graph_ids[second_index],
                        "full_difference_a_minus_b": float(difference),
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "ordering_reversal_probability": reversal_probability,
                        "ordering_resolved": bool(
                            np.quantile(draws, 0.025) > 0 or np.quantile(draws, 0.975) < 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def render_report(
    integrity: dict[str, Any],
    graph_summary: pd.DataFrame,
    node_summary: pd.DataFrame,
    pairs: pd.DataFrame,
) -> str:
    lines = ["# Task-conditioned topology-ranking stability", "", "## Integrity", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in integrity.items())
    lines.extend(
        [
            "",
            "## Graph split-half rank stability",
            "",
            "| stratum | outcome | valid | mean Spearman | 95% split interval |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in graph_summary.iterrows():
        lines.append(
            f"| {row['stratum']} | {row['outcome']} | {int(row['valid_correlations'])} | "
            f"{row['mean_spearman']:.3f} | "
            f"[{row['q025_spearman']:.3f}, {row['q975_spearman']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Vulnerable-node stability",
            "",
            f"- Valid node-rank correlations: {int(node_summary['valid_correlations'].iloc[0])}",
            f"- Mean node-rank Spearman: {node_summary['mean_spearman'].iloc[0]:.3f}",
            f"- Mean top-node overlap: {node_summary['mean_top_overlap'].iloc[0]:.3f}",
            "",
            "## Pairwise graph ordering",
            "",
            f"- Evaluated graph pairs: {len(pairs)}",
            f"- Resolved orderings: {int(pairs['ordering_resolved'].sum())}",
            f"- Unresolved orderings: {int((~pairs['ordering_resolved']).sum())}",
            "",
            "## Claim guardrails",
            "",
            "- Comparisons are confined to matched `(n,m)` strata.",
            "- Instability may reflect finite positive-event counts, not task-specific semantics.",
            "- Stability on GSM8K does not establish cross-dataset or cross-model transfer.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.split_replicates < 100 or args.bootstrap_replicates < 100:
        raise ValueError("split and bootstrap replicates must each be at least 100")
    run_root = args.run_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "orchestrator_status.json"
    status = read_json(status_path)
    if status.get("status") != "completed":
        raise RuntimeError("pilot must be completed before ranking-stability analysis")
    attacks = load_attack_table(run_root, status)
    keys = ["task_id", "graph_id", "attack_node", "experiment_seed", "assignment_seed"]
    duplicate_rows = int(attacks.duplicated(keys).sum())
    rng = np.random.default_rng(args.seed)
    graph_splits, node_splits = split_half_stability(
        attacks, replicates=args.split_replicates, rng=rng
    )
    graph_summary = summarize_stability(graph_splits, ["stratum", "outcome"])
    node_for_summary = node_splits.rename(columns={"induced_target_spearman": "spearman"})
    node_summary = pd.DataFrame(
        [
            {
                "observations": len(node_splits),
                "valid_correlations": int(node_for_summary["spearman"].notna().sum()),
                "mean_spearman": float(node_for_summary["spearman"].mean()),
                "median_spearman": float(node_for_summary["spearman"].median()),
                "mean_top_overlap": float(node_splits["top_vulnerable_overlap"].mean()),
            }
        ]
    )
    pairs = pairwise_reversal_table(
        attacks, replicates=args.bootstrap_replicates, rng=rng
    )
    task_count = int(attacks["task_id"].nunique())
    graph_count = int(attacks["graph_id"].nunique())
    integrity = {
        "passed": bool(
            duplicate_rows == 0
            and task_count == int(status["task_count"])
            and len(graph_splits) > 0
            and len(node_splits) == args.split_replicates * graph_count
        ),
        "attack_conditions": len(attacks),
        "tasks": task_count,
        "graphs": graph_count,
        "duplicate_attack_keys": duplicate_rows,
        "split_replicates": args.split_replicates,
        "pairwise_bootstrap_replicates": args.bootstrap_replicates,
        "rankable_strata": int(graph_splits["stratum"].nunique()),
    }
    if not integrity["passed"]:
        raise RuntimeError("ranking-stability integrity audit failed")
    graph_splits.to_csv(output / "graph_split_half_stability.csv", index=False)
    node_splits.to_csv(output / "node_split_half_stability.csv", index=False)
    graph_summary.to_csv(output / "graph_stability_summary.csv", index=False)
    node_summary.to_csv(output / "node_stability_summary.csv", index=False)
    pairs.to_csv(output / "pairwise_ordering_uncertainty.csv", index=False)
    (output / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(
        render_report(integrity, graph_summary, node_summary, pairs), encoding="utf-8"
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(run_root),
        "source_status_sha256": sha256_file(status_path),
        "split_replicates": args.split_replicates,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "integrity_passed": integrity["passed"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
