"""Compare the fixed-T=3 pilot with graph-depth Experiment C."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, deque
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRIC_LABELS = {
    "utility": "Clean accuracy",
    "r_mean": "Mean attack accuracy",
    "d_mean": "Mean paired accuracy drop",
    "induced_target_rate": "Induced target-error rate",
    "propagation_count": "Maximum induced non-attacker count",
    "clean_model_calls": "Clean logical model calls",
    "attack_model_calls": "Attack logical model calls",
    "clean_input_tokens": "Clean input tokens",
    "attack_input_tokens": "Attack input tokens",
}


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fieldnames = list(rows[0])
    output = []
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output.append(buffer.getvalue())
    atomic_text(path, "".join(output))


def task_metrics(analysis_dir: Path, graph_ids: set[str] | None = None) -> pd.DataFrame:
    runs = read_jsonl(analysis_dir / "run_metrics.jsonl")
    paired = read_jsonl(analysis_dir / "paired_attacks.jsonl")
    if graph_ids is not None:
        runs = runs[runs.graph_id.isin(graph_ids)]
        paired = paired[paired.graph_id.isin(graph_ids)]
    clean = runs[runs.condition == "clean"]
    attack = runs[runs.condition == "attack"]
    result = pd.DataFrame(index=sorted(runs.task_id.unique()))
    result.index.name = "task_id"
    result["utility"] = clean.groupby("task_id").final_correct.mean()
    result["r_mean"] = attack.groupby("task_id").final_correct.mean()
    result["d_mean"] = result.utility - result.r_mean
    result["induced_target_rate"] = paired.groupby("task_id").induced_readout_target.mean()
    result["propagation_count"] = paired.groupby("task_id").max_induced_nonattacker_count.mean()
    result["clean_model_calls"] = clean.groupby("task_id").model_calls.mean()
    result["attack_model_calls"] = attack.groupby("task_id").model_calls.mean()
    result["clean_input_tokens"] = clean.groupby("task_id").input_tokens.mean()
    result["attack_input_tokens"] = attack.groupby("task_id").input_tokens.mean()
    if result.isna().any().any():
        raise ValueError(f"incomplete task metric table in {analysis_dir}")
    return result


def bootstrap_mean(values: np.ndarray, *, seed: int, replicates: int) -> tuple[float, float, float]:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("bootstrap requires a one-dimensional sample")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975)
    )


def summarize_task_metrics(
    regimes: dict[str, pd.DataFrame], *, seed: int, replicates: int
) -> list[dict[str, Any]]:
    rows = []
    for regime, frame in regimes.items():
        for metric, label in METRIC_LABELS.items():
            point, low, high = bootstrap_mean(
                frame[metric].to_numpy(dtype=float), seed=seed, replicates=replicates
            )
            rows.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "metric_label": label,
                    "estimate": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "task_clusters": len(frame),
                }
            )
    return rows


def paired_differences(
    comparisons: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    seed: int,
    replicates: int,
) -> list[dict[str, Any]]:
    rows = []
    for comparison, (left, right) in comparisons.items():
        if not left.index.equals(right.index):
            raise ValueError(f"task pairing differs for {comparison}")
        for metric, label in METRIC_LABELS.items():
            delta = left[metric].to_numpy(dtype=float) - right[metric].to_numpy(dtype=float)
            point, low, high = bootstrap_mean(delta, seed=seed, replicates=replicates)
            rows.append(
                {
                    "comparison": comparison,
                    "estimand": "left_minus_right",
                    "metric": metric,
                    "metric_label": label,
                    "estimate": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "task_clusters": len(delta),
                    "ci_excludes_zero": low > 0 or high < 0,
                }
            )
    return rows


def load_graph_metrics(path: Path, *, regime: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "graph_metrics.csv")
    frame.insert(0, "regime", regime)
    return frame


def batch_resource_rows(batch_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for regime, batch_dir in batch_dirs.items():
        summary = json.loads((batch_dir / "summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "regime": regime,
                "completed_runs": summary["completed_runs"],
                "logical_model_calls": summary["trace_model_calls"],
                "backend_model_calls": summary["trace_backend_calls"],
                "state_replay_cache_hits": summary["state_replay_cache_hits"],
                "input_tokens": summary["known_input_tokens"],
                "output_tokens": summary["known_output_tokens"],
            }
        )
    return rows


def graph_depths(batch_dir: Path) -> dict[str, int]:
    values = {}
    for line in (batch_dir / "inputs" / "graphs.jsonl").read_text(encoding="utf-8").splitlines():
        graph = json.loads(line)
        predecessors = [[] for _ in range(graph["node_count"])]
        for edge in graph["edges"]:
            predecessors[edge["target"]].append(edge["source"])
        distances: list[int | None] = [None] * graph["node_count"]
        distances[graph["readout_node"]] = 0
        queue = deque([graph["readout_node"]])
        while queue:
            target = queue.popleft()
            for source in predecessors[target]:
                if distances[source] is None:
                    distances[source] = int(distances[target]) + 1
                    queue.append(source)
        if any(distance is None for distance in distances):
            raise ValueError(f"unreachable node in {graph['graph_id']}")
        values[graph["graph_id"]] = max(int(distance) for distance in distances)
    return values


def graph_and_node_horizon_tables(
    fixed_dir: Path, depth_dir: Path, depths: dict[str, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixed_graphs = {
        item["graph_id"]: item
        for item in json.loads((fixed_dir / "graph_metrics.json").read_text(encoding="utf-8"))
    }
    depth_graphs = {
        item["graph_id"]: item
        for item in json.loads((depth_dir / "graph_metrics.json").read_text(encoding="utf-8"))
    }
    graph_rows = []
    node_rows = []
    for graph_id in sorted(fixed_graphs):
        fixed = fixed_graphs[graph_id]
        depth = depth_graphs[graph_id]
        fixed_nodes = {item["node_id"]: item for item in fixed["node_metrics"]}
        depth_nodes = {item["node_id"]: item for item in depth["node_metrics"]}
        fixed_losses = np.array(
            [fixed_nodes[node]["paired_accuracy_drop"] for node in sorted(fixed_nodes)]
        )
        depth_losses = np.array(
            [depth_nodes[node]["paired_accuracy_drop"] for node in sorted(depth_nodes)]
        )
        correlation = spearmanr(fixed_losses, depth_losses).statistic
        graph_rows.append(
            {
                "graph_id": graph_id,
                "graph_depth": depths[graph_id],
                "utility_fixed_t3": fixed["utility"],
                "utility_graph_depth": depth["utility"],
                "utility_delta": depth["utility"] - fixed["utility"],
                "r_mean_fixed_t3": fixed["r_mean"],
                "r_mean_graph_depth": depth["r_mean"],
                "r_mean_delta": depth["r_mean"] - fixed["r_mean"],
                "r_worst_fixed_t3": fixed["r_worst"],
                "r_worst_graph_depth": depth["r_worst"],
                "r_worst_delta": depth["r_worst"] - fixed["r_worst"],
                "induced_target_fixed_t3": fixed["induced_readout_target_rate"],
                "induced_target_graph_depth": depth["induced_readout_target_rate"],
                "induced_target_delta": depth["induced_readout_target_rate"]
                - fixed["induced_readout_target_rate"],
                "vulnerability_rank_spearman": float(correlation),
                "worst_node_fixed_t3": min(
                    fixed_nodes, key=lambda node: fixed_nodes[node]["attack_accuracy"]
                ),
                "worst_node_graph_depth": min(
                    depth_nodes, key=lambda node: depth_nodes[node]["attack_accuracy"]
                ),
            }
        )
        for node_id in sorted(fixed_nodes):
            node_rows.append(
                {
                    "graph_id": graph_id,
                    "graph_depth": depths[graph_id],
                    "node_id": node_id,
                    "attack_accuracy_fixed_t3": fixed_nodes[node_id]["attack_accuracy"],
                    "attack_accuracy_graph_depth": depth_nodes[node_id]["attack_accuracy"],
                    "paired_drop_fixed_t3": fixed_nodes[node_id]["paired_accuracy_drop"],
                    "paired_drop_graph_depth": depth_nodes[node_id]["paired_accuracy_drop"],
                    "induced_target_fixed_t3": fixed_nodes[node_id][
                        "induced_readout_target_rate"
                    ],
                    "induced_target_graph_depth": depth_nodes[node_id][
                        "induced_readout_target_rate"
                    ],
                }
            )
    return graph_rows, node_rows


def transition_table(
    fixed_dir: Path, depth_dir: Path, depth_two_graphs: set[str]
) -> list[dict[str, Any]]:
    fixed_runs = read_jsonl(fixed_dir / "run_metrics.jsonl")
    depth_runs = read_jsonl(depth_dir / "run_metrics.jsonl")
    keys = ["task_id", "graph_id", "condition", "attack_node"]
    fixed_runs = fixed_runs[fixed_runs.graph_id.isin(depth_two_graphs)]
    depth_runs = depth_runs[depth_runs.graph_id.isin(depth_two_graphs)]
    merged = fixed_runs.merge(
        depth_runs,
        on=keys,
        suffixes=("_fixed", "_depth"),
        validate="one_to_one",
    )
    rows = []
    for condition in ("clean", "attack"):
        subset = merged[merged.condition == condition]
        categories = {
            "both_correct": subset.final_correct_fixed & subset.final_correct_depth,
            "fixed_only_correct": subset.final_correct_fixed & ~subset.final_correct_depth,
            "depth_only_correct": ~subset.final_correct_fixed & subset.final_correct_depth,
            "both_wrong": ~subset.final_correct_fixed & ~subset.final_correct_depth,
        }
        for transition, mask in categories.items():
            rows.append(
                {
                    "condition": condition,
                    "transition": transition,
                    "count": int(mask.sum()),
                    "rate": float(mask.mean()),
                    "paired_cells": len(subset),
                }
            )
    return rows


def audit_truncation(
    source_batch: Path, target_batch: Path, depths: dict[str, int]
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    examples = []
    target_files = sorted((target_batch / "traces").glob("*.json"))
    for target_path in target_files:
        source_path = source_batch / "traces" / target_path.name
        counters["target_traces"] += 1
        if not source_path.exists():
            counters["missing_source"] += 1
            continue
        target = json.loads(target_path.read_text(encoding="utf-8"))
        source = json.loads(source_path.read_text(encoding="utf-8"))
        new_trace = target["trace"]
        old_trace = source["trace"]
        graph_id = target["run_spec"]["graph_id"]
        depth = depths[graph_id]
        readout = next(
            graph["readout_node"]
            for graph in (
                json.loads(line)
                for line in (target_batch / "inputs" / "graphs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if graph["graph_id"] == graph_id
        )
        counters[f"depth_{depth}_traces"] += 1
        if new_trace["schedule"].get("effective_horizon") != depth:
            counters["bad_effective_horizon"] += 1
        if max(turn["round_index"] for turn in new_trace["turns"]) != depth:
            counters["bad_max_turn"] += 1
        new_readout = [
            turn
            for turn in new_trace["turns"]
            if turn["node_id"] == readout and turn["round_index"] == depth
        ]
        old_readout = [
            turn
            for turn in old_trace["turns"]
            if turn["node_id"] == readout and turn["round_index"] == depth
        ]
        if len(new_readout) != 1 or len(old_readout) != 1:
            counters["missing_cutoff_readout"] += 1
            continue
        for suffix in ("raw_output", "parsed_answer", "answer_state"):
            if new_trace[f"final_{suffix}"] != old_readout[0][suffix]:
                counters["final_differs_from_fixed_prefix"] += 1
                break
        old_turns = {
            (turn["round_index"], turn["node_id"]): turn for turn in old_trace["turns"]
        }
        for turn in new_trace["turns"]:
            previous = old_turns.get((turn["round_index"], turn["node_id"]))
            same = previous is not None and all(
                turn[field] == previous[field]
                for field in (
                    "raw_output",
                    "parsed_answer",
                    "answer_state",
                    "prompt_messages",
                    "generation_seed",
                )
            )
            if not same:
                counters["prefix_turn_mismatch"] += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            "trace": target_path.name,
                            "round": turn["round_index"],
                            "node": turn["node_id"],
                        }
                    )
                break
    expected_keys = {"target_traces", "depth_2_traces", "depth_3_traces"}
    failure_keys = [key for key in counters if key not in expected_keys]
    return {
        "source_batch": str(source_batch.resolve()),
        "target_batch": str(target_batch.resolve()),
        "counts": dict(sorted(counters.items())),
        "examples": examples,
        "passed": not failure_keys and counters["target_traces"] > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-m10-analysis", type=Path, required=True)
    parser.add_argument("--fixed-m21-analysis", type=Path, required=True)
    parser.add_argument("--depth-m21-analysis", type=Path, required=True)
    parser.add_argument("--fixed-m10-batch", type=Path, required=True)
    parser.add_argument("--fixed-m21-batch", type=Path, required=True)
    parser.add_argument("--depth-m21-batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    args = parser.parse_args()

    depths = graph_depths(args.depth_m21_batch)
    depth_two = {graph_id for graph_id, depth in depths.items() if depth == 2}
    depth_three = {graph_id for graph_id, depth in depths.items() if depth == 3}
    fixed_m10 = task_metrics(args.fixed_m10_analysis)
    fixed_m21 = task_metrics(args.fixed_m21_analysis)
    depth_m21 = task_metrics(args.depth_m21_analysis)
    fixed_m21_d2 = task_metrics(args.fixed_m21_analysis, depth_two)
    depth_m21_d2 = task_metrics(args.depth_m21_analysis, depth_two)
    fixed_m21_d3 = task_metrics(args.fixed_m21_analysis, depth_three)
    depth_m21_d3 = task_metrics(args.depth_m21_analysis, depth_three)

    regimes = {
        "fixed_t3_m10": fixed_m10,
        "fixed_t3_m21": fixed_m21,
        "graph_depth_m10_exact_reuse": fixed_m10,
        "graph_depth_m21": depth_m21,
    }
    summary_rows = summarize_task_metrics(
        regimes, seed=args.bootstrap_seed, replicates=args.bootstrap_replicates
    )
    comparison_rows = paired_differences(
        {
            "fixed_t3_m21_minus_fixed_t3_m10": (fixed_m21, fixed_m10),
            "graph_depth_m21_minus_graph_depth_m10": (depth_m21, fixed_m10),
            "graph_depth_minus_fixed_t3_m21_depth2_only": (
                depth_m21_d2,
                fixed_m21_d2,
            ),
            "graph_depth_minus_fixed_t3_m21_depth3_control": (
                depth_m21_d3,
                fixed_m21_d3,
            ),
        },
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    graph_frames = [
        load_graph_metrics(args.fixed_m10_analysis, regime="fixed_t3_m10"),
        load_graph_metrics(args.fixed_m21_analysis, regime="fixed_t3_m21"),
        load_graph_metrics(args.fixed_m10_analysis, regime="graph_depth_m10_exact_reuse"),
        load_graph_metrics(args.depth_m21_analysis, regime="graph_depth_m21"),
    ]
    graph_table = pd.concat(graph_frames, ignore_index=True)
    graph_rows, node_rows = graph_and_node_horizon_tables(
        args.fixed_m21_analysis, args.depth_m21_analysis, depths
    )
    transitions = transition_table(
        args.fixed_m21_analysis, args.depth_m21_analysis, depth_two
    )
    audit = audit_truncation(args.fixed_m21_batch, args.depth_m21_batch, depths)
    if not audit["passed"]:
        raise ValueError("graph-depth truncation audit failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "task_cluster_summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)
    atomic_text(args.output_dir / "graph_metrics_all_regimes.csv", graph_table.to_csv(index=False))
    write_csv(args.output_dir / "per_graph_horizon_comparison.csv", graph_rows)
    write_csv(args.output_dir / "per_node_horizon_comparison.csv", node_rows)
    write_csv(args.output_dir / "round2_round3_transitions.csv", transitions)
    write_csv(
        args.output_dir / "batch_resource_summary.csv",
        batch_resource_rows(
            {
                "fixed_t3_m10": args.fixed_m10_batch,
                "fixed_t3_m21": args.fixed_m21_batch,
                "graph_depth_m21": args.depth_m21_batch,
            }
        ),
    )
    atomic_text(
        args.output_dir / "truncation_audit.json",
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    manifest = {
        "analysis_version": "fixed-vs-graph-depth-v1",
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "confidence_interval": "task-clustered paired percentile bootstrap",
        "fixed_m10_manifest_sha256": sha256_file(args.fixed_m10_analysis / "manifest.json"),
        "fixed_m21_manifest_sha256": sha256_file(args.fixed_m21_analysis / "manifest.json"),
        "depth_m21_manifest_sha256": sha256_file(args.depth_m21_analysis / "manifest.json"),
        "graph_depths": depths,
        "depth_two_graphs": sorted(depth_two),
        "depth_three_control_graphs": sorted(depth_three),
        "claim_limits": [
            "single model, dataset, experiment seed, and assignment seed",
            "task-cluster confidence intervals are conditional on the selected graph sets",
            "fixed-T versus graph-depth changes benign updates and persistent attack dose together",
        ],
        "truncation_audit_passed": audit["passed"],
    }
    atomic_text(
        args.output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
