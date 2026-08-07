"""Audit and summarize one completed multi-stratum topology pilot.

This is the first post-hoc analysis stage.  It deliberately avoids fitting a
structural explanation model.  Its estimands are conditional on the selected
model, prompt, assignment seed, and experiment seed recorded by the pilot.

Uncertainty is estimated with a crossed graph-by-task bootstrap.  Attack nodes
are design positions and are therefore kept complete within every resampled
graph/task cell rather than resampled as if they were independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")


ANALYSIS_VERSION = "scale-pilot-descriptive-v1"
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_807


@dataclass(frozen=True)
class StratumData:
    key: str
    n: int
    m: int
    selected_graphs: int
    available_graphs: int
    clean: pd.DataFrame
    attacks: pd.DataFrame
    graph_metrics: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def load_stratum(run_root: Path, descriptor: dict[str, Any]) -> StratumData:
    key = descriptor["key"]
    analysis_root = run_root / "strata" / key / "analysis-v1"
    run_metrics = read_jsonl(analysis_root / "run_metrics.jsonl")
    attacks = read_jsonl(analysis_root / "paired_attacks.jsonl")
    graph_metrics = pd.read_csv(analysis_root / "graph_metrics.csv")
    clean = run_metrics.loc[run_metrics["condition"] == "clean"].copy()
    return StratumData(
        key=key,
        n=int(descriptor["n"]),
        m=int(descriptor["m"]),
        selected_graphs=int(descriptor["selected_graphs"]),
        available_graphs=int(descriptor["available_graphs"]),
        clean=clean,
        attacks=attacks,
        graph_metrics=graph_metrics,
    )


def audit_stratum(data: StratumData, task_count: int) -> dict[str, Any]:
    errors: list[str] = []
    expected_clean = task_count * data.selected_graphs
    expected_attacks = expected_clean * (data.n - 1)
    clean_key = ["task_id", "graph_id", "experiment_seed", "assignment_seed"]
    attack_key = clean_key + ["attack_node"]

    require(
        len(data.clean) == expected_clean,
        f"clean rows {len(data.clean)} != {expected_clean}",
        errors,
    )
    require(
        len(data.attacks) == expected_attacks,
        f"paired attack rows {len(data.attacks)} != {expected_attacks}",
        errors,
    )
    require(not data.clean.duplicated(clean_key).any(), "duplicate clean keys", errors)
    require(not data.attacks.duplicated(attack_key).any(), "duplicate paired attack keys", errors)
    require(
        data.clean["graph_id"].nunique() == data.selected_graphs,
        "clean graph count differs from selected graph count",
        errors,
    )
    require(
        data.attacks["graph_id"].nunique() == data.selected_graphs,
        "attack graph count differs from selected graph count",
        errors,
    )
    require(
        data.graph_metrics["graph_id"].nunique() == data.selected_graphs,
        "graph metric count differs from selected graph count",
        errors,
    )
    require(data.clean["task_id"].nunique() == task_count, "clean task coverage incomplete", errors)
    require(
        data.attacks["task_id"].nunique() == task_count,
        "attack task coverage incomplete",
        errors,
    )

    expected_nodes = set(range(data.n - 1))
    for graph_id, frame in data.attacks.groupby("graph_id", sort=False):
        observed_nodes = set(int(value) for value in frame["attack_node"].unique())
        require(
            observed_nodes == expected_nodes,
            f"{graph_id}: attack nodes {sorted(observed_nodes)} != {sorted(expected_nodes)}",
            errors,
        )
        counts = frame.groupby("attack_node")["task_id"].nunique()
        require(
            bool((counts == task_count).all()),
            f"{graph_id}: at least one attack node lacks complete task coverage",
            errors,
        )

    recomputed_utility = data.clean.groupby("graph_id")["final_correct"].mean().sort_index()
    recorded_utility = data.graph_metrics.set_index("graph_id")["utility"].sort_index()
    require(
        np.allclose(recomputed_utility.to_numpy(), recorded_utility.to_numpy()),
        "recorded utility differs from clean rows",
        errors,
    )

    recomputed_r_mean = data.attacks.groupby("graph_id")["attack_correct"].mean().sort_index()
    recorded_r_mean = data.graph_metrics.set_index("graph_id")["r_mean"].sort_index()
    require(
        np.allclose(recomputed_r_mean.to_numpy(), recorded_r_mean.to_numpy()),
        "recorded r_mean differs from paired attack rows",
        errors,
    )

    return {
        "key": data.key,
        "passed": not errors,
        "errors": errors,
        "n": data.n,
        "m": data.m,
        "selected_graphs": data.selected_graphs,
        "available_graphs": data.available_graphs,
        "topology_generalization_identifiable": data.selected_graphs > 1,
        "clean_rows": len(data.clean),
        "paired_attack_rows": len(data.attacks),
        "task_count": task_count,
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def estimate_metrics(
    data: StratumData,
    graph_weights: dict[str, int] | None = None,
    task_weights: dict[str, int] | None = None,
) -> dict[str, float]:
    graph_ids = list(data.clean["graph_id"].unique())
    task_ids = list(data.clean["task_id"].unique())
    graph_weights = graph_weights or dict.fromkeys(graph_ids, 1)
    task_weights = task_weights or dict.fromkeys(task_ids, 1)

    clean = data.clean.copy()
    clean["_weight"] = clean["graph_id"].map(graph_weights) * clean["task_id"].map(task_weights)
    clean = clean.loc[clean["_weight"] > 0]

    attacks = data.attacks.copy()
    attacks["_weight"] = (
        attacks["graph_id"].map(graph_weights) * attacks["task_id"].map(task_weights)
    )
    attacks = attacks.loc[attacks["_weight"] > 0]

    utility = weighted_mean(clean["final_correct"], clean["_weight"])
    r0_accuracy = weighted_mean(clean["readout_round_zero_correct"], clean["_weight"])
    r_mean = weighted_mean(attacks["attack_correct"], attacks["_weight"])
    d_mean = weighted_mean(attacks["paired_accuracy_drop"], attacks["_weight"])
    induced_target_rate = weighted_mean(attacks["induced_readout_target"], attacks["_weight"])

    clean_correct = attacks.loc[attacks["clean_correct"]]
    if clean_correct.empty:
        corruption_rate = float("nan")
        target_flip_rate = float("nan")
        non_target_corruption_rate = float("nan")
    else:
        corruption = (~clean_correct["attack_correct"]).astype(int)
        target_flip = clean_correct["correct_to_target_flip"].astype(int)
        non_target_corruption = (
            (~clean_correct["attack_correct"]) & (~clean_correct["attack_final_matches_target"])
        ).astype(int)
        corruption_rate = weighted_mean(corruption, clean_correct["_weight"])
        target_flip_rate = weighted_mean(target_flip, clean_correct["_weight"])
        non_target_corruption_rate = weighted_mean(
            non_target_corruption, clean_correct["_weight"]
        )

    r_worst_values: list[tuple[float, int]] = []
    d_max_values: list[tuple[float, int]] = []
    for graph_id, graph_weight in graph_weights.items():
        if graph_weight <= 0:
            continue
        frame = attacks.loc[attacks["graph_id"] == graph_id]
        node_accuracy: list[float] = []
        node_drop: list[float] = []
        for _, node_frame in frame.groupby("attack_node", sort=False):
            node_accuracy.append(weighted_mean(node_frame["attack_correct"], node_frame["_weight"]))
            node_drop.append(
                weighted_mean(node_frame["paired_accuracy_drop"], node_frame["_weight"])
            )
        r_worst_values.append((min(node_accuracy), graph_weight))
        d_max_values.append((max(node_drop), graph_weight))

    r_worst_mean = float(
        np.average([x[0] for x in r_worst_values], weights=[x[1] for x in r_worst_values])
    )
    d_max_mean = float(
        np.average([x[0] for x in d_max_values], weights=[x[1] for x in d_max_values])
    )

    return {
        "utility": utility,
        "readout_round_zero_accuracy": r0_accuracy,
        "utility_minus_round_zero": utility - r0_accuracy,
        "r_mean": r_mean,
        "d_mean": d_mean,
        "mean_graph_r_worst": r_worst_mean,
        "mean_graph_d_max": d_max_mean,
        "induced_readout_target_rate": induced_target_rate,
        "correct_to_any_error_rate": corruption_rate,
        "correct_to_target_flip_rate": target_flip_rate,
        "correct_to_non_target_error_rate": non_target_corruption_rate,
    }


def sampled_counts(values: list[str], rng: np.random.Generator) -> dict[str, int]:
    sampled = rng.choice(values, size=len(values), replace=True)
    unique, counts = np.unique(sampled, return_counts=True)
    result = dict.fromkeys(values, 0)
    result.update({str(key): int(value) for key, value in zip(unique, counts, strict=True)})
    return result


def bootstrap_metrics(
    data: StratumData,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    graph_ids = [str(value) for value in data.clean["graph_id"].unique()]
    task_ids = [str(value) for value in data.clean["task_id"].unique()]
    draws: dict[str, list[float]] = {}
    for _ in range(replicates):
        graph_weights = (
            sampled_counts(graph_ids, rng) if len(graph_ids) > 1 else {graph_ids[0]: 1}
        )
        task_weights = sampled_counts(task_ids, rng)
        estimates = estimate_metrics(data, graph_weights, task_weights)
        for metric, value in estimates.items():
            if np.isfinite(value):
                draws.setdefault(metric, []).append(value)
    return {
        metric: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in draws.items()
    }


def summarize_stratum(
    data: StratumData,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    estimates = estimate_metrics(data)
    intervals = bootstrap_metrics(data, replicates, rng)
    rows: list[dict[str, Any]] = []
    for metric, estimate in estimates.items():
        low, high = intervals.get(metric, (float("nan"), float("nan")))
        rows.append(
            {
                "stratum": data.key,
                "n": data.n,
                "m": data.m,
                "edge_density": data.m / ((data.n - 1) ** 2),
                "selected_graphs": data.selected_graphs,
                "tasks": data.clean["task_id"].nunique(),
                "metric": metric,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_unit": (
                    "task_only_unique_topology"
                    if data.selected_graphs == 1
                    else "crossed_graph_by_task"
                ),
            }
        )
    return rows


def write_table(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def plot_strata(estimates: pd.DataFrame, path: Path) -> None:
    metrics = ["utility", "r_mean", "d_mean", "correct_to_target_flip_rate"]
    titles = [
        "Clean utility",
        "Mean attacked accuracy",
        "Paired accuracy drop",
        "Target flip | clean correct",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric, title in zip(axes.flat, metrics, titles, strict=True):
        frame = estimates.loc[estimates["metric"] == metric]
        for n, group in frame.groupby("n"):
            group = group.sort_values("edge_density")
            lower = group["estimate"] - group["ci95_low"]
            upper = group["ci95_high"] - group["estimate"]
            axis.errorbar(
                group["edge_density"],
                group["estimate"],
                yerr=np.vstack([lower, upper]),
                marker="o",
                capsize=3,
                label=f"n={n}",
            )
        axis.set_title(title)
        axis.set_xlabel("Directed edge density")
        axis.set_ylabel("Rate")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_graph_scatter(graphs: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for n, group in graphs.groupby("node_count"):
        axis.scatter(group["utility"], group["r_worst"], label=f"n={n}", alpha=0.8)
    axis.set_xlabel("Clean utility")
    axis.set_ylabel("Worst attacked-node accuracy")
    axis.set_title("Selected graph outcomes (descriptive only)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(
    status: dict[str, Any],
    audits: list[dict[str, Any]],
    estimates: pd.DataFrame,
) -> str:
    lines = [
        "# Scale pilot descriptive audit",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`",
        "",
        "## Scope",
        "",
        (
            "These estimates are conditional on the recorded model, prompt, task sample, "
            "assignment seed, and experiment seed. They do not yet identify a graph mechanism."
        ),
        "",
        "## Integrity",
        "",
        f"- Pilot status: `{status['status']}`",
        f"- Tasks: {status['task_count']}",
        f"- Expected traces: {status['expected_total_traces']}",
        f"- All stratum audits passed: `{all(item['passed'] for item in audits)}`",
        "",
        "## Primary descriptive estimates",
        "",
        "| stratum | graphs | utility | r_mean | d_mean | target flip |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    pivot = estimates.pivot(index="stratum", columns="metric", values="estimate")
    graph_counts = estimates.groupby("stratum")["selected_graphs"].first()
    for stratum in [item["key"] for item in status["strata"]]:
        row = pivot.loc[stratum]
        lines.append(
            f"| {stratum} | {int(graph_counts.loc[stratum])} | "
            f"{row['utility']:.3f} | {row['r_mean']:.3f} | {row['d_mean']:.3f} | "
            f"{row['correct_to_target_flip_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Confidence intervals resample tasks and, where available, selected graphs.",
            "- Attack nodes are exhaustively enumerated design positions, not IID samples.",
            (
                "- The complete n=5, m=16 stratum has one unique topology; graph-level "
                "generalization is not identifiable."
            ),
            (
                "- One model and one assignment/experiment seed do not establish cross-model "
                "or stochastic generalization."
            ),
            (
                "- No monotonicity, classical-graph failure, or semantic mechanism claim is "
                "made in this stage."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_root / "orchestrator_status.json"
    status = read_json(status_path)
    if status.get("status") != "completed":
        raise RuntimeError("pilot must be completed before post-hoc analysis")

    strata = [load_stratum(run_root, descriptor) for descriptor in status["strata"]]
    audits = [audit_stratum(data, int(status["task_count"])) for data in strata]
    if not all(item["passed"] for item in audits):
        (output_dir / "integrity_audit.json").write_text(
            json.dumps(audits, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError("integrity audit failed; see integrity_audit.json")

    rng = np.random.default_rng(args.seed)
    estimate_rows: list[dict[str, Any]] = []
    graph_frames: list[pd.DataFrame] = []
    for data in strata:
        estimate_rows.extend(summarize_stratum(data, args.bootstrap_replicates, rng))
        frame = data.graph_metrics.copy()
        frame.insert(0, "stratum", data.key)
        frame["edge_density"] = data.m / ((data.n - 1) ** 2)
        graph_frames.append(frame)

    estimates = pd.DataFrame(estimate_rows)
    graphs = pd.concat(graph_frames, ignore_index=True)

    (output_dir / "integrity_audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )
    write_table(output_dir / "stratum_estimates.csv", estimates)
    write_table(output_dir / "graph_metrics_all.csv", graphs)
    plot_strata(estimates, output_dir / "stratum_descriptive_intervals.png")
    plot_graph_scatter(graphs, output_dir / "graph_utility_worst_robustness.png")
    (output_dir / "report.md").write_text(
        render_report(status, audits, estimates), encoding="utf-8"
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(run_root),
        "source_status_sha256": sha256_file(status_path),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "task_count": status["task_count"],
        "strata": [data.key for data in strata],
        "selected_graph_count": int(sum(data.selected_graphs for data in strata)),
        "output_files": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
