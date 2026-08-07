"""Compare frozen LLM pilot outcomes with parameter-free classical graph dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, r2_score

plt.switch_backend("Agg")

ANALYSIS_VERSION = "classical-dynamics-v1"
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_807
BASELINES = ("frozen_round_zero", "inertial_majority", "degroot_equal")
OUTCOMES = ("paired_accuracy_drop", "induced_readout_target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def stable_state(value: Any, node_id: int) -> str:
    if value is None or not str(value).strip():
        return f"__unparsed_node_{node_id}__"
    return str(value).strip()


def incoming_neighbors(graph: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    incoming: list[list[int]] = [[] for _ in range(int(graph["node_count"]))]
    for edge in graph["edges"]:
        incoming[int(edge["target"])].append(int(edge["source"]))
    return tuple(tuple(sorted(values)) for values in incoming)


def frozen_round_zero(
    graph: dict[str, Any],
    initial_states: tuple[str, ...],
    *,
    attack_node: int | None,
    target_state: str,
) -> str:
    del attack_node, target_state
    return initial_states[int(graph["readout_node"])]


def inertial_majority(
    graph: dict[str, Any],
    initial_states: tuple[str, ...],
    *,
    attack_node: int | None,
    target_state: str,
) -> str:
    states = list(initial_states)
    if attack_node is not None:
        states[attack_node] = target_state
    incoming = incoming_neighbors(graph)
    for _ in range(int(graph["max_rounds"])):
        updated = list(states)
        for node_id in range(int(graph["node_count"])):
            if node_id == attack_node:
                updated[node_id] = target_state
                continue
            votes = [states[node_id], *(states[source] for source in incoming[node_id])]
            counts = Counter(votes)
            maximum = max(counts.values())
            leaders = [state for state, count in counts.items() if count == maximum]
            updated[node_id] = leaders[0] if len(leaders) == 1 else states[node_id]
        states = updated
    return states[int(graph["readout_node"])]


def degroot_equal(
    graph: dict[str, Any],
    initial_states: tuple[str, ...],
    *,
    attack_node: int | None,
    target_state: str,
) -> str:
    labels = tuple(sorted(set(initial_states) | {target_state}))
    label_index = {label: index for index, label in enumerate(labels)}
    node_count = int(graph["node_count"])
    beliefs = np.zeros((node_count, len(labels)), dtype=float)
    discrete = list(initial_states)
    for node_id, state in enumerate(initial_states):
        beliefs[node_id, label_index[state]] = 1.0
    if attack_node is not None:
        beliefs[attack_node, :] = 0.0
        beliefs[attack_node, label_index[target_state]] = 1.0
        discrete[attack_node] = target_state

    incoming = incoming_neighbors(graph)
    weights = np.zeros((node_count, node_count), dtype=float)
    for node_id, sources in enumerate(incoming):
        participants = (node_id, *sources)
        weight = 1.0 / len(participants)
        for source in participants:
            weights[node_id, source] = weight

    for _ in range(int(graph["max_rounds"])):
        beliefs = weights @ beliefs
        if attack_node is not None:
            beliefs[attack_node, :] = 0.0
            beliefs[attack_node, label_index[target_state]] = 1.0
        updated: list[str] = []
        for node_id in range(node_count):
            row = beliefs[node_id]
            leaders = np.flatnonzero(np.isclose(row, row.max()))
            if len(leaders) == 1:
                updated.append(labels[int(leaders[0])])
            else:
                updated.append(discrete[node_id])
        discrete = updated
    return discrete[int(graph["readout_node"])]


SIMULATORS = {
    "frozen_round_zero": frozen_round_zero,
    "inertial_majority": inertial_majority,
    "degroot_equal": degroot_equal,
}


def load_inputs(
    run_root: Path, status: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    graph_specs: dict[str, Any] = {}
    for descriptor in status["strata"]:
        key = descriptor["key"]
        stratum_root = run_root / "strata" / key
        analysis_root = stratum_root / "analysis-v1"
        graphs = read_jsonl(stratum_root / "selected_graphs.jsonl")
        attacks = pd.DataFrame(read_jsonl(analysis_root / "paired_attacks.jsonl"))
        clean = pd.DataFrame(read_jsonl(analysis_root / "run_metrics.jsonl"))
        clean = clean.loc[clean["condition"] == "clean"].copy()
        initial = read_jsonl(analysis_root / "classical_initial_states.jsonl")
        for graph in graphs:
            graph_id = str(graph["graph_id"])
            if graph_id in graph_specs:
                raise ValueError(f"duplicate graph id: {graph_id}")
            graph_specs[graph_id] = graph
        records.append(
            {
                "stratum": key,
                "n": int(descriptor["n"]),
                "m": int(descriptor["m"]),
                "selected_graphs": int(descriptor["selected_graphs"]),
                "graphs": graphs,
                "attacks": attacks,
                "clean": clean,
                "initial": initial,
            }
        )
    return records, graph_specs


def audit_inputs(records: list[dict[str, Any]], task_count: int) -> dict[str, Any]:
    errors: list[str] = []
    assignment_rows: list[dict[str, Any]] = []
    observed_initial = 0
    observed_attacks = 0
    for record in records:
        graph_ids = {str(graph["graph_id"]) for graph in record["graphs"]}
        initial = record["initial"]
        attacks = record["attacks"]
        clean = record["clean"]
        expected_initial = task_count * record["selected_graphs"]
        expected_attacks = expected_initial * (record["n"] - 1)
        observed_initial += len(initial)
        observed_attacks += len(attacks)
        if len(initial) != expected_initial:
            errors.append(f"{record['stratum']}: initial rows {len(initial)} != {expected_initial}")
        if len(attacks) != expected_attacks:
            errors.append(f"{record['stratum']}: attack rows {len(attacks)} != {expected_attacks}")
        initial_keys = [(item["task_id"], item["graph_id"]) for item in initial]
        if len(set(initial_keys)) != len(initial_keys):
            errors.append(f"{record['stratum']}: duplicate initial-state keys")
        if set(attacks["graph_id"].astype(str).unique()) != graph_ids:
            errors.append(f"{record['stratum']}: attack graph coverage mismatch")
        if set(clean["graph_id"].astype(str).unique()) != graph_ids:
            errors.append(f"{record['stratum']}: clean graph coverage mismatch")
        expected_nodes = set(range(record["n"] - 1))
        for graph_id, frame in attacks.groupby("graph_id", sort=False):
            if set(frame["attack_node"].astype(int).unique()) != expected_nodes:
                errors.append(f"{graph_id}: incomplete non-readout attack coverage")
        for item in initial:
            if len(item["node_parsed_answers"]) != record["n"]:
                errors.append(f"{item['graph_id']}/{item['task_id']}: wrong initial-state length")
            assignment_rows.append(
                {
                    "n": record["n"],
                    "task_id": item["task_id"],
                    "assignment": json.dumps(item["structural_node_to_replica"]),
                    "answers": json.dumps(item["node_parsed_answers"]),
                }
            )

    assignment = pd.DataFrame(assignment_rows)
    if not assignment.empty:
        variation = assignment.groupby(["n", "task_id"]).agg(
            assignment_versions=("assignment", "nunique"),
            answer_versions=("answers", "nunique"),
        )
        if bool((variation["assignment_versions"] != 1).any()):
            errors.append("Round-zero structural assignments vary across graphs within n/task")
        if bool((variation["answer_versions"] != 1).any()):
            errors.append("Round-zero answers vary across graphs within n/task")
    return {
        "passed": not errors,
        "errors": errors,
        "strata": len(records),
        "graphs": sum(record["selected_graphs"] for record in records),
        "initial_state_rows": observed_initial,
        "paired_attack_rows": observed_attacks,
        "task_count": task_count,
        "round_zero_graph_independence_passed": not any(
            "Round-zero" in error for error in errors
        ),
    }


def simulate_records(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    for record in records:
        attack_lookup = record["attacks"].set_index(["task_id", "graph_id", "attack_node"])
        clean_lookup = record["clean"].set_index(["task_id", "graph_id"])
        graph_lookup = {str(graph["graph_id"]): graph for graph in record["graphs"]}
        for item in record["initial"]:
            task_id = str(item["task_id"])
            graph_id = str(item["graph_id"])
            graph = graph_lookup[graph_id]
            reference = str(item["reference_answer"]).strip()
            target = str(item["target_answer"]).strip()
            initial = tuple(
                stable_state(value, node_id)
                for node_id, value in enumerate(item["node_parsed_answers"])
            )
            observed_clean = clean_lookup.loc[(task_id, graph_id)]
            for baseline, simulator in SIMULATORS.items():
                clean_final = simulator(
                    graph, initial, attack_node=None, target_state=target
                )
                clean_correct = clean_final == reference
                clean_matches_target = clean_final == target
                clean_rows.append(
                    {
                        "baseline": baseline,
                        "stratum": record["stratum"],
                        "task_id": task_id,
                        "graph_id": graph_id,
                        "classical_final_state": clean_final,
                        "classical_clean_correct": clean_correct,
                        "classical_clean_matches_target": clean_matches_target,
                        "llm_clean_correct": bool(observed_clean["final_correct"]),
                        "llm_round_zero_readout_correct": bool(
                            observed_clean["readout_round_zero_correct"]
                        ),
                    }
                )
                for attack_node in range(int(graph["node_count"]) - 1):
                    observed = attack_lookup.loc[(task_id, graph_id, attack_node)]
                    if str(observed["target_answer"]).strip() != target:
                        raise ValueError(f"target mismatch: {task_id}/{graph_id}/{attack_node}")
                    attack_final = simulator(
                        graph,
                        initial,
                        attack_node=attack_node,
                        target_state=target,
                    )
                    attack_correct = attack_final == reference
                    attack_matches_target = attack_final == target
                    condition_rows.append(
                        {
                            "baseline": baseline,
                            "stratum": record["stratum"],
                            "task_id": task_id,
                            "graph_id": graph_id,
                            "attack_node": attack_node,
                            "reference_answer": reference,
                            "target_answer": target,
                            "classical_clean_state": clean_final,
                            "classical_attack_state": attack_final,
                            "classical_clean_correct": clean_correct,
                            "classical_attack_correct": attack_correct,
                            "classical_paired_accuracy_drop": int(clean_correct)
                            - int(attack_correct),
                            "classical_induced_readout_target": attack_matches_target
                            and not clean_matches_target,
                            "llm_clean_correct": bool(observed["clean_correct"]),
                            "llm_attack_correct": bool(observed["attack_correct"]),
                            "llm_paired_accuracy_drop": int(observed["paired_accuracy_drop"]),
                            "llm_induced_readout_target": bool(
                                observed["induced_readout_target"]
                            ),
                        }
                    )
    return pd.DataFrame(condition_rows), pd.DataFrame(clean_rows)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["prediction"].to_numpy(dtype=float)
    within: list[float] = []
    top1: list[float] = []
    for _, group in frame.groupby("graph_id", sort=False):
        correlation = safe_spearman(
            group["observed"].to_numpy(dtype=float),
            group["prediction"].to_numpy(dtype=float),
        )
        if math.isfinite(correlation):
            within.append(correlation)
        observed_top = set(
            group.loc[np.isclose(group["observed"], group["observed"].max()), "attack_node"]
        )
        predicted_top = set(
            group.loc[
                np.isclose(group["prediction"], group["prediction"].max()), "attack_node"
            ]
        )
        top1.append(len(observed_top & predicted_top) / len(predicted_top))
    residual = observed - predicted
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.mean(residual**2) ** 0.5),
        "r2": float(r2_score(observed, predicted)),
        "spearman": safe_spearman(observed, predicted),
        "mean_within_graph_spearman": float(np.mean(within)) if within else float("nan"),
        "top1_vulnerable_accuracy": float(np.mean(top1)),
    }


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    if float(weights.sum()) <= 0:
        return float("nan")
    mean_x = float(np.average(x, weights=weights))
    mean_y = float(np.average(y, weights=weights))
    centered_x = x - mean_x
    centered_y = y - mean_y
    covariance = float(np.average(centered_x * centered_y, weights=weights))
    variance_x = float(np.average(centered_x**2, weights=weights))
    variance_y = float(np.average(centered_y**2, weights=weights))
    if variance_x <= 0 or variance_y <= 0:
        return float("nan")
    return covariance / math.sqrt(variance_x * variance_y)


def node_predictions(condition_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = condition_rows.groupby(
        ["baseline", "stratum", "graph_id", "attack_node"], sort=False
    ).agg(
        llm_paired_accuracy_drop=("llm_paired_accuracy_drop", "mean"),
        classical_paired_accuracy_drop=("classical_paired_accuracy_drop", "mean"),
        llm_induced_readout_target=("llm_induced_readout_target", "mean"),
        classical_induced_readout_target=("classical_induced_readout_target", "mean"),
    )
    grouped = grouped.reset_index()
    rows: list[pd.DataFrame] = []
    for outcome in OUTCOMES:
        renamed = grouped[
            [
                "baseline",
                "stratum",
                "graph_id",
                "attack_node",
                f"llm_{outcome}",
                f"classical_{outcome}",
            ]
        ].rename(
            columns={f"llm_{outcome}": "observed", f"classical_{outcome}": "prediction"}
        )
        renamed["outcome"] = outcome
        rows.append(renamed)
    return pd.concat(rows, ignore_index=True)


def bootstrap_metrics(
    frame: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    graph_ids = [str(value) for value in frame["graph_id"].unique()]
    graph_index = {graph_id: index for index, graph_id in enumerate(graph_ids)}
    row_graph_index = frame["graph_id"].map(graph_index).to_numpy(dtype=int)
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["prediction"].to_numpy(dtype=float)
    observed_rank = pd.Series(observed).rank(method="average").to_numpy(dtype=float)
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy(dtype=float)
    per_graph_within: list[float] = []
    per_graph_top1: list[float] = []
    for graph_id in graph_ids:
        metrics = prediction_metrics(frame.loc[frame["graph_id"] == graph_id])
        per_graph_within.append(metrics["mean_within_graph_spearman"])
        per_graph_top1.append(metrics["top1_vulnerable_accuracy"])
    values: dict[str, list[float]] = {}
    for _ in range(replicates):
        sampled = rng.integers(0, len(graph_ids), size=len(graph_ids))
        graph_weights = np.bincount(sampled, minlength=len(graph_ids)).astype(float)
        row_weights = graph_weights[row_graph_index]
        active = row_weights > 0
        active_weights = row_weights[active]
        active_observed = observed[active]
        active_predicted = predicted[active]
        residual = active_observed - active_predicted
        observed_mean = float(np.average(active_observed, weights=active_weights))
        residual_sum = float(np.sum(active_weights * residual**2))
        total_sum = float(
            np.sum(active_weights * (active_observed - observed_mean) ** 2)
        )
        finite_within = np.isfinite(per_graph_within)
        within_weights = graph_weights[finite_within]
        metrics = {
            "mae": float(np.average(np.abs(residual), weights=active_weights)),
            "rmse": float(np.average(residual**2, weights=active_weights) ** 0.5),
            "r2": 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan"),
            "spearman": weighted_correlation(
                observed_rank[active], predicted_rank[active], active_weights
            ),
            "mean_within_graph_spearman": (
                float(
                    np.average(
                        np.asarray(per_graph_within)[finite_within],
                        weights=within_weights,
                    )
                )
                if within_weights.sum() > 0
                else float("nan")
            ),
            "top1_vulnerable_accuracy": float(
                np.average(per_graph_top1, weights=graph_weights)
            ),
        }
        for metric, value in metrics.items():
            if math.isfinite(value):
                values.setdefault(metric, []).append(value)
    return {
        metric: (float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975)))
        for metric, items in values.items()
    }


def summarize_node_predictions(
    predictions: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (outcome, baseline), frame in predictions.groupby(
        ["outcome", "baseline"], sort=False
    ):
        estimates = prediction_metrics(frame)
        intervals = bootstrap_metrics(frame, replicates, rng)
        for metric, estimate in estimates.items():
            low, high = intervals.get(metric, (np.nan, np.nan))
            rows.append(
                {
                    "outcome": outcome,
                    "baseline": baseline,
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_unit": "graph",
                }
            )
    return pd.DataFrame(rows)


def compare_against_frozen(
    predictions: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        outcome_frame = predictions.loc[predictions["outcome"] == outcome]
        reference = outcome_frame.loc[
            outcome_frame["baseline"] == "frozen_round_zero",
            ["graph_id", "attack_node", "observed", "prediction"],
        ].rename(columns={"prediction": "reference_prediction"})
        for candidate_name in ("inertial_majority", "degroot_equal"):
            candidate = outcome_frame.loc[
                outcome_frame["baseline"] == candidate_name,
                ["graph_id", "attack_node", "prediction"],
            ].rename(columns={"prediction": "candidate_prediction"})
            paired = reference.merge(
                candidate, on=["graph_id", "attack_node"], validate="one_to_one"
            )
            paired["reference_absolute_error"] = (
                paired["observed"] - paired["reference_prediction"]
            ).abs()
            paired["candidate_absolute_error"] = (
                paired["observed"] - paired["candidate_prediction"]
            ).abs()
            graph_errors = paired.groupby("graph_id", sort=False).agg(
                reference_mae=("reference_absolute_error", "mean"),
                candidate_mae=("candidate_absolute_error", "mean"),
            )
            graph_errors["improvement"] = (
                graph_errors["reference_mae"] - graph_errors["candidate_mae"]
            )
            values = graph_errors["improvement"].to_numpy(dtype=float)
            draws = np.mean(
                values[
                    rng.integers(0, len(values), size=(replicates, len(values)))
                ],
                axis=1,
            )
            rows.append(
                {
                    "outcome": outcome,
                    "candidate": candidate_name,
                    "reference": "frozen_round_zero",
                    "graph_equal_mae_improvement": float(values.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "fraction_graphs_candidate_better": float(np.mean(values > 0)),
                    "positive_means_candidate_better": True,
                }
            )
    return pd.DataFrame(rows)


def task_diagnostics(condition_rows: pd.DataFrame, clean_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline in BASELINES:
        clean = clean_rows.loc[clean_rows["baseline"] == baseline]
        attack = condition_rows.loc[condition_rows["baseline"] == baseline]
        pairs = (
            ("clean_correct", clean["llm_clean_correct"], clean["classical_clean_correct"]),
            ("attack_correct", attack["llm_attack_correct"], attack["classical_attack_correct"]),
            (
                "induced_readout_target",
                attack["llm_induced_readout_target"],
                attack["classical_induced_readout_target"],
            ),
        )
        for outcome, observed_series, predicted_series in pairs:
            observed = observed_series.astype(int).to_numpy()
            predicted = predicted_series.astype(int).to_numpy()
            rows.append(
                {
                    "baseline": baseline,
                    "outcome": outcome,
                    "rows": len(observed),
                    "llm_positive_rate": float(observed.mean()),
                    "classical_positive_rate": float(predicted.mean()),
                    "exact_agreement": float(np.mean(observed == predicted)),
                    "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
                    "matthews_correlation": float(matthews_corrcoef(observed, predicted)),
                }
            )
        observed_drop = attack["llm_paired_accuracy_drop"].to_numpy(dtype=float)
        predicted_drop = attack["classical_paired_accuracy_drop"].to_numpy(dtype=float)
        rows.append(
            {
                "baseline": baseline,
                "outcome": "paired_accuracy_drop",
                "rows": len(observed_drop),
                "llm_positive_rate": float(np.mean(observed_drop > 0)),
                "classical_positive_rate": float(np.mean(predicted_drop > 0)),
                "exact_agreement": float(np.mean(observed_drop == predicted_drop)),
                "balanced_accuracy": np.nan,
                "matthews_correlation": safe_spearman(observed_drop, predicted_drop),
            }
        )
    return pd.DataFrame(rows)


def clean_summary(clean_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        clean_rows.groupby("baseline", sort=False)
        .agg(
            cells=("task_id", "size"),
            llm_clean_utility=("llm_clean_correct", "mean"),
            classical_clean_utility=("classical_clean_correct", "mean"),
            clean_correct_agreement=(
                "classical_clean_correct",
                lambda values: float(
                    np.mean(
                        values.to_numpy(dtype=bool)
                        == clean_rows.loc[values.index, "llm_clean_correct"].to_numpy(dtype=bool)
                    )
                ),
            ),
        )
        .reset_index()
    )


def plot_predictions(predictions: pd.DataFrame, outcome: str, path: Path) -> None:
    figure, axes = plt.subplots(1, len(BASELINES), figsize=(12, 5.2), constrained_layout=True)
    for axis, baseline in zip(axes, BASELINES, strict=True):
        frame = predictions.loc[
            (predictions["outcome"] == outcome) & (predictions["baseline"] == baseline)
        ]
        for stratum, group in frame.groupby("stratum", sort=False):
            axis.scatter(group["observed"], group["prediction"], alpha=0.65, label=stratum)
        low = float(min(frame["observed"].min(), frame["prediction"].min()))
        high = float(max(frame["observed"].max(), frame["prediction"].max()))
        axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        axis.set_title(baseline)
        axis.set_xlabel("LLM observed")
        axis.set_ylabel("Classical prediction")
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7, ncol=2)
    figure.suptitle(outcome)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(
    audit: dict[str, Any],
    clean: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    lines = [
        "# Classical dynamics comparison",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`",
        "",
        "## Integrity",
        "",
        f"- Passed: `{audit['passed']}`",
        f"- Graphs: {audit['graphs']}",
        f"- Tasks: {audit['task_count']}",
        f"- Paired attack rows: {audit['paired_attack_rows']}",
        f"- Round-zero graph independence: `{audit['round_zero_graph_independence_passed']}`",
        "",
        "## Clean utility from the same Round-zero states",
        "",
        "| baseline | LLM utility | classical utility | correctness agreement |",
        "|---|---:|---:|---:|",
    ]
    for _, row in clean.iterrows():
        lines.append(
            f"| {row['baseline']} | {row['llm_clean_utility']:.3f} | "
            f"{row['classical_clean_utility']:.3f} | {row['clean_correct_agreement']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Node-level aggregate prediction",
            "",
            "| outcome | baseline | MAE | R2 | Spearman | within-graph Spearman | top-1 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    pivot = summary.pivot(index=["outcome", "baseline"], columns="metric", values="estimate")
    for outcome in OUTCOMES:
        for baseline in BASELINES:
            row = pivot.loc[(outcome, baseline)]
            lines.append(
                f"| {outcome} | {baseline} | {row['mae']:.4f} | {row['r2']:.3f} | "
                f"{row['spearman']:.3f} | {row['mean_within_graph_spearman']:.3f} | "
                f"{row['top1_vulnerable_accuracy']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Paired MAE improvement over frozen Round zero",
            "",
            "Positive values favor the dynamic baseline.",
            "",
            "| outcome | candidate | improvement | 95% graph-bootstrap CI | graphs better |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['outcome']} | {row['candidate']} | "
            f"{row['graph_equal_mae_improvement']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['fraction_graphs_candidate_better']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Task-level diagnostics",
            "",
            "| baseline | outcome | rows | LLM positive | classical positive | "
            "agreement | balanced acc. | MCC/rank |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in diagnostics.iterrows():
        lines.append(
            f"| {row['baseline']} | {row['outcome']} | {int(row['rows'])} | "
            f"{row['llm_positive_rate']:.3f} | {row['classical_positive_rate']:.3f} | "
            f"{row['exact_agreement']:.3f} | {row['balanced_accuracy']:.3f} | "
            f"{row['matthews_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim guardrails",
            "",
            "- Both baselines use the exact frozen Round-zero node answers, graphs, attacks, "
            "and horizon.",
            "- Agreement supports compatibility, not equivalence to the LLM's internal "
            "update rule.",
            "- Disagreement does not identify a semantic cause without a matched intervention.",
            "- Graph-bootstrap intervals omit task, seed, model, and graph-population uncertainty.",
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
        raise RuntimeError("pilot must be completed before classical dynamics analysis")
    records, _ = load_inputs(run_root, status)
    audit = audit_inputs(records, int(status["task_count"]))
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if not audit["passed"]:
        raise RuntimeError("input audit failed; see integrity_audit.json")

    conditions, clean = simulate_records(records)
    predictions = node_predictions(conditions)
    rng = np.random.default_rng(args.seed)
    summary = summarize_node_predictions(predictions, args.bootstrap_replicates, rng)
    comparisons = compare_against_frozen(predictions, args.bootstrap_replicates, rng)
    diagnostics = task_diagnostics(conditions, clean)
    clean_metrics = clean_summary(clean)

    conditions.to_csv(output_dir / "paired_condition_predictions.csv", index=False)
    clean.to_csv(output_dir / "clean_predictions.csv", index=False)
    predictions.to_csv(output_dir / "node_predictions.csv", index=False)
    summary.to_csv(output_dir / "node_prediction_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "baseline_comparisons.csv", index=False)
    diagnostics.to_csv(output_dir / "task_diagnostics.csv", index=False)
    clean_metrics.to_csv(output_dir / "clean_summary.csv", index=False)
    for outcome in OUTCOMES:
        plot_predictions(predictions, outcome, output_dir / f"observed_predicted_{outcome}.png")
    (output_dir / "report.md").write_text(
        render_report(audit, clean_metrics, summary, comparisons, diagnostics),
        encoding="utf-8",
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(run_root),
        "source_status_sha256": sha256_file(status_path),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "baselines": list(BASELINES),
        "graphs": audit["graphs"],
        "tasks": audit["task_count"],
        "paired_attack_rows": audit["paired_attack_rows"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
