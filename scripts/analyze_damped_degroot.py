"""Fit one global damped-DeGroot susceptibility with graph-held-out validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_classical_dynamics import (
    audit_inputs,
    bootstrap_metrics,
    incoming_neighbors,
    load_inputs,
    prediction_metrics,
    read_json,
    stable_state,
)

plt.switch_backend("Agg")

ANALYSIS_VERSION = "damped-degroot-loo-graph-v1"
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_807
ALPHAS = tuple(float(value) for value in np.linspace(0.0, 1.0, 51))
OUTCOMES = ("induced_readout_target", "paired_accuracy_drop")
MODELS = ("frozen_round_zero", "degroot_equal", "damped_degroot_loo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def damped_grid_final_states(
    graph: dict[str, Any],
    initial_states: tuple[str, ...],
    *,
    attack_node: int | None,
    target_state: str,
) -> tuple[str, ...]:
    """Return the final readout state for every alpha in the fixed grid."""
    labels = tuple(sorted(set(initial_states) | {target_state}))
    label_index = {label: index for index, label in enumerate(labels)}
    node_count = int(graph["node_count"])
    base = np.zeros((node_count, len(labels)), dtype=float)
    base_discrete = np.zeros(node_count, dtype=int)
    for node_id, state in enumerate(initial_states):
        base[node_id, label_index[state]] = 1.0
        base_discrete[node_id] = label_index[state]

    beliefs = np.repeat(base[np.newaxis, :, :], len(ALPHAS), axis=0)
    discrete = np.repeat(base_discrete[np.newaxis, :], len(ALPHAS), axis=0)
    target_index = label_index[target_state]
    if attack_node is not None:
        beliefs[:, attack_node, :] = 0.0
        beliefs[:, attack_node, target_index] = 1.0
        discrete[:, attack_node] = target_index

    incoming = incoming_neighbors(graph)
    weights = np.zeros((node_count, node_count), dtype=float)
    for node_id, sources in enumerate(incoming):
        participants = (node_id, *sources)
        weight = 1.0 / len(participants)
        for source in participants:
            weights[node_id, source] = weight

    alpha = np.asarray(ALPHAS, dtype=float)[:, np.newaxis, np.newaxis]
    for _ in range(int(graph["max_rounds"])):
        neighbor_average = np.einsum("ij,ajk->aik", weights, beliefs)
        beliefs = (1.0 - alpha) * beliefs + alpha * neighbor_average
        if attack_node is not None:
            beliefs[:, attack_node, :] = 0.0
            beliefs[:, attack_node, target_index] = 1.0
        maximum = beliefs.max(axis=2, keepdims=True)
        leader_mask = np.isclose(beliefs, maximum)
        unique = leader_mask.sum(axis=2) == 1
        leaders = beliefs.argmax(axis=2)
        discrete = np.where(unique, leaders, discrete)

    readout = int(graph["readout_node"])
    return tuple(labels[int(index)] for index in discrete[:, readout])


def build_candidate_tables(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    for record in records:
        attack_lookup = record["attacks"].set_index(["task_id", "graph_id", "attack_node"])
        clean_lookup = record["clean"].set_index(["task_id", "graph_id"])
        graph_lookup = {str(graph["graph_id"]): graph for graph in record["graphs"]}
        initial_by_graph: dict[str, list[dict[str, Any]]] = {}
        for item in record["initial"]:
            initial_by_graph.setdefault(str(item["graph_id"]), []).append(item)

        for graph_id, graph in graph_lookup.items():
            node_count = int(graph["node_count"])
            readout = int(graph["readout_node"])
            attack_nodes = [node for node in range(node_count) if node != readout]
            clean_correct_sum = np.zeros(len(ALPHAS), dtype=float)
            target_sum = np.zeros((len(ALPHAS), len(attack_nodes)), dtype=float)
            drop_sum = np.zeros((len(ALPHAS), len(attack_nodes)), dtype=float)
            observed_target_sum = np.zeros(len(attack_nodes), dtype=float)
            observed_drop_sum = np.zeros(len(attack_nodes), dtype=float)
            observed_clean_correct = 0.0
            task_items = initial_by_graph[graph_id]

            for item in task_items:
                task_id = str(item["task_id"])
                reference = str(item["reference_answer"]).strip()
                target = str(item["target_answer"]).strip()
                initial = tuple(
                    stable_state(value, node_id)
                    for node_id, value in enumerate(item["node_parsed_answers"])
                )
                clean_states = damped_grid_final_states(
                    graph, initial, attack_node=None, target_state=target
                )
                clean_correct = np.asarray(
                    [state == reference for state in clean_states], dtype=float
                )
                clean_matches_target = np.asarray(
                    [state == target for state in clean_states], dtype=bool
                )
                clean_correct_sum += clean_correct
                observed_clean = clean_lookup.loc[(task_id, graph_id)]
                observed_clean_correct += float(bool(observed_clean["final_correct"]))

                for node_position, attack_node in enumerate(attack_nodes):
                    observed = attack_lookup.loc[(task_id, graph_id, attack_node)]
                    if str(observed["target_answer"]).strip() != target:
                        raise ValueError(
                            f"target mismatch: {task_id}/{graph_id}/{attack_node}"
                        )
                    attack_states = damped_grid_final_states(
                        graph,
                        initial,
                        attack_node=attack_node,
                        target_state=target,
                    )
                    attack_correct = np.asarray(
                        [state == reference for state in attack_states], dtype=float
                    )
                    attack_matches_target = np.asarray(
                        [state == target for state in attack_states], dtype=bool
                    )
                    target_sum[:, node_position] += (
                        attack_matches_target & ~clean_matches_target
                    ).astype(float)
                    drop_sum[:, node_position] += clean_correct - attack_correct
                    observed_target_sum[node_position] += float(
                        bool(observed["induced_readout_target"])
                    )
                    observed_drop_sum[node_position] += float(
                        observed["paired_accuracy_drop"]
                    )

            tasks = len(task_items)
            for alpha_index, alpha in enumerate(ALPHAS):
                clean_rows.append(
                    {
                        "alpha": alpha,
                        "stratum": record["stratum"],
                        "graph_id": graph_id,
                        "llm_clean_utility": observed_clean_correct / tasks,
                        "classical_clean_utility": clean_correct_sum[alpha_index] / tasks,
                    }
                )
                for node_position, attack_node in enumerate(attack_nodes):
                    node_rows.append(
                        {
                            "alpha": alpha,
                            "stratum": record["stratum"],
                            "graph_id": graph_id,
                            "attack_node": attack_node,
                            "llm_induced_readout_target": (
                                observed_target_sum[node_position] / tasks
                            ),
                            "classical_induced_readout_target": (
                                target_sum[alpha_index, node_position] / tasks
                            ),
                            "llm_paired_accuracy_drop": observed_drop_sum[node_position]
                            / tasks,
                            "classical_paired_accuracy_drop": (
                                drop_sum[alpha_index, node_position] / tasks
                            ),
                        }
                    )
    return pd.DataFrame(node_rows), pd.DataFrame(clean_rows)


def alpha_grid_summary(nodes: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for alpha, frame in nodes.groupby("alpha", sort=True):
        graph_target_mae = frame.assign(
            error=(
                frame["llm_induced_readout_target"]
                - frame["classical_induced_readout_target"]
            ).abs()
        ).groupby("graph_id")["error"].mean()
        graph_drop_mae = frame.assign(
            error=(
                frame["llm_paired_accuracy_drop"]
                - frame["classical_paired_accuracy_drop"]
            ).abs()
        ).groupby("graph_id")["error"].mean()
        clean_frame = clean.loc[np.isclose(clean["alpha"], alpha)]
        clean_mae = (
            clean_frame["llm_clean_utility"] - clean_frame["classical_clean_utility"]
        ).abs()
        rows.append(
            {
                "alpha": float(alpha),
                "graph_equal_target_mae": float(graph_target_mae.mean()),
                "graph_equal_drop_mae": float(graph_drop_mae.mean()),
                "graph_equal_clean_utility_mae": float(clean_mae.mean()),
            }
        )
    return pd.DataFrame(rows)


def select_alpha_by_outer_graph(nodes: pd.DataFrame) -> pd.DataFrame:
    graph_ids = list(nodes["graph_id"].unique())
    rows: list[dict[str, Any]] = []
    for held_graph in graph_ids:
        train = nodes.loc[nodes["graph_id"] != held_graph].copy()
        train["absolute_error"] = (
            train["llm_induced_readout_target"]
            - train["classical_induced_readout_target"]
        ).abs()
        loss = (
            train.groupby(["alpha", "graph_id"], sort=True)["absolute_error"]
            .mean()
            .groupby("alpha")
            .mean()
        )
        minimum = float(loss.min())
        selected = min(float(alpha) for alpha in loss.index if np.isclose(loss[alpha], minimum))
        held = nodes.loc[
            (nodes["graph_id"] == held_graph) & np.isclose(nodes["alpha"], selected)
        ]
        held_target_mae = float(
            (
                held["llm_induced_readout_target"]
                - held["classical_induced_readout_target"]
            )
            .abs()
            .mean()
        )
        rows.append(
            {
                "held_out_graph": held_graph,
                "stratum": str(held["stratum"].iloc[0]),
                "selected_alpha": selected,
                "training_graph_equal_target_mae": minimum,
                "held_out_target_mae": held_target_mae,
            }
        )
    return pd.DataFrame(rows)


def held_out_predictions(nodes: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, fold in folds.iterrows():
        graph_id = str(fold["held_out_graph"])
        selected_alpha = float(fold["selected_alpha"])
        graph_rows = nodes.loc[nodes["graph_id"] == graph_id]
        configurations = (
            ("frozen_round_zero", 0.0),
            ("degroot_equal", 1.0),
            ("damped_degroot_loo", selected_alpha),
        )
        for model, alpha in configurations:
            selected = graph_rows.loc[np.isclose(graph_rows["alpha"], alpha)]
            for outcome in OUTCOMES:
                frame = selected[
                    [
                        "stratum",
                        "graph_id",
                        "attack_node",
                        f"llm_{outcome}",
                        f"classical_{outcome}",
                    ]
                ].rename(
                    columns={
                        f"llm_{outcome}": "observed",
                        f"classical_{outcome}": "prediction",
                    }
                )
                frame["model"] = model
                frame["outcome"] = outcome
                frame["selected_alpha"] = alpha
                rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def held_out_clean_predictions(clean: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, fold in folds.iterrows():
        graph_id = str(fold["held_out_graph"])
        selected_alpha = float(fold["selected_alpha"])
        graph_rows = clean.loc[clean["graph_id"] == graph_id]
        for model, alpha in (
            ("frozen_round_zero", 0.0),
            ("degroot_equal", 1.0),
            ("damped_degroot_loo", selected_alpha),
        ):
            row = graph_rows.loc[np.isclose(graph_rows["alpha"], alpha)].iloc[0]
            rows.append(
                {
                    "model": model,
                    "stratum": row["stratum"],
                    "graph_id": graph_id,
                    "selected_alpha": alpha,
                    "observed_clean_utility": row["llm_clean_utility"],
                    "predicted_clean_utility": row["classical_clean_utility"],
                    "absolute_error": abs(
                        row["llm_clean_utility"] - row["classical_clean_utility"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_predictions(
    predictions: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (outcome, model), frame in predictions.groupby(["outcome", "model"], sort=False):
        estimates = prediction_metrics(frame)
        intervals = bootstrap_metrics(frame, replicates, rng)
        alpha_values = frame["selected_alpha"].dropna().to_numpy(dtype=float)
        alpha_mode = Counter(alpha_values).most_common(1)[0][0]
        for metric, estimate in estimates.items():
            low, high = intervals.get(metric, (np.nan, np.nan))
            rows.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "selected_alpha_mode": alpha_mode,
                    "outer_unit": "held_out_graph",
                    "bootstrap_unit": "held_out_graph",
                }
            )
    return pd.DataFrame(rows)


def compare_damped(
    predictions: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        outcome_frame = predictions.loc[predictions["outcome"] == outcome]
        candidate = outcome_frame.loc[
            outcome_frame["model"] == "damped_degroot_loo",
            ["graph_id", "attack_node", "observed", "prediction"],
        ].rename(columns={"prediction": "candidate_prediction"})
        for reference_name in ("frozen_round_zero", "degroot_equal"):
            reference = outcome_frame.loc[
                outcome_frame["model"] == reference_name,
                ["graph_id", "attack_node", "prediction"],
            ].rename(columns={"prediction": "reference_prediction"})
            paired = candidate.merge(
                reference, on=["graph_id", "attack_node"], validate="one_to_one"
            )
            paired["candidate_error"] = (
                paired["observed"] - paired["candidate_prediction"]
            ).abs()
            paired["reference_error"] = (
                paired["observed"] - paired["reference_prediction"]
            ).abs()
            graph = paired.groupby("graph_id", sort=False).agg(
                candidate_mae=("candidate_error", "mean"),
                reference_mae=("reference_error", "mean"),
            )
            values = (graph["reference_mae"] - graph["candidate_mae"]).to_numpy()
            draws = np.mean(
                values[
                    rng.integers(0, len(values), size=(replicates, len(values)))
                ],
                axis=1,
            )
            rows.append(
                {
                    "outcome": outcome,
                    "candidate": "damped_degroot_loo",
                    "reference": reference_name,
                    "graph_equal_mae_improvement": float(values.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "fraction_graphs_candidate_better": float(np.mean(values > 0)),
                    "positive_means_candidate_better": True,
                }
            )
    return pd.DataFrame(rows)


def clean_summary(clean_predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        clean_predictions.groupby("model", sort=False)
        .agg(
            graph_equal_mae=("absolute_error", "mean"),
            observed_clean_utility=("observed_clean_utility", "mean"),
            predicted_clean_utility=("predicted_clean_utility", "mean"),
        )
        .reset_index()
    )


def plot_alpha_curve(grid: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.plot(grid["alpha"], grid["graph_equal_target_mae"], label="target induction")
    axis.plot(grid["alpha"], grid["graph_equal_drop_mae"], label="accuracy drop")
    axis.plot(
        grid["alpha"], grid["graph_equal_clean_utility_mae"], label="clean utility"
    )
    axis.set_xlabel("alpha")
    axis.set_ylabel("Graph-equal MAE (diagnostic full set)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_observed_predicted(predictions: pd.DataFrame, outcome: str, path: Path) -> None:
    frame = predictions.loc[
        (predictions["outcome"] == outcome)
        & (predictions["model"] == "damped_degroot_loo")
    ]
    figure, axis = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
    for stratum, group in frame.groupby("stratum", sort=False):
        axis.scatter(group["observed"], group["prediction"], alpha=0.7, label=stratum)
    low = float(min(frame["observed"].min(), frame["prediction"].min()))
    high = float(max(frame["observed"].max(), frame["prediction"].max()))
    axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("LLM observed")
    axis.set_ylabel("Graph-held-out damped DeGroot")
    axis.set_title(outcome)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(
    audit: dict[str, Any],
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    clean: pd.DataFrame,
) -> str:
    selected = folds["selected_alpha"]
    lines = [
        "# Damped DeGroot held-out-graph calibration",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`",
        "",
        "## Integrity and selection",
        "",
        f"- Input audit passed: `{audit['passed']}`",
        f"- Graphs: {audit['graphs']}",
        f"- Tasks: {audit['task_count']}",
        f"- Alpha grid: {len(ALPHAS)} values from {min(ALPHAS):.2f} to {max(ALPHAS):.2f}",
        f"- Selected alpha mode: {selected.mode().iloc[0]:.2f}",
        f"- Selected alpha median: {selected.median():.2f}",
        f"- Selected alpha range: [{selected.min():.2f}, {selected.max():.2f}]",
        "",
        "## Held-out node-level prediction",
        "",
        (
            "| outcome | model | MAE | R2 | Spearman | within-graph Spearman | "
            "rank graphs | top-1 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    pivot = summary.pivot(index=["outcome", "model"], columns="metric", values="estimate")
    for outcome in OUTCOMES:
        for model in MODELS:
            row = pivot.loc[(outcome, model)]
            rank_frame = predictions.loc[
                (predictions["outcome"] == outcome)
                & (predictions["model"] == model)
            ]
            valid_rank_graphs = sum(
                group["observed"].nunique() > 1
                and group["prediction"].nunique() > 1
                for _, group in rank_frame.groupby("graph_id", sort=False)
            )
            lines.append(
                f"| {outcome} | {model} | {row['mae']:.4f} | {row['r2']:.3f} | "
                f"{row['spearman']:.3f} | {row['mean_within_graph_spearman']:.3f} | "
                f"{valid_rank_graphs}/{audit['graphs']} | "
                f"{row['top1_vulnerable_accuracy']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Paired graph-equal MAE improvement",
            "",
            "Positive values favor held-out damped DeGroot.",
            "",
            "| outcome | reference | improvement | 95% graph-bootstrap CI | graphs better |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['outcome']} | {row['reference']} | "
            f"{row['graph_equal_mae_improvement']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['fraction_graphs_candidate_better']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Held-out prediction activation",
            "",
            "| outcome | observed mean | damped mean | rows changed from frozen |",
            "|---|---:|---:|---:|",
        ]
    )
    for outcome in OUTCOMES:
        frame = predictions.loc[predictions["outcome"] == outcome]
        table = frame.pivot(
            index=["graph_id", "attack_node"], columns="model", values="prediction"
        )
        observed = frame.drop_duplicates(["graph_id", "attack_node"])["observed"]
        changed = (
            table["damped_degroot_loo"] != table["frozen_round_zero"]
        ).mean()
        lines.append(
            f"| {outcome} | {observed.mean():.4f} | "
            f"{table['damped_degroot_loo'].mean():.4f} | {changed:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Clean utility without retuning",
            "",
            "| model | observed | predicted | graph-equal MAE |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in clean.iterrows():
        lines.append(
            f"| {row['model']} | {row['observed_clean_utility']:.3f} | "
            f"{row['predicted_clean_utility']:.3f} | {row['graph_equal_mae']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim guardrails",
            "",
            "- Alpha is selected without any node from the held-out graph.",
            (
                "- Only target induction is optimized; accuracy drop and clean utility "
                "are secondary tests."
            ),
            (
                "- Better calibration would support a global susceptibility explanation, "
                "not DeGroot equivalence."
            ),
            (
                "- Remaining residuals cannot be attributed to language semantics "
                "without intervention."
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
        raise RuntimeError("pilot must be completed before damped DeGroot analysis")
    records, _ = load_inputs(run_root, status)
    audit = audit_inputs(records, int(status["task_count"]))
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if not audit["passed"]:
        raise RuntimeError("input audit failed; see integrity_audit.json")

    candidates, clean_candidates = build_candidate_tables(records)
    grid = alpha_grid_summary(candidates, clean_candidates)
    folds = select_alpha_by_outer_graph(candidates)
    predictions = held_out_predictions(candidates, folds)
    clean_predictions = held_out_clean_predictions(clean_candidates, folds)
    rng = np.random.default_rng(args.seed)
    summary = summarize_predictions(predictions, args.bootstrap_replicates, rng)
    comparisons = compare_damped(predictions, args.bootstrap_replicates, rng)
    clean = clean_summary(clean_predictions)

    candidates.to_csv(output_dir / "alpha_node_candidates.csv", index=False)
    clean_candidates.to_csv(output_dir / "alpha_clean_candidates.csv", index=False)
    grid.to_csv(output_dir / "alpha_grid_summary.csv", index=False)
    folds.to_csv(output_dir / "outer_fold_selected_alpha.csv", index=False)
    predictions.to_csv(output_dir / "held_out_graph_predictions.csv", index=False)
    summary.to_csv(output_dir / "predictive_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "model_comparisons.csv", index=False)
    clean_predictions.to_csv(output_dir / "held_out_clean_predictions.csv", index=False)
    clean.to_csv(output_dir / "clean_summary.csv", index=False)
    plot_alpha_curve(grid, output_dir / "alpha_diagnostic_curve.png")
    for outcome in OUTCOMES:
        plot_observed_predicted(
            predictions, outcome, output_dir / f"observed_predicted_{outcome}.png"
        )
    (output_dir / "report.md").write_text(
        render_report(audit, folds, predictions, summary, comparisons, clean),
        encoding="utf-8",
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(run_root),
        "source_status_sha256": sha256_file(status_path),
        "alpha_grid": list(ALPHAS),
        "selection_outcome": "induced_readout_target",
        "outer_unit": "graph",
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "graphs": audit["graphs"],
        "tasks": audit["task_count"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
