"""Predict LLM target adoption from Round-zero states and classical exposure."""

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
    incoming_neighbors,
    load_inputs,
    prediction_metrics,
    read_json,
    stable_state,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

plt.switch_backend("Agg")

ANALYSIS_VERSION = "conditional-classical-exposure-v1"
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_807
GRAPH_FOLDS = 5
TASK_FOLDS = 5
EPSILON = 1e-6

STATE_FEATURES = (
    "readout_initial_correct",
    "readout_initial_target",
    "attacker_initial_correct",
    "attacker_initial_target",
    "benign_correct_fraction",
    "benign_target_fraction",
    "benign_distinct_answer_fraction",
    "benign_largest_consensus_fraction",
)
EXPOSURE_FEATURES = ("degroot_target_exposure",)
MODEL_FEATURES = {
    "intercept_only": (),
    "round0_state": STATE_FEATURES,
    "classical_exposure": EXPOSURE_FEATURES,
    "state_plus_exposure": (*STATE_FEATURES, *EXPOSURE_FEATURES),
}


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


def degroot_target_exposure(
    graph: dict[str, Any],
    initial_answers: tuple[str, ...],
    *,
    attack_node: int,
    target_answer: str,
) -> float:
    """Return continuous target mass at readout after equal-weight DeGroot updates."""
    labels = tuple(sorted(set(initial_answers) | {target_answer}))
    label_index = {label: index for index, label in enumerate(labels)}
    node_count = int(graph["node_count"])
    beliefs = np.zeros((node_count, len(labels)), dtype=float)
    for node_id, answer in enumerate(initial_answers):
        beliefs[node_id, label_index[answer]] = 1.0
    target_index = label_index[target_answer]
    beliefs[attack_node, :] = 0.0
    beliefs[attack_node, target_index] = 1.0

    incoming = incoming_neighbors(graph)
    weights = np.zeros((node_count, node_count), dtype=float)
    for node_id, sources in enumerate(incoming):
        participants = (node_id, *sources)
        weights[node_id, list(participants)] = 1.0 / len(participants)

    for _ in range(int(graph["max_rounds"])):
        beliefs = weights @ beliefs
        beliefs[attack_node, :] = 0.0
        beliefs[attack_node, target_index] = 1.0
    value = float(beliefs[int(graph["readout_node"]), target_index])
    if not -EPSILON <= value <= 1.0 + EPSILON:
        raise ValueError(f"invalid exposure: {value}")
    return float(np.clip(value, 0.0, 1.0))


def state_features(
    initial_answers: tuple[str, ...],
    *,
    reference_answer: str,
    target_answer: str,
    attack_node: int,
    readout_node: int,
) -> dict[str, float]:
    """Build fixed non-textual features from the Round-zero answer configuration."""

    def category(answer: str) -> str:
        if answer == reference_answer:
            return "correct"
        if answer == target_answer:
            return "target"
        return "other"

    categories = tuple(category(answer) for answer in initial_answers)
    benign_answers = [
        answer for node_id, answer in enumerate(initial_answers) if node_id != attack_node
    ]
    benign_categories = [
        value for node_id, value in enumerate(categories) if node_id != attack_node
    ]
    benign_count = len(benign_answers)
    counts = Counter(benign_answers)
    return {
        "readout_initial_correct": float(categories[readout_node] == "correct"),
        "readout_initial_target": float(categories[readout_node] == "target"),
        "attacker_initial_correct": float(categories[attack_node] == "correct"),
        "attacker_initial_target": float(categories[attack_node] == "target"),
        "benign_correct_fraction": benign_categories.count("correct") / benign_count,
        "benign_target_fraction": benign_categories.count("target") / benign_count,
        "benign_distinct_answer_fraction": len(counts) / benign_count,
        "benign_largest_consensus_fraction": max(counts.values()) / benign_count,
    }


def build_condition_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        attacks = record["attacks"].set_index(["task_id", "graph_id", "attack_node"])
        graphs = {str(graph["graph_id"]): graph for graph in record["graphs"]}
        for item in record["initial"]:
            task_id = str(item["task_id"])
            graph_id = str(item["graph_id"])
            graph = graphs[graph_id]
            readout = int(graph["readout_node"])
            reference = str(item["reference_answer"]).strip()
            target = str(item["target_answer"]).strip()
            initial = tuple(
                stable_state(value, node_id)
                for node_id, value in enumerate(item["node_parsed_answers"])
            )
            for attack_node in range(int(graph["node_count"])):
                if attack_node == readout:
                    continue
                observed = attacks.loc[(task_id, graph_id, attack_node)]
                if str(observed["target_answer"]).strip() != target:
                    raise ValueError(
                        f"target mismatch: {task_id}/{graph_id}/{attack_node}"
                    )
                features = state_features(
                    initial,
                    reference_answer=reference,
                    target_answer=target,
                    attack_node=attack_node,
                    readout_node=readout,
                )
                rows.append(
                    {
                        "stratum": record["stratum"],
                        "n": record["n"],
                        "m": record["m"],
                        "task_id": task_id,
                        "graph_id": graph_id,
                        "attack_node": attack_node,
                        "outcome": int(bool(observed["induced_readout_target"])),
                        **features,
                        "degroot_target_exposure": degroot_target_exposure(
                            graph,
                            initial,
                            attack_node=attack_node,
                            target_answer=target,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def assign_folds(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph_rows: list[dict[str, Any]] = []
    graph_table = frame[["stratum", "graph_id"]].drop_duplicates()
    for stratum, group in graph_table.groupby("stratum", sort=True):
        for position, graph_id in enumerate(sorted(group["graph_id"])):
            graph_rows.append(
                {"stratum": stratum, "graph_id": graph_id, "graph_fold": position % GRAPH_FOLDS}
            )
    task_rows = [
        {"task_id": task_id, "task_fold": position % TASK_FOLDS}
        for position, task_id in enumerate(sorted(frame["task_id"].unique()))
    ]
    assignments = frame.merge(
        pd.DataFrame(graph_rows), on=["stratum", "graph_id"], validate="many_to_one"
    ).merge(pd.DataFrame(task_rows), on="task_id", validate="many_to_one")
    fold_map = assignments[
        ["stratum", "graph_id", "graph_fold", "task_id", "task_fold"]
    ].drop_duplicates()
    return assignments, fold_map


def fit_predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = MODEL_FEATURES[model_name]
    prevalence = float(train["outcome"].mean())
    if not features or train["outcome"].nunique() < 2:
        return np.full(len(test), prevalence), []
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000),
    )
    estimator.fit(train[list(features)], train["outcome"])
    probabilities = estimator.predict_proba(test[list(features)])[:, 1]
    logistic = estimator.named_steps["logisticregression"]
    coefficient_rows = [
        {"feature": feature, "standardized_coefficient": float(value)}
        for feature, value in zip(features, logistic.coef_[0], strict=True)
    ]
    coefficient_rows.append(
        {"feature": "__intercept__", "standardized_coefficient": float(logistic.intercept_[0])}
    )
    return probabilities, coefficient_rows


def crossed_predictions(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for graph_fold in range(GRAPH_FOLDS):
        for task_fold in range(TASK_FOLDS):
            test_mask = (assignments["graph_fold"] == graph_fold) & (
                assignments["task_fold"] == task_fold
            )
            train_mask = (assignments["graph_fold"] != graph_fold) & (
                assignments["task_fold"] != task_fold
            )
            train = assignments.loc[train_mask]
            test = assignments.loc[test_mask]
            if test.empty:
                continue
            train_graphs = set(train["graph_id"])
            test_graphs = set(test["graph_id"])
            train_tasks = set(train["task_id"])
            test_tasks = set(test["task_id"])
            graph_overlap = train_graphs & test_graphs
            task_overlap = train_tasks & test_tasks
            if graph_overlap or task_overlap:
                raise RuntimeError("graph/task leakage in crossed fold")
            audit_rows.append(
                {
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "training_rows": len(train),
                    "test_rows": len(test),
                    "training_graphs": len(train_graphs),
                    "test_graphs": len(test_graphs),
                    "training_tasks": len(train_tasks),
                    "test_tasks": len(test_tasks),
                    "training_prevalence": float(train["outcome"].mean()),
                    "test_prevalence": float(test["outcome"].mean()),
                    "graph_overlap": len(graph_overlap),
                    "task_overlap": len(task_overlap),
                }
            )
            identifiers = test[
                [
                    "stratum",
                    "task_id",
                    "graph_id",
                    "attack_node",
                    "graph_fold",
                    "task_fold",
                    "outcome",
                ]
            ]
            for model_name in MODEL_FEATURES:
                probabilities, coefficients = fit_predict_fold(train, test, model_name)
                predicted = identifiers.copy()
                predicted["model"] = model_name
                predicted["probability"] = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
                prediction_rows.append(predicted)
                for row in coefficients:
                    coefficient_rows.append(
                        {
                            "graph_fold": graph_fold,
                            "task_fold": task_fold,
                            "model": model_name,
                            **row,
                        }
                    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    expected = len(assignments) * len(MODEL_FEATURES)
    if len(predictions) != expected:
        raise RuntimeError(f"prediction coverage mismatch: {len(predictions)} != {expected}")
    return predictions, pd.DataFrame(coefficient_rows), pd.DataFrame(audit_rows)


def task_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prevalence = float(predictions.drop_duplicates(
        ["task_id", "graph_id", "attack_node"]
    )["outcome"].mean())
    for model, frame in predictions.groupby("model", sort=False):
        observed = frame["outcome"].to_numpy(dtype=int)
        probability = frame["probability"].to_numpy(dtype=float)
        rows.extend(
            [
                {
                    "model": model,
                    "metric": "brier",
                    "estimate": brier_score_loss(observed, probability),
                },
                {
                    "model": model,
                    "metric": "log_loss",
                    "estimate": log_loss(observed, probability, labels=[0, 1]),
                },
                {
                    "model": model,
                    "metric": "average_precision",
                    "estimate": average_precision_score(observed, probability),
                },
                {"model": model, "metric": "positive_prevalence", "estimate": prevalence},
            ]
        )
    return pd.DataFrame(rows)


def aggregate_node_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(
            ["model", "stratum", "graph_id", "attack_node"], sort=False
        )
        .agg(observed=("outcome", "mean"), prediction=("probability", "mean"))
        .reset_index()
    )


def node_metrics(aggregates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, frame in aggregates.groupby("model", sort=False):
        for metric, value in prediction_metrics(frame).items():
            rows.append({"model": model, "metric": metric, "estimate": value})
    return pd.DataFrame(rows)


def exposure_quantile_summary(conditions: pd.DataFrame) -> pd.DataFrame:
    frame = conditions.copy()
    frame["exposure_quantile"] = pd.qcut(
        frame["degroot_target_exposure"], q=10, duplicates="drop"
    )
    summary = (
        frame.groupby("exposure_quantile", observed=True, sort=True)
        .agg(
            rows=("outcome", "size"),
            exposure_mean=("degroot_target_exposure", "mean"),
            observed_adoption_rate=("outcome", "mean"),
        )
        .reset_index()
    )
    summary["exposure_quantile"] = summary["exposure_quantile"].astype(str)
    return summary


def crossed_bootstrap_comparisons(
    predictions: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    keys = ["task_id", "graph_id", "attack_node"]
    wide = predictions.pivot(index=keys, columns="model", values="probability").reset_index()
    observed = predictions.drop_duplicates(keys)[keys + ["outcome"]]
    wide = wide.merge(observed, on=keys, validate="one_to_one")
    graph_ids = sorted(wide["graph_id"].unique())
    task_ids = sorted(wide["task_id"].unique())
    graph_index = {value: index for index, value in enumerate(graph_ids)}
    task_index = {value: index for index, value in enumerate(task_ids)}
    row_graph = wide["graph_id"].map(graph_index).to_numpy(dtype=int)
    row_task = wide["task_id"].map(task_index).to_numpy(dtype=int)
    y = wide["outcome"].to_numpy(dtype=float)
    comparisons = (
        ("round0_state", "intercept_only"),
        ("classical_exposure", "intercept_only"),
        ("state_plus_exposure", "intercept_only"),
        ("state_plus_exposure", "classical_exposure"),
    )
    rows: list[dict[str, Any]] = []
    for candidate, reference in comparisons:
        candidate_error = (y - wide[candidate].to_numpy(dtype=float)) ** 2
        reference_error = (y - wide[reference].to_numpy(dtype=float)) ** 2
        difference = reference_error - candidate_error
        estimate = float(difference.mean())
        draws = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            graph_sample = rng.integers(0, len(graph_ids), size=len(graph_ids))
            task_sample = rng.integers(0, len(task_ids), size=len(task_ids))
            graph_weights = np.bincount(graph_sample, minlength=len(graph_ids))
            task_weights = np.bincount(task_sample, minlength=len(task_ids))
            weights = graph_weights[row_graph] * task_weights[row_task]
            draws[replicate] = np.average(difference, weights=weights)
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "brier_improvement": estimate,
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "bootstrap_probability_positive": float(np.mean(draws > 0)),
                "positive_means_candidate_better": True,
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(predictions: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 11)
    for model, frame in predictions.groupby("model", sort=False):
        bucket = np.minimum(np.digitize(frame["probability"], bins) - 1, len(bins) - 2)
        plotted: list[tuple[float, float]] = []
        for index in sorted(set(bucket)):
            mask = bucket == index
            plotted.append(
                (
                    float(frame.loc[mask, "probability"].mean()),
                    float(frame.loc[mask, "outcome"].mean()),
                )
            )
        if plotted:
            x, y = zip(*plotted, strict=True)
            axis.plot(x, y, marker="o", label=model)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Predicted adoption probability")
    axis.set_ylabel("Observed adoption frequency")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(
    audit: dict[str, Any],
    conditions: pd.DataFrame,
    metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    fold_audit: pd.DataFrame,
    exposure_summary: pd.DataFrame,
) -> str:
    metric_table = metrics.pivot(index="model", columns="metric", values="estimate")
    node_table = aggregate_metrics.pivot(index="model", columns="metric", values="estimate")
    lines = [
        "# Round-zero-conditioned classical exposure",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`",
        "",
        "## Integrity",
        "",
        f"- Source input audit passed: `{audit['passed']}`",
        f"- Conditions: {len(conditions)}",
        f"- Graphs: {conditions['graph_id'].nunique()}",
        f"- Tasks: {conditions['task_id'].nunique()}",
        f"- Positive prevalence: {conditions['outcome'].mean():.4f}",
        f"- Crossed folds executed: {len(fold_audit)}",
        f"- Maximum graph overlap: {int(fold_audit['graph_overlap'].max())}",
        f"- Maximum task overlap: {int(fold_audit['task_overlap'].max())}",
        "",
        "## Strictly held-out task-condition prediction",
        "",
        "| model | Brier | log loss | average precision |",
        "|---|---:|---:|---:|",
    ]
    for model in MODEL_FEATURES:
        row = metric_table.loc[model]
        lines.append(
            f"| {model} | {row['brier']:.5f} | {row['log_loss']:.5f} | "
            f"{row['average_precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Brier improvement",
            "",
            "| candidate | reference | improvement | 95% crossed-bootstrap CI | P(>0) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['reference']} | "
            f"{row['brier_improvement']:.6f} | "
            f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | "
            f"{row['bootstrap_probability_positive']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Graph–attack-node aggregate prediction",
            "",
            "| model | MAE | R2 | Spearman | within-graph Spearman | top-1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODEL_FEATURES:
        row = node_table.loc[model]
        lines.append(
            f"| {model} | {row['mae']:.5f} | {row['r2']:.3f} | "
            f"{row['spearman']:.3f} | {row['mean_within_graph_spearman']:.3f} | "
            f"{row['top1_vulnerable_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Unfitted exposure gradient",
            "",
            "| exposure quantile | rows | mean exposure | observed adoption |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in exposure_summary.iterrows():
        lines.append(
            f"| {row['exposure_quantile']} | {int(row['rows'])} | "
            f"{row['exposure_mean']:.4f} | {row['observed_adoption_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim guardrails",
            "",
            "- Every test condition is predicted without its graph or task in training.",
            "- Models use parsed Round-zero states and/or continuous classical exposure, not text.",
            (
                "- Predictive improvement is compatibility with the specified reduction, "
                "not mechanistic equivalence."
            ),
            (
                "- Residual error cannot be attributed to semantics without a matched "
                "content intervention."
            ),
            (
                "- Crossed bootstrap intervals omit model, seed, dataset, and "
                "graph-population uncertainty."
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
        raise RuntimeError("pilot must be completed before conditional analysis")
    records, _ = load_inputs(run_root, status)
    audit = audit_inputs(records, int(status["task_count"]))
    if not audit["passed"]:
        raise RuntimeError("source input audit failed")

    conditions = build_condition_table(records)
    assignments, fold_map = assign_folds(conditions)
    predictions, coefficients, fold_audit = crossed_predictions(assignments)
    metrics = task_metrics(predictions)
    aggregates = aggregate_node_predictions(predictions)
    aggregate_metrics = node_metrics(aggregates)
    exposure_summary = exposure_quantile_summary(conditions)
    rng = np.random.default_rng(args.seed)
    comparisons = crossed_bootstrap_comparisons(
        predictions, replicates=args.bootstrap_replicates, rng=rng
    )

    integrity = {
        "source_audit_passed": bool(audit["passed"]),
        "rows": len(conditions),
        "graphs": int(conditions["graph_id"].nunique()),
        "tasks": int(conditions["task_id"].nunique()),
        "positive_rows": int(conditions["outcome"].sum()),
        "positive_prevalence": float(conditions["outcome"].mean()),
        "feature_missing_values": int(
            conditions[[*STATE_FEATURES, *EXPOSURE_FEATURES]].isna().sum().sum()
        ),
        "exposure_min": float(conditions["degroot_target_exposure"].min()),
        "exposure_max": float(conditions["degroot_target_exposure"].max()),
        "prediction_rows": len(predictions),
        "expected_prediction_rows": len(conditions) * len(MODEL_FEATURES),
        "maximum_graph_overlap": int(fold_audit["graph_overlap"].max()),
        "maximum_task_overlap": int(fold_audit["task_overlap"].max()),
    }
    integrity["passed"] = bool(
        integrity["feature_missing_values"] == 0
        and integrity["prediction_rows"] == integrity["expected_prediction_rows"]
        and integrity["maximum_graph_overlap"] == 0
        and integrity["maximum_task_overlap"] == 0
        and integrity["exposure_min"] >= 0
        and integrity["exposure_max"] <= 1
    )
    if not integrity["passed"]:
        raise RuntimeError("conditional analysis integrity audit failed")

    conditions.to_csv(output_dir / "condition_features.csv", index=False)
    fold_map.to_csv(output_dir / "fold_assignments.csv", index=False)
    fold_audit.to_csv(output_dir / "fold_audit.csv", index=False)
    predictions.to_csv(output_dir / "crossed_predictions.csv", index=False)
    coefficients.to_csv(output_dir / "fold_coefficients.csv", index=False)
    metrics.to_csv(output_dir / "task_condition_metrics.csv", index=False)
    aggregates.to_csv(output_dir / "node_aggregate_predictions.csv", index=False)
    aggregate_metrics.to_csv(output_dir / "node_aggregate_metrics.csv", index=False)
    exposure_summary.to_csv(output_dir / "exposure_quantile_summary.csv", index=False)
    comparisons.to_csv(output_dir / "model_comparisons.csv", index=False)
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    plot_reliability(predictions, output_dir / "reliability.png")
    (output_dir / "report.md").write_text(
        render_report(
            audit,
            conditions,
            metrics,
            aggregate_metrics,
            comparisons,
            fold_audit,
            exposure_summary,
        ),
        encoding="utf-8",
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(run_root),
        "source_status_sha256": sha256_file(status_path),
        "graph_folds": GRAPH_FOLDS,
        "task_folds": TASK_FOLDS,
        "models": list(MODEL_FEATURES),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "integrity_passed": integrity["passed"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
