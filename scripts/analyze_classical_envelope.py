"""Evaluate nonlinear trajectory-level classical predictors of LLM target adoption."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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
from analyze_classical_structure import FULL_FEATURES
from analyze_conditional_classical_exposure import (
    EPSILON,
    GRAPH_FOLDS,
    STATE_FEATURES,
    TASK_FOLDS,
    assign_folds,
    build_condition_table,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ANALYSIS_VERSION = "nonlinear-classical-envelope-v1"
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_807
ROUND_INDICES = tuple(range(4))
READOUT_TRAJECTORY = tuple(f"readout_target_mass_r{index}" for index in ROUND_INDICES)
BENIGN_TRAJECTORY = tuple(f"benign_mean_target_mass_r{index}" for index in ROUND_INDICES)
TRAJECTORY_SUMMARIES = (
    "readout_target_mass_mean",
    "readout_target_mass_peak",
    "benign_mean_target_mass_mean",
    "benign_mean_target_mass_peak",
)
TRAJECTORY_FEATURES = (*READOUT_TRAJECTORY, *BENIGN_TRAJECTORY, *TRAJECTORY_SUMMARIES)
CLASSICAL_FEATURES = (*STATE_FEATURES, *TRAJECTORY_FEATURES, *FULL_FEATURES)
MODEL_NAMES = (
    "intercept_only",
    "final_exposure_linear",
    "final_exposure_spline",
    "classical_trajectory_linear",
    "classical_trajectory_hgb",
)


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


def degroot_trajectory_features(
    graph: dict[str, Any],
    initial_answers: tuple[str, ...],
    *,
    attack_node: int,
    target_answer: str,
) -> dict[str, float]:
    """Return content-free target-mass trajectories for one targeted attack."""

    rounds = int(graph["max_rounds"])
    if rounds != max(ROUND_INDICES):
        raise ValueError(f"expected {max(ROUND_INDICES)} rounds, found {rounds}")
    labels = tuple(sorted(set(initial_answers) | {target_answer}))
    target_index = labels.index(target_answer)
    node_count = int(graph["node_count"])
    readout = int(graph["readout_node"])
    beliefs = np.zeros((node_count, len(labels)), dtype=float)
    for node_id, answer in enumerate(initial_answers):
        beliefs[node_id, labels.index(answer)] = 1.0
    beliefs[attack_node, :] = 0.0
    beliefs[attack_node, target_index] = 1.0

    incoming = incoming_neighbors(graph)
    weights = np.zeros((node_count, node_count), dtype=float)
    for node_id, sources in enumerate(incoming):
        participants = (node_id, *sources)
        weights[node_id, list(participants)] = 1.0 / len(participants)

    readout_values: list[float] = []
    benign_values: list[float] = []
    benign_nodes = [node_id for node_id in range(node_count) if node_id != attack_node]
    for round_index in ROUND_INDICES:
        if round_index:
            beliefs = weights @ beliefs
            beliefs[attack_node, :] = 0.0
            beliefs[attack_node, target_index] = 1.0
        readout_values.append(float(beliefs[readout, target_index]))
        benign_values.append(float(beliefs[benign_nodes, target_index].mean()))

    values = {
        **{
            feature: readout_values[index]
            for index, feature in enumerate(READOUT_TRAJECTORY)
        },
        **{
            feature: benign_values[index]
            for index, feature in enumerate(BENIGN_TRAJECTORY)
        },
        "readout_target_mass_mean": float(np.mean(readout_values)),
        "readout_target_mass_peak": float(np.max(readout_values)),
        "benign_mean_target_mass_mean": float(np.mean(benign_values)),
        "benign_mean_target_mass_peak": float(np.max(benign_values)),
    }
    if any(not -EPSILON <= value <= 1.0 + EPSILON for value in values.values()):
        raise ValueError("invalid DeGroot trajectory mass")
    return {key: float(np.clip(value, 0.0, 1.0)) for key, value in values.items()}


def trajectory_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        graphs = {str(graph["graph_id"]): graph for graph in record["graphs"]}
        for item in record["initial"]:
            task_id = str(item["task_id"])
            graph_id = str(item["graph_id"])
            graph = graphs[graph_id]
            readout = int(graph["readout_node"])
            target = str(item["target_answer"]).strip()
            initial = tuple(
                stable_state(value, node_id)
                for node_id, value in enumerate(item["node_parsed_answers"])
            )
            for attack_node in range(int(graph["node_count"])):
                if attack_node == readout:
                    continue
                rows.append(
                    {
                        "task_id": task_id,
                        "graph_id": graph_id,
                        "attack_node": attack_node,
                        **degroot_trajectory_features(
                            graph,
                            initial,
                            attack_node=attack_node,
                            target_answer=target,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_feature_table(
    records: list[dict[str, Any]],
    *,
    static_features_path: Path,
) -> pd.DataFrame:
    conditions = build_condition_table(records)
    trajectories = trajectory_table(records)
    keys = ["task_id", "graph_id", "attack_node"]
    frame = conditions.merge(trajectories, on=keys, validate="one_to_one")
    static = pd.read_csv(static_features_path).rename(columns={"node_id": "attack_node"})
    static = static[["graph_id", "attack_node", *FULL_FEATURES]]
    frame = frame.merge(static, on=["graph_id", "attack_node"], validate="many_to_one")
    final_difference = np.abs(
        frame[READOUT_TRAJECTORY[-1]] - frame["degroot_target_exposure"]
    )
    if float(final_difference.max()) > EPSILON:
        raise RuntimeError("trajectory final exposure differs from existing exposure")
    return frame


def fit_predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
) -> np.ndarray:
    prevalence = float(train["outcome"].mean())
    if model_name == "intercept_only" or train["outcome"].nunique() < 2:
        return np.full(len(test), prevalence)
    if model_name == "final_exposure_linear":
        features = ["degroot_target_exposure"]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000),
        )
    elif model_name == "final_exposure_spline":
        features = ["degroot_target_exposure"]
        model = make_pipeline(
            SplineTransformer(n_knots=5, degree=3, include_bias=False),
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000),
        )
    elif model_name == "classical_trajectory_linear":
        features = list(CLASSICAL_FEATURES)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000),
        )
    elif model_name == "classical_trajectory_hgb":
        features = list(CLASSICAL_FEATURES)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=0,
        )
    else:
        raise ValueError(f"unknown model: {model_name}")
    model.fit(train[features], train["outcome"])
    return model.predict_proba(test[features])[:, 1]


def crossed_predictions(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    identifiers = [
        "stratum",
        "task_id",
        "graph_id",
        "attack_node",
        "graph_fold",
        "task_fold",
        "outcome",
    ]
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
            graph_overlap = set(train["graph_id"]) & set(test["graph_id"])
            task_overlap = set(train["task_id"]) & set(test["task_id"])
            if graph_overlap or task_overlap:
                raise RuntimeError("graph/task leakage in crossed fold")
            audit_rows.append(
                {
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "training_rows": len(train),
                    "test_rows": len(test),
                    "graph_overlap": len(graph_overlap),
                    "task_overlap": len(task_overlap),
                }
            )
            for model_name in MODEL_NAMES:
                prediction = test[identifiers].copy()
                prediction["model"] = model_name
                prediction["probability"] = np.clip(
                    fit_predict_fold(train, test, model_name), EPSILON, 1.0 - EPSILON
                )
                prediction_rows.append(prediction)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    expected = len(assignments) * len(MODEL_NAMES)
    if len(predictions) != expected:
        raise RuntimeError(f"prediction coverage mismatch: {len(predictions)} != {expected}")
    return predictions, pd.DataFrame(audit_rows)


def prediction_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, frame in predictions.groupby("model", sort=False):
        observed = frame["outcome"].to_numpy(dtype=int)
        probability = frame["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model_name,
                "brier": brier_score_loss(observed, probability),
                "log_loss": log_loss(observed, probability, labels=[0, 1]),
                "average_precision": average_precision_score(observed, probability),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics_table(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = (
        predictions.groupby(["model", "stratum", "graph_id", "attack_node"], sort=False)
        .agg(observed=("outcome", "mean"), prediction=("probability", "mean"))
        .reset_index()
    )
    rows = [
        {"model": model_name, **prediction_metrics(frame)}
        for model_name, frame in aggregate.groupby("model", sort=False)
    ]
    return aggregate, pd.DataFrame(rows)


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
    reference = "final_exposure_linear"
    reference_error = (y - wide[reference].to_numpy(dtype=float)) ** 2
    rows: list[dict[str, Any]] = []
    for candidate in MODEL_NAMES:
        if candidate == reference:
            continue
        candidate_error = (y - wide[candidate].to_numpy(dtype=float)) ** 2
        difference = reference_error - candidate_error
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
                "brier_improvement": float(difference.mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "bootstrap_probability_positive": float(np.mean(draws > 0)),
            }
        )
    return pd.DataFrame(rows)


def render_report(
    integrity: dict[str, Any],
    metrics: pd.DataFrame,
    aggregates: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    lines = [
        "# Nonlinear classical envelope",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`",
        "",
        "## Integrity",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in integrity.items())
    lines.extend(
        [
            "",
            "## Strict crossed-held-out prediction",
            "",
            "| model | Brier | log loss | average precision |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['model']} | {row['brier']:.5f} | {row['log_loss']:.5f} | "
            f"{row['average_precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Graph--attack-node aggregate prediction",
            "",
            "| model | MAE | R2 | Spearman | within-graph Spearman | top-1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in aggregates.iterrows():
        lines.append(
            f"| {row['model']} | {row['mae']:.5f} | {row['r2']:.3f} | "
            f"{row['spearman']:.3f} | {row['mean_within_graph_spearman']:.3f} | "
            f"{row['top1_vulnerable_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Brier improvement over linear final exposure",
            "",
            "| candidate | improvement | 95% crossed-bootstrap CI | P(>0) |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['brier_improvement']:.6f} | "
            f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | "
            f"{row['bootstrap_probability_positive']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim guardrails",
            "",
            "- All predictors are content-free and tested on unseen graphs and unseen tasks.",
            "- Better prediction strengthens a classical baseline; it does not prove mechanism.",
            "- Remaining error is not evidence of semantics without a matched intervention.",
            "- This pilot omits model, dataset, assignment-seed, and graph-population uncertainty.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    run_root = args.run_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    status_path = run_root / "orchestrator_status.json"
    status = read_json(status_path)
    if status.get("status") != "completed":
        raise RuntimeError("pilot must be completed before classical-envelope analysis")
    records, _ = load_inputs(run_root, status)
    source_audit = audit_inputs(records, int(status["task_count"]))
    if not source_audit["passed"]:
        raise RuntimeError("source input audit failed")
    static_path = run_root / "posthoc-classical-structure-v1" / "node_structural_features.csv"
    frame = build_feature_table(records, static_features_path=static_path)
    feature_columns = list(dict.fromkeys(CLASSICAL_FEATURES))
    assignments, fold_map = assign_folds(frame)
    predictions, fold_audit = crossed_predictions(assignments)
    metrics = prediction_metrics_table(predictions)
    aggregate_predictions, aggregate_metrics = aggregate_metrics_table(predictions)
    comparisons = crossed_bootstrap_comparisons(
        predictions,
        replicates=args.bootstrap_replicates,
        rng=np.random.default_rng(args.seed),
    )
    duplicate_keys = int(frame.duplicated(["task_id", "graph_id", "attack_node"]).sum())
    integrity = {
        "passed": bool(
            len(frame) == 20_400
            and duplicate_keys == 0
            and frame[feature_columns].notna().all().all()
            and fold_audit["graph_overlap"].max() == 0
            and fold_audit["task_overlap"].max() == 0
        ),
        "conditions": len(frame),
        "graphs": int(frame["graph_id"].nunique()),
        "tasks": int(frame["task_id"].nunique()),
        "positive_events": int(frame["outcome"].sum()),
        "duplicate_condition_keys": duplicate_keys,
        "classical_feature_count": len(feature_columns),
        "missing_feature_values": int(frame[feature_columns].isna().sum().sum()),
        "maximum_graph_overlap": int(fold_audit["graph_overlap"].max()),
        "maximum_task_overlap": int(fold_audit["task_overlap"].max()),
    }
    if not integrity["passed"]:
        raise RuntimeError("classical-envelope integrity audit failed")

    frame.to_csv(output / "condition_features.csv", index=False)
    fold_map.to_csv(output / "fold_assignments.csv", index=False)
    fold_audit.to_csv(output / "fold_audit.csv", index=False)
    predictions.to_csv(output / "crossed_predictions.csv", index=False)
    metrics.to_csv(output / "task_condition_metrics.csv", index=False)
    aggregate_predictions.to_csv(output / "node_aggregate_predictions.csv", index=False)
    aggregate_metrics.to_csv(output / "node_aggregate_metrics.csv", index=False)
    comparisons.to_csv(output / "model_comparisons.csv", index=False)
    (output / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(
        render_report(integrity, metrics, aggregate_metrics, comparisons),
        encoding="utf-8",
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(run_root),
        "source_status_sha256": sha256_file(status_path),
        "models": list(MODEL_NAMES),
        "classical_features": feature_columns,
        "graph_folds": GRAPH_FOLDS,
        "task_folds": TASK_FOLDS,
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
