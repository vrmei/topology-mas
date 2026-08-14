"""Predict LLM node transitions from C/T/O/U message composition only.

The learned models never receive message text, graph structure, density, task identity,
or graph identity. Out-of-fold predictions use crossed graph--task holdout: every test
row belongs to an unseen graph and an unseen task relative to its training fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_node_round_adoption import extract_updates, read_json
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

STATES = ("correct", "target", "other", "unparsed")
STATE_INDEX = {state: index for index, state in enumerate(STATES)}
MODELS = ("persistence", "degroot_equal", "ctou_table", "ctou_logit")
COUNT_COLUMNS = tuple(f"incoming_{state}_count" for state in STATES)
TABLE_KEYS = ("previous_attack_state", "round_index", *COUNT_COLUMNS)
DEFAULT_SEED = 20_260_814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--table-prior-strength", type=float, default=5.0)
    return parser.parse_args()


def stable_fold(value: str, folds: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def load_updates(
    run_root: Path, folds: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    status = read_json(run_root / "orchestrator_status.json")
    if status.get("status") != "completed":
        raise RuntimeError("run must be complete")
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for descriptor in status["strata"]:
        frame, audit = extract_updates(run_root, {"strata": [descriptor]})
        if not audit["passed"]:
            raise RuntimeError("integrity audit failed: " + "; ".join(audit["errors"][:10]))
        frames.append(frame)
        audits.append(audit)
    result = pd.concat(frames, ignore_index=True)
    result["receiver_scope"] = np.where(
        result.receiver_is_readout.eq(1), "readout", "internal"
    )
    result["graph_fold"] = result.graph_id.map(
        lambda value: stable_fold(str(value), folds)
    )
    result["task_fold"] = result.task_id.map(
        lambda value: stable_fold(str(value), folds)
    )
    result["current_state_index"] = result.current_attack_state.map(STATE_INDEX)
    if result.current_state_index.isna().any():
        raise ValueError("unknown current state")
    audit = {
        "passed": True,
        "updates": len(result),
        "tasks": int(result.task_id.nunique()),
        "graphs": int(result.graph_id.nunique()),
        "paired_conditions": int(sum(int(item["paired_conditions"]) for item in audits)),
        "strata": int(result.stratum.nunique()),
    }
    return result, audit


def static_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    rows = len(frame)
    previous = frame.previous_attack_state.map(STATE_INDEX).to_numpy(int)
    persistence = np.zeros((rows, len(STATES)), dtype=np.float32)
    persistence[np.arange(rows), previous] = 1.0
    counts = frame[list(COUNT_COLUMNS)].to_numpy(float)
    degroot = counts + persistence
    degroot /= degroot.sum(axis=1, keepdims=True)
    return {"persistence": persistence, "degroot_equal": degroot.astype(np.float32)}


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    previous = frame.previous_attack_state.map(STATE_INDEX).to_numpy(int)
    rounds = frame.round_index.to_numpy(int)
    previous_one_hot = np.eye(len(STATES), dtype=float)[previous]
    round_one_hot = np.eye(4, dtype=float)[rounds]
    counts = frame[list(COUNT_COLUMNS)].to_numpy(float)
    total = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    return np.column_stack((previous_one_hot, round_one_hot, counts, fractions))


def table_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prior_strength: float,
) -> np.ndarray:
    base_counts = (
        train.groupby(["previous_attack_state", "round_index", "current_attack_state"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATES, fill_value=0)
    )
    cell_counts = (
        train.groupby([*TABLE_KEYS, "current_attack_state"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATES, fill_value=0)
    )
    base_lookup = {
        key: values.to_numpy(float)
        for key, values in base_counts.iterrows()
    }
    cell_lookup = {key: values.to_numpy(float) for key, values in cell_counts.iterrows()}
    global_counts = train.current_attack_state.value_counts().reindex(STATES, fill_value=0)
    global_prior = (global_counts.to_numpy(float) + 1.0) / (
        float(global_counts.sum()) + len(STATES)
    )
    predictions = np.zeros((len(test), len(STATES)), dtype=np.float32)
    columns = list(test.columns)
    indices = {name: columns.index(name) for name in TABLE_KEYS}
    for row_index, row in enumerate(test.itertuples(index=False, name=None)):
        base_key = (
            row[indices["previous_attack_state"]],
            row[indices["round_index"]],
        )
        base = base_lookup.get(base_key)
        prior = (
            global_prior
            if base is None
            else (base + 1.0) / (base.sum() + len(STATES))
        )
        cell_key = tuple(row[indices[name]] for name in TABLE_KEYS)
        cell = cell_lookup.get(cell_key)
        if cell is None:
            predictions[row_index] = prior
        else:
            predictions[row_index] = (cell + prior_strength * prior) / (
                cell.sum() + prior_strength
            )
    return predictions


def crossed_predictions(
    frame: pd.DataFrame,
    folds: int,
    prior_strength: float,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    predictions = static_predictions(frame)
    predictions["ctou_table"] = np.full(
        (len(frame), len(STATES)), np.nan, dtype=np.float32
    )
    predictions["ctou_logit"] = np.full(
        (len(frame), len(STATES)), np.nan, dtype=np.float32
    )
    matrix = design_matrix(frame)
    labels = frame.current_state_index.to_numpy(int)
    audit_rows: list[dict[str, object]] = []
    for graph_fold in range(folds):
        for task_fold in range(folds):
            test_mask = frame.graph_fold.eq(graph_fold) & frame.task_fold.eq(task_fold)
            train_mask = frame.graph_fold.ne(graph_fold) & frame.task_fold.ne(task_fold)
            test_indices = np.flatnonzero(test_mask.to_numpy())
            train_indices = np.flatnonzero(train_mask.to_numpy())
            if len(test_indices) == 0:
                continue
            train = frame.iloc[train_indices]
            test = frame.iloc[test_indices]
            predictions["ctou_table"][test_indices] = table_predictions(
                train, test, prior_strength
            )
            model = LogisticRegression(
                C=1.0,
                max_iter=300,
                solver="lbfgs",
                random_state=0,
            )
            model.fit(matrix[train_indices], labels[train_indices])
            fold_probability = model.predict_proba(matrix[test_indices])
            aligned = np.zeros((len(test_indices), len(STATES)), dtype=np.float32)
            aligned[:, model.classes_.astype(int)] = fold_probability
            predictions["ctou_logit"][test_indices] = aligned
            audit_rows.append(
                {
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "train_rows": len(train_indices),
                    "test_rows": len(test_indices),
                    "train_graphs": int(train.graph_id.nunique()),
                    "test_graphs": int(test.graph_id.nunique()),
                    "train_tasks": int(train.task_id.nunique()),
                    "test_tasks": int(test.task_id.nunique()),
                    "graph_overlap": len(set(train.graph_id) & set(test.graph_id)),
                    "task_overlap": len(set(train.task_id) & set(test.task_id)),
                }
            )
    for model in ("ctou_table", "ctou_logit"):
        if np.isnan(predictions[model]).any():
            raise RuntimeError(f"incomplete out-of-fold predictions for {model}")
    return predictions, pd.DataFrame(audit_rows)


def multiclass_losses(probability: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    one_hot = np.eye(len(STATES), dtype=float)[labels]
    return {
        "log_loss": -np.log(clipped[np.arange(len(labels)), labels]),
        "brier": ((probability - one_hot) ** 2).sum(axis=1),
        "error": (probability.argmax(axis=1) != labels).astype(float),
    }


def binary_losses(probability: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "log_loss": -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)),
        "brier": (probability - labels) ** 2,
        "error": ((probability >= 0.5) != labels.astype(bool)).astype(float),
    }


def evaluation_subsets(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, int | None]]:
    all_rows = np.ones(len(frame), dtype=bool)
    adoption = (
        frame.previous_attack_state.eq("correct")
        & frame.received_induced_target.eq(1)
    ).to_numpy()
    adoption_labels = (
        frame.current_attack_state.eq("target")
        & ~frame.current_clean_state.eq("target")
    ).astype(int).to_numpy()
    recovery = frame.previous_induced_target_state.eq(1).to_numpy()
    recovery_labels = frame.current_attack_state.eq("correct").astype(int).to_numpy()
    return {
        "next_state": (all_rows, frame.current_state_index.to_numpy(int), None),
        "adoption": (adoption, adoption_labels, STATE_INDEX["target"]),
        "recovery": (recovery, recovery_labels, STATE_INDEX["correct"]),
    }


def task_losses(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    scopes = {
        "all": np.ones(len(frame), dtype=bool),
        "internal": frame.receiver_scope.eq("internal").to_numpy(),
        "readout": frame.receiver_scope.eq("readout").to_numpy(),
    }
    for model, probability in predictions.items():
        for evaluation, (eligible, labels, positive_index) in evaluation_subsets(frame).items():
            for scope, scope_mask in scopes.items():
                mask = eligible & scope_mask
                if positive_index is None:
                    losses = multiclass_losses(probability[mask], labels[mask])
                else:
                    losses = binary_losses(probability[mask, positive_index], labels[mask])
                selected = frame.loc[mask, ["task_id"]].copy()
                for metric, values in losses.items():
                    part = selected.copy()
                    part["value"] = values
                    part["model"] = model
                    part["evaluation"] = evaluation
                    part["scope"] = scope
                    part["metric"] = metric
                    rows.append(
                        part.groupby(
                            ["task_id", "model", "evaluation", "scope", "metric"],
                            as_index=False,
                        ).value.mean()
                    )
    return pd.concat(rows, ignore_index=True)


def summarize_task_losses(
    task: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in task.groupby(
        ["model", "evaluation", "scope", "metric"], sort=True
    ):
        model, evaluation, scope, metric = keys
        values = group.value.to_numpy(float)
        draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
        rows.append(
            {
                "model": model,
                "evaluation": evaluation,
                "scope": scope,
                "metric": metric,
                "estimate": float(values.mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "tasks": len(values),
            }
        )
    return pd.DataFrame(rows)


def paired_model_comparisons(
    task: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    reference = "degroot_equal"
    candidates = ("ctou_table", "ctou_logit")
    for (evaluation, scope, metric), selected in task.groupby(
        ["evaluation", "scope", "metric"], sort=True
    ):
        pivot = selected.pivot(index="task_id", columns="model", values="value")
        for candidate in candidates:
            paired = pivot[[candidate, reference]].dropna()
            difference = (paired[candidate] - paired[reference]).to_numpy(float)
            draws = difference[
                rng.integers(0, len(difference), size=(replicates, len(difference)))
            ].mean(axis=1)
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "evaluation": evaluation,
                    "scope": scope,
                    "metric": metric,
                    "loss_difference": float(difference.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "negative_favors_candidate": True,
                    "tasks": len(difference),
                }
            )
    return pd.DataFrame(rows)


def curve_predictions(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    subsets = evaluation_subsets(frame)
    for evaluation in ("adoption", "recovery"):
        eligible, labels, positive_index = subsets[evaluation]
        for model, probability in predictions.items():
            selected = frame.loc[
                eligible, ["n", "m", "stratum", "receiver_scope", "task_id"]
            ].copy()
            selected["observed"] = labels[eligible]
            selected["prediction"] = probability[eligible, int(positive_index)]
            selected["model"] = model
            selected["evaluation"] = evaluation
            rows.append(selected)
    combined = pd.concat(rows, ignore_index=True)
    return (
        combined.groupby(
            ["model", "evaluation", "receiver_scope", "n", "m", "stratum"],
            as_index=False,
        )
        .agg(
            updates=("observed", "size"),
            observed_rate=("observed", "mean"),
            predicted_rate=("prediction", "mean"),
            tasks=("task_id", "nunique"),
        )
        .assign(absolute_error=lambda value: (value.observed_rate - value.predicted_rate).abs())
    )


def curve_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, frame in curves.groupby(
        ["model", "evaluation", "receiver_scope", "n"], sort=True
    ):
        model, evaluation, scope, n = keys
        correlation = spearmanr(frame.observed_rate, frame.predicted_rate).statistic
        rows.append(
            {
                "model": model,
                "evaluation": evaluation,
                "receiver_scope": scope,
                "n": int(n),
                "m_levels": len(frame),
                "curve_mae": float(frame.absolute_error.mean()),
                "curve_spearman": float(correlation),
            }
        )
    return pd.DataFrame(rows)


def binary_diagnostics(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subsets = evaluation_subsets(frame)
    for evaluation in ("adoption", "recovery"):
        eligible, labels, positive_index = subsets[evaluation]
        for scope in ("all", "internal", "readout"):
            mask = eligible & (
                np.ones(len(frame), dtype=bool)
                if scope == "all"
                else frame.receiver_scope.eq(scope).to_numpy()
            )
            y = labels[mask]
            for model, probability in predictions.items():
                score = probability[mask, int(positive_index)]
                rows.append(
                    {
                        "model": model,
                        "evaluation": evaluation,
                        "scope": scope,
                        "updates": int(mask.sum()),
                        "positive_rate": float(y.mean()),
                        "mean_prediction": float(score.mean()),
                        "average_precision": float(average_precision_score(y, score)),
                        "roc_auc": float(roc_auc_score(y, score)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, audit = load_updates(args.run_root, args.folds)
    predictions, folds = crossed_predictions(
        frame, args.folds, args.table_prior_strength
    )
    if (folds.graph_overlap != 0).any() or (folds.task_overlap != 0).any():
        raise RuntimeError("crossed holdout leakage detected")
    rng = np.random.default_rng(args.seed)
    task = task_losses(frame, predictions)
    summary = summarize_task_losses(task, args.bootstrap_replicates, rng)
    comparisons = paired_model_comparisons(task, args.bootstrap_replicates, rng)
    curves = curve_predictions(frame, predictions)
    curve_metrics = curve_summary(curves)
    diagnostics = binary_diagnostics(frame, predictions)

    folds.to_csv(args.output_dir / "fold_audit.csv", index=False)
    task.to_csv(args.output_dir / "task_loss_sufficient_statistics.csv", index=False)
    summary.to_csv(args.output_dir / "model_loss_summary.csv", index=False)
    comparisons.to_csv(args.output_dir / "model_loss_comparisons.csv", index=False)
    curves.to_csv(args.output_dir / "transition_curve_predictions.csv", index=False)
    curve_metrics.to_csv(args.output_dir / "transition_curve_metrics.csv", index=False)
    diagnostics.to_csv(args.output_dir / "binary_diagnostics.csv", index=False)
    manifest = {
        "analysis_version": "ctou-crossed-graph-task-v1",
        "run_root": str(args.run_root.resolve()),
        "states": list(STATES),
        "features": ["previous_state", "round", "#C", "#T", "#O", "#U"],
        "excluded_features": [
            "message text",
            "task identity",
            "graph identity",
            "graph structure",
            "n",
            "m",
            "density",
            "receiver scope",
        ],
        "models": list(MODELS),
        "crossed_holdout": (
            "25 graph-fold x task-fold cells; test graph and task are both absent from training"
        ),
        "table_prior_strength": args.table_prior_strength,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "integrity": audit,
        "claim_limits": [
            "prediction from realized composition is descriptive and post-treatment",
            "good prediction does not prove the LLM implements the fitted transition law",
            "residual prediction error is not by itself evidence of a semantic mechanism",
            "task bootstrap is conditional on sampled graphs and one model/run configuration",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
