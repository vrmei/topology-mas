"""Validate local CTOU laws and a correlated Round-0 initializer across scale."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import LogisticRegression

from topology_mas.simulation.ctou_scale import (
    COUNT_COLUMNS,
    CTOU_STATES,
    LOCAL_LAW_VARIANTS,
    HierarchicalRoundZeroModel,
    ctou_design_matrix,
    extract_round_zero_groups,
    fit_hierarchical_round_zero,
    local_law_feature_names,
)

SELECTION_SIZES = (6, 7, 8)
STRESS_SIZE = 10
FOLDS = 5
SEED = 20_260_825
UPDATE_COLUMNS = (
    "task_id",
    "graph_id",
    "n",
    "m",
    "previous_attack_state",
    "round_index",
    *COUNT_COLUMNS,
    "current_state_index",
    "current_attack_state",
    "task_fold",
    "receiver_scope",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n5-n8-cache", type=Path, required=True)
    parser.add_argument("--n6-n7-cache", type=Path, required=True)
    parser.add_argument("--n10-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round0-draws-per-task", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_selected(path: Path, sizes: set[int]) -> dict[str, Any]:
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    result: dict[str, Any] = {}
    for condition in ("attack", "clean"):
        updates = bundle[f"{condition}_updates"]
        selected = updates.loc[updates.n.isin(sizes), list(UPDATE_COLUMNS)].copy()
        result[f"{condition}_updates"] = selected
    clean_cases = bundle["clean_cases"]
    result["round0_groups"] = extract_round_zero_groups(clean_cases.loc[clean_cases.n.isin(sizes)])
    result["audits"] = bundle["audits"]
    return result


def verify_sources(parts: dict[int, dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    task_sets: list[set[str]] = []
    for size, part in sorted(parts.items()):
        for name, audit in part["audits"].items():
            if not audit.get("passed"):
                errors.append(f"n={size} source audit failed: {name}")
        for condition in ("attack", "clean"):
            frame = part[f"{condition}_updates"]
            if set(map(int, frame.n.unique())) != {size}:
                errors.append(f"n={size}/{condition} has the wrong size values")
            task_sets.append(set(frame.task_id.astype(str)))
        groups = part["round0_groups"]
        if set(map(int, groups.n.unique())) != {size}:
            errors.append(f"n={size} Round-0 groups have the wrong size values")
    if task_sets and any(value != task_sets[0] for value in task_sets[1:]):
        errors.append("task collections differ across sizes or conditions")
    audit = {
        "passed": not errors,
        "errors": errors,
        "sizes": sorted(parts),
        "tasks": len(task_sets[0]) if task_sets else 0,
        "round0_groups": {
            str(size): len(part["round0_groups"]) for size, part in sorted(parts.items())
        },
    }
    if errors:
        raise RuntimeError("; ".join(errors))
    return audit


def aligned_probability(model: LogisticRegression, frame: pd.DataFrame, variant: str) -> np.ndarray:
    raw = model.predict_proba(ctou_design_matrix(frame, variant))
    probability = np.zeros((len(frame), len(CTOU_STATES)), dtype=np.float64)
    probability[:, model.classes_.astype(int)] = raw
    return probability


def task_loss_rows(
    *,
    frame: pd.DataFrame,
    probability: np.ndarray,
    condition: str,
    variant: str,
    size: int,
) -> pd.DataFrame:
    labels = frame.current_state_index.to_numpy(int)
    one_hot = np.eye(len(CTOU_STATES))[labels]
    clipped = np.clip(probability, 1e-9, 1.0)
    row = frame[["task_id", "receiver_scope", "round_index"]].copy()
    row["log_loss"] = -np.log(clipped[np.arange(len(frame)), labels])
    row["brier"] = ((probability - one_hot) ** 2).sum(axis=1)
    row["classification_error"] = (probability.argmax(axis=1) != labels).astype(float)
    row["condition"] = condition
    row["model"] = variant
    row["n"] = size
    return row.groupby(["condition", "model", "n", "task_id"], as_index=False).agg(
        updates=("log_loss", "size"),
        log_loss=("log_loss", "mean"),
        brier=("brier", "mean"),
        classification_error=("classification_error", "mean"),
    )


def fit_and_evaluate_local_laws(
    parts: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    task_rows: list[pd.DataFrame] = []
    fitted: dict[str, dict[int, dict[str, LogisticRegression]]] = {
        "attack": {},
        "clean": {},
    }
    for condition in ("attack", "clean"):
        train_all = parts[5][f"{condition}_updates"]
        for fold in range(FOLDS):
            train = train_all.loc[train_all.task_fold.ne(fold)]
            fitted[condition][fold] = {}
            for variant in LOCAL_LAW_VARIANTS:
                model = LogisticRegression(
                    C=1.0,
                    max_iter=500,
                    solver="lbfgs",
                    random_state=0,
                )
                model.fit(
                    ctou_design_matrix(train, variant),
                    train.current_state_index.to_numpy(int),
                )
                fitted[condition][fold][variant] = model
            for size in (*SELECTION_SIZES, STRESS_SIZE):
                test_all = parts[size][f"{condition}_updates"]
                test = test_all.loc[test_all.task_fold.eq(fold)].reset_index(drop=True)
                train_tasks = set(train.task_id.astype(str))
                test_tasks = set(test.task_id.astype(str))
                if train_tasks & test_tasks:
                    raise RuntimeError(f"task leakage for {condition}/fold={fold}/n={size}")
                for variant, model in fitted[condition][fold].items():
                    probability = aligned_probability(model, test, variant)
                    task_rows.append(
                        task_loss_rows(
                            frame=test,
                            probability=probability,
                            condition=condition,
                            variant=variant,
                            size=size,
                        )
                    )
    task_metrics = pd.concat(task_rows, ignore_index=True)
    summary = task_metrics.groupby(["condition", "model", "n"], as_index=False).agg(
        tasks=("task_id", "nunique"),
        updates=("updates", "sum"),
        multiclass_log_loss=("log_loss", "mean"),
        multiclass_brier=("brier", "mean"),
        classification_error=("classification_error", "mean"),
    )
    selection = task_metrics.loc[task_metrics.n.isin(SELECTION_SIZES)]
    score = (
        selection.groupby("model", as_index=False)
        .agg(
            selection_log_loss=("log_loss", "mean"),
            selection_brier=("brier", "mean"),
            selection_classification_error=("classification_error", "mean"),
        )
        .set_index("model")
    )
    cells = summary.loc[summary.n.isin(SELECTION_SIZES)].groupby("model").multiclass_log_loss.max()
    stress = summary.loc[summary.n.eq(STRESS_SIZE)].groupby("model").multiclass_log_loss.mean()
    score["worst_selection_cell_log_loss"] = cells
    score["n10_stress_log_loss"] = stress
    baseline = float(stress.loc["proportions"])
    score["n10_relative_to_proportions"] = score.n10_stress_log_loss / baseline
    score["stable_at_n10"] = np.isfinite(score.n10_stress_log_loss) & score[
        "n10_relative_to_proportions"
    ].le(1.25)
    eligible = score.loc[score.stable_at_n10]
    selected_model = None if eligible.empty else str(eligible.selection_log_loss.idxmin())
    task_level = (
        selection.groupby(["model", "task_id"], as_index=False)
        .log_loss.mean()
        .pivot(index="task_id", columns="model", values="log_loss")
    )
    if selected_model is not None:
        for variant in score.index:
            difference = task_level[variant] - task_level[selected_model]
            half_width = 1.96 * float(difference.std(ddof=1)) / np.sqrt(len(difference))
            score.loc[variant, "paired_difference_vs_selected"] = float(difference.mean())
            score.loc[variant, "paired_difference_ci95_low"] = float(
                difference.mean() - half_width
            )
            score.loc[variant, "paired_difference_ci95_high"] = float(
                difference.mean() + half_width
            )
            score.loc[variant, "indistinguishable_from_selected"] = bool(
                difference.mean() - half_width <= 0 <= difference.mean() + half_width
            )
    score["selected"] = score.index.to_numpy(dtype=str) == selected_model
    score = score.reset_index().sort_values("selection_log_loss")
    frozen = {
        "schema_version": 1,
        "train_sizes": [5],
        "selection_sizes": list(SELECTION_SIZES),
        "stress_size": STRESS_SIZE,
        "selected_model": selected_model,
        "equivalence_set": (
            tuple(score.loc[score.indistinguishable_from_selected, "model"].astype(str))
            if selected_model is not None
            else ()
        ),
        "feature_names": {
            variant: local_law_feature_names(variant) for variant in LOCAL_LAW_VARIANTS
        },
        "condition_fold_models": fitted,
    }
    return task_metrics, summary, {"score": score, "frozen": frozen}


def _beta_binomial_tail_probability(n: int, alpha: float, beta: float, minimum: int) -> float:
    values = []
    for count in range(minimum, n + 1):
        log_choose = gammaln(n + 1) - gammaln(count + 1) - gammaln(n - count + 1)
        values.append(
            np.exp(log_choose + betaln(count + alpha, n - count + beta) - betaln(alpha, beta))
        )
    return float(np.sum(values))


def _predictive_summary(
    model: HierarchicalRoundZeroModel,
    task_ids: list[str],
    n: int,
) -> dict[str, float]:
    means = np.vstack([model.mean_for_task(task_id) for task_id in task_ids])
    correct = means[:, 0]
    kappa = model.concentration
    conditional_variance = correct * (1.0 - correct) * (n + kappa) / (n * (1.0 + kappa))
    mixture_variance = float(conditional_variance.mean() + correct.var())
    all_correct = np.mean(
        np.exp(
            gammaln(kappa)
            - gammaln(kappa + n)
            + gammaln(kappa * correct + n)
            - gammaln(kappa * correct)
        )
    )
    majority_minimum = n // 2 + 1
    majority = np.mean(
        [
            _beta_binomial_tail_probability(
                n,
                kappa * probability,
                kappa * (1.0 - probability),
                majority_minimum,
            )
            for probability in correct
        ]
    )
    return {
        "correct_proportion": float(means[:, 0].mean()),
        "other_proportion": float(means[:, 1].mean()),
        "unparsed_proportion": float(means[:, 2].mean()),
        "correct_fraction_variance": mixture_variance,
        "all_correct_rate": float(all_correct),
        "majority_correct_rate": float(majority),
    }


def _iid_summary(global_mean: np.ndarray, n: int) -> dict[str, float]:
    correct = float(global_mean[0])
    majority_minimum = n // 2 + 1
    majority = 0.0
    for count in range(majority_minimum, n + 1):
        log_probability = (
            gammaln(n + 1)
            - gammaln(count + 1)
            - gammaln(n - count + 1)
            + count * np.log(correct)
            + (n - count) * np.log1p(-correct)
        )
        majority += float(np.exp(log_probability))
    return {
        "correct_proportion": correct,
        "other_proportion": float(global_mean[1]),
        "unparsed_proportion": float(global_mean[2]),
        "correct_fraction_variance": correct * (1.0 - correct) / n,
        "all_correct_rate": correct**n,
        "majority_correct_rate": majority,
    }


def validate_round_zero(
    parts: dict[int, dict[str, Any]],
    *,
    draws_per_task: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, HierarchicalRoundZeroModel, dict[str, Any]]:
    train = parts[5]["round0_groups"]
    model = fit_hierarchical_round_zero(train, required_sizes={5})
    rng = np.random.default_rng(seed)
    validation_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    count_columns = ["correct_count", "other_count", "unparsed_count"]
    for size in (*SELECTION_SIZES, STRESS_SIZE):
        observed = parts[size]["round0_groups"].copy()
        task_ids = sorted(observed.task_id.astype(str).unique())
        group_counts = observed[count_columns].to_numpy(float)
        group_fractions = group_counts / size
        observed_summary = {
            "correct_proportion": float(group_fractions[:, 0].mean()),
            "other_proportion": float(group_fractions[:, 1].mean()),
            "unparsed_proportion": float(group_fractions[:, 2].mean()),
            "correct_fraction_variance": float(group_fractions[:, 0].var()),
            "all_correct_rate": float(np.mean(group_counts[:, 0] == size)),
            "majority_correct_rate": float(np.mean(group_counts[:, 0] > size / 2)),
        }
        summaries = {
            "hierarchical": _predictive_summary(model, task_ids, size),
            "iid_global": _iid_summary(np.asarray(model.global_mean), size),
        }
        observed_task = (
            observed.assign(correct_fraction=observed.correct_count / size)
            .groupby("task_id", as_index=False)
            .correct_fraction.mean()
        )
        task_predictions = {
            "hierarchical": np.asarray(
                [model.mean_for_task(task_id)[0] for task_id in observed_task.task_id]
            ),
            "iid_global": np.repeat(model.global_mean[0], len(observed_task)),
        }
        predictive_samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        hierarchical_values: list[float] = []
        hierarchical_weights: list[float] = []
        iid_values: list[float] = []
        iid_weights: list[float] = []
        for task_id in task_ids:
            sampled = model.sample_counts(
                task_id=task_id,
                node_count=size,
                draws=draws_per_task,
                rng=rng,
            )
            hierarchical_values.extend((sampled[:, 0] / size).tolist())
            hierarchical_weights.extend([1.0 / draws_per_task] * draws_per_task)
            iid = rng.multinomial(size, np.asarray(model.global_mean), size=draws_per_task)
            iid_values.extend((iid[:, 0] / size).tolist())
            iid_weights.extend([1.0 / draws_per_task] * draws_per_task)
        predictive_samples["hierarchical"] = (
            np.asarray(hierarchical_values),
            np.asarray(hierarchical_weights),
        )
        predictive_samples["iid_global"] = (np.asarray(iid_values), np.asarray(iid_weights))
        observed_weights = observed.task_id.map(1.0 / observed.groupby("task_id").size()).to_numpy(
            float
        )
        for name, predicted_summary in summaries.items():
            predicted_task = task_predictions[name]
            actual_task = observed_task.correct_fraction.to_numpy(float)
            task_rho = (
                float(spearmanr(actual_task, predicted_task).statistic)
                if np.unique(predicted_task).size > 1
                else float("nan")
            )
            values, weights = predictive_samples[name]
            validation_rows.append(
                {
                    "model": name,
                    "n": size,
                    "groups": len(observed),
                    "tasks": len(task_ids),
                    **{f"observed_{key}": value for key, value in observed_summary.items()},
                    **{f"predicted_{key}": value for key, value in predicted_summary.items()},
                    "mean_correct_bias": predicted_summary["correct_proportion"]
                    - observed_summary["correct_proportion"],
                    "absolute_variance_error": abs(
                        predicted_summary["correct_fraction_variance"]
                        - observed_summary["correct_fraction_variance"]
                    ),
                    "correct_fraction_wasserstein": wasserstein_distance(
                        group_fractions[:, 0],
                        values,
                        u_weights=observed_weights,
                        v_weights=weights,
                    ),
                    "task_correct_mae": float(np.mean(np.abs(predicted_task - actual_task))),
                    "task_correct_spearman": task_rho,
                }
            )
            for task_id, actual, predicted in zip(
                observed_task.task_id, actual_task, predicted_task, strict=True
            ):
                task_rows.append(
                    {
                        "model": name,
                        "n": size,
                        "task_id": task_id,
                        "observed_correct_fraction": actual,
                        "predicted_correct_fraction": predicted,
                    }
                )
    validation = pd.DataFrame(validation_rows)
    hierarchical = validation.loc[validation.model.eq("hierarchical")]
    iid = validation.loc[validation.model.eq("iid_global")]
    wasserstein_ratio = float(
        hierarchical.correct_fraction_wasserstein.mean() / iid.correct_fraction_wasserstein.mean()
    )
    variance_ratio = float(
        hierarchical.absolute_variance_error.mean() / iid.absolute_variance_error.mean()
    )
    maximum_bias = float(hierarchical.mean_correct_bias.abs().max())
    gate = {
        "wasserstein_ratio_vs_iid": wasserstein_ratio,
        "variance_error_ratio_vs_iid": variance_ratio,
        "maximum_absolute_mean_correct_bias": maximum_bias,
        "thresholds": {
            "maximum_wasserstein_ratio": 0.9,
            "maximum_variance_error_ratio": 0.9,
            "maximum_absolute_mean_correct_bias": 0.03,
        },
        "passed": wasserstein_ratio <= 0.9 and variance_ratio <= 0.9 and maximum_bias <= 0.03,
    }
    return validation, pd.DataFrame(task_rows), model, gate


def main() -> None:
    args = parse_args()
    if args.round0_draws_per_task < 100:
        raise ValueError("round0-draws-per-task must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model_validation"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_sources", flush=True)
    n5n8 = load_selected(args.n5_n8_cache, {5, 8})
    parts = {
        5: {
            key: value.loc[value.n.eq(5)].copy()
            if hasattr(value, "columns") and "n" in value
            else value
            for key, value in n5n8.items()
        },
        8: {
            key: value.loc[value.n.eq(8)].copy()
            if hasattr(value, "columns") and "n" in value
            else value
            for key, value in n5n8.items()
        },
    }
    del n5n8
    n6n7 = load_selected(args.n6_n7_cache, {6, 7})
    for size in (6, 7):
        parts[size] = {
            key: value.loc[value.n.eq(size)].copy()
            if hasattr(value, "columns") and "n" in value
            else value
            for key, value in n6n7.items()
        }
    del n6n7
    parts[10] = load_selected(args.n10_cache, {10})
    boundary = verify_sources(parts)
    atomic_json(model_dir / "boundary_audit.json", boundary)

    print("stage=validate_local_laws", flush=True)
    task_metrics, law_summary, law_result = fit_and_evaluate_local_laws(parts)
    task_metrics.to_csv(model_dir / "local_law_task_metrics.csv.gz", index=False)
    law_summary.to_csv(model_dir / "local_law_n6_n7_n8_n10.csv", index=False)
    law_score = law_result["score"]
    law_score.to_csv(model_dir / "local_law_selection.csv", index=False)
    atomic_pickle(model_dir / "frozen_strict_n5_local_laws.pkl", law_result["frozen"])

    print("stage=validate_round0_initializer", flush=True)
    round0, round0_tasks, initializer, round0_gate = validate_round_zero(
        parts,
        draws_per_task=args.round0_draws_per_task,
        seed=args.seed,
    )
    round0.to_csv(model_dir / "round0_initializer_validation.csv", index=False)
    round0_tasks.to_csv(model_dir / "round0_task_validation.csv.gz", index=False)
    initializer_payload = {
        "schema_version": 1,
        "train_sizes": [5],
        "states": ["correct", "other", "unparsed"],
        "global_mean": initializer.global_mean,
        "task_means": initializer.task_means,
        "concentration": initializer.concentration,
        "smoothing_strength": initializer.smoothing_strength,
    }
    atomic_json(model_dir / "frozen_strict_n5_round0_initializer.json", initializer_payload)
    selected_model = law_result["frozen"]["selected_model"]
    local_gate = selected_model is not None
    phase2_recommended = bool(local_gate and round0_gate["passed"])
    manifest = {
        "analysis_version": "ctou-scale-simulation-n5-to-n50-v1-phase1",
        "phase": "input_validation",
        "real_anchor_sizes": [5, 6, 7, 8, 10],
        "selection_sizes": list(SELECTION_SIZES),
        "stress_size": STRESS_SIZE,
        "local_law_candidates": list(LOCAL_LAW_VARIANTS),
        "selected_local_law": selected_model,
        "local_law_gate_passed": local_gate,
        "round0_gate": round0_gate,
        "phase2_recommended": phase2_recommended,
        "round0_concentration": initializer.concentration,
        "source_files": {
            "n5_n8": str(args.n5_n8_cache.resolve()),
            "n6_n7": str(args.n6_n7_cache.resolve()),
            "n10": str(args.n10_cache.resolve()),
        },
        "source_sha256": {
            "n5_n8": sha256_file(args.n5_n8_cache),
            "n6_n7": sha256_file(args.n6_n7_cache),
            "n10": sha256_file(args.n10_cache),
        },
        "claim_limits": [
            "n=6,7,8 are model-selection anchors and n=10 is a stress check",
            "one-step realized composition is post-treatment",
            "Round-0 initializer is fitted from clean n=5 task-run vectors",
            "no n>10 performance claim is produced in Phase 1",
        ],
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
