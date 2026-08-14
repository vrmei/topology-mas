"""Evaluate clean CTOU utility and clean/attack local-law transfer."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_ctou_recursive_rollout import (
    COUNT_COLUMNS,
    STATE_INDEX,
    STATES,
    dense_lookup,
    load_rollout_cases,
    mean_field_rollout,
    particle_rollout_from_particles,
    query_frame,
    stable_seed,
)
from analyze_ctou_round_zero_free import benign_state_pool, draw_initial_particles
from analyze_ctou_transition_prediction import TABLE_KEYS, design_matrix, load_updates, stable_fold
from analyze_node_round_adoption import read_json, read_jsonl
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

DEFAULT_SEED = 20_260_817
EPSILON = 1e-6
LAWS = ("clean_specific", "attack_specific", "pooled_balanced")
STORED_STATE_ALIASES = {
    "correct": "correct",
    "target": "target",
    "target_error": "target",
    "other": "other",
    "other_error": "other",
    "unparsed": "unparsed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--attack-oracle-dir", type=Path, required=True)
    parser.add_argument("--attack-prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--particles", type=int, default=2_048)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def stored_state(item: dict[str, Any]) -> str:
    raw = item.get("answer_state")
    state = STORED_STATE_ALIASES.get(str(raw))
    if state is None:
        raise ValueError(f"invalid stored answer_state: {raw!r}")
    return state


def graph_file(root: Path) -> Path:
    selected = root / "selected_graphs.jsonl"
    return selected if selected.exists() else root / "batch/inputs/graphs.jsonl"


def load_clean_data(
    run_root: Path,
    status: dict[str, Any],
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], dict[str, object]]:
    """Read each clean `(task, graph)` trace exactly once."""

    cases: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []
    graphs: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    seen_clean_ids: set[str] = set()
    task_graph_to_clean: dict[tuple[str, str], str] = {}
    for descriptor in status["strata"]:
        stratum = str(descriptor["key"])
        root = run_root / "strata" / stratum
        stratum_graphs = {str(x["graph_id"]): x for x in read_jsonl(graph_file(root))}
        graphs.update(stratum_graphs)
        trace_root = root / "batch/traces"
        for pair in read_jsonl(root / "analysis-v1/paired_attacks.jsonl"):
            clean_id = str(pair["clean_run_spec_id"])
            task_id = str(pair["task_id"])
            graph_id = str(pair["graph_id"])
            key = (task_id, graph_id)
            previous = task_graph_to_clean.setdefault(key, clean_id)
            if previous != clean_id:
                errors.append(f"multiple clean traces for {key}: {previous}, {clean_id}")
                continue
            if clean_id in seen_clean_ids:
                continue
            seen_clean_ids.add(clean_id)
            path = trace_root / f"{clean_id}.json"
            if not path.exists():
                errors.append(f"missing clean trace {path}")
                continue
            trace = read_json(path)["trace"]
            graph = stratum_graphs[graph_id]
            n = int(graph["node_count"])
            readout = int(graph["readout_node"])
            turns = {
                (int(turn["node_id"]), int(turn["round_index"])): turn
                for turn in trace["turns"]
            }
            messages = {str(message["message_id"]): message for message in trace["messages"]}
            round_zero = {node: turns.get((node, 0)) for node in range(n)}
            if any(turn is None for turn in round_zero.values()):
                errors.append(f"incomplete clean Round 0: {clean_id}")
                continue
            initial_states = tuple(
                STATE_INDEX[stored_state(round_zero[node])] for node in range(n)  # type: ignore[arg-type]
            )
            try:
                final_state = stored_state({"answer_state": trace.get("final_answer_state")})
            except ValueError:
                errors.append(f"invalid clean final state: {clean_id}")
                continue
            graph_fold = stable_fold(graph_id, folds)
            task_fold = stable_fold(task_id, folds)
            cases.append(
                {
                    "stratum": stratum,
                    "task_id": task_id,
                    "graph_id": graph_id,
                    "attack_node": -1,
                    "n": n,
                    "m": len(graph["edges"]),
                    "readout_node": readout,
                    "horizon": int(graph["max_rounds"]),
                    "initial_states": initial_states,
                    "round0_state": STATES[initial_states[readout]],
                    "round0_correct": int(initial_states[readout] == STATE_INDEX["correct"]),
                    "actual_state": final_state,
                    "actual_state_index": STATE_INDEX[final_state],
                    "actual_target": int(final_state == "target"),
                    "actual_correct": int(final_state == "correct"),
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                }
            )
            for (receiver, round_index), turn in sorted(turns.items()):
                if round_index == 0:
                    continue
                previous = turns.get((receiver, round_index - 1))
                if previous is None:
                    errors.append(
                        f"missing previous clean turn: {clean_id}, {receiver}, {round_index}"
                    )
                    continue
                incoming_states: list[str] = []
                for message_id in turn["incoming_message_ids"]:
                    message = messages.get(str(message_id))
                    if message is None:
                        errors.append(f"missing clean message: {clean_id}, {message_id}")
                        continue
                    incoming_states.append(stored_state(message))
                counts = Counter(incoming_states)
                current_state = stored_state(turn)
                row: dict[str, object] = {
                    "condition": "clean",
                    "stratum": stratum,
                    "task_id": task_id,
                    "graph_id": graph_id,
                    "receiver_node": receiver,
                    "round_index": round_index,
                    "previous_attack_state": stored_state(previous),
                    "current_attack_state": current_state,
                    "current_state_index": STATE_INDEX[current_state],
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "n": n,
                    "m": len(graph["edges"]),
                }
                for state, column in zip(STATES, COUNT_COLUMNS, strict=True):
                    row[column] = counts[state]
                updates.append(row)
    case_frame = pd.DataFrame(cases)
    update_frame = pd.DataFrame(updates)
    if case_frame.duplicated(["task_id", "graph_id"]).any():
        errors.append("duplicate clean endpoint cases")
    if update_frame.duplicated(["task_id", "graph_id", "receiver_node", "round_index"]).any():
        errors.append("duplicate clean transition rows")
    audit = {
        "passed": not errors,
        "errors": errors[:100],
        "clean_traces": len(case_frame),
        "clean_updates": len(update_frame),
        "tasks": int(case_frame.task_id.nunique()),
        "graphs": int(case_frame.graph_id.nunique()),
        "strata": int(case_frame.stratum.nunique()),
    }
    return case_frame, update_frame, graphs, audit


def standardized_attack_updates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["condition"] = "attack"
    return result


def fit_logit(
    train: pd.DataFrame,
    *,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression:
    model = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=0)
    model.fit(
        design_matrix(train),
        train.current_state_index.to_numpy(int),
        sample_weight=sample_weight,
    )
    return model


def aligned_probabilities(model: LogisticRegression, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(design_matrix(frame))
    aligned = np.zeros((len(frame), len(STATES)), dtype=np.float32)
    aligned[:, model.classes_.astype(int)] = raw
    return aligned


def balanced_pooled_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.condition.value_counts()
    total = len(frame)
    return frame.condition.map(lambda value: total / (len(counts) * counts[value])).to_numpy(float)


def fit_fold_laws(
    clean: pd.DataFrame,
    attack: pd.DataFrame,
    *,
    maximum_neighbors: int,
    horizon: int,
) -> tuple[dict[str, LogisticRegression], dict[str, np.ndarray]]:
    pooled = pd.concat([clean, attack], ignore_index=True)
    models = {
        "clean_specific": fit_logit(clean),
        "attack_specific": fit_logit(attack),
        "pooled_balanced": fit_logit(pooled, sample_weight=balanced_pooled_weights(pooled)),
    }
    query = query_frame(maximum_neighbors, horizon)
    lookups = {
        name: dense_lookup(
            query,
            aligned_probabilities(model, query),
            horizon,
            maximum_neighbors,
        )
        for name, model in models.items()
    }
    return models, lookups


def exact_keys(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    return set(frame[list(TABLE_KEYS)].itertuples(index=False, name=None))


def probability_losses(probability: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    one_hot = np.eye(len(STATES))[labels]
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    result = {
        "multiclass_brier": ((probability - one_hot) ** 2).sum(axis=1),
        "multiclass_log_loss": -np.log(clipped[np.arange(len(labels)), labels]),
    }
    for state in ("correct", "target"):
        observed = (labels == STATE_INDEX[state]).astype(float)
        predicted = probability[:, STATE_INDEX[state]]
        clipped_binary = np.clip(predicted, EPSILON, 1.0 - EPSILON)
        result[f"{state}_brier"] = (predicted - observed) ** 2
        result[f"{state}_log_loss"] = -(
            observed * np.log(clipped_binary)
            + (1.0 - observed) * np.log(1.0 - clipped_binary)
        )
    return result


def one_step_law_transfer(
    clean_updates: pd.DataFrame,
    attack_updates: pd.DataFrame,
    *,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    maximum_neighbors = int(max(clean_updates.n.max(), attack_updates.n.max()) - 1)
    horizon = int(max(clean_updates.round_index.max(), attack_updates.round_index.max()))
    for graph_fold in range(folds):
        for task_fold in range(folds):
            clean_train = clean_updates[
                clean_updates.graph_fold.ne(graph_fold)
                & clean_updates.task_fold.ne(task_fold)
            ]
            attack_train = attack_updates[
                attack_updates.graph_fold.ne(graph_fold)
                & attack_updates.task_fold.ne(task_fold)
            ]
            clean_test = clean_updates[
                clean_updates.graph_fold.eq(graph_fold)
                & clean_updates.task_fold.eq(task_fold)
            ]
            attack_test = attack_updates[
                attack_updates.graph_fold.eq(graph_fold)
                & attack_updates.task_fold.eq(task_fold)
            ]
            if clean_test.empty and attack_test.empty:
                continue
            models, _ = fit_fold_laws(
                clean_train,
                attack_train,
                maximum_neighbors=maximum_neighbors,
                horizon=horizon,
            )
            shared = exact_keys(clean_train) & exact_keys(attack_train)
            audits.append(
                {
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "clean_train": len(clean_train),
                    "attack_train": len(attack_train),
                    "clean_test": len(clean_test),
                    "attack_test": len(attack_test),
                    "shared_training_cells": len(shared),
                    "graph_overlap": len(
                        (set(clean_train.graph_id) | set(attack_train.graph_id))
                        & (set(clean_test.graph_id) | set(attack_test.graph_id))
                    ),
                    "task_overlap": len(
                        (set(clean_train.task_id) | set(attack_train.task_id))
                        & (set(clean_test.task_id) | set(attack_test.task_id))
                    ),
                }
            )
            for condition, test in (("clean", clean_test), ("attack", attack_test)):
                labels = test.current_state_index.to_numpy(int)
                shared_mask = np.asarray(
                    [
                        key in shared
                        for key in test[list(TABLE_KEYS)].itertuples(index=False, name=None)
                    ]
                )
                for law, model in models.items():
                    probability = aligned_probabilities(model, test)
                    losses = probability_losses(probability, labels)
                    for scope, mask in (
                        ("all", np.ones(len(test), dtype=bool)),
                        ("shared_cell", shared_mask),
                    ):
                        if not mask.any():
                            continue
                        part = pd.DataFrame(
                            {
                                "task_id": test.loc[mask, "task_id"].to_numpy(),
                                **{name: values[mask] for name, values in losses.items()},
                            }
                        )
                        part = part.groupby("task_id", sort=False).agg(
                            rows=("task_id", "size"),
                            **{
                                metric: (metric, "mean")
                                for metric in losses
                            },
                        ).reset_index()
                        part["condition"] = condition
                        part["law"] = law
                        part["support_scope"] = scope
                        part["graph_fold"] = graph_fold
                        part["task_fold"] = task_fold
                        task_rows.append(part)
    return pd.concat(task_rows, ignore_index=True), pd.DataFrame(audits)


def summarize_task_losses(
    task_losses: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [column for column in task_losses if column.endswith(("brier", "log_loss"))]
    task = (
        task_losses.groupby(["condition", "law", "support_scope", "task_id"], sort=False)
        .apply(
            lambda frame: pd.Series(
                {
                    metric: np.average(frame[metric], weights=frame.rows)
                    for metric in metrics
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    summaries: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for (condition, law, scope), group in task.groupby(
        ["condition", "law", "support_scope"], sort=True
    ):
        values = group[metrics].to_numpy(float)
        samples = rng.integers(0, len(values), size=(replicates, len(values)))
        draws = values[samples].mean(axis=1)
        for index, metric in enumerate(metrics):
            low, high = np.quantile(draws[:, index], [0.025, 0.975])
            summaries.append(
                {
                    "condition": condition,
                    "law": law,
                    "support_scope": scope,
                    "metric": metric,
                    "estimate": values[:, index].mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "tasks": len(values),
                }
            )
    for (condition, scope), group in task.groupby(["condition", "support_scope"], sort=True):
        reference = "clean_specific" if condition == "clean" else "attack_specific"
        reference_frame = group[group.law.eq(reference)].set_index("task_id")
        for law in ("pooled_balanced", "attack_specific", "clean_specific"):
            if law == reference:
                continue
            candidate = group[group.law.eq(law)].set_index("task_id")
            common = sorted(set(reference_frame.index) & set(candidate.index))
            for metric in metrics:
                difference = (
                    candidate.loc[common, metric].to_numpy(float)
                    - reference_frame.loc[common, metric].to_numpy(float)
                )
                samples = rng.integers(0, len(difference), size=(replicates, len(difference)))
                draws = difference[samples].mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975])
                comparisons.append(
                    {
                        "condition": condition,
                        "support_scope": scope,
                        "candidate_law": law,
                        "reference_law": reference,
                        "metric": metric,
                        "loss_difference": difference.mean(),
                        "ci95_low": low,
                        "ci95_high": high,
                        "tasks": len(difference),
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def clean_initial_pool(train_cases: pd.DataFrame, n: int) -> np.ndarray:
    selected = train_cases[train_cases.n.eq(n)]
    if selected.empty:
        raise ValueError(f"empty clean initialization pool for n={n}")
    return np.asarray(selected.initial_states.tolist(), dtype=np.int8)


def draw_clean_particles(
    pool: np.ndarray,
    *,
    particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = pool[rng.integers(0, len(pool), size=particles)]
    permutations = np.argsort(rng.random(selected.shape), axis=1)
    return np.take_along_axis(selected, permutations, axis=1)


def endpoint_rows(
    probability: np.ndarray,
    case: Any,
    *,
    condition: str,
    law: str,
    initialization: str,
    rollout_mode: str,
    predicted_round0_correct: float,
) -> dict[str, object]:
    return {
        "condition": condition,
        "law": law,
        "initialization": initialization,
        "rollout_mode": rollout_mode,
        "task_id": case.task_id,
        "graph_id": case.graph_id,
        "attack_node": int(case.attack_node),
        "n": int(case.n),
        "m": int(case.m),
        "actual_state": case.actual_state,
        "actual_state_index": int(case.actual_state_index),
        "actual_target": int(case.actual_target),
        "actual_correct": int(case.actual_correct),
        "actual_round0_correct": int(getattr(case, "round0_correct", -1)),
        "predicted_round0_correct": predicted_round0_correct,
        **{
            f"p_{state}": float(probability[index])
            for index, state in enumerate(STATES)
        },
    }


def empirical_endpoint_rollouts(
    clean_cases: pd.DataFrame,
    attack_cases: pd.DataFrame,
    clean_updates: pd.DataFrame,
    attack_updates: pd.DataFrame,
    graphs: dict[str, dict[str, Any]],
    *,
    folds: int,
    particles: int,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[Path] = []
    audits: list[dict[str, object]] = []
    maximum_neighbors = int(max(clean_cases.n.max(), attack_cases.n.max()) - 1)
    horizon = int(max(clean_cases.horizon.max(), attack_cases.horizon.max()))
    for graph_fold in range(folds):
        for task_fold in range(folds):
            path = checkpoint_dir / f"graph-{graph_fold}_task-{task_fold}.csv"
            checkpoint_paths.append(path)
            clean_train_updates = clean_updates[
                clean_updates.graph_fold.ne(graph_fold)
                & clean_updates.task_fold.ne(task_fold)
            ]
            attack_train_updates = attack_updates[
                attack_updates.graph_fold.ne(graph_fold)
                & attack_updates.task_fold.ne(task_fold)
            ]
            clean_train_cases = clean_cases[
                clean_cases.graph_fold.ne(graph_fold) & clean_cases.task_fold.ne(task_fold)
            ]
            attack_train_cases = attack_cases[
                attack_cases.graph_fold.ne(graph_fold) & attack_cases.task_fold.ne(task_fold)
            ]
            clean_test = clean_cases[
                clean_cases.graph_fold.eq(graph_fold) & clean_cases.task_fold.eq(task_fold)
            ]
            attack_test = attack_cases[
                attack_cases.graph_fold.eq(graph_fold) & attack_cases.task_fold.eq(task_fold)
            ]
            if clean_test.empty and attack_test.empty:
                continue
            audits.append(
                {
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "clean_test": len(clean_test),
                    "attack_test": len(attack_test),
                    "graph_overlap": len(
                        (set(clean_train_cases.graph_id) | set(attack_train_cases.graph_id))
                        & (set(clean_test.graph_id) | set(attack_test.graph_id))
                    ),
                    "task_overlap": len(
                        (set(clean_train_cases.task_id) | set(attack_train_cases.task_id))
                        & (set(clean_test.task_id) | set(attack_test.task_id))
                    ),
                }
            )
            if path.exists():
                print(f"resume endpoint fold graph={graph_fold} task={task_fold}", flush=True)
                continue
            _, lookups = fit_fold_laws(
                clean_train_updates,
                attack_train_updates,
                maximum_neighbors=maximum_neighbors,
                horizon=horizon,
            )
            clean_pools = {
                int(n): clean_initial_pool(clean_train_cases, int(n))
                for n in sorted(clean_test.n.unique())
            }
            attack_pools = {
                int(n): benign_state_pool(attack_train_cases, int(n))[1]
                for n in sorted(attack_test.n.unique())
            }
            rows: list[dict[str, object]] = []
            clean_empirical_cache: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
            for case in clean_test.itertuples(index=False):
                graph = graphs[case.graph_id]
                initial = tuple(int(x) for x in case.initial_states)
                for law in ("clean_specific", "pooled_balanced"):
                    probability = mean_field_rollout(
                        graph=graph,
                        initial_states=initial,
                        attack_node=None,
                        model="ctou_logit",
                        lookup=lookups[law],
                    )
                    rows.append(
                        endpoint_rows(
                            probability,
                            case,
                            condition="clean",
                            law=law,
                            initialization="oracle",
                            rollout_mode="mean_field",
                            predicted_round0_correct=float(case.round0_correct),
                        )
                    )
                for baseline in ("persistence", "degroot_equal"):
                    probability = mean_field_rollout(
                        graph=graph,
                        initial_states=initial,
                        attack_node=None,
                        model=baseline,
                        lookup=None,
                    )
                    rows.append(
                        endpoint_rows(
                            probability,
                            case,
                            condition="clean",
                            law=baseline,
                            initialization="oracle",
                            rollout_mode="mean_field",
                            predicted_round0_correct=float(case.round0_correct),
                        )
                    )
                for law in ("clean_specific", "pooled_balanced"):
                    key = (law, case.graph_id)
                    if key not in clean_empirical_cache:
                        case_seed = stable_seed(
                            graph_fold,
                            task_fold,
                            case.graph_id,
                            law,
                            "clean-correlated",
                            seed=seed,
                        )
                        particles_initial = draw_clean_particles(
                            clean_pools[int(case.n)],
                            particles=particles,
                            rng=np.random.default_rng(case_seed),
                        )
                        initial_correct = float(
                            np.mean(
                                particles_initial[:, int(case.readout_node)]
                                == STATE_INDEX["correct"]
                            )
                        )
                        probability = particle_rollout_from_particles(
                            graph=graph,
                            initial_particles=particles_initial,
                            attack_node=None,
                            model="ctou_logit",
                            lookup=lookups[law],
                            seed=case_seed,
                        )
                        clean_empirical_cache[key] = probability, initial_correct
                    probability, initial_correct = clean_empirical_cache[key]
                    rows.append(
                        endpoint_rows(
                            probability,
                            case,
                            condition="clean",
                            law=law,
                            initialization="correlated_empirical",
                            rollout_mode="particle",
                            predicted_round0_correct=initial_correct,
                        )
                    )
            attack_oracle_cache: dict[tuple[str, int, tuple[int, ...]], np.ndarray] = {}
            attack_empirical_cache: dict[tuple[str, int], np.ndarray] = {}
            for case in attack_test.itertuples(index=False):
                graph = graphs[case.graph_id]
                initial = tuple(int(x) for x in case.initial_states)
                oracle_key = (case.graph_id, int(case.attack_node), initial)
                if oracle_key not in attack_oracle_cache:
                    attack_oracle_cache[oracle_key] = mean_field_rollout(
                        graph=graph,
                        initial_states=initial,
                        attack_node=int(case.attack_node),
                        model="ctou_logit",
                        lookup=lookups["pooled_balanced"],
                    )
                rows.append(
                    endpoint_rows(
                        attack_oracle_cache[oracle_key],
                        case,
                        condition="attack",
                        law="pooled_balanced",
                        initialization="oracle",
                        rollout_mode="mean_field",
                        predicted_round0_correct=np.nan,
                    )
                )
                empirical_key = (case.graph_id, int(case.attack_node))
                if empirical_key not in attack_empirical_cache:
                    case_seed = stable_seed(
                        graph_fold,
                        task_fold,
                        case.graph_id,
                        case.attack_node,
                        "attack-correlated-pooled",
                        seed=seed,
                    )
                    iid_probability = np.bincount(
                        attack_pools[int(case.n)].ravel(), minlength=len(STATES)
                    ).astype(float)
                    iid_probability /= iid_probability.sum()
                    particles_initial = draw_initial_particles(
                        n=int(case.n),
                        attack_node=int(case.attack_node),
                        mode="correlated_empirical",
                        particles=particles,
                        iid_probability=iid_probability,
                        correlated_pool=attack_pools[int(case.n)],
                        rng=np.random.default_rng(case_seed),
                    )
                    attack_empirical_cache[empirical_key] = particle_rollout_from_particles(
                        graph=graph,
                        initial_particles=particles_initial,
                        attack_node=int(case.attack_node),
                        model="ctou_logit",
                        lookup=lookups["pooled_balanced"],
                        seed=case_seed,
                    )
                rows.append(
                    endpoint_rows(
                        attack_empirical_cache[empirical_key],
                        case,
                        condition="attack",
                        law="pooled_balanced",
                        initialization="correlated_empirical",
                        rollout_mode="particle",
                        predicted_round0_correct=np.nan,
                    )
                )
            temporary = path.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(path)
            print(
                f"endpoint fold graph={graph_fold} task={task_fold}: rows={len(rows)}",
                flush=True,
            )
            del lookups, rows, clean_empirical_cache, attack_oracle_cache, attack_empirical_cache
            gc.collect()
    missing = [path for path in checkpoint_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing endpoint checkpoints: {missing}")
    return (
        pd.concat((pd.read_csv(path) for path in checkpoint_paths), ignore_index=True),
        pd.DataFrame(audits),
    )


def saved_attack_specific(
    attack_oracle_dir: Path,
    attack_prior_dir: Path,
) -> pd.DataFrame:
    oracle = pd.read_csv(attack_oracle_dir / "endpoint_predictions.csv")
    oracle = oracle[
        oracle.model.eq("ctou_logit") & oracle.rollout_mode.eq("mean_field")
    ].copy()
    oracle["initialization"] = "oracle"
    prior = pd.read_csv(attack_prior_dir / "endpoint_predictions.csv")
    prior = prior[prior.initialization.eq("correlated_empirical")].copy()
    selected = pd.concat([oracle, prior], ignore_index=True, sort=False)
    selected["condition"] = "attack"
    selected["law"] = "attack_specific"
    selected["actual_round0_correct"] = -1
    selected["predicted_round0_correct"] = np.nan
    return selected


def score_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    probability = result[[f"p_{state}" for state in STATES]].to_numpy(float)
    labels = result.actual_state_index.to_numpy(int)
    for name, values in probability_losses(probability, labels).items():
        result[name] = values
    return result


def endpoint_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    graph = (
        frame.groupby(
            ["condition", "law", "initialization", "rollout_mode", "graph_id", "n", "m"],
            sort=False,
        )
        .agg(
            cases=("actual_correct", "size"),
            observed_correct=("actual_correct", "mean"),
            predicted_correct=("p_correct", "mean"),
            observed_target=("actual_target", "mean"),
            predicted_target=("p_target", "mean"),
            observed_round0_correct=("actual_round0_correct", "mean"),
            predicted_round0_correct=("predicted_round0_correct", "mean"),
        )
        .reset_index()
    )
    curves = (
        graph.groupby(["condition", "law", "initialization", "rollout_mode", "n", "m"])
        .agg(
            graphs=("graph_id", "size"),
            observed_correct=("observed_correct", "mean"),
            predicted_correct=("predicted_correct", "mean"),
            observed_target=("observed_target", "mean"),
            predicted_target=("predicted_target", "mean"),
            observed_round0_correct=("observed_round0_correct", "mean"),
            predicted_round0_correct=("predicted_round0_correct", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for keys, group in graph.groupby(
        ["condition", "law", "initialization", "rollout_mode", "n"], sort=True
    ):
        condition, law, initialization, mode, n = keys
        for outcome in ("correct", "target"):
            observed = group[f"observed_{outcome}"].to_numpy(float)
            predicted = group[f"predicted_{outcome}"].to_numpy(float)
            rows.append(
                {
                    "condition": condition,
                    "law": law,
                    "initialization": initialization,
                    "rollout_mode": mode,
                    "n": int(n),
                    "outcome": outcome,
                    "graphs": len(group),
                    "graph_mae": np.abs(predicted - observed).mean(),
                    "graph_spearman": spearmanr(observed, predicted).statistic,
                }
            )
    return graph, curves, pd.DataFrame(rows)


def utility_robustness_tables(graph: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = graph[
        graph.law.eq("pooled_balanced")
        & graph.initialization.eq("correlated_empirical")
        & graph.rollout_mode.eq("particle")
    ]
    clean = primary[primary.condition.eq("clean")].set_index(["graph_id", "n", "m"])
    attack = primary[primary.condition.eq("attack")].set_index(["graph_id", "n", "m"])
    common = clean.index.intersection(attack.index)
    rows = pd.DataFrame(
        {
            "graph_id": [index[0] for index in common],
            "n": [index[1] for index in common],
            "m": [index[2] for index in common],
            "observed_u0": clean.loc[common, "observed_round0_correct"].to_numpy(float),
            "predicted_u0": clean.loc[common, "predicted_round0_correct"].to_numpy(float),
            "observed_utility": clean.loc[common, "observed_correct"].to_numpy(float),
            "predicted_utility": clean.loc[common, "predicted_correct"].to_numpy(float),
            "observed_robustness": attack.loc[common, "observed_correct"].to_numpy(float),
            "predicted_robustness": attack.loc[common, "predicted_correct"].to_numpy(float),
        }
    )
    rows["observed_delta_u"] = rows.observed_utility - rows.observed_u0
    rows["predicted_delta_u"] = rows.predicted_utility - rows.predicted_u0
    curves = (
        rows.groupby(["n", "m"], sort=True)
        .agg(
            graphs=("graph_id", "size"),
            observed_u0=("observed_u0", "mean"),
            predicted_u0=("predicted_u0", "mean"),
            observed_utility=("observed_utility", "mean"),
            predicted_utility=("predicted_utility", "mean"),
            observed_robustness=("observed_robustness", "mean"),
            predicted_robustness=("predicted_robustness", "mean"),
            observed_delta_u=("observed_delta_u", "mean"),
            predicted_delta_u=("predicted_delta_u", "mean"),
        )
        .reset_index()
    )
    return rows, curves


def main() -> None:
    args = parse_args()
    if args.particles < 256:
        raise ValueError("particles must be at least 256")
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = read_json(args.run_root / "orchestrator_status.json")
    clean_cases, clean_updates, graphs, clean_audit = load_clean_data(
        args.run_root, status, args.folds
    )
    attack_updates, attack_update_audit = load_updates(args.run_root, args.folds)
    attack_updates = standardized_attack_updates(attack_updates)
    attack_cases, attack_graphs, attack_case_audit = load_rollout_cases(
        args.run_root, status, args.folds
    )
    graphs.update(attack_graphs)
    if (
        not clean_audit["passed"]
        or not attack_update_audit["passed"]
        or not attack_case_audit["passed"]
    ):
        raise RuntimeError("input integrity audit failed")

    task_losses, one_step_audit = one_step_law_transfer(
        clean_updates,
        attack_updates,
        folds=args.folds,
    )
    if (one_step_audit.graph_overlap != 0).any() or (one_step_audit.task_overlap != 0).any():
        raise RuntimeError("one-step crossed holdout leakage")
    one_step_summary, one_step_comparisons = summarize_task_losses(
        task_losses,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    task_losses.to_csv(args.output_dir / "one_step_task_losses.csv", index=False)
    one_step_audit.to_csv(args.output_dir / "one_step_fold_audit.csv", index=False)
    one_step_summary.to_csv(args.output_dir / "one_step_loss_summary.csv", index=False)
    one_step_comparisons.to_csv(
        args.output_dir / "one_step_paired_law_comparisons.csv", index=False
    )

    generated, endpoint_audit = empirical_endpoint_rollouts(
        clean_cases,
        attack_cases,
        clean_updates,
        attack_updates,
        graphs,
        folds=args.folds,
        particles=args.particles,
        seed=args.seed,
        checkpoint_dir=args.output_dir / "endpoint-fold-checkpoints",
    )
    if (endpoint_audit.graph_overlap != 0).any() or (endpoint_audit.task_overlap != 0).any():
        raise RuntimeError("endpoint crossed holdout leakage")
    saved = saved_attack_specific(args.attack_oracle_dir, args.attack_prior_dir)
    endpoints = score_endpoints(pd.concat([generated, saved], ignore_index=True, sort=False))
    graph, curves, metrics = endpoint_summaries(endpoints)
    utility_graphs, utility_curves = utility_robustness_tables(graph)
    endpoint_audit.to_csv(args.output_dir / "endpoint_fold_audit.csv", index=False)
    endpoints.to_csv(args.output_dir / "endpoint_predictions.csv", index=False)
    graph.to_csv(args.output_dir / "graph_endpoint_predictions.csv", index=False)
    curves.to_csv(args.output_dir / "m_curve_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "graph_endpoint_metrics.csv", index=False)
    utility_graphs.to_csv(args.output_dir / "utility_robustness_graphs.csv", index=False)
    utility_curves.to_csv(args.output_dir / "utility_robustness_curves.csv", index=False)
    manifest = {
        "analysis_version": "ctou-clean-utility-v1",
        "run_root": str(args.run_root.resolve()),
        "states": list(STATES),
        "laws": list(LAWS),
        "primary_surrogate": "pooled_balanced + correlated_empirical",
        "particles": args.particles,
        "folds": args.folds,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "clean_integrity": clean_audit,
        "attack_update_integrity": attack_update_audit,
        "attack_case_integrity": attack_case_audit,
        "claim_limits": [
            "similar endpoint performance does not prove identical clean and attack laws",
            "shared-cell analysis controls exact categorical inputs but not message semantics",
            "the pooled model uses condition-balanced weights and no condition feature",
            "utility and robustness remain conditional on the current model/task/attack regime",
            "the empirical initializer is distribution-conditioned, not task-conditioned",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
