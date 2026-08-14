"""Recursively predict MAS endpoints from Round-0 C/T/O/U states only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_ctou_transition_prediction import (
    COUNT_COLUMNS,
    STATE_INDEX,
    STATES,
    design_matrix,
    load_updates,
    stable_fold,
    table_predictions,
)
from analyze_node_round_adoption import read_json, read_jsonl, trace_category
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

MODELS = ("persistence", "degroot_equal", "ctou_table", "ctou_logit")
ROLLOUT_MODES = ("particle", "mean_field")
DEFAULT_SEED = 20_260_815
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--particles", type=int, default=2_048)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--table-prior-strength", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def graph_maps(graph: dict[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    n = int(graph["node_count"])
    incoming = [[] for _ in range(n)]
    outgoing = [[] for _ in range(n)]
    for edge in graph["edges"]:
        source, target = int(edge["source"]), int(edge["target"])
        outgoing[source].append(target)
        incoming[target].append(source)
    return [sorted(x) for x in incoming], [sorted(x) for x in outgoing]


def distances_to_readout(graph: dict[str, Any]) -> list[int]:
    _, outgoing = graph_maps(graph)
    readout = int(graph["readout_node"])
    distances: list[int] = []
    for source in range(len(outgoing)):
        frontier = [(source, 0)]
        seen = {source}
        found = None
        for node, distance in frontier:
            if node == readout:
                found = distance
                break
            for target in outgoing[node]:
                if target not in seen:
                    seen.add(target)
                    frontier.append((target, distance + 1))
        if found is None:
            raise ValueError(f"node {source} cannot reach readout in {graph['graph_id']}")
        distances.append(found)
    return distances


def load_rollout_cases(
    run_root: Path,
    status: dict[str, Any],
    folds: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, object]]:
    cases: list[dict[str, object]] = []
    graph_lookup: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for descriptor in status["strata"]:
        stratum = str(descriptor["key"])
        root = run_root / "strata" / stratum
        graph_path = root / "selected_graphs.jsonl"
        if not graph_path.exists():
            graph_path = root / "batch/inputs/graphs.jsonl"
        graphs = {str(item["graph_id"]): item for item in read_jsonl(graph_path)}
        graph_lookup.update(graphs)
        tasks = {
            str(item["task_id"]): item for item in read_jsonl(root / "batch/inputs/tasks.jsonl")
        }
        trace_root = root / "batch/traces"
        for pair in read_jsonl(root / "analysis-v1/paired_attacks.jsonl"):
            task_id = str(pair["task_id"])
            graph_id = str(pair["graph_id"])
            attack_node = int(pair["attack_node"])
            trace_path = trace_root / f"{pair['attack_run_spec_id']}.json"
            if not trace_path.exists():
                errors.append(f"missing attack trace {trace_path}")
                continue
            stored = read_json(trace_path)
            trace = stored["trace"]
            graph = graphs[graph_id]
            task = tasks[task_id]
            reference = str(task["reference_answer"])
            target = str(pair["target_answer"])
            n = int(graph["node_count"])
            round_zero = {
                int(turn["node_id"]): turn
                for turn in trace["turns"]
                if int(turn["round_index"]) == 0
            }
            if set(round_zero) != set(range(n)):
                errors.append(f"incomplete Round-0 state for {pair['attack_run_spec_id']}")
                continue
            initial_states = tuple(
                STATE_INDEX[trace_category(round_zero[node], reference=reference, target=target)]
                for node in range(n)
            )
            if initial_states[attack_node] != STATE_INDEX["target"]:
                errors.append(f"attacker is not T at Round 0: {pair['attack_run_spec_id']}")
                continue
            final_state = trace_category(
                {
                    "answer_state": trace.get("final_answer_state"),
                    "parsed_answer": trace.get("final_parsed_answer"),
                },
                reference=reference,
                target=target,
            )
            cases.append(
                {
                    "stratum": stratum,
                    "task_id": task_id,
                    "graph_id": graph_id,
                    "attack_node": attack_node,
                    "n": n,
                    "m": len(graph["edges"]),
                    "readout_node": int(graph["readout_node"]),
                    "horizon": int(graph["max_rounds"]),
                    "initial_states": initial_states,
                    "actual_state": final_state,
                    "actual_state_index": STATE_INDEX[final_state],
                    "actual_target": int(final_state == "target"),
                    "actual_correct": int(final_state == "correct"),
                    "graph_fold": stable_fold(graph_id, folds),
                    "task_fold": stable_fold(task_id, folds),
                }
            )
    frame = pd.DataFrame(cases)
    duplicate_keys = int(frame.duplicated(["task_id", "graph_id", "attack_node"]).sum())
    if duplicate_keys:
        errors.append(f"duplicate rollout cases: {duplicate_keys}")
    audit = {
        "passed": not errors,
        "errors": errors[:100],
        "cases": len(frame),
        "tasks": int(frame.task_id.nunique()),
        "graphs": int(frame.graph_id.nunique()),
        "strata": int(frame.stratum.nunique()),
        "unique_initial_patterns": int(
            frame[["graph_id", "task_fold", "attack_node", "initial_states"]]
            .drop_duplicates()
            .shape[0]
        ),
    }
    return frame, graph_lookup, audit


def count_compositions(maximum: int) -> list[tuple[int, int, int, int]]:
    rows: list[tuple[int, int, int, int]] = []
    for total in range(maximum + 1):
        for c in range(total + 1):
            for t in range(total - c + 1):
                for o in range(total - c - t + 1):
                    u = total - c - t - o
                    rows.append((c, t, o, u))
    return rows


def query_frame(maximum_neighbors: int, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for round_index, previous, counts in itertools.product(
        range(1, horizon + 1), STATES, count_compositions(maximum_neighbors)
    ):
        row: dict[str, object] = {
            "previous_attack_state": previous,
            "round_index": round_index,
        }
        row.update(dict(zip(COUNT_COLUMNS, counts, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def dense_lookup(
    query: pd.DataFrame,
    probabilities: np.ndarray,
    horizon: int,
    maximum_neighbors: int,
) -> np.ndarray:
    lookup = np.full(
        (
            horizon + 1,
            len(STATES),
            *(maximum_neighbors + 1 for _ in STATES),
            len(STATES),
        ),
        np.nan,
        dtype=np.float32,
    )
    for position, row in enumerate(query.itertuples(index=False)):
        counts = tuple(int(getattr(row, column)) for column in COUNT_COLUMNS)
        index = (
            int(row.round_index),
            STATE_INDEX[str(row.previous_attack_state)],
            *counts,
        )
        lookup[index] = probabilities[position]
    return lookup


def fit_fold_lookups(
    train: pd.DataFrame,
    *,
    maximum_neighbors: int,
    horizon: int,
    table_prior_strength: float,
) -> dict[str, np.ndarray]:
    query = query_frame(maximum_neighbors, horizon)
    table = table_predictions(train, query, table_prior_strength)
    matrix = design_matrix(train)
    labels = train.current_state_index.to_numpy(int)
    logistic = LogisticRegression(
        C=1.0,
        max_iter=300,
        solver="lbfgs",
        random_state=0,
    )
    logistic.fit(matrix, labels)
    raw = logistic.predict_proba(design_matrix(query))
    aligned = np.zeros((len(query), len(STATES)), dtype=np.float32)
    aligned[:, logistic.classes_.astype(int)] = raw
    return {
        "ctou_table": dense_lookup(query, table, horizon, maximum_neighbors),
        "ctou_logit": dense_lookup(query, aligned, horizon, maximum_neighbors),
    }


def stable_seed(*parts: object, seed: int) -> int:
    text = "\x1f".join(str(part) for part in (*parts, seed))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sample_categories(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cumulative = probabilities.cumsum(axis=1)
    cumulative[:, -1] = 1.0
    return (uniforms[:, None] > cumulative).sum(axis=1).astype(np.int8)


def particle_rollout(
    *,
    graph: dict[str, Any],
    initial_states: tuple[int, ...],
    attack_node: int,
    model: str,
    lookup: np.ndarray | None,
    particles: int,
    seed: int,
) -> np.ndarray:
    states = np.tile(np.asarray(initial_states, dtype=np.int8), (particles, 1))
    return particle_rollout_from_particles(
        graph=graph,
        initial_particles=states,
        attack_node=attack_node,
        model=model,
        lookup=lookup,
        seed=seed,
    )


def particle_rollout_from_particles(
    *,
    graph: dict[str, Any],
    initial_particles: np.ndarray,
    attack_node: int | None,
    model: str,
    lookup: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    """Roll out a joint particle population with caller-supplied Round-0 states."""

    n = int(graph["node_count"])
    horizon = int(graph["max_rounds"])
    readout = int(graph["readout_node"])
    incoming, _ = graph_maps(graph)
    distances = distances_to_readout(graph)
    states = np.asarray(initial_particles, dtype=np.int8).copy()
    if states.ndim != 2 or states.shape[1] != n:
        raise ValueError(f"initial_particles must have shape (particles, {n})")
    if len(states) == 0:
        raise ValueError("initial_particles must not be empty")
    particles = len(states)
    if attack_node is not None:
        states[:, attack_node] = STATE_INDEX["target"]
    rng = np.random.default_rng(seed)
    for round_index in range(1, horizon + 1):
        updated = states.copy()
        for node in range(n):
            if round_index + distances[node] > horizon:
                continue
            if attack_node is not None and node == attack_node:
                updated[:, node] = STATE_INDEX["target"]
                continue
            if model == "persistence":
                continue
            sources = incoming[node]
            previous = states[:, node]
            if model == "degroot_equal":
                participants = states[:, [node, *sources]]
                counts = np.column_stack(
                    [(participants == state).sum(axis=1) for state in range(len(STATES))]
                )
                probabilities = counts / counts.sum(axis=1, keepdims=True)
            else:
                incoming_counts = np.column_stack(
                    [
                        (states[:, sources] == state).sum(axis=1)
                        if sources
                        else np.zeros(particles, dtype=int)
                        for state in range(len(STATES))
                    ]
                )
                probabilities = lookup[
                    round_index,
                    previous,
                    incoming_counts[:, 0],
                    incoming_counts[:, 1],
                    incoming_counts[:, 2],
                    incoming_counts[:, 3],
                ]
                if np.isnan(probabilities).any():
                    raise RuntimeError("missing transition probability in dense lookup")
            updated[:, node] = sample_categories(probabilities, rng.random(particles))
        states = updated
    return np.bincount(states[:, readout], minlength=len(STATES)) / particles


def composition_distribution(marginals: list[np.ndarray]) -> dict[tuple[int, ...], float]:
    distribution: dict[tuple[int, ...], float] = {(0, 0, 0, 0): 1.0}
    for marginal in marginals:
        updated: dict[tuple[int, ...], float] = {}
        for counts, mass in distribution.items():
            for state, probability in enumerate(marginal):
                next_counts = list(counts)
                next_counts[state] += 1
                key = tuple(next_counts)
                updated[key] = updated.get(key, 0.0) + mass * float(probability)
        distribution = updated
    return distribution


def mean_field_rollout(
    *,
    graph: dict[str, Any],
    initial_states: tuple[int, ...],
    attack_node: int | None,
    model: str,
    lookup: np.ndarray | None,
) -> np.ndarray:
    n = int(graph["node_count"])
    horizon = int(graph["max_rounds"])
    readout = int(graph["readout_node"])
    incoming, _ = graph_maps(graph)
    distances = distances_to_readout(graph)
    marginals = np.eye(len(STATES), dtype=float)[np.asarray(initial_states)]
    for round_index in range(1, horizon + 1):
        updated = marginals.copy()
        for node in range(n):
            if round_index + distances[node] > horizon:
                continue
            if attack_node is not None and node == attack_node:
                updated[node] = np.eye(len(STATES))[STATE_INDEX["target"]]
                continue
            if model == "persistence":
                continue
            sources = incoming[node]
            if model == "degroot_equal":
                updated[node] = marginals[[node, *sources]].mean(axis=0)
                continue
            result = np.zeros(len(STATES), dtype=float)
            compositions = composition_distribution([marginals[source] for source in sources])
            for previous, previous_mass in enumerate(marginals[node]):
                if previous_mass == 0:
                    continue
                for counts, composition_mass in compositions.items():
                    index = (round_index, previous, *counts)
                    result += previous_mass * composition_mass * lookup[index]
            updated[node] = result / result.sum()
        marginals = updated
    return marginals[readout]


def execute_rollouts(
    cases: pd.DataFrame,
    updates: pd.DataFrame,
    graphs: dict[str, dict[str, Any]],
    *,
    folds: int,
    particles: int,
    table_prior_strength: float,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[Path] = []
    audit_rows: list[dict[str, object]] = []
    maximum_neighbors = int(cases.n.max() - 1)
    horizon = int(cases.horizon.max())
    for graph_fold in range(folds):
        for task_fold in range(folds):
            checkpoint_path = checkpoint_dir / f"graph-{graph_fold}_task-{task_fold}.csv"
            checkpoint_paths.append(checkpoint_path)
            train = updates[updates.graph_fold.ne(graph_fold) & updates.task_fold.ne(task_fold)]
            test = cases[cases.graph_fold.eq(graph_fold) & cases.task_fold.eq(task_fold)]
            if test.empty:
                continue
            print(
                f"fold graph={graph_fold} task={task_fold}: "
                f"train_updates={len(train)} test_cases={len(test)}",
                flush=True,
            )
            train_graphs = set(train.graph_id)
            test_graphs = set(test.graph_id)
            train_tasks = set(train.task_id)
            test_tasks = set(test.task_id)
            audit_rows.append(
                {
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "train_rows": len(train),
                    "test_cases": len(test),
                    "train_graphs": len(train_graphs),
                    "test_graphs": len(test_graphs),
                    "train_tasks": len(train_tasks),
                    "test_tasks": len(test_tasks),
                    "graph_overlap": len(train_graphs & test_graphs),
                    "task_overlap": len(train_tasks & test_tasks),
                }
            )
            if checkpoint_path.exists():
                print(
                    f"fold graph={graph_fold} task={task_fold}: resume from checkpoint",
                    flush=True,
                )
                del train, test
                continue
            learned = fit_fold_lookups(
                train,
                maximum_neighbors=maximum_neighbors,
                horizon=horizon,
                table_prior_strength=table_prior_strength,
            )
            unique = test.drop_duplicates(
                ["graph_id", "task_fold", "attack_node", "initial_states"]
            )
            print(f"  unique rollout inputs={len(unique)}", flush=True)
            cache: dict[tuple[object, ...], np.ndarray] = {}
            for case in unique.itertuples(index=False):
                common_seed = stable_seed(
                    case.graph_id,
                    case.task_fold,
                    case.attack_node,
                    case.initial_states,
                    seed=seed,
                )
                graph = graphs[case.graph_id]
                for model in MODELS:
                    model_lookup = learned.get(model)
                    particle_key = (
                        model,
                        "particle",
                        case.graph_id,
                        case.attack_node,
                        case.initial_states,
                    )
                    cache[particle_key] = particle_rollout(
                        graph=graph,
                        initial_states=case.initial_states,
                        attack_node=case.attack_node,
                        model=model,
                        lookup=model_lookup,
                        particles=particles,
                        seed=common_seed,
                    )
                    mean_field_key = (
                        model,
                        "mean_field",
                        case.graph_id,
                        case.attack_node,
                        case.initial_states,
                    )
                    cache[mean_field_key] = mean_field_rollout(
                        graph=graph,
                        initial_states=case.initial_states,
                        attack_node=case.attack_node,
                        model=model,
                        lookup=model_lookup,
                    )
            fold_prediction_rows: list[dict[str, object]] = []
            for case in test.itertuples(index=False):
                base = {
                    "stratum": case.stratum,
                    "task_id": case.task_id,
                    "graph_id": case.graph_id,
                    "attack_node": case.attack_node,
                    "n": case.n,
                    "m": case.m,
                    "graph_fold": graph_fold,
                    "task_fold": task_fold,
                    "actual_state": case.actual_state,
                    "actual_state_index": case.actual_state_index,
                    "actual_target": case.actual_target,
                    "actual_correct": case.actual_correct,
                }
                for model, mode in itertools.product(MODELS, ROLLOUT_MODES):
                    probability = cache[
                        (model, mode, case.graph_id, case.attack_node, case.initial_states)
                    ]
                    fold_prediction_rows.append(
                        {
                            **base,
                            "model": model,
                            "rollout_mode": mode,
                            **{
                                f"p_{state}": float(probability[index])
                                for index, state in enumerate(STATES)
                            },
                        }
                    )
            # Persist each crossed-holdout cell atomically. Besides enabling resume,
            # this bounds Python-object memory instead of retaining ~300k dictionaries.
            temporary = checkpoint_path.with_suffix(".csv.tmp")
            pd.DataFrame(fold_prediction_rows).to_csv(temporary, index=False)
            temporary.replace(checkpoint_path)
            print(
                f"  checkpoint rows={len(fold_prediction_rows)} path={checkpoint_path.name}",
                flush=True,
            )
            del fold_prediction_rows, cache, learned, train, test, unique
            gc.collect()
    missing = [path for path in checkpoint_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing rollout checkpoints: {missing}")
    predictions = pd.concat(
        (pd.read_csv(path) for path in checkpoint_paths),
        ignore_index=True,
    )
    return predictions, pd.DataFrame(audit_rows)


def attach_losses(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    probability = result[[f"p_{state}" for state in STATES]].to_numpy(float)
    labels = result.actual_state_index.to_numpy(int)
    one_hot = np.eye(len(STATES))[labels]
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    result["multiclass_brier"] = ((probability - one_hot) ** 2).sum(axis=1)
    result["multiclass_log_loss"] = -np.log(clipped[np.arange(len(result)), labels])
    for outcome in ("target", "correct"):
        observed = result[f"actual_{outcome}"].to_numpy(float)
        predicted = result[f"p_{outcome}"].to_numpy(float)
        clipped_binary = np.clip(predicted, EPSILON, 1.0 - EPSILON)
        result[f"{outcome}_brier"] = (predicted - observed) ** 2
        result[f"{outcome}_log_loss"] = -(
            observed * np.log(clipped_binary) + (1.0 - observed) * np.log(1.0 - clipped_binary)
        )
    return result


def bootstrap_task_summary(
    frame: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> pd.DataFrame:
    metrics = (
        "multiclass_brier",
        "multiclass_log_loss",
        "target_brier",
        "target_log_loss",
        "correct_brier",
        "correct_log_loss",
    )
    task = (
        frame.groupby(["model", "rollout_mode", "task_id"], sort=False)[list(metrics)]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for (model, mode), group in task.groupby(["model", "rollout_mode"], sort=False):
        values = group[list(metrics)].to_numpy(float)
        sample = rng.integers(0, len(values), size=(replicates, len(values)))
        bootstrap = values[sample].mean(axis=1)
        estimates = values.mean(axis=0)
        for index, metric in enumerate(metrics):
            low, high = np.quantile(bootstrap[:, index], [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    "rollout_mode": mode,
                    "metric": metric,
                    "estimate": estimates[index],
                    "ci95_low": low,
                    "ci95_high": high,
                    "tasks": len(values),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(
    frame: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> pd.DataFrame:
    metrics = (
        "multiclass_brier",
        "multiclass_log_loss",
        "target_brier",
        "target_log_loss",
        "correct_brier",
        "correct_log_loss",
    )
    task = (
        frame.groupby(["model", "rollout_mode", "task_id"], sort=False)[list(metrics)]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for mode in ROLLOUT_MODES:
        for reference_name in ("degroot_equal", "persistence"):
            reference = task[task.model.eq(reference_name) & task.rollout_mode.eq(mode)].set_index(
                "task_id"
            )
            for candidate in ("ctou_table", "ctou_logit"):
                selected = task[task.model.eq(candidate) & task.rollout_mode.eq(mode)].set_index(
                    "task_id"
                )
                common = sorted(set(reference.index) & set(selected.index))
                for metric in metrics:
                    differences = selected.loc[common, metric].to_numpy(float) - reference.loc[
                        common, metric
                    ].to_numpy(float)
                    sample = rng.integers(0, len(differences), size=(replicates, len(differences)))
                    bootstrap = differences[sample].mean(axis=1)
                    low, high = np.quantile(bootstrap, [0.025, 0.975])
                    rows.append(
                        {
                            "candidate": candidate,
                            "reference": reference_name,
                            "rollout_mode": mode,
                            "metric": metric,
                            "loss_difference": differences.mean(),
                            "ci95_low": low,
                            "ci95_high": high,
                            "negative_favors_candidate": True,
                            "tasks": len(differences),
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_curves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = (
        frame.groupby(["model", "rollout_mode", "n", "m"], sort=False)
        .agg(
            cases=("actual_target", "size"),
            observed_target=("actual_target", "mean"),
            predicted_target=("p_target", "mean"),
            observed_correct=("actual_correct", "mean"),
            predicted_correct=("p_correct", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for (model, mode, n), group in curves.groupby(["model", "rollout_mode", "n"], sort=False):
        for outcome in ("target", "correct"):
            observed = group[f"observed_{outcome}"].to_numpy(float)
            predicted = group[f"predicted_{outcome}"].to_numpy(float)
            correlation = spearmanr(observed, predicted).statistic
            rows.append(
                {
                    "model": model,
                    "rollout_mode": mode,
                    "n": n,
                    "outcome": outcome,
                    "m_levels": len(group),
                    "curve_mae": np.abs(predicted - observed).mean(),
                    "curve_spearman": correlation,
                }
            )
    return curves, pd.DataFrame(rows)


def bootstrap_spearman(
    observed: np.ndarray,
    predicted: np.ndarray,
    sample: np.ndarray,
) -> np.ndarray:
    observed_ranks = rankdata(observed[sample], axis=1)
    predicted_ranks = rankdata(predicted[sample], axis=1)
    observed_centered = observed_ranks - observed_ranks.mean(axis=1, keepdims=True)
    predicted_centered = predicted_ranks - predicted_ranks.mean(axis=1, keepdims=True)
    denominator = np.sqrt((observed_centered**2).sum(axis=1) * (predicted_centered**2).sum(axis=1))
    numerator = (observed_centered * predicted_centered).sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def aggregate_graphs(
    frame: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    graphs = (
        frame.groupby(["model", "rollout_mode", "graph_id", "n", "m"], sort=False)
        .agg(
            cases=("actual_target", "size"),
            observed_target=("actual_target", "mean"),
            predicted_target=("p_target", "mean"),
            observed_correct=("actual_correct", "mean"),
            predicted_correct=("p_correct", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for (model, mode, n), group in graphs.groupby(["model", "rollout_mode", "n"], sort=False):
        for outcome in ("target", "correct"):
            observed = group[f"observed_{outcome}"].to_numpy(float)
            predicted = group[f"predicted_{outcome}"].to_numpy(float)
            sample = rng.integers(0, len(group), size=(replicates, len(group)))
            bootstrap_mae = np.abs(predicted[sample] - observed[sample]).mean(axis=1)
            bootstrap_correlation = bootstrap_spearman(observed, predicted, sample)
            mae_low, mae_high = np.quantile(bootstrap_mae, [0.025, 0.975])
            valid_correlation = bootstrap_correlation[~np.isnan(bootstrap_correlation)]
            correlation_low, correlation_high = np.quantile(valid_correlation, [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    "rollout_mode": mode,
                    "n": n,
                    "outcome": outcome,
                    "graphs": len(group),
                    "graph_mae": np.abs(predicted - observed).mean(),
                    "graph_mae_ci95_low": mae_low,
                    "graph_mae_ci95_high": mae_high,
                    "graph_spearman": spearmanr(observed, predicted).statistic,
                    "graph_spearman_ci95_low": correlation_low,
                    "graph_spearman_ci95_high": correlation_high,
                }
            )
    return graphs, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.particles < 256:
        raise ValueError("particles must be at least 256")
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = read_json(args.run_root / "orchestrator_status.json")
    updates, update_audit = load_updates(args.run_root, args.folds)
    print(f"loaded updates: {json.dumps(update_audit, ensure_ascii=False)}", flush=True)
    case_cache = args.output_dir / "normalized_rollout_cases.pkl"
    graph_cache = args.output_dir / "normalized_graphs.json"
    audit_cache = args.output_dir / "normalized_case_audit.json"
    if case_cache.exists() and graph_cache.exists() and audit_cache.exists():
        cases = pd.read_pickle(case_cache)
        graphs = json.loads(graph_cache.read_text(encoding="utf-8"))
        case_audit = read_json(audit_cache)
        print("loaded normalized rollout cases from checkpoint", flush=True)
    else:
        cases, graphs, case_audit = load_rollout_cases(args.run_root, status, args.folds)
        case_temporary = case_cache.with_suffix(".pkl.tmp")
        graph_temporary = graph_cache.with_suffix(".json.tmp")
        audit_temporary = audit_cache.with_suffix(".json.tmp")
        cases.to_pickle(case_temporary)
        graph_temporary.write_text(json.dumps(graphs, ensure_ascii=False), encoding="utf-8")
        audit_temporary.write_text(
            json.dumps(case_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        case_temporary.replace(case_cache)
        graph_temporary.replace(graph_cache)
        audit_temporary.replace(audit_cache)
    print(f"loaded cases: {json.dumps(case_audit, ensure_ascii=False)}", flush=True)
    if not update_audit["passed"] or not case_audit["passed"]:
        raise RuntimeError("input integrity audit failed")
    predictions, folds = execute_rollouts(
        cases,
        updates,
        graphs,
        folds=args.folds,
        particles=args.particles,
        table_prior_strength=args.table_prior_strength,
        seed=args.seed,
        checkpoint_dir=args.output_dir / "fold-checkpoints",
    )
    if (folds.graph_overlap != 0).any() or (folds.task_overlap != 0).any():
        raise RuntimeError("crossed holdout leakage detected")
    scored = attach_losses(predictions)
    rng = np.random.default_rng(args.seed)
    summary = bootstrap_task_summary(scored, args.bootstrap_replicates, rng)
    comparisons = paired_comparisons(scored, args.bootstrap_replicates, rng)
    curves, curve_metrics = aggregate_curves(scored)
    graph_predictions, graph_metrics = aggregate_graphs(
        scored,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )
    folds.to_csv(args.output_dir / "fold_audit.csv", index=False)
    scored.to_csv(args.output_dir / "endpoint_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "endpoint_loss_summary.csv", index=False)
    comparisons.to_csv(args.output_dir / "model_comparisons_vs_baselines.csv", index=False)
    comparisons[comparisons.reference.eq("degroot_equal")].to_csv(
        args.output_dir / "model_comparisons_vs_degroot.csv", index=False
    )
    curves.to_csv(args.output_dir / "m_curve_predictions.csv", index=False)
    curve_metrics.to_csv(args.output_dir / "m_curve_metrics.csv", index=False)
    graph_predictions.to_csv(args.output_dir / "graph_endpoint_predictions.csv", index=False)
    graph_metrics.to_csv(args.output_dir / "graph_endpoint_metrics.csv", index=False)
    manifest = {
        "analysis_version": "ctou-recursive-rollout-v1",
        "run_root": str(args.run_root.resolve()),
        "states": list(STATES),
        "models": list(MODELS),
        "rollout_modes": list(ROLLOUT_MODES),
        "particles": args.particles,
        "folds": args.folds,
        "table_prior_strength": args.table_prior_strength,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "information_boundary": (
            "observed Round-0 categorical states plus graph, attack node, and schedule; "
            "no observed Round-1+ states, compositions, answers, or text"
        ),
        "update_integrity": update_audit,
        "case_integrity": case_audit,
        "claim_limits": [
            "Round-0 states are observed, so this is not topology-only prediction",
            "particle rollout approximates the induced joint distribution",
            "mean-field rollout assumes factorized node marginals",
            "successful prediction does not prove the LLM implements the fitted law",
            "failure is not by itself evidence of a semantic mechanism",
            "bootstrap is conditional on sampled graphs and one model/dataset configuration",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
