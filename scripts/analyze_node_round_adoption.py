"""Decompose target-error delivery and benign-node adoption at each active update."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANALYSIS_VERSION = "node-round-adoption-v1"
GRAPH_FOLDS = 5
TASK_FOLDS = 5
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_807
EPSILON = 1e-7
SUBSETS = ("all_updates", "received_target", "received_induced_target")
MODEL_NAMES = (
    "intercept_only",
    "degroot_exposure",
    "target_exposure",
    "neighborhood_state",
    "all_content_free_linear",
    "all_content_free_hgb",
)
STATE_NAMES = ("correct", "target", "other", "unparsed")
TRACE_STATE_NAMES = {
    "correct": "correct",
    "target_error": "target",
    "other_error": "other",
    "unparsed": "unparsed",
}

DEGROOT_FEATURES = ("degroot_receiver_target_mass",)
TARGET_FEATURES = (
    "incoming_target_count",
    "incoming_target_fraction",
    "incoming_induced_target_count",
    "incoming_induced_target_fraction",
)
NEIGHBORHOOD_FEATURES = (
    *TARGET_FEATURES,
    *(f"receiver_previous_{name}" for name in STATE_NAMES),
    *(f"incoming_{name}_count" for name in STATE_NAMES),
    *(f"incoming_{name}_fraction" for name in STATE_NAMES),
    "incoming_distinct_answer_count",
    "incoming_unique_plurality",
)
STATIC_FEATURES = (
    "round_index",
    "n",
    "m",
    "receiver_indegree",
    "receiver_outdegree",
    "receiver_distance_to_readout",
    "receiver_is_readout",
    "attacker_distance_to_readout",
    "attacker_distance_to_receiver",
    "attacker_directly_incoming",
)
ALL_FEATURES = (*DEGROOT_FEATURES, *NEIGHBORHOOD_FEATURES, *STATIC_FEATURES)
MODEL_FEATURES = {
    "intercept_only": (),
    "degroot_exposure": DEGROOT_FEATURES,
    "target_exposure": TARGET_FEATURES,
    "neighborhood_state": NEIGHBORHOOD_FEATURES,
    "all_content_free_linear": ALL_FEATURES,
    "all_content_free_hgb": ALL_FEATURES,
}


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


def normalized(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def category(value: Any, *, reference: str, target: str) -> str:
    parsed = normalized(value)
    if parsed is None:
        return "unparsed"
    if parsed == normalized(reference):
        return "correct"
    if parsed == normalized(target):
        return "target"
    return "other"


def trace_category(record: dict[str, Any], *, reference: str, target: str) -> str:
    """Use execution-time oracle state, falling back for legacy traces."""
    answer_state = record.get("answer_state")
    if answer_state in TRACE_STATE_NAMES:
        return TRACE_STATE_NAMES[str(answer_state)]
    return category(record.get("parsed_answer"), reference=reference, target=target)


def graph_maps(graph: dict[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    n = int(graph["node_count"])
    incoming = [[] for _ in range(n)]
    outgoing = [[] for _ in range(n)]
    for edge in graph["edges"]:
        source, target = int(edge["source"]), int(edge["target"])
        outgoing[source].append(target)
        incoming[target].append(source)
    return [sorted(x) for x in incoming], [sorted(x) for x in outgoing]


def shortest_distance(outgoing: list[list[int]], source: int, target: int) -> int:
    queue: deque[tuple[int, int]] = deque([(source, 0)])
    seen = {source}
    while queue:
        node, distance = queue.popleft()
        if node == target:
            return distance
        for neighbor in outgoing[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return len(outgoing) + 1


def degroot_target_masses(
    graph: dict[str, Any],
    round_zero: dict[int, dict[str, Any]],
    *,
    attack_node: int,
    target: str,
) -> dict[tuple[int, int], float]:
    n = int(graph["node_count"])
    rounds = int(graph["max_rounds"])
    incoming, _ = graph_maps(graph)
    values = np.array(
        [
            float(
                trace_category(round_zero[node], reference="__never__", target=target) == "target"
            )
            for node in range(n)
        ],
        dtype=float,
    )
    values[attack_node] = 1.0
    output: dict[tuple[int, int], float] = {}
    for round_index in range(rounds + 1):
        for node in range(n):
            output[(node, round_index)] = float(values[node])
        updated = values.copy()
        for node, sources in enumerate(incoming):
            participants = [node, *sources]
            updated[node] = float(values[participants].mean())
        updated[attack_node] = 1.0
        values = updated
    return output


def paired_trace_rows(
    *,
    pair: dict[str, Any],
    graph: dict[str, Any],
    task: dict[str, Any],
    clean_stored: dict[str, Any],
    attack_stored: dict[str, Any],
    stratum: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    clean = clean_stored["trace"]
    attack = attack_stored["trace"]
    attack_node = int(pair["attack_node"])
    target = str(pair["target_answer"])
    reference = str(task["reference_answer"])
    readout = int(graph["readout_node"])
    n = int(graph["node_count"])
    incoming, outgoing = graph_maps(graph)

    paired_fields = (
        "task_id",
        "graph_id",
        "seed",
        "prompt_version",
        "initial_assignment_id",
        "initial_assignment_seed",
        "structural_node_to_replica",
    )
    for field in paired_fields:
        if clean.get(field) != attack.get(field):
            errors.append(f"{pair['attack_run_spec_id']}: paired {field} mismatch")
    clean_turns = {(int(x["node_id"]), int(x["round_index"])): x for x in clean["turns"]}
    attack_turns = {(int(x["node_id"]), int(x["round_index"])): x for x in attack["turns"]}
    if clean_turns.keys() != attack_turns.keys():
        errors.append(f"{pair['attack_run_spec_id']}: paired schedule mismatch")
        return [], errors
    clean_messages = {(int(x["sender"]), int(x["round_index"])): x for x in clean["messages"]}
    attack_messages_by_id = {x["message_id"]: x for x in attack["messages"]}
    round_zero = {node: attack_turns[(node, 0)] for node in range(n)}
    degroot = degroot_target_masses(graph, round_zero, attack_node=attack_node, target=target)
    attacker_to_readout = shortest_distance(outgoing, attack_node, readout)
    rows: list[dict[str, Any]] = []

    for (receiver, round_index), attack_turn in sorted(attack_turns.items()):
        if round_index == 0 or receiver == attack_node:
            continue
        clean_turn = clean_turns[(receiver, round_index)]
        previous_attack = attack_turns[(receiver, round_index - 1)]
        previous_clean = clean_turns[(receiver, round_index - 1)]
        incoming_attack = []
        incoming_clean = []
        for message_id in attack_turn["incoming_message_ids"]:
            message = attack_messages_by_id.get(message_id)
            if message is None:
                errors.append(
                    f"{pair['attack_run_spec_id']}: missing incoming message {message_id}"
                )
                continue
            sender = int(message["sender"])
            message_round = int(message["round_index"])
            if sender not in incoming[receiver] or message_round != round_index - 1:
                errors.append(f"{pair['attack_run_spec_id']}: invalid synchronous incoming message")
            clean_message = clean_messages.get((sender, message_round))
            if clean_message is None:
                errors.append(f"{pair['attack_run_spec_id']}: missing paired clean message")
                continue
            incoming_attack.append(message)
            incoming_clean.append(clean_message)
        attack_categories = [
            trace_category(x, reference=reference, target=target) for x in incoming_attack
        ]
        clean_categories = [
            trace_category(x, reference=reference, target=target) for x in incoming_clean
        ]
        incoming_count = len(attack_categories)
        state_counts = Counter(attack_categories)
        target_count = state_counts["target"]
        induced_count = sum(
            attack_value == "target" and clean_value != "target"
            for attack_value, clean_value in zip(attack_categories, clean_categories, strict=True)
        )
        parsed_values = [normalized(x["parsed_answer"]) for x in incoming_attack]
        parsed_counts = Counter(parsed_values)
        plurality = max(parsed_counts.values()) if parsed_counts else 0
        unique_plurality = bool(
            parsed_counts and list(parsed_counts.values()).count(plurality) == 1
        )

        current_attack_state = trace_category(attack_turn, reference=reference, target=target)
        current_clean_state = trace_category(clean_turn, reference=reference, target=target)
        previous_attack_state = trace_category(previous_attack, reference=reference, target=target)
        previous_clean_state = trace_category(previous_clean, reference=reference, target=target)
        now_attack_target = current_attack_state == "target"
        now_clean_target = current_clean_state == "target"
        previous_attack_target = previous_attack_state == "target"
        previous_clean_target = previous_clean_state == "target"
        induced_now = now_attack_target and not now_clean_target
        induced_previous = previous_attack_target and not previous_clean_target

        receiver_distance = shortest_distance(outgoing, receiver, readout)
        attacker_receiver_distance = shortest_distance(outgoing, attack_node, receiver)
        row = {
            "stratum": stratum,
            "task_id": str(pair["task_id"]),
            "graph_id": str(pair["graph_id"]),
            "attack_node": attack_node,
            "receiver_node": receiver,
            "round_index": round_index,
            "outcome": int(induced_now and not induced_previous),
            "induced_target_state": int(induced_now),
            "induced_target_recovery": int(induced_previous and not induced_now),
            "current_attack_state": current_attack_state,
            "current_clean_state": current_clean_state,
            "previous_attack_state": previous_attack_state,
            "previous_clean_state": previous_clean_state,
            "previous_induced_target_state": int(induced_previous),
            "received_target": int(target_count > 0),
            "received_induced_target": int(induced_count > 0),
            "incoming_target_count": target_count,
            "incoming_target_fraction": target_count / incoming_count if incoming_count else 0.0,
            "incoming_induced_target_count": induced_count,
            "incoming_induced_target_fraction": (
                induced_count / incoming_count if incoming_count else 0.0
            ),
            "incoming_distinct_answer_count": len(parsed_counts),
            "incoming_unique_plurality": float(unique_plurality),
            "degroot_receiver_target_mass": degroot[(receiver, round_index)],
            "n": n,
            "m": len(graph["edges"]),
            "receiver_indegree": len(incoming[receiver]),
            "receiver_outdegree": len(outgoing[receiver]),
            "receiver_distance_to_readout": receiver_distance,
            "receiver_is_readout": float(receiver == readout),
            "attacker_distance_to_readout": attacker_to_readout,
            "attacker_distance_to_receiver": attacker_receiver_distance,
            "attacker_directly_incoming": float(attack_node in incoming[receiver]),
            "graph_depth": max(shortest_distance(outgoing, node, readout) for node in range(n)),
        }
        for name in STATE_NAMES:
            row[f"receiver_previous_{name}"] = float(previous_attack_state == name)
            row[f"incoming_{name}_count"] = state_counts[name]
            row[f"incoming_{name}_fraction"] = (
                state_counts[name] / incoming_count if incoming_count else 0.0
            )
        rows.append(row)
    return rows, errors


def extract_updates(
    run_root: Path,
    status: dict[str, Any],
    *,
    graph_ids: set[str] | None = None,
    task_ids: set[str] | None = None,
    clean_cache_size: int = 32,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    paired_conditions = 0
    for descriptor in status["strata"]:
        stratum = str(descriptor["key"])
        root = run_root / "strata" / stratum
        graph_path = root / "selected_graphs.jsonl"
        if not graph_path.exists():
            graph_path = root / "batch" / "inputs" / "graphs.jsonl"
        graphs = {str(x["graph_id"]): x for x in read_jsonl(graph_path)}
        tasks = {str(x["task_id"]): x for x in read_jsonl(root / "batch/inputs/tasks.jsonl")}
        trace_root = root / "batch/traces"
        clean_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for pair in read_jsonl(root / "analysis-v1/paired_attacks.jsonl"):
            if graph_ids is not None and str(pair["graph_id"]) not in graph_ids:
                continue
            if task_ids is not None and str(pair["task_id"]) not in task_ids:
                continue
            paired_conditions += 1
            clean_id = str(pair["clean_run_spec_id"])
            clean_path = trace_root / f"{clean_id}.json"
            attack_path = trace_root / f"{pair['attack_run_spec_id']}.json"
            if not clean_path.exists() or not attack_path.exists():
                errors.append(f"missing trace for {pair['attack_run_spec_id']}")
                continue
            if clean_id not in clean_cache:
                clean_cache[clean_id] = read_json(clean_path)
                if len(clean_cache) > clean_cache_size:
                    clean_cache.popitem(last=False)
            else:
                clean_cache.move_to_end(clean_id)
            extracted, pair_errors = paired_trace_rows(
                pair=pair,
                graph=graphs[str(pair["graph_id"])],
                task=tasks[str(pair["task_id"])],
                clean_stored=clean_cache[clean_id],
                attack_stored=read_json(attack_path),
                stratum=stratum,
            )
            rows.extend(extracted)
            errors.extend(pair_errors)
    frame = pd.DataFrame(rows)
    duplicate_keys = int(
        frame.duplicated(
            ["task_id", "graph_id", "attack_node", "receiver_node", "round_index"]
        ).sum()
    )
    if duplicate_keys:
        errors.append(f"duplicate update keys: {duplicate_keys}")
    audit = {
        "passed": not errors,
        "errors": errors[:100],
        "paired_conditions": paired_conditions,
        "eligible_updates": len(frame),
        "tasks": int(frame["task_id"].nunique()),
        "graphs": int(frame["graph_id"].nunique()),
        "new_induced_adoptions": int(frame["outcome"].sum()),
        "updates_receiving_target": int(frame["received_target"].sum()),
        "updates_receiving_induced_target": int(frame["received_induced_target"].sum()),
        "duplicate_update_keys": duplicate_keys,
    }
    return frame, audit


def assign_folds(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph_rows = []
    for stratum, group in frame[["stratum", "graph_id"]].drop_duplicates().groupby("stratum"):
        for position, graph_id in enumerate(sorted(group["graph_id"])):
            graph_rows.append(
                {"stratum": stratum, "graph_id": graph_id, "graph_fold": position % GRAPH_FOLDS}
            )
    task_rows = [
        {"task_id": task_id, "task_fold": position % TASK_FOLDS}
        for position, task_id in enumerate(sorted(frame["task_id"].unique()))
    ]
    assigned = frame.merge(
        pd.DataFrame(graph_rows),
        on=["stratum", "graph_id"],
        validate="many_to_one",
    )
    assigned = assigned.merge(pd.DataFrame(task_rows), on="task_id", validate="many_to_one")
    fold_map = assigned[
        ["stratum", "graph_id", "graph_fold", "task_id", "task_fold"]
    ].drop_duplicates()
    return assigned, fold_map


def subset_frame(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all_updates":
        return frame
    if subset == "received_target":
        return frame.loc[frame["received_target"] == 1]
    if subset == "received_induced_target":
        return frame.loc[frame["received_induced_target"] == 1]
    raise ValueError(subset)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, model_name: str) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prevalence = float(train["outcome"].mean()) if len(train) else 0.0
    features = list(MODEL_FEATURES[model_name])
    if not features or train["outcome"].nunique() < 2:
        return np.full(len(test), prevalence)
    if model_name == "all_content_free_hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=DEFAULT_SEED,
        )
    else:
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000)
        )
    model.fit(train[features], train["outcome"])
    return model.predict_proba(test[features])[:, 1]


def crossed_predictions(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    identifiers = [
        "stratum",
        "task_id",
        "graph_id",
        "attack_node",
        "receiver_node",
        "round_index",
        "graph_fold",
        "task_fold",
        "outcome",
    ]
    for subset in SUBSETS:
        selected = subset_frame(assignments, subset)
        for graph_fold in range(GRAPH_FOLDS):
            for task_fold in range(TASK_FOLDS):
                test = selected.loc[
                    (selected["graph_fold"] == graph_fold) & (selected["task_fold"] == task_fold)
                ]
                train = selected.loc[
                    (selected["graph_fold"] != graph_fold) & (selected["task_fold"] != task_fold)
                ]
                if test.empty:
                    continue
                graph_overlap = set(train["graph_id"]) & set(test["graph_id"])
                task_overlap = set(train["task_id"]) & set(test["task_id"])
                if graph_overlap or task_overlap:
                    raise RuntimeError("graph/task leakage in crossed fold")
                audits.append(
                    {
                        "subset": subset,
                        "graph_fold": graph_fold,
                        "task_fold": task_fold,
                        "training_rows": len(train),
                        "test_rows": len(test),
                        "training_prevalence": float(train["outcome"].mean()),
                        "test_prevalence": float(test["outcome"].mean()),
                        "graph_overlap": len(graph_overlap),
                        "task_overlap": len(task_overlap),
                    }
                )
                for model_name in MODEL_NAMES:
                    result = test[identifiers].copy()
                    result["subset"] = subset
                    result["model"] = model_name
                    result["probability"] = np.clip(
                        fit_predict(train, test, model_name), EPSILON, 1.0 - EPSILON
                    )
                    outputs.append(result)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audits)


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    rows = []
    for (subset, model), group in predictions.groupby(["subset", "model"], sort=True):
        outcome = group["outcome"].to_numpy(dtype=int)
        probability = group["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "subset": subset,
                "model": model,
                "rows": len(group),
                "positives": int(outcome.sum()),
                "prevalence": float(outcome.mean()),
                "brier": float(brier_score_loss(outcome, probability)),
                "log_loss": float(log_loss(outcome, probability, labels=[0, 1])),
                "average_precision": float(average_precision_score(outcome, probability)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_comparisons(
    predictions: pd.DataFrame, *, replicates: int, rng: np.random.Generator
) -> pd.DataFrame:
    reference = "degroot_exposure"
    rows: list[dict[str, Any]] = []
    squared = predictions.assign(
        squared_error=(predictions["outcome"] - predictions["probability"]) ** 2
    )
    for subset in SUBSETS:
        selected = squared.loc[squared["subset"] == subset]
        task_ids = np.array(sorted(selected["task_id"].unique()))
        graph_table = selected[["stratum", "graph_id"]].drop_duplicates()
        graph_groups = {
            stratum: np.array(sorted(group["graph_id"]))
            for stratum, group in graph_table.groupby("stratum")
        }
        losses = (
            selected.groupby(["model", "task_id", "graph_id"], sort=False)["squared_error"]
            .mean()
            .unstack("model")
        )
        for candidate in MODEL_NAMES:
            if candidate == reference:
                continue
            delta = (losses[reference] - losses[candidate]).rename("improvement").reset_index()
            task_index = {value: index for index, value in enumerate(task_ids)}
            graph_ids = np.array(sorted(delta["graph_id"].unique()))
            graph_index = {value: index for index, value in enumerate(graph_ids)}
            matrix = np.full((len(graph_ids), len(task_ids)), np.nan)
            for item in delta.itertuples(index=False):
                matrix[graph_index[item.graph_id], task_index[item.task_id]] = item.improvement
            draws = np.empty(replicates)
            for replicate in range(replicates):
                task_weights = np.bincount(
                    rng.integers(0, len(task_ids), len(task_ids)), minlength=len(task_ids)
                )
                graph_weights = np.zeros(len(graph_ids), dtype=int)
                for values in graph_groups.values():
                    sampled = rng.choice(values, size=len(values), replace=True)
                    for value in sampled:
                        graph_weights[graph_index[value]] += 1
                weights = np.outer(graph_weights, task_weights)
                valid = ~np.isnan(matrix)
                draws[replicate] = np.nansum(matrix * weights) / np.sum(weights * valid)
            point = float(delta["improvement"].mean())
            rows.append(
                {
                    "subset": subset,
                    "candidate": candidate,
                    "reference": reference,
                    "brier_improvement": point,
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_probability_positive": float(np.mean(draws > 0)),
                }
            )
    return pd.DataFrame(rows)


def exposure_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["round_index", "incoming_target_count", "incoming_induced_target_count"],
            sort=True,
        )
        .agg(
            updates=("outcome", "size"),
            new_induced_adoptions=("outcome", "sum"),
            adoption_rate=("outcome", "mean"),
            induced_state_rate=("induced_target_state", "mean"),
            recovery_rate=("induced_target_recovery", "mean"),
        )
        .reset_index()
    )


def render_report(audit: dict[str, Any], metrics: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    lines = [
        "# Node-round exposure and adoption",
        "",
        "## Integrity",
        "",
        f"- passed: `{audit['passed']}`",
        f"- paired attack conditions: `{audit['paired_conditions']}`",
        f"- eligible benign updates: `{audit['eligible_updates']}`",
        f"- new induced target adoptions: `{audit['new_induced_adoptions']}`",
        f"- updates receiving a target: `{audit['updates_receiving_target']}`",
        "- updates receiving an attack-induced target: "
        f"`{audit['updates_receiving_induced_target']}`",
        "",
        "## Strict crossed-holdout prediction",
        "",
        "| subset | model | rows | positives | Brier | log loss | AP |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.subset} | {row.model} | {row.rows} | {row.positives} | "
            f"{row.brier:.6f} | {row.log_loss:.6f} | {row.average_precision:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Brier improvement over DeGroot receiver exposure",
            "",
            "| subset | candidate | improvement | 95% crossed-bootstrap CI | P(>0) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.subset} | {row.candidate} | {row.brier_improvement:.6f} | "
            f"[{row.ci95_low:.6f}, {row.ci95_high:.6f}] | "
            f"{row.bootstrap_probability_positive:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim guardrails",
            "",
            "- Intermediate categorical states are observed post-treatment variables.",
            "- Better prediction from them supports a finite-state propagation description, "
            "not a semantic mechanism.",
            "- Residual error may reflect omitted confidence, stochasticity, or lossy "
            "answer parsing.",
            "- Message content causality requires the deferred matched rationale intervention.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    status_path = args.run_root / "orchestrator_status.json"
    status = read_json(status_path)
    frame, audit = extract_updates(args.run_root, status)
    if not audit["passed"]:
        raise RuntimeError("integrity audit failed: " + "; ".join(audit["errors"][:10]))
    assignments, fold_map = assign_folds(frame)
    predictions, fold_audit = crossed_predictions(assignments)
    metrics = prediction_metrics(predictions)
    comparisons = bootstrap_comparisons(
        predictions,
        replicates=args.bootstrap_replicates,
        rng=np.random.default_rng(args.seed),
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "node_round_features.csv", index=False)
    fold_map.to_csv(output / "fold_assignments.csv", index=False)
    fold_audit.to_csv(output / "fold_audit.csv", index=False)
    predictions.to_csv(output / "crossed_predictions.csv", index=False)
    metrics.to_csv(output / "prediction_metrics.csv", index=False)
    comparisons.to_csv(output / "model_comparisons.csv", index=False)
    exposure_summary(frame).to_csv(output / "exposure_adoption_summary.csv", index=False)
    (output / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(args.run_root.resolve()),
        "source_status_sha256": sha256_file(status_path),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "integrity_passed": audit["passed"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(audit, metrics, comparisons), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
