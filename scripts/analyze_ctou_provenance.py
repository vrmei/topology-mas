"""Test whether message provenance explains variation hidden by CTOU counts."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analyze_node_round_adoption import graph_maps, read_json, read_jsonl, trace_category


ANALYSIS_VERSION = "ctou-provenance-v1"
CELL_COLUMNS = (
    "previous_state",
    "round_index",
    "incoming_correct_count",
    "incoming_target_count",
    "incoming_other_count",
    "incoming_unparsed_count",
)
DEFAULT_THRESHOLDS = (10, 30, 50)
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def _pair_overlap(sets: list[set[int]]) -> tuple[bool, int]:
    overlaps = [len(left & right) for left, right in combinations(sets, 2)]
    return bool(overlaps and max(overlaps) > 0), max(overlaps, default=0)


def _lineage_builder(
    turns: dict[tuple[int, int], dict[str, Any]],
    messages: dict[str, dict[str, Any]],
):
    memo: dict[tuple[int, int], frozenset[int]] = {}

    def lineage(node: int, round_index: int) -> frozenset[int]:
        key = (node, round_index)
        if key in memo:
            return memo[key]
        members = {node}
        turn = turns.get(key)
        if turn is not None and round_index > 0:
            if (node, round_index - 1) in turns:
                members.update(lineage(node, round_index - 1))
            for message_id in turn.get("incoming_message_ids", []):
                message = messages.get(str(message_id))
                if message is None:
                    continue
                sender = int(message["sender"])
                message_round = int(message["round_index"])
                members.update(lineage(sender, message_round))
        memo[key] = frozenset(members)
        return memo[key]

    return lineage


def provenance_trace_rows(
    *,
    pair: dict[str, Any],
    graph: dict[str, Any],
    task: dict[str, Any],
    clean_stored: dict[str, Any],
    attack_stored: dict[str, Any],
    stratum: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reconstruct exact CTOU counts and source ancestry from one attack trace."""
    errors: list[str] = []
    trace = attack_stored["trace"]
    clean_trace = clean_stored["trace"]
    attack_node = int(pair["attack_node"])
    target = str(pair["target_answer"])
    reference = str(task["reference_answer"])
    readout = int(graph["readout_node"])
    n = int(graph["node_count"])
    m = len(graph["edges"])
    incoming, _ = graph_maps(graph)
    turns = {
        (int(turn["node_id"]), int(turn["round_index"])): turn
        for turn in trace["turns"]
    }
    messages = {str(message["message_id"]): message for message in trace["messages"]}
    clean_messages = {
        (int(message["sender"]), int(message["round_index"])): message
        for message in clean_trace["messages"]
    }
    lineage = _lineage_builder(turns, messages)
    rows: list[dict[str, Any]] = []

    for (receiver, round_index), turn in sorted(turns.items()):
        if round_index == 0 or receiver == attack_node:
            continue
        previous = turns.get((receiver, round_index - 1))
        if previous is None:
            errors.append(f"{pair['attack_run_spec_id']}: missing receiver previous turn")
            continue
        incoming_messages: list[dict[str, Any]] = []
        for message_id in turn.get("incoming_message_ids", []):
            message = messages.get(str(message_id))
            if message is None:
                errors.append(f"{pair['attack_run_spec_id']}: missing message {message_id}")
                continue
            sender = int(message["sender"])
            message_round = int(message["round_index"])
            if sender not in incoming[receiver] or message_round != round_index - 1:
                errors.append(f"{pair['attack_run_spec_id']}: invalid synchronous message")
            incoming_messages.append(message)

        categories = [
            trace_category(message, reference=reference, target=target)
            for message in incoming_messages
        ]
        counts = {state: categories.count(state) for state in ("correct", "target", "other", "unparsed")}
        target_messages = [
            message
            for message, state in zip(incoming_messages, categories)
            if state == "target"
        ]
        correct_messages = [
            message
            for message, state in zip(incoming_messages, categories)
            if state == "correct"
        ]
        direct_target_count = 0
        relayed_target_count = 0
        natural_target_count = 0
        for message in target_messages:
            sender = int(message["sender"])
            message_round = int(message["round_index"])
            if sender == attack_node:
                direct_target_count += 1
                continue
            clean_message = clean_messages.get((sender, message_round))
            if clean_message is None:
                errors.append(f"{pair['attack_run_spec_id']}: missing paired clean message")
                continue
            clean_state = trace_category(clean_message, reference=reference, target=target)
            if clean_state == "target":
                natural_target_count += 1
            else:
                relayed_target_count += 1
        if direct_target_count > 1:
            errors.append(f"{pair['attack_run_spec_id']}: multiple direct attacker messages")
        active_origins = sum(
            value > 0
            for value in (direct_target_count, relayed_target_count, natural_target_count)
        )
        if active_origins > 1:
            target_origin = "mixed"
        elif direct_target_count:
            target_origin = "direct_only"
        elif relayed_target_count:
            target_origin = "relayed_only"
        elif natural_target_count:
            target_origin = "natural_only"
        else:
            target_origin = "no_target"

        immediate_parent_sets: list[set[int]] = []
        recursive_lineage_sets: list[set[int]] = []
        for message in correct_messages:
            sender = int(message["sender"])
            message_round = int(message["round_index"])
            parents = {sender}
            sender_turn = turns.get((sender, message_round))
            if sender_turn is not None:
                for parent_message_id in sender_turn.get("incoming_message_ids", []):
                    parent_message = messages.get(str(parent_message_id))
                    if parent_message is not None:
                        parents.add(int(parent_message["sender"]))
            immediate_parent_sets.append(parents)
            recursive_lineage_sets.append(set(lineage(sender, message_round)))
        immediate_overlap, immediate_max_overlap = _pair_overlap(immediate_parent_sets)
        recursive_overlap, recursive_max_overlap = _pair_overlap(recursive_lineage_sets)

        previous_state = trace_category(previous, reference=reference, target=target)
        next_state = trace_category(turn, reference=reference, target=target)
        rows.append(
            {
                "stratum": stratum,
                "task_id": str(pair["task_id"]),
                "graph_id": str(pair["graph_id"]),
                "run_spec_id": str(pair["attack_run_spec_id"]),
                "attack_node": attack_node,
                "receiver_node": receiver,
                "receiver_scope": "readout" if receiver == readout else "internal",
                "round_index": round_index,
                "n": n,
                "m": m,
                "previous_state": previous_state,
                "next_state": next_state,
                "next_is_target": int(next_state == "target"),
                "next_is_correct": int(next_state == "correct"),
                "incoming_correct_count": counts["correct"],
                "incoming_target_count": counts["target"],
                "incoming_other_count": counts["other"],
                "incoming_unparsed_count": counts["unparsed"],
                "direct_target_count": direct_target_count,
                "relayed_target_count": relayed_target_count,
                "natural_target_count": natural_target_count,
                "has_direct_target": int(direct_target_count > 0),
                "target_origin": target_origin,
                "correct_sender_count": len(correct_messages),
                "immediate_correct_overlap": int(immediate_overlap),
                "immediate_correct_max_overlap": immediate_max_overlap,
                "recursive_correct_overlap": int(recursive_overlap),
                "recursive_correct_max_overlap": recursive_max_overlap,
            }
        )
    return rows, errors


def extract_provenance_updates(
    run_root: Path,
    *,
    max_pairs: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    status = read_json(run_root / "orchestrator_status.json")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    pair_count = 0
    stop = False
    for descriptor in status["strata"]:
        stratum = str(descriptor["key"])
        root = run_root / "strata" / stratum
        graph_path = root / "selected_graphs.jsonl"
        if not graph_path.exists():
            graph_path = root / "batch" / "inputs" / "graphs.jsonl"
        graphs = {str(record["graph_id"]): record for record in read_jsonl(graph_path)}
        tasks = {
            str(record["task_id"]): record
            for record in read_jsonl(root / "batch" / "inputs" / "tasks.jsonl")
        }
        trace_root = root / "batch" / "traces"
        clean_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for pair in read_jsonl(root / "analysis-v1" / "paired_attacks.jsonl"):
            if max_pairs is not None and pair_count >= max_pairs:
                stop = True
                break
            attack_path = trace_root / f"{pair['attack_run_spec_id']}.json"
            clean_id = str(pair["clean_run_spec_id"])
            clean_path = trace_root / f"{clean_id}.json"
            if not attack_path.exists() or not clean_path.exists():
                errors.append(f"missing trace {attack_path}")
                continue
            if clean_id not in clean_cache:
                clean_cache[clean_id] = read_json(clean_path)
                if len(clean_cache) > 32:
                    clean_cache.popitem(last=False)
            else:
                clean_cache.move_to_end(clean_id)
            extracted, pair_errors = provenance_trace_rows(
                pair=pair,
                graph=graphs[str(pair["graph_id"])],
                task=tasks[str(pair["task_id"])],
                clean_stored=clean_cache[clean_id],
                attack_stored=read_json(attack_path),
                stratum=stratum,
            )
            rows.extend(extracted)
            errors.extend(pair_errors)
            pair_count += 1
        if stop:
            break
    frame = pd.DataFrame(rows)
    keys = ["task_id", "graph_id", "attack_node", "receiver_node", "round_index"]
    duplicates = int(frame.duplicated(keys).sum()) if not frame.empty else 0
    if duplicates:
        errors.append(f"duplicate update keys: {duplicates}")
    audit = {
        "passed": not errors,
        "errors": errors[:100],
        "paired_conditions": pair_count,
        "eligible_updates": len(frame),
        "tasks": int(frame["task_id"].nunique()) if not frame.empty else 0,
        "graphs": int(frame["graph_id"].nunique()) if not frame.empty else 0,
        "duplicate_update_keys": duplicates,
    }
    return frame, audit


def _scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return frame
    return frame.loc[frame["receiver_scope"] == scope]


def cell_rates(
    frame: pd.DataFrame,
    *,
    group_column: str,
    outcomes: Iterable[str],
) -> pd.DataFrame:
    columns = [*CELL_COLUMNS, "receiver_scope", group_column]
    aggregations: dict[str, tuple[str, str]] = {
        "rows": ("next_is_target", "size"),
        "tasks": ("task_id", "nunique"),
        "graphs": ("graph_id", "nunique"),
        "runs": ("run_spec_id", "nunique"),
    }
    for outcome in outcomes:
        aggregations[f"events_{outcome}"] = (outcome, "sum")
        aggregations[f"probability_{outcome}"] = (outcome, "mean")
    return frame.groupby(columns, dropna=False).agg(**aggregations).reset_index()


def _point_effect(
    frame: pd.DataFrame,
    *,
    group_column: str,
    group_a: Any,
    group_b: Any,
    outcome_column: str,
    minimum_cell_group_rows: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    stats = (
        frame.groupby([*CELL_COLUMNS, group_column], dropna=False)[outcome_column]
        .agg(["size", "sum"])
        .reset_index()
    )
    count = stats.pivot(index=list(CELL_COLUMNS), columns=group_column, values="size")
    events = stats.pivot(index=list(CELL_COLUMNS), columns=group_column, values="sum")
    if group_a not in count or group_b not in count:
        return {
            "matched_cells": 0,
            "matched_rows": 0,
            "group_a_probability": np.nan,
            "group_b_probability": np.nan,
            "risk_difference": np.nan,
        }, pd.DataFrame()
    matched = count[[group_a, group_b]].fillna(0).ge(minimum_cell_group_rows).all(axis=1)
    selected_count = count.loc[matched, [group_a, group_b]]
    selected_events = events.loc[matched, [group_a, group_b]].fillna(0)
    if selected_count.empty:
        return {
            "matched_cells": 0,
            "matched_rows": 0,
            "group_a_probability": np.nan,
            "group_b_probability": np.nan,
            "risk_difference": np.nan,
        }, pd.DataFrame()
    rates = selected_events / selected_count
    weights = selected_count.min(axis=1)
    probability_a = float(np.average(rates[group_a], weights=weights))
    probability_b = float(np.average(rates[group_b], weights=weights))
    cell_frame = rates.rename(columns={group_a: "rate_a", group_b: "rate_b"}).reset_index()
    cell_frame["count_a"] = selected_count[group_a].to_numpy()
    cell_frame["count_b"] = selected_count[group_b].to_numpy()
    cell_frame["cell_weight"] = weights.to_numpy()
    cell_frame["risk_difference"] = cell_frame["rate_a"] - cell_frame["rate_b"]
    return {
        "matched_cells": int(len(selected_count)),
        "matched_rows": int(selected_count.to_numpy().sum()),
        "group_a_probability": probability_a,
        "group_b_probability": probability_b,
        "risk_difference": probability_a - probability_b,
    }, cell_frame


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    selected_cells: pd.DataFrame,
    group_column: str,
    group_a: Any,
    group_b: Any,
    outcome_column: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    if selected_cells.empty or replicates <= 0:
        return np.nan, np.nan, 0
    cell_lookup = {
        tuple(row[column] for column in CELL_COLUMNS): index
        for index, row in selected_cells.reset_index(drop=True).iterrows()
    }
    working = frame.loc[frame[group_column].isin([group_a, group_b])].copy()
    working["cell_index"] = [
        cell_lookup.get(tuple(row[column] for column in CELL_COLUMNS), -1)
        for _, row in working.iterrows()
    ]
    working = working.loc[working["cell_index"] >= 0].copy()
    if working.empty:
        return np.nan, np.nan, 0
    working["group_index"] = (working[group_column] == group_b).astype(int)
    tasks = sorted(working["task_id"].unique())
    task_index = {value: index for index, value in enumerate(tasks)}
    graphs = working[["stratum", "graph_id"]].drop_duplicates().reset_index(drop=True)
    graph_index = {
        (row.stratum, row.graph_id): index for index, row in graphs.iterrows()
    }
    aggregated = (
        working.assign(
            task_index=working["task_id"].map(task_index),
            graph_index=[graph_index[(row.stratum, row.graph_id)] for _, row in working.iterrows()],
        )
        .groupby(["task_index", "graph_index", "cell_index", "group_index"], as_index=False)
        .agg(rows=(outcome_column, "size"), events=(outcome_column, "sum"))
    )
    task_ids = aggregated["task_index"].to_numpy(dtype=int)
    graph_ids = aggregated["graph_index"].to_numpy(dtype=int)
    bin_ids = (
        aggregated["cell_index"].to_numpy(dtype=int) * 2
        + aggregated["group_index"].to_numpy(dtype=int)
    )
    row_counts = aggregated["rows"].to_numpy(dtype=float)
    event_counts = aggregated["events"].to_numpy(dtype=float)
    graph_groups: dict[str, np.ndarray] = {}
    for stratum, group in graphs.groupby("stratum"):
        graph_groups[str(stratum)] = group.index.to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    bins = len(cell_lookup) * 2
    for _ in range(replicates):
        task_weights = np.bincount(rng.integers(0, len(tasks), size=len(tasks)), minlength=len(tasks))
        graph_weights = np.zeros(len(graphs), dtype=int)
        for members in graph_groups.values():
            sampled = rng.choice(members, size=len(members), replace=True)
            graph_weights += np.bincount(sampled, minlength=len(graphs))
        weights = task_weights[task_ids] * graph_weights[graph_ids]
        if not np.any(weights):
            continue
        counts = np.bincount(bin_ids, weights=row_counts * weights, minlength=bins).reshape(-1, 2)
        events = np.bincount(bin_ids, weights=event_counts * weights, minlength=bins).reshape(-1, 2)
        valid = (counts > 0).all(axis=1)
        if not np.any(valid):
            continue
        rates = events[valid] / counts[valid]
        cell_weights = counts[valid].min(axis=1)
        estimates.append(float(np.average(rates[:, 0] - rates[:, 1], weights=cell_weights)))
    if not estimates:
        return np.nan, np.nan, 0
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper), len(estimates)


def analyze_comparison(
    frame: pd.DataFrame,
    *,
    comparison: str,
    group_column: str,
    group_a: Any,
    group_b: Any,
    outcome_column: str,
    scopes: Iterable[str],
    thresholds: Iterable[int],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    cells: list[pd.DataFrame] = []
    for scope_index, scope in enumerate(scopes):
        scoped = _scope(frame, scope)
        for threshold in thresholds:
            summary, selected_cells = _point_effect(
                scoped,
                group_column=group_column,
                group_a=group_a,
                group_b=group_b,
                outcome_column=outcome_column,
                minimum_cell_group_rows=threshold,
            )
            lower = upper = np.nan
            successful = 0
            if threshold == 30 and summary["matched_cells"]:
                lower, upper, successful = _cluster_bootstrap(
                    scoped,
                    selected_cells=selected_cells,
                    group_column=group_column,
                    group_a=group_a,
                    group_b=group_b,
                    outcome_column=outcome_column,
                    replicates=bootstrap_replicates,
                    seed=seed + scope_index,
                )
            summaries.append(
                {
                    "comparison": comparison,
                    "scope": scope,
                    "outcome": outcome_column,
                    "group_a": str(group_a),
                    "group_b": str(group_b),
                    "minimum_cell_group_rows": threshold,
                    **summary,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "successful_bootstraps": successful,
                    "eligible_rows": int(len(scoped)),
                    "eligible_tasks": int(scoped["task_id"].nunique()),
                    "eligible_graphs": int(scoped["graph_id"].nunique()),
                }
            )
            if not selected_cells.empty:
                selected_cells.insert(0, "comparison", comparison)
                selected_cells.insert(1, "scope", scope)
                selected_cells.insert(2, "outcome", outcome_column)
                selected_cells.insert(3, "minimum_cell_group_rows", threshold)
                cells.append(selected_cells)
    return pd.DataFrame(summaries), pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, audit = extract_provenance_updates(args.run_root, max_pairs=args.max_pairs)
    if not audit["passed"]:
        raise RuntimeError(json.dumps(audit, indent=2))
    frame.to_csv(args.output_dir / "provenance_updates.csv.gz", index=False, compression="gzip")

    primary = frame.loc[
        (frame["previous_state"] == "correct")
        & (frame["incoming_target_count"] >= 1)
        & frame["target_origin"].isin(["direct_only", "relayed_only"])
    ].copy()
    primary["target_comparison_group"] = primary["target_origin"]
    target_rates = cell_rates(
        primary,
        group_column="target_comparison_group",
        outcomes=("next_is_target", "next_is_correct"),
    )
    target_rates.to_csv(args.output_dir / "target_origin_cell_rates.csv", index=False)
    target_summary, target_cells = analyze_comparison(
        primary,
        comparison="target_direct_vs_relayed",
        group_column="target_comparison_group",
        group_a="direct_only",
        group_b="relayed_only",
        outcome_column="next_is_target",
        scopes=("all", "internal", "readout"),
        thresholds=DEFAULT_THRESHOLDS,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    overlap_base = frame.loc[
        (frame["previous_state"] == "correct")
        & (frame["incoming_target_count"] >= 1)
        & (frame["incoming_correct_count"] >= 2)
    ].copy()
    summaries = [target_summary]
    cell_outputs = [target_cells]
    overlap_rate_outputs: list[pd.DataFrame] = []
    for offset, overlap_column in enumerate(("immediate_correct_overlap", "recursive_correct_overlap"), 1):
        label = overlap_column.replace("_correct_overlap", "")
        rates = cell_rates(
            overlap_base,
            group_column=overlap_column,
            outcomes=("next_is_target", "next_is_correct"),
        )
        rates.insert(0, "overlap_definition", label)
        overlap_rate_outputs.append(rates)
        for outcome_offset, outcome in enumerate(("next_is_target", "next_is_correct")):
            summary, cells = analyze_comparison(
                overlap_base,
                comparison=f"correct_{label}_shared_vs_independent",
                group_column=overlap_column,
                group_a=1,
                group_b=0,
                outcome_column=outcome,
                scopes=("all", "internal", "readout"),
                thresholds=DEFAULT_THRESHOLDS,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed + offset * 100 + outcome_offset * 10,
            )
            summaries.append(summary)
            cell_outputs.append(cells)

    pd.concat(overlap_rate_outputs, ignore_index=True).to_csv(
        args.output_dir / "correct_overlap_cell_rates.csv", index=False
    )
    pd.concat(summaries, ignore_index=True).to_csv(args.output_dir / "matched_effects.csv", index=False)
    nonempty_cells = [item for item in cell_outputs if not item.empty]
    if nonempty_cells:
        pd.concat(nonempty_cells, ignore_index=True).to_csv(
            args.output_dir / "matched_cell_effects.csv", index=False
        )

    support = (
        frame.groupby(["n", "m", "receiver_scope"], dropna=False)
        .agg(
            updates=("next_is_target", "size"),
            tasks=("task_id", "nunique"),
            graphs=("graph_id", "nunique"),
            direct_target_updates=("has_direct_target", "sum"),
            relayed_only_updates=("target_origin", lambda values: int((values == "relayed_only").sum())),
            natural_only_updates=("target_origin", lambda values: int((values == "natural_only").sum())),
            immediate_overlap_updates=("immediate_correct_overlap", "sum"),
            recursive_overlap_updates=("recursive_correct_overlap", "sum"),
        )
        .reset_index()
    )
    support.to_csv(args.output_dir / "support_summary.csv", index=False)
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_root": str(args.run_root),
        "audit": audit,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "primary_rows": int(len(primary)),
        "overlap_rows": int(len(overlap_base)),
        "practical_effect_threshold": 0.02,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
