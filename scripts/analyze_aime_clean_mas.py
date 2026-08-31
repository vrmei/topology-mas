#!/usr/bin/env python3
"""Analyze paired Round-0-to-final utility in a clean AIME MAS batch."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
import pandas as pd

from topology_mas.analysis.loader import load_complete_batch
from topology_mas.execution.aime import AIME_BOUNDED_PROMPT_VERSION
from topology_mas.models import AnswerState, RunCondition

ANALYSIS_VERSION = "aime-clean-mas-paired-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--round-zero-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    return parser.parse_args()


def coarse_state(state: AnswerState) -> str:
    if state is AnswerState.CORRECT:
        return "C"
    if state is AnswerState.UNPARSED:
        return "U"
    if state is AnswerState.OTHER_ERROR:
        return "O"
    raise ValueError("clean AIME traces cannot contain target-error states")


def difficulty_band(rate: float) -> str:
    if 0.0 <= rate <= 0.1:
        return "floor"
    if 0.2 <= rate <= 0.8:
        return "informative"
    if 0.9 <= rate <= 1.0:
        return "ceiling"
    raise ValueError(f"uncovered external solve rate {rate}")


def read_reference(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        task_id = row["task_id"]
        rate = float(row["solve_rate"])
        result[task_id] = {
            "external_round_zero_rate": rate,
            "difficulty_band": difficulty_band(rate),
        }
    return result


def readout_turn(trace, *, node_id: int, round_index: int):
    matches = [
        turn
        for turn in trace.turns
        if turn.node_id == node_id and turn.round_index == round_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"trace {trace.run_id} lacks exactly one readout turn at round {round_index}"
        )
    return matches[0]


def build_run_frame(batch, reference: dict[str, dict[str, Any]]) -> pd.DataFrame:
    graph_by_id = {graph.graph_id: graph for graph in batch.graphs}
    rows = []
    for stored in batch.runs:
        spec = stored.run_spec
        trace = stored.trace
        graph = graph_by_id[spec.graph_id]
        if spec.condition is not RunCondition.CLEAN:
            raise ValueError("AIME clean analysis received a non-clean run")
        if trace.prompt_version != AIME_BOUNDED_PROMPT_VERSION:
            raise ValueError("trace does not use the frozen AIME bounded-message protocol")
        round_zero = readout_turn(trace, node_id=graph.readout_node, round_index=0)
        final = readout_turn(
            trace,
            node_id=graph.readout_node,
            round_index=trace.schedule.effective_horizon,
        )
        initial_state = coarse_state(round_zero.answer_state)
        final_state = coarse_state(final.answer_state)
        rows.append(
            {
                "run_spec_id": spec.run_spec_id,
                "task_id": spec.task_id,
                "graph_id": spec.graph_id,
                "edge_count": len(graph.edges),
                "experiment_seed": spec.experiment_seed,
                "initial_state": initial_state,
                "final_state": final_state,
                "initial_correct": int(initial_state == "C"),
                "final_correct": int(final_state == "C"),
                "paired_delta": int(final_state == "C") - int(initial_state == "C"),
                "model_calls": trace.total_model_calls,
                "backend_calls": trace.total_backend_calls,
                "input_tokens": trace.total_input_tokens,
                "output_tokens": trace.total_output_tokens,
                "private_output_tokens": sum(
                    turn.metadata.get("private_output_tokens") or 0
                    for turn in trace.turns
                ),
                "public_output_tokens": sum(
                    turn.metadata.get("public_output_tokens") or 0
                    for turn in trace.turns
                ),
                "private_length_turns": sum(
                    turn.metadata.get("private_finish_reason") == "length"
                    for turn in trace.turns
                ),
                "private_parsed_turns": sum(
                    turn.metadata.get("private_parsed_answer") is not None
                    for turn in trace.turns
                ),
                "summary_answer_mismatch_turns": sum(
                    not turn.metadata.get("summary_answer_matches_private", False)
                    for turn in trace.turns
                    if turn.metadata.get("private_parsed_answer") is not None
                ),
                "length_turns": sum(
                    turn.finish_reason == "length" for turn in trace.turns
                ),
                "unparsed_turns": sum(
                    turn.answer_state is AnswerState.UNPARSED for turn in trace.turns
                ),
                "serial_generation_latency_seconds": sum(
                    turn.latency_ms or 0.0 for turn in trace.turns
                )
                / 1000.0,
                **reference[spec.task_id],
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(batch.tasks) * len(batch.graphs):
        raise ValueError("clean batch does not contain one run per task and graph")
    if frame[["task_id", "graph_id"]].duplicated().any():
        raise ValueError("clean batch contains duplicate task-graph cells")
    return frame


def conditional_rate(frame: pd.DataFrame, initial: str, final: str) -> float | None:
    selected = frame.loc[frame.initial_state == initial]
    if selected.empty:
        return None
    return float((selected.final_state == final).mean())


def transition_summary(frame: pd.DataFrame, group: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(zip(frame.initial_state, frame.final_state, strict=True))
    result: dict[str, Any] = {
        **group,
        "runs": len(frame),
        "initial_utility": float(frame.initial_correct.mean()),
        "final_utility": float(frame.final_correct.mean()),
        "paired_delta": float(frame.paired_delta.mean()),
        "correct_preservation_C_to_C": conditional_rate(frame, "C", "C"),
        "correct_corruption_C_to_not_C": (
            1.0 - conditional_rate(frame, "C", "C")
            if conditional_rate(frame, "C", "C") is not None
            else None
        ),
        "other_error_correction_O_to_C": conditional_rate(frame, "O", "C"),
        "unparsed_correction_U_to_C": conditional_rate(frame, "U", "C"),
    }
    for initial in ("C", "O", "U"):
        for final in ("C", "O", "U"):
            result[f"count_{initial}_to_{final}"] = counts[(initial, final)]
    return result


def bootstrap_delta(
    frame: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    task_ids = sorted(frame.task_id.unique())
    by_task = {
        task_id: frame.loc[frame.task_id == task_id, "paired_delta"].to_numpy(dtype=float)
        for task_id in task_ids
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.choice(task_ids, size=len(task_ids), replace=True)
        estimates[index] = np.concatenate([by_task[task] for task in selected]).mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    batch = load_complete_batch(args.batch_dir)
    if batch.manifest.config.include_attacks:
        raise ValueError("clean analysis requires include_attacks=false")
    reference = read_reference(args.round_zero_reference)
    if set(reference) != set(batch.manifest.task_ids):
        raise ValueError("external Round-zero reference tasks differ from the clean batch")
    frame = build_run_frame(batch, reference)

    graph_rows = []
    for graph_id, group_frame in frame.groupby("graph_id", sort=False):
        row = transition_summary(group_frame, {"graph_id": graph_id})
        row["edge_count"] = int(group_frame.edge_count.iloc[0])
        row["mean_model_calls"] = float(group_frame.model_calls.mean())
        row["mean_backend_calls"] = float(group_frame.backend_calls.mean())
        row["mean_input_tokens"] = float(group_frame.input_tokens.mean())
        row["mean_output_tokens"] = float(group_frame.output_tokens.mean())
        row["mean_private_output_tokens"] = float(
            group_frame.private_output_tokens.mean()
        )
        row["mean_public_output_tokens"] = float(
            group_frame.public_output_tokens.mean()
        )
        row["mean_serial_generation_latency_seconds"] = float(
            group_frame.serial_generation_latency_seconds.mean()
        )
        row["length_turn_rate"] = float(
            group_frame.length_turns.sum() / group_frame.model_calls.sum()
        )
        row["private_length_turn_rate"] = float(
            group_frame.private_length_turns.sum() / group_frame.model_calls.sum()
        )
        row["summary_answer_mismatch_rate"] = float(
            group_frame.summary_answer_mismatch_turns.sum()
            / max(1, group_frame.private_parsed_turns.sum())
        )
        graph_rows.append(row)
    graph_frame = pd.DataFrame(graph_rows).sort_values(["edge_count", "graph_id"])

    edge_rows = []
    for edge_count, group_frame in frame.groupby("edge_count", sort=True):
        row = transition_summary(group_frame, {"edge_count": int(edge_count)})
        graph_slice = graph_frame.loc[graph_frame.edge_count == edge_count]
        row["graphs"] = len(graph_slice)
        row["graph_final_utility_sd"] = (
            float(pstdev(graph_slice.final_utility)) if len(graph_slice) > 1 else None
        )
        row["paired_delta_task_bootstrap_95_low"], row[
            "paired_delta_task_bootstrap_95_high"
        ] = bootstrap_delta(
            group_frame,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + int(edge_count),
        )
        edge_rows.append(row)
    edge_frame = pd.DataFrame(edge_rows).sort_values("edge_count")

    task_rows = []
    for task_id, group_frame in frame.groupby("task_id", sort=True):
        task_rows.append(
            transition_summary(
                group_frame,
                {
                    "task_id": task_id,
                    "difficulty_band": group_frame.difficulty_band.iloc[0],
                    "external_round_zero_rate": float(
                        group_frame.external_round_zero_rate.iloc[0]
                    ),
                },
            )
        )
    task_frame = pd.DataFrame(task_rows)

    band_rows = []
    for band, group_frame in frame.groupby("difficulty_band", sort=False):
        row = transition_summary(group_frame, {"difficulty_band": band})
        row["tasks"] = group_frame.task_id.nunique()
        row["paired_delta_task_bootstrap_95_low"], row[
            "paired_delta_task_bootstrap_95_high"
        ] = bootstrap_delta(
            group_frame,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + {"floor": 1, "informative": 2, "ceiling": 3}[band],
        )
        band_rows.append(row)
    band_frame = pd.DataFrame(band_rows)

    overall = transition_summary(frame, {"scope": "all_tasks_all_graphs"})
    overall["paired_delta_task_bootstrap_95_low"], overall[
        "paired_delta_task_bootstrap_95_high"
    ] = bootstrap_delta(
        frame,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    overall["graph_final_utility_mean"] = fmean(graph_frame.final_utility)
    overall["graph_final_utility_sd"] = pstdev(graph_frame.final_utility)
    overall["external_full_rationale_round_zero_utility"] = 0.5133333333333333
    overall["external_reference_is_paired"] = False

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "run_transitions.csv", index=False, lineterminator="\n")
    graph_frame.to_csv(output / "graph_metrics.csv", index=False, lineterminator="\n")
    edge_frame.to_csv(output / "edge_level_metrics.csv", index=False, lineterminator="\n")
    task_frame.to_csv(output / "task_metrics.csv", index=False, lineterminator="\n")
    band_frame.to_csv(output / "difficulty_band_metrics.csv", index=False, lineterminator="\n")
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "overall": overall,
        "edge_levels": edge_frame.to_dict(orient="records"),
        "difficulty_bands": band_frame.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
