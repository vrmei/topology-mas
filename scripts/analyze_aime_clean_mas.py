#!/usr/bin/env python3
"""Analyze paired Round-0-to-final utility in a clean AIME MAS batch."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
import pandas as pd

from topology_mas.analysis.loader import load_complete_batch
from topology_mas.execution.aime import AIME_BOUNDED_PROMPT_VERSION
from topology_mas.models import AnswerState, RunCondition

ANALYSIS_VERSION = "aime-clean-mas-paired-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-dir",
        type=Path,
        action="append",
        required=True,
        help="A complete clean batch. Repeat to combine disjoint graph batches.",
    )
    parser.add_argument(
        "--batch-label",
        action="append",
        help="Optional cohort label corresponding to each --batch-dir.",
    )
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


def build_run_frame(
    batch,
    reference: dict[str, dict[str, Any]],
    *,
    cohort: str = "batch_1",
) -> pd.DataFrame:
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
                "cohort": cohort,
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


def validate_compatible_batches(batches) -> None:
    """Reject combinations that differ in anything except graph collection."""

    if not batches:
        raise ValueError("at least one clean batch is required")
    first = batches[0].manifest
    comparable_fields = (
        "task_collection_fingerprint",
        "task_ids",
        "node_count",
        "readout_node",
        "max_rounds",
        "prompt_version",
        "runner_version",
        "config",
        "execution_settings",
    )
    for batch in batches[1:]:
        for field in comparable_fields:
            if getattr(batch.manifest, field) != getattr(first, field):
                raise ValueError(f"batch manifests differ on {field}")
    graph_ids = [graph.graph_id for batch in batches for graph in batch.graphs]
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("combined batches contain duplicate graph IDs")
    run_ids = [run.run_spec.run_spec_id for batch in batches for run in batch.runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("combined batches contain duplicate run IDs")


def complete_task_graph_matrix(frame: pd.DataFrame, value: str) -> np.ndarray:
    matrix = frame.pivot(index="graph_id", columns="task_id", values=value)
    if matrix.isna().any().any():
        raise ValueError("task-graph matrix is incomplete")
    return matrix.to_numpy(dtype=float)


def hierarchical_bootstrap_mean(
    frame: pd.DataFrame,
    *,
    value: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Resample graph and task axes independently for a crossed design."""

    matrix = complete_task_graph_matrix(frame, value)
    graph_count, task_count = matrix.shape
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        graph_indices = rng.integers(graph_count, size=graph_count)
        task_indices = rng.integers(task_count, size=task_count)
        estimates[index] = matrix[np.ix_(graph_indices, task_indices)].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def hierarchical_bootstrap_conditional_rate(
    frame: pd.DataFrame,
    *,
    initial_state: str,
    final_state: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    denominator = (frame.initial_state == initial_state).astype(float)
    numerator = (
        (frame.initial_state == initial_state) & (frame.final_state == final_state)
    ).astype(float)
    working = frame.assign(_denominator=denominator, _numerator=numerator)
    denominator_matrix = complete_task_graph_matrix(working, "_denominator")
    numerator_matrix = complete_task_graph_matrix(working, "_numerator")
    graph_count, task_count = denominator_matrix.shape
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        graph_indices = rng.integers(graph_count, size=graph_count)
        task_indices = rng.integers(task_count, size=task_count)
        selected_denominator = denominator_matrix[np.ix_(graph_indices, task_indices)].sum()
        if selected_denominator == 0:
            continue
        estimates.append(
            numerator_matrix[np.ix_(graph_indices, task_indices)].sum()
            / selected_denominator
        )
    if not estimates:
        return float("nan"), float("nan")
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return float(low), float(high)


def conditional_rate_from_matrices(
    numerator: np.ndarray,
    denominator: np.ndarray,
    graph_indices: np.ndarray,
    task_indices: np.ndarray,
) -> float | None:
    selected_denominator = denominator[np.ix_(graph_indices, task_indices)].sum()
    if selected_denominator == 0:
        return None
    return float(
        numerator[np.ix_(graph_indices, task_indices)].sum() / selected_denominator
    )


def bootstrap_conditional_group_difference(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    initial_state: str,
    final_state: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    task_ids = sorted(set(first.task_id) & set(second.task_id))
    if set(task_ids) != set(first.task_id) or set(task_ids) != set(second.task_id):
        raise ValueError("conditional comparison groups do not contain identical tasks")

    def matrices(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        working = group.assign(
            _denominator=(group.initial_state == initial_state).astype(float),
            _numerator=(
                (group.initial_state == initial_state)
                & (group.final_state == final_state)
            ).astype(float),
        )
        numerator = working.pivot(
            index="graph_id", columns="task_id", values="_numerator"
        )[task_ids].to_numpy(dtype=float)
        denominator = working.pivot(
            index="graph_id", columns="task_id", values="_denominator"
        )[task_ids].to_numpy(dtype=float)
        return numerator, denominator

    first_num, first_den = matrices(first)
    second_num, second_den = matrices(second)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        task_indices = rng.integers(len(task_ids), size=len(task_ids))
        first_graphs = rng.integers(first_num.shape[0], size=first_num.shape[0])
        second_graphs = rng.integers(second_num.shape[0], size=second_num.shape[0])
        first_rate = conditional_rate_from_matrices(
            first_num, first_den, first_graphs, task_indices
        )
        second_rate = conditional_rate_from_matrices(
            second_num, second_den, second_graphs, task_indices
        )
        if first_rate is not None and second_rate is not None:
            estimates.append(first_rate - second_rate)
    if not estimates:
        return float("nan"), float("nan")
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return float(low), float(high)


def design_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    pieces = [np.ones((len(frame), 1), dtype=float)]
    for column in columns:
        dummies = pd.get_dummies(frame[column].astype(str), drop_first=True, dtype=float)
        pieces.append(dummies.to_numpy(dtype=float))
    return np.concatenate(pieces, axis=1)


def residual_sum_of_squares(y: np.ndarray, q: np.ndarray) -> float:
    residual = y - q @ (q.T @ y)
    return float(residual @ residual)


def permutation_structure_test(
    frame: pd.DataFrame,
    *,
    value: str,
    reduced_columns: tuple[str, ...],
    full_columns: tuple[str, ...],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Test added structure terms by permuting outcomes within each task."""

    y = frame[value].to_numpy(dtype=float)
    reduced_q, _ = np.linalg.qr(design_matrix(frame, reduced_columns), mode="reduced")
    full_q, _ = np.linalg.qr(design_matrix(frame, full_columns), mode="reduced")
    reduced_sse = residual_sum_of_squares(y, reduced_q)
    full_sse = residual_sum_of_squares(y, full_q)
    observed = (reduced_sse - full_sse) / reduced_sse if reduced_sse else 0.0
    task_indices = [
        indices.to_numpy(dtype=int)
        for _, indices in frame.groupby("task_id", sort=False).groups.items()
    ]
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = y.copy()
        for indices in task_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        null_reduced = residual_sum_of_squares(permuted, reduced_q)
        null_full = residual_sum_of_squares(permuted, full_q)
        null_effect = (
            (null_reduced - null_full) / null_reduced if null_reduced else 0.0
        )
        exceedances += null_effect >= observed
    return {
        "metric": value,
        "reduced_terms": list(reduced_columns),
        "full_terms": list(full_columns),
        "partial_r_squared": float(observed),
        "permutations": permutations,
        "permutation_p_value": float((exceedances + 1) / (permutations + 1)),
    }


def bootstrap_group_difference(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    value: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a first-minus-second difference with paired task resampling."""

    first_pivot = first.pivot(index="graph_id", columns="task_id", values=value)
    second_pivot = second.pivot(index="graph_id", columns="task_id", values=value)
    if set(first_pivot.columns) != set(second_pivot.columns):
        raise ValueError("comparison groups do not contain identical tasks")
    task_ids = sorted(first_pivot.columns)
    first_matrix = first_pivot[task_ids].to_numpy(dtype=float)
    second_matrix = second_pivot[task_ids].to_numpy(dtype=float)
    if np.isnan(first_matrix).any() or np.isnan(second_matrix).any():
        raise ValueError("comparison groups contain incomplete task-graph cells")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        task_indices = rng.integers(len(task_ids), size=len(task_ids))
        first_graphs = rng.integers(first_matrix.shape[0], size=first_matrix.shape[0])
        second_graphs = rng.integers(second_matrix.shape[0], size=second_matrix.shape[0])
        estimates[index] = (
            first_matrix[np.ix_(first_graphs, task_indices)].mean()
            - second_matrix[np.ix_(second_graphs, task_indices)].mean()
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_density_slope(
    frame: pd.DataFrame,
    *,
    value: str,
    edge_counts: tuple[int, ...],
    samples: int,
    seed: int,
) -> dict[str, float | list[int] | str]:
    """Bootstrap the slope of density-level means under the crossed design."""

    groups = {edge: frame.loc[frame.edge_count == edge] for edge in edge_counts}
    task_ids = sorted(frame.task_id.unique())
    matrices: dict[int, np.ndarray] = {}
    for edge, group in groups.items():
        pivot = group.pivot(index="graph_id", columns="task_id", values=value)
        if set(pivot.columns) != set(task_ids) or pivot.isna().any().any():
            raise ValueError(f"edge stratum {edge} is not a complete crossed design")
        matrices[edge] = pivot[task_ids].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    slopes = np.empty(samples, dtype=float)
    x = np.asarray(edge_counts, dtype=float)
    for index in range(samples):
        task_indices = rng.integers(len(task_ids), size=len(task_ids))
        means = []
        for edge in edge_counts:
            matrix = matrices[edge]
            graph_indices = rng.integers(matrix.shape[0], size=matrix.shape[0])
            means.append(matrix[np.ix_(graph_indices, task_indices)].mean())
        slopes[index] = np.polyfit(x, np.asarray(means), 1)[0]
    observed_means = np.asarray([groups[edge][value].mean() for edge in edge_counts])
    observed = float(np.polyfit(x, observed_means, 1)[0])
    low, high = np.quantile(slopes, [0.025, 0.975])
    return {
        "metric": value,
        "edge_counts": list(edge_counts),
        "observed_slope_per_edge": observed,
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "bootstrap_probability_positive": float((slopes > 0).mean()),
    }


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


def bootstrap_band_difference(
    frame: pd.DataFrame,
    *,
    first_band: str,
    second_band: str,
    samples: int,
    seed: int,
) -> dict[str, float | str]:
    first = frame.loc[frame.difficulty_band == first_band]
    second = frame.loc[frame.difficulty_band == second_band]
    first_tasks = sorted(first.task_id.unique())
    second_tasks = sorted(second.task_id.unique())
    first_by_task = {
        task_id: first.loc[first.task_id == task_id, "paired_delta"].to_numpy(dtype=float)
        for task_id in first_tasks
    }
    second_by_task = {
        task_id: second.loc[second.task_id == task_id, "paired_delta"].to_numpy(dtype=float)
        for task_id in second_tasks
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected_first = rng.choice(first_tasks, size=len(first_tasks), replace=True)
        selected_second = rng.choice(second_tasks, size=len(second_tasks), replace=True)
        first_mean = np.concatenate(
            [first_by_task[task] for task in selected_first]
        ).mean()
        second_mean = np.concatenate(
            [second_by_task[task] for task in selected_second]
        ).mean()
        estimates[index] = first_mean - second_mean
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "first_band": first_band,
        "second_band": second_band,
        "observed_difference": float(first.paired_delta.mean() - second.paired_delta.mean()),
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    batches = [load_complete_batch(path) for path in args.batch_dir]
    validate_compatible_batches(batches)
    if any(batch.manifest.config.include_attacks for batch in batches):
        raise ValueError("clean analysis requires include_attacks=false")
    if args.batch_label is not None and len(args.batch_label) != len(batches):
        raise ValueError("--batch-label count must equal --batch-dir count")
    labels = args.batch_label or [f"batch_{index + 1}" for index in range(len(batches))]
    if len(labels) != len(set(labels)):
        raise ValueError("batch labels must be unique")
    reference = read_reference(args.round_zero_reference)
    if set(reference) != set(batches[0].manifest.task_ids):
        raise ValueError("external Round-zero reference tasks differ from the clean batch")
    frame = pd.concat(
        [
            build_run_frame(batch, reference, cohort=label)
            for batch, label in zip(batches, labels, strict=True)
        ],
        ignore_index=True,
    )
    if frame[["task_id", "graph_id"]].duplicated().any():
        raise ValueError("combined batches contain duplicate task-graph cells")

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
        for offset, metric in enumerate(
            ("initial_correct", "final_correct", "paired_delta"), start=1
        ):
            low, high = hierarchical_bootstrap_mean(
                group_frame,
                value=metric,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + 10 * int(edge_count) + offset,
            )
            prefix = {
                "initial_correct": "initial_utility",
                "final_correct": "final_utility",
                "paired_delta": "paired_delta",
            }[metric]
            row[f"{prefix}_hierarchical_bootstrap_95_low"] = low
            row[f"{prefix}_hierarchical_bootstrap_95_high"] = high
        for offset, (initial_state, final_state, prefix) in enumerate(
            (
                ("C", "C", "correct_preservation_C_to_C"),
                ("O", "C", "other_error_correction_O_to_C"),
                ("U", "C", "unparsed_correction_U_to_C"),
            ),
            start=1,
        ):
            low, high = hierarchical_bootstrap_conditional_rate(
                group_frame,
                initial_state=initial_state,
                final_state=final_state,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + 500 + 10 * int(edge_count) + offset,
            )
            row[f"{prefix}_hierarchical_bootstrap_95_low"] = low
            row[f"{prefix}_hierarchical_bootstrap_95_high"] = high
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

    density_band_rows = []
    for (edge_count, band), group_frame in frame.groupby(
        ["edge_count", "difficulty_band"], sort=True
    ):
        row = transition_summary(
            group_frame,
            {"edge_count": int(edge_count), "difficulty_band": band},
        )
        row["graphs"] = group_frame.graph_id.nunique()
        row["tasks"] = group_frame.task_id.nunique()
        for offset, metric in enumerate(
            ("initial_correct", "final_correct", "paired_delta"), start=1
        ):
            low, high = hierarchical_bootstrap_mean(
                group_frame,
                value=metric,
                samples=args.bootstrap_samples,
                seed=(
                    args.bootstrap_seed
                    + 1000
                    + int(edge_count) * 10
                    + offset
                    + {"floor": 100, "informative": 200, "ceiling": 300}[band]
                ),
            )
            prefix = {
                "initial_correct": "initial_utility",
                "final_correct": "final_utility",
                "paired_delta": "paired_delta",
            }[metric]
            row[f"{prefix}_hierarchical_bootstrap_95_low"] = low
            row[f"{prefix}_hierarchical_bootstrap_95_high"] = high
        density_band_rows.append(row)
    density_band_frame = pd.DataFrame(density_band_rows)

    cohort_edge_rows = []
    for (cohort, edge_count), group_frame in frame.groupby(
        ["cohort", "edge_count"], sort=True
    ):
        row = transition_summary(
            group_frame,
            {"cohort": cohort, "edge_count": int(edge_count)},
        )
        row["graphs"] = group_frame.graph_id.nunique()
        row["summary_answer_mismatch_rate"] = float(
            group_frame.summary_answer_mismatch_turns.sum()
            / max(1, group_frame.private_parsed_turns.sum())
        )
        row["private_length_turn_rate"] = float(
            group_frame.private_length_turns.sum() / group_frame.model_calls.sum()
        )
        row["unparsed_turn_rate"] = float(
            group_frame.unparsed_turns.sum() / group_frame.model_calls.sum()
        )
        cohort_edge_rows.append(row)
    cohort_edge_frame = pd.DataFrame(cohort_edge_rows)

    cohort_comparison_rows = []
    if len(labels) == 2:
        first_label, second_label = labels
        common_edges = sorted(
            set(frame.loc[frame.cohort == first_label, "edge_count"])
            & set(frame.loc[frame.cohort == second_label, "edge_count"])
        )
        for edge_count in common_edges:
            first = frame.loc[
                (frame.cohort == first_label) & (frame.edge_count == edge_count)
            ]
            second = frame.loc[
                (frame.cohort == second_label) & (frame.edge_count == edge_count)
            ]
            for offset, metric in enumerate(("final_correct", "paired_delta"), start=1):
                low, high = bootstrap_group_difference(
                    second,
                    first,
                    value=metric,
                    samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + 2000 + int(edge_count) * 10 + offset,
                )
                cohort_comparison_rows.append(
                    {
                        "edge_count": int(edge_count),
                        "metric": metric,
                        "first_cohort": second_label,
                        "second_cohort": first_label,
                        "observed_first_minus_second": float(
                            second[metric].mean() - first[metric].mean()
                        ),
                        "hierarchical_bootstrap_95_low": low,
                        "hierarchical_bootstrap_95_high": high,
                    }
                )
    cohort_comparison_frame = pd.DataFrame(cohort_comparison_rows)

    edge_difference_rows = []
    conditional_edge_difference_rows = []
    edge_values = sorted(int(value) for value in frame.edge_count.unique())
    for first_edge, second_edge in zip(edge_values[1:], edge_values[:-1], strict=True):
        first = frame.loc[frame.edge_count == first_edge]
        second = frame.loc[frame.edge_count == second_edge]
        for offset, metric in enumerate(
            ("initial_correct", "final_correct", "paired_delta"), start=1
        ):
            low, high = bootstrap_group_difference(
                first,
                second,
                value=metric,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + 3000 + first_edge * 10 + offset,
            )
            edge_difference_rows.append(
                {
                    "first_edge_count": first_edge,
                    "second_edge_count": second_edge,
                    "metric": metric,
                    "observed_first_minus_second": float(
                        first[metric].mean() - second[metric].mean()
                    ),
                    "hierarchical_bootstrap_95_low": low,
                    "hierarchical_bootstrap_95_high": high,
                }
            )
        for offset, (initial_state, final_state, metric) in enumerate(
            (
                ("C", "C", "correct_preservation_C_to_C"),
                ("O", "C", "other_error_correction_O_to_C"),
                ("U", "C", "unparsed_correction_U_to_C"),
            ),
            start=1,
        ):
            low, high = bootstrap_conditional_group_difference(
                first,
                second,
                initial_state=initial_state,
                final_state=final_state,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + 3500 + first_edge * 10 + offset,
            )
            first_rate = conditional_rate(first, initial_state, final_state)
            second_rate = conditional_rate(second, initial_state, final_state)
            conditional_edge_difference_rows.append(
                {
                    "first_edge_count": first_edge,
                    "second_edge_count": second_edge,
                    "metric": metric,
                    "observed_first_minus_second": (
                        first_rate - second_rate
                        if first_rate is not None and second_rate is not None
                        else None
                    ),
                    "hierarchical_bootstrap_95_low": low,
                    "hierarchical_bootstrap_95_high": high,
                }
            )
    edge_difference_frame = pd.DataFrame(edge_difference_rows)
    conditional_edge_difference_frame = pd.DataFrame(conditional_edge_difference_rows)

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
    density_band_frame.to_csv(
        output / "density_difficulty_metrics.csv", index=False, lineterminator="\n"
    )
    cohort_edge_frame.to_csv(
        output / "cohort_edge_metrics.csv", index=False, lineterminator="\n"
    )
    cohort_comparison_frame.to_csv(
        output / "cohort_comparisons.csv", index=False, lineterminator="\n"
    )
    edge_difference_frame.to_csv(
        output / "edge_level_differences.csv", index=False, lineterminator="\n"
    )
    conditional_edge_difference_frame.to_csv(
        output / "conditional_edge_level_differences.csv",
        index=False,
        lineterminator="\n",
    )
    primary_edges = tuple(
        int(row.edge_count)
        for row in edge_frame.itertuples()
        if int(row.graphs) > 1
    )
    density_slopes = (
        [
            bootstrap_density_slope(
                frame,
                value=metric,
                edge_counts=primary_edges,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + 4000 + offset,
            )
            for offset, metric in enumerate(
                ("initial_correct", "final_correct", "paired_delta"), start=1
            )
        ]
        if len(primary_edges) >= 2
        else []
    )
    structure_tests = []
    primary_frame = frame.loc[frame.edge_count.isin(primary_edges)].reset_index(drop=True)
    if not primary_frame.empty:
        for metric_offset, metric in enumerate(
            ("initial_correct", "final_correct", "paired_delta"), start=1
        ):
            for test_offset, (reduced, full, label) in enumerate(
                (
                    (("task_id",), ("task_id", "edge_count"), "density_beyond_task"),
                    (("task_id",), ("task_id", "graph_id"), "graph_beyond_task"),
                    (
                        ("task_id", "edge_count"),
                        ("task_id", "graph_id"),
                        "graph_arrangement_beyond_task_and_density",
                    ),
                ),
                start=1,
            ):
                result = permutation_structure_test(
                    primary_frame,
                    value=metric,
                    reduced_columns=reduced,
                    full_columns=full,
                    permutations=5000,
                    seed=args.bootstrap_seed + 5000 + metric_offset * 10 + test_offset,
                )
                result["test"] = label
                result["edge_counts"] = list(primary_edges)
                structure_tests.append(result)
    summary = json_safe({
        "analysis_version": ANALYSIS_VERSION,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "batch_dirs": [str(path) for path in args.batch_dir],
        "batch_labels": labels,
        "overall": overall,
        "edge_levels": edge_frame.to_dict(orient="records"),
        "density_slopes_excluding_single_complete_graph": density_slopes,
        "structure_permutation_tests_excluding_single_complete_graph": structure_tests,
        "edge_level_adjacent_differences": edge_difference_frame.to_dict(
            orient="records"
        ),
        "conditional_edge_level_adjacent_differences": (
            conditional_edge_difference_frame.to_dict(orient="records")
        ),
        "cohort_comparisons": cohort_comparison_frame.to_dict(orient="records"),
        "difficulty_bands": band_frame.to_dict(orient="records"),
        "difficulty_band_pairwise_differences": [
            bootstrap_band_difference(
                frame,
                first_band="informative",
                second_band=other,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + offset,
            )
            for other, offset in (("floor", 101), ("ceiling", 102))
        ],
    })
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
