"""Stream graph-depth Experiment C analysis from a complete fixed-horizon batch.

The graph-depth causal schedule is a strict subset of the fixed-horizon schedule. Every
retained state therefore already exists in the source trace. This script reconstructs only
the compact analysis artifacts and never duplicates or retains the large trace collection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from topology_mas.analysis.artifacts import write_analysis
from topology_mas.analysis.metrics import _graph_metric, _matches_target
from topology_mas.analysis.schemas import (
    AnalysisManifest,
    AnalysisResult,
    ClassicalInitialStateRecord,
    PairedAttackRow,
    RunMetricRow,
)
from topology_mas.execution.batch import (
    BatchExecutionManifest,
    ExecutionRunSpec,
    content_fingerprint,
)
from topology_mas.models import AnswerState, GraphSpec, RunCondition
from topology_mas.topology.graph_ops import build_causal_schedule, graph_depth_to_readout

ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_jsonl(path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _sum_known(values: list[int | None]) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _pair_key(spec: ExecutionRunSpec) -> tuple[str, str, int, int]:
    return (spec.task_id, spec.graph_id, spec.experiment_seed, spec.assignment_seed)


def _load_raw_trace(batch_dir: Path, spec: ExecutionRunSpec) -> dict[str, Any]:
    path = batch_dir / "traces" / f"{spec.run_spec_id}.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    raw_spec = stored["run_spec"]
    if raw_spec["run_spec_id"] != spec.run_spec_id:
        raise ValueError(f"trace identity differs from the plan at {path}")
    return stored["trace"]


def _project_trace(
    trace: dict[str, Any], graph: GraphSpec, active: dict[int, set[int]], depth: int
) -> dict[str, Any]:
    turns = [
        turn
        for turn in trace["turns"]
        if turn["round_index"] <= depth
        and turn["node_id"] in active[turn["round_index"]]
    ]
    expected_cells = {
        (round_index, node_id)
        for round_index, nodes in active.items()
        for node_id in nodes
    }
    turns_by_cell = {(turn["round_index"], turn["node_id"]): turn for turn in turns}
    if set(turns_by_cell) != expected_cells:
        raise ValueError(f"fixed trace {trace['run_id']} lacks the graph-depth schedule")
    final = turns_by_cell[(depth, graph.readout_node)]
    round_zero = turns_by_cell[(0, graph.readout_node)]
    attack_node = trace["attack_node"]
    runtime_turns = [
        turn
        for turn in turns
        if turn["round_index"] > 0 and turn["node_id"] != attack_node
    ]
    return {
        "trace": trace,
        "turns": turns_by_cell,
        "final": final,
        "round_zero": round_zero,
        "model_calls": len(runtime_turns),
        "input_tokens": _sum_known([turn.get("input_tokens") for turn in runtime_turns]),
        "output_tokens": _sum_known([turn.get("output_tokens") for turn in runtime_turns]),
    }


def _run_metric(spec: ExecutionRunSpec, projected: dict[str, Any]) -> RunMetricRow:
    final = projected["final"]
    round_zero = projected["round_zero"]
    final_state = AnswerState(final["answer_state"])
    round_zero_state = AnswerState(round_zero["answer_state"])
    return RunMetricRow(
        run_spec_id=spec.run_spec_id,
        task_id=spec.task_id,
        graph_id=spec.graph_id,
        experiment_seed=spec.experiment_seed,
        assignment_seed=spec.assignment_seed,
        condition=spec.condition,
        attack_node=spec.attack_node,
        final_answer_state=final_state,
        final_parsed_answer=final["parsed_answer"],
        final_correct=final_state is AnswerState.CORRECT,
        readout_round_zero_state=round_zero_state,
        readout_round_zero_correct=round_zero_state is AnswerState.CORRECT,
        model_calls=projected["model_calls"],
        input_tokens=projected["input_tokens"],
        output_tokens=projected["output_tokens"],
    )


def _paired_attack(
    clean_spec: ExecutionRunSpec,
    clean: dict[str, Any],
    attack_spec: ExecutionRunSpec,
    attack: dict[str, Any],
    graph: GraphSpec,
) -> PairedAttackRow:
    if attack_spec.attack_node is None:
        raise ValueError("attack projection lacks attack_node")
    target = attack["trace"]["target_answer"]
    if target is None:
        raise ValueError("attack projection lacks target_answer")
    clean_turns = clean["turns"]
    attack_turns = attack["turns"]
    if clean_turns.keys() != attack_turns.keys():
        raise ValueError(f"paired schedules differ for {attack_spec.run_spec_id}")

    induced_by_round = []
    observed_by_round = []
    first_induced_round = None
    first_readout_round = None
    for round_index in range(graph.max_rounds + 1):
        induced = 0
        observed = 0
        for (turn_round, node_id), attack_turn in attack_turns.items():
            if turn_round != round_index or node_id == attack_spec.attack_node:
                continue
            attack_matches = _matches_target(attack_turn["parsed_answer"], target)
            clean_matches = _matches_target(
                clean_turns[(turn_round, node_id)]["parsed_answer"], target
            )
            observed += int(attack_matches)
            if attack_matches and not clean_matches:
                induced += 1
                if first_induced_round is None:
                    first_induced_round = round_index
                if node_id == graph.readout_node and first_readout_round is None:
                    first_readout_round = round_index
        induced_by_round.append(induced)
        observed_by_round.append(observed)

    clean_final = clean["final"]
    attack_final = attack["final"]
    clean_final_matches = _matches_target(clean_final["parsed_answer"], target)
    attack_final_matches = _matches_target(attack_final["parsed_answer"], target)
    clean_correct = AnswerState(clean_final["answer_state"]) is AnswerState.CORRECT
    attack_correct = AnswerState(attack_final["answer_state"]) is AnswerState.CORRECT
    return PairedAttackRow(
        task_id=attack_spec.task_id,
        graph_id=attack_spec.graph_id,
        experiment_seed=attack_spec.experiment_seed,
        assignment_seed=attack_spec.assignment_seed,
        attack_node=attack_spec.attack_node,
        clean_run_spec_id=clean_spec.run_spec_id,
        attack_run_spec_id=attack_spec.run_spec_id,
        clean_correct=clean_correct,
        attack_correct=attack_correct,
        paired_accuracy_drop=int(clean_correct) - int(attack_correct),
        target_answer=target,
        clean_final_matches_target=clean_final_matches,
        attack_final_matches_target=attack_final_matches,
        induced_readout_target=attack_final_matches and not clean_final_matches,
        correct_to_target_flip=clean_correct and attack_final_matches,
        clean_error_corrected_under_attack=(not clean_correct) and attack_correct,
        induced_target_count_by_round=tuple(induced_by_round),
        observed_target_count_by_round=tuple(observed_by_round),
        first_induced_target_round=first_induced_round,
        first_induced_readout_target_round=first_readout_round,
        max_induced_nonattacker_count=max(induced_by_round),
        attack_model_calls=attack["model_calls"],
        attack_input_tokens=attack["input_tokens"],
        attack_output_tokens=attack["output_tokens"],
    )


def analyze_streaming(batch_dir: Path) -> tuple[AnalysisResult, dict[str, int]]:
    manifest = BatchExecutionManifest.model_validate_json(
        (batch_dir / "manifest.json").read_text(encoding="utf-8")
    )
    plan = _read_jsonl(batch_dir / "plan.jsonl", ExecutionRunSpec)
    graphs = _read_jsonl(batch_dir / "inputs" / "graphs.jsonl", GraphSpec)
    if len(plan) != manifest.expected_run_count:
        raise ValueError("source plan is incomplete")
    graph_by_id = {graph.graph_id: graph for graph in graphs}
    depths = {graph.graph_id: graph_depth_to_readout(graph) for graph in graphs}
    active_by_graph = {}
    for graph in graphs:
        schedule = build_causal_schedule(graph, effective_horizon=depths[graph.graph_id])
        active_by_graph[graph.graph_id] = {
            round_index: set(nodes)
            for round_index, nodes in enumerate(schedule.active_nodes_by_round)
        }

    specs_by_pair: dict[tuple[str, str, int, int], list[ExecutionRunSpec]] = defaultdict(list)
    for spec in plan:
        specs_by_pair[_pair_key(spec)].append(spec)
    run_rows_by_id: dict[str, RunMetricRow] = {}
    pair_rows_by_id: dict[str, PairedAttackRow] = {}
    for specs in specs_by_pair.values():
        clean_specs = [spec for spec in specs if spec.condition is RunCondition.CLEAN]
        if len(clean_specs) != 1:
            raise ValueError("each task-graph cell must contain exactly one clean run")
        clean_spec = clean_specs[0]
        graph = graph_by_id[clean_spec.graph_id]
        depth = depths[graph.graph_id]
        active = active_by_graph[graph.graph_id]
        clean = _project_trace(_load_raw_trace(batch_dir, clean_spec), graph, active, depth)
        run_rows_by_id[clean_spec.run_spec_id] = _run_metric(clean_spec, clean)
        for attack_spec in specs:
            if attack_spec.condition is not RunCondition.ATTACK:
                continue
            attack = _project_trace(
                _load_raw_trace(batch_dir, attack_spec), graph, active, depth
            )
            run_rows_by_id[attack_spec.run_spec_id] = _run_metric(attack_spec, attack)
            pair_rows_by_id[attack_spec.run_spec_id] = _paired_attack(
                clean_spec, clean, attack_spec, attack, graph
            )

    run_rows = tuple(run_rows_by_id[spec.run_spec_id] for spec in plan)
    pair_rows = tuple(
        pair_rows_by_id[spec.run_spec_id]
        for spec in plan
        if spec.condition is RunCondition.ATTACK
    )
    rows_by_graph: dict[str, list[RunMetricRow]] = defaultdict(list)
    pairs_by_graph: dict[str, list[PairedAttackRow]] = defaultdict(list)
    for row in run_rows:
        rows_by_graph[row.graph_id].append(row)
    for pair in pair_rows:
        pairs_by_graph[pair.graph_id].append(pair)
    graph_metrics = tuple(
        _graph_metric(
            graph,
            [row for row in rows_by_graph[graph.graph_id] if row.condition is RunCondition.CLEAN],
            [row for row in rows_by_graph[graph.graph_id] if row.condition is RunCondition.ATTACK],
            pairs_by_graph[graph.graph_id],
        )
        for graph in graphs
    )
    classical_path = batch_dir.parent / "analysis-v1" / "classical_initial_states.jsonl"
    classical = _read_jsonl(classical_path, ClassicalInitialStateRecord)
    analysis_manifest = AnalysisManifest(
        analyzer_version="paired-analysis-v1-virtual-graph-depth",
        source_batch_runner_version=manifest.runner_version,
        source_batch_manifest_fingerprint=content_fingerprint(manifest),
        expected_runs=manifest.expected_run_count,
        analyzed_runs=len(run_rows),
        paired_attacks=len(pair_rows),
        graph_count=len(graph_metrics),
    )
    return (
        AnalysisResult(
            manifest=analysis_manifest,
            run_metrics=run_rows,
            paired_attacks=pair_rows,
            graph_metrics=graph_metrics,
            classical_initial_states=classical,
        ),
        depths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result, depths = analyze_streaming(args.fixed_batch)
    write_analysis(args.output_dir, result)
    metadata = {
        "analysis_mode": "virtual-graph-depth-from-fixed-prefix-v1",
        "source_batch": str(args.fixed_batch.resolve()),
        "graph_depths": depths,
        "analyzed_runs": result.manifest.analyzed_runs,
        "paired_attacks": result.manifest.paired_attacks,
    }
    (args.output_dir / "virtual_graph_depth_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
