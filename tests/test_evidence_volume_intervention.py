from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from topology_mas.experiments.evidence_volume import (
    RATIO_DESIGNS,
    build_request_plan,
    render_request_messages,
)
from topology_mas.models import TaskInstance


def make_pool() -> tuple[dict[tuple[str, str], list[str]], dict[str, dict[str, str]]]:
    pool: dict[tuple[str, str], list[str]] = {}
    stimuli: dict[str, dict[str, str]] = {}
    for state, count in (("correct", 20), ("target_error", 10), ("other_error", 10)):
        ids = []
        for index in range(count):
            item_id = f"{state}-{index}"
            ids.append(item_id)
            stimuli[item_id] = {
                "stimulus_id": item_id,
                "state": state,
                "raw_text": f"Distinct {state} rationale {index}. FINAL_ANSWER: {index + 1}",
                "parsed_answer": str(index + 1),
            }
        pool[("task-1", state)] = ids
    return pool, stimuli


def test_plan_is_nested_and_paired() -> None:
    pool, _ = make_pool()
    plan = build_request_plan(task_ids=["task-1"], pool_by_task_state=pool, replicates=2)
    assert len(plan) == 2 * len(RATIO_DESIGNS) * 2 * 3
    keys = ["scenario", "ratio_id", "replicate"]
    for _, group in pd.DataFrame(plan).groupby(keys):
        by_multiplier = {row.multiplier: row for row in group.itertuples(index=False)}
        assert by_multiplier[1].generation_seed == by_multiplier[3].generation_seed
        assert by_multiplier[1].previous_stimulus_id == by_multiplier[3].previous_stimulus_id
        assert (
            set(by_multiplier[1].peer_stimulus_ids) < set(by_multiplier[2].peer_stimulus_ids)
            or not by_multiplier[1].peer_stimulus_ids
        )
        assert set(by_multiplier[2].peer_stimulus_ids).issubset(
            set(by_multiplier[3].peer_stimulus_ids)
        )
        assert len(by_multiplier[3].peer_stimulus_ids) == by_multiplier[3].incoming_degree
        assert len(set(by_multiplier[3].peer_stimulus_ids)) == by_multiplier[3].incoming_degree


def test_rendered_prompt_has_no_source_identity_or_duplicate_peer_text() -> None:
    pool, stimuli = make_pool()
    plan = build_request_plan(task_ids=["task-1"], pool_by_task_state=pool, replicates=1)
    row = next(
        item for item in plan if item["scenario"] == "attack_adoption" and item["multiplier"] == 3
    )
    task = TaskInstance(
        task_id="task-1",
        dataset="gsm8k",
        split="test",
        prompt="What is 1+1?",
        reference_answer="2",
        oracle_type="numeric",
    )
    messages = render_request_messages(task=task, plan_row=row, stimuli=stimuli)
    prompt = messages[-1].content
    assert prompt.count("<peer_message>") == row["incoming_degree"]
    assert "source_node" not in prompt
    assert "graph_id" not in prompt
    assert len(set(row["peer_stimulus_ids"])) == row["incoming_degree"]


def load_analysis_module():
    path = Path(__file__).parents[1] / "scripts" / "analyze_evidence_volume_intervention.py"
    spec = importlib.util.spec_from_file_location("evidence_analysis", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_contrast_uses_task_clusters() -> None:
    module = load_analysis_module()
    rows = []
    for task_index in range(8):
        for multiplier in (1, 2, 3):
            rows.append(
                {
                    "task_id": f"t{task_index}",
                    "scenario": "attack_adoption",
                    "ratio_id": "c50_e50",
                    "replicate": 0,
                    "multiplier": multiplier,
                    "is_primary_outcome": multiplier == 3,
                }
            )
    contrasts, tasks = module.paired_contrasts(
        pd.DataFrame(rows), bootstrap_replicates=200, bootstrap_seed=7
    )
    primary = contrasts[(contrasts.ratio_id == "pooled") & (contrasts.contrast == "3x-1x")].iloc[0]
    assert primary.estimate == 1.0
    assert primary.directional
    assert tasks.task_id.nunique() == 8


def test_self_plus_peers_degroot_is_not_scale_invariant() -> None:
    # Previous state is C, so it contributes no target-error mass. At fixed 1:1
    # peer composition, scaling 1T+1C to 3T+3C dilutes the one self-state vote.
    small_target_mass = 1 / (1 + 1 + 1)
    large_target_mass = 3 / (1 + 3 + 3)
    assert large_target_mass > small_target_mass

    # A peer-only equal-weight mixture remains exactly scale invariant.
    assert 1 / (1 + 1) == 3 / (3 + 3)
