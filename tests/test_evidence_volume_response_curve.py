from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path

import pandas as pd

from topology_mas.experiments.evidence_volume_curve import (
    ATTACK_DESIGNS,
    build_curve_request_plan,
    build_token_matched_plan,
    select_supported_tasks,
)
from topology_mas.models import AnswerState


def load_analysis_module():
    path = Path(__file__).parents[1] / "scripts" / "analyze_evidence_volume_response_curve.py"
    spec = importlib.util.spec_from_file_location("evidence_curve_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_pool(tasks: int = 41, per_state: int = 80):
    pool: dict[tuple[str, str], list[str]] = defaultdict(list)
    lengths: dict[str, int] = {}
    for task_index in range(tasks):
        task_id = f"task-{task_index:02d}"
        for state in (
            AnswerState.CORRECT.value,
            AnswerState.TARGET_ERROR.value,
            AnswerState.OTHER_ERROR.value,
        ):
            for item_index in range(per_state):
                item = f"{task_id}-{state}-{item_index:03d}"
                pool[(task_id, state)].append(item)
                lengths[item] = 20 + item_index * 3
    return pool, lengths


def test_curve_plan_has_frozen_size_and_nested_peers() -> None:
    pool, _lengths = make_pool()
    tasks = select_supported_tasks(pool)
    plan = build_curve_request_plan(task_ids=tasks, pool_by_task_state=pool)
    assert len(tasks) == 40
    assert len(plan) == 13_480
    attack = [
        row
        for row in plan
        if row["scenario"] == "attack_adoption"
        and row["task_id"] == tasks[0]
        and row["ratio_id"] == "c50_t50"
        and row["replicate"] == 0
        and row["previous_mode"] == "omit"
    ]
    attack.sort(key=lambda row: int(row["incoming_degree"]))
    assert [row["incoming_degree"] for row in attack] == list(
        ATTACK_DESIGNS[-1].degrees
    )
    for smaller, larger in zip(attack, attack[1:], strict=False):
        assert set(smaller["peer_stimulus_ids"]) < set(larger["peer_stimulus_ids"])
    for row in attack:
        assert row["correct_count"] == row["error_count"]
        assert len(row["peer_stimulus_ids"]) == row["incoming_degree"]


def test_previous_modes_share_peer_sets_and_seeds() -> None:
    pool, _lengths = make_pool()
    task = select_supported_tasks(pool)[0]
    plan = build_curve_request_plan(task_ids=[task], pool_by_task_state=pool)
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in plan:
        if row["scenario"] != "attack_adoption":
            continue
        key = (row["ratio_id"], row["replicate"], row["incoming_degree"])
        groups[key].append(row)
    assert groups
    for rows in groups.values():
        assert len(rows) == 2
        assert rows[0]["peer_stimulus_ids"] == rows[1]["peer_stimulus_ids"]
        assert rows[0]["generation_seed"] == rows[1]["generation_seed"]


def test_token_matched_plan_is_disjoint_and_within_tolerance() -> None:
    pool, lengths = make_pool(tasks=40)
    tasks = select_supported_tasks(pool)[:1]
    plan = build_token_matched_plan(
        task_ids=tasks,
        pool_by_task_state=pool,
        token_lengths=lengths,
    )
    assert len(plan) == 10
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in plan:
        groups[str(row["token_match_pair_id"])].append(row)
    for rows in groups.values():
        assert len(rows) == 2
        by_condition = {str(row["token_match_condition"]): row for row in rows}
        long = by_condition["four_long"]
        short = by_condition["eight_short"]
        assert long["incoming_degree"] == 4
        assert short["incoming_degree"] == 8
        assert set(long["peer_stimulus_ids"]).isdisjoint(short["peer_stimulus_ids"])
        long_tokens = int(long["peer_message_tokens"])
        short_tokens = int(short["peer_message_tokens"])
        tolerance = max(96, 0.10 * ((long_tokens + short_tokens) / 2))
        assert abs(long_tokens - short_tokens) <= tolerance
        assert long_tokens / 4 > short_tokens / 8
        assert long["generation_seed"] == short["generation_seed"]


def test_analysis_builds_curve_and_token_contrasts() -> None:
    module = load_analysis_module()
    rows: list[dict[str, object]] = []
    grids = {
        "c80_t20": (5, 10, 15, 30, 50),
        "c67_t33": (3, 6, 9, 30, 48),
        "c50_t50": (2, 4, 6, 30, 50),
    }
    for task_index in range(4):
        for mode in ("include", "omit"):
            for ratio, degrees in grids.items():
                for degree in degrees:
                    outcome = int(degree >= 30 and task_index % 2 == 0)
                    rows.append(
                        {
                            "request_kind": "response_curve",
                            "scenario": "attack_adoption",
                            "previous_mode": mode,
                            "ratio_id": ratio,
                            "task_id": f"task-{task_index}",
                            "incoming_degree": degree,
                            "token_match_condition": None,
                            "is_primary_outcome": outcome,
                            "is_target": outcome,
                            "is_correct": 1 - outcome,
                            "is_other": 0,
                            "is_unparsed": 0,
                            "input_tokens": 100 + degree,
                            "output_tokens": 20,
                            "latency_ms": 10,
                            "peer_message_tokens": 80 + degree,
                        }
                    )
        for condition, outcome, degree in (
            ("four_long", 0, 4),
            ("eight_short", task_index % 2, 8),
        ):
            rows.append(
                {
                    "request_kind": "token_matched",
                    "scenario": "attack_adoption",
                    "previous_mode": "omit",
                    "ratio_id": "c50_t50_token_matched",
                    "task_id": f"task-{task_index}",
                    "incoming_degree": degree,
                    "token_match_condition": condition,
                    "is_primary_outcome": outcome,
                    "is_target": outcome,
                    "is_correct": 1 - outcome,
                    "is_other": 0,
                    "is_unparsed": 0,
                    "input_tokens": 1000,
                    "output_tokens": 20,
                    "latency_ms": 10,
                    "peer_message_tokens": 800,
                }
            )
    frame = pd.DataFrame(rows)
    cells = module.cell_summary(frame, bootstraps=100)
    contrasts = module.degree_contrasts(frame, bootstraps=100)
    token = module.token_matched_contrast(frame, bootstraps=100)
    links = module.out_of_range_link_evaluation(frame)
    assert not cells.empty
    assert (cells.primary_parsed_rate == cells.primary_rate).all()
    assert set(contrasts.contrast) >= {"adjacent", "high_tail", "pooled_high_tail"}
    assert token.iloc[0].effect == 0.5
    assert len(links) == 2 * 3 * 8
