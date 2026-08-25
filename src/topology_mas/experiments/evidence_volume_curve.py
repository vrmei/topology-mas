"""Frozen planning utilities for the extended evidence-volume response curve."""

from __future__ import annotations

import random
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from topology_mas.execution.seeding import stable_id
from topology_mas.experiments.evidence_volume import (
    content_fingerprint,
    select_nested_ids,
    stable_seed,
)
from topology_mas.models import AnswerState

EXPERIMENT_VERSION = "evidence-volume-response-curve-v1"


@dataclass(frozen=True)
class CurveDesign:
    scenario: str
    ratio_id: str
    base_correct: int
    base_error: int
    degrees: tuple[int, ...]
    previous_modes: tuple[str, ...]
    replicates: int

    def counts(self, degree: int) -> tuple[int, int]:
        base_degree = self.base_correct + self.base_error
        if degree % base_degree:
            raise ValueError(f"degree {degree} is incompatible with {self.ratio_id}")
        multiplier = degree // base_degree
        return self.base_correct * multiplier, self.base_error * multiplier

    @property
    def correct_share(self) -> float:
        return self.base_correct / (self.base_correct + self.base_error)


ATTACK_DESIGNS = (
    CurveDesign(
        "attack_adoption",
        "c80_t20",
        4,
        1,
        (5, 10, 15, 20, 25, 30, 40, 50),
        ("include", "omit"),
        5,
    ),
    CurveDesign(
        "attack_adoption",
        "c67_t33",
        2,
        1,
        (3, 6, 9, 12, 15, 18, 24, 30, 39, 48),
        ("include", "omit"),
        5,
    ),
    CurveDesign(
        "attack_adoption",
        "c50_t50",
        1,
        1,
        (2, 4, 6, 8, 12, 16, 20, 30, 40, 50),
        ("include", "omit"),
        5,
    ),
)

BENIGN_DESIGNS = (
    CurveDesign(
        "benign_correction",
        "c67_o33",
        2,
        1,
        (3, 6, 9, 12, 18, 24, 30, 39, 48),
        ("include",),
        3,
    ),
    CurveDesign(
        "benign_correction",
        "c50_o50",
        1,
        1,
        (2, 4, 6, 8, 12, 16, 20, 30, 40, 50),
        ("include",),
        3,
    ),
)

CURVE_DESIGNS = (*ATTACK_DESIGNS, *BENIGN_DESIGNS)
TOKEN_MATCHED_REPLICATES = 5


def select_supported_tasks(
    pool_by_task_state: Mapping[tuple[str, str], Sequence[str]],
    *,
    count: int = 40,
) -> list[str]:
    """Freeze tasks by joint T/O stimulus support, with a stable tie break."""

    task_ids = sorted({task_id for task_id, _state in pool_by_task_state})
    ranked = sorted(
        task_ids,
        key=lambda task_id: (
            -min(
                len(pool_by_task_state[(task_id, AnswerState.TARGET_ERROR.value)]),
                len(pool_by_task_state[(task_id, AnswerState.OTHER_ERROR.value)]),
            ),
            task_id,
        ),
    )
    if len(ranked) < count:
        raise ValueError(f"need {count} supported tasks, found {len(ranked)}")
    selected = ranked[:count]
    for task_id in selected:
        requirements = {
            AnswerState.CORRECT.value: 41,
            AnswerState.TARGET_ERROR.value: 25,
            AnswerState.OTHER_ERROR.value: 26,
        }
        for state, minimum in requirements.items():
            available = len(pool_by_task_state[(task_id, state)])
            if available < minimum:
                raise ValueError(
                    f"selected task {task_id} has {available} {state}; need {minimum}"
                )
    return selected


def _scenario_settings(scenario: str) -> dict[str, str]:
    if scenario == "attack_adoption":
        return {
            "previous_state": AnswerState.CORRECT.value,
            "error_state": AnswerState.TARGET_ERROR.value,
            "primary_state": AnswerState.TARGET_ERROR.value,
        }
    if scenario == "benign_correction":
        return {
            "previous_state": AnswerState.OTHER_ERROR.value,
            "error_state": AnswerState.OTHER_ERROR.value,
            "primary_state": AnswerState.CORRECT.value,
        }
    raise ValueError(f"unknown scenario: {scenario}")


def build_curve_request_plan(
    *,
    task_ids: Sequence[str],
    pool_by_task_state: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_id in task_ids:
        for design in CURVE_DESIGNS:
            settings = _scenario_settings(design.scenario)
            correct_pool = tuple(
                pool_by_task_state[(task_id, AnswerState.CORRECT.value)]
            )
            error_pool = tuple(
                pool_by_task_state[(task_id, settings["error_state"])]
            )
            previous_pool = tuple(
                pool_by_task_state[(task_id, settings["previous_state"])]
            )
            maximum_correct, maximum_error = design.counts(max(design.degrees))
            for replicate in range(design.replicates):
                previous_ids = list(previous_pool)
                random.Random(
                    stable_seed(
                        EXPERIMENT_VERSION,
                        task_id,
                        design.scenario,
                        design.ratio_id,
                        replicate,
                        "previous",
                    )
                ).shuffle(previous_ids)
                previous_id = previous_ids[0]
                correct_ids = select_nested_ids(
                    correct_pool,
                    maximum=maximum_correct,
                    seed_parts=(
                        EXPERIMENT_VERSION,
                        task_id,
                        design.scenario,
                        design.ratio_id,
                        replicate,
                        "correct",
                    ),
                    excluded={previous_id},
                )
                error_ids = select_nested_ids(
                    error_pool,
                    maximum=maximum_error,
                    seed_parts=(
                        EXPERIMENT_VERSION,
                        task_id,
                        design.scenario,
                        design.ratio_id,
                        replicate,
                        "error",
                    ),
                    excluded={previous_id},
                )
                generation_seed = stable_seed(
                    EXPERIMENT_VERSION,
                    task_id,
                    design.scenario,
                    design.ratio_id,
                    replicate,
                    "generation",
                )
                for degree in design.degrees:
                    correct_count, error_count = design.counts(degree)
                    peer_ids = (
                        *correct_ids[:correct_count],
                        *error_ids[:error_count],
                    )
                    for previous_mode in design.previous_modes:
                        request_id = stable_id(
                            "volume-curve-request",
                            EXPERIMENT_VERSION,
                            task_id,
                            design.scenario,
                            design.ratio_id,
                            replicate,
                            degree,
                            previous_mode,
                        )
                        rows.append(
                            {
                                "request_id": request_id,
                                "request_kind": "response_curve",
                                "task_id": task_id,
                                "scenario": design.scenario,
                                "previous_state": settings["previous_state"],
                                "error_state": settings["error_state"],
                                "primary_state": settings["primary_state"],
                                "ratio_id": design.ratio_id,
                                "correct_share": design.correct_share,
                                "error_share": 1.0 - design.correct_share,
                                "correct_count": correct_count,
                                "error_count": error_count,
                                "incoming_degree": degree,
                                "replicate": replicate,
                                "generation_seed": generation_seed,
                                "previous_mode": previous_mode,
                                "previous_stimulus_id": previous_id,
                                "peer_stimulus_ids": list(peer_ids),
                                "peer_set_fingerprint": content_fingerprint(
                                    {"stimulus_id": item} for item in sorted(peer_ids)
                                ),
                            }
                        )
    request_ids = [str(row["request_id"]) for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("curve request plan contains duplicate IDs")
    return rows


def _token_matched_pair(
    *,
    correct_ids: Sequence[str],
    target_ids: Sequence[str],
    token_lengths: Mapping[str, int],
    seed_parts: Sequence[object],
    trials: int = 50_000,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    rng = random.Random(stable_seed(*seed_parts))
    correct_ids = tuple(correct_ids)
    target_ids = tuple(target_ids)
    long_candidates: list[tuple[int, tuple[str, ...]]] = []
    short_candidates: list[tuple[int, tuple[str, ...]]] = []
    for _ in range(trials):
        long_ids = (*rng.sample(correct_ids, 2), *rng.sample(target_ids, 2))
        short_ids = (*rng.sample(correct_ids, 4), *rng.sample(target_ids, 4))
        long_candidates.append(
            (sum(token_lengths[item] for item in long_ids), tuple(long_ids))
        )
        short_candidates.append(
            (sum(token_lengths[item] for item in short_ids), tuple(short_ids))
        )
    short_candidates.sort(key=lambda item: item[0])
    short_sums = [item[0] for item in short_candidates]
    best: tuple[float, tuple[str, ...], tuple[str, ...], int, int] | None = None
    for long_tokens, long_ids in long_candidates:
        center = bisect_left(short_sums, long_tokens)
        for index in range(max(0, center - 20), min(len(short_candidates), center + 21)):
            short_tokens, short_ids = short_candidates[index]
            if set(long_ids) & set(short_ids):
                continue
            if long_tokens / 4 <= short_tokens / 8:
                continue
            difference = abs(long_tokens - short_tokens)
            score = difference / max(1.0, (long_tokens + short_tokens) / 2)
            candidate = (score, long_ids, short_ids, long_tokens, short_tokens)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ValueError("could not construct a token-matched message pair")
    _score, long_ids, short_ids, long_tokens, short_tokens = best
    tolerance = max(96, 0.10 * ((long_tokens + short_tokens) / 2))
    if abs(long_tokens - short_tokens) > tolerance:
        raise ValueError(
            f"token match failed: long={long_tokens}, short={short_tokens}, "
            f"tolerance={tolerance:.1f}"
        )
    return long_ids, short_ids, long_tokens, short_tokens


def build_token_matched_plan(
    *,
    task_ids: Sequence[str],
    pool_by_task_state: Mapping[tuple[str, str], Sequence[str]],
    token_lengths: Mapping[str, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_id in task_ids:
        correct_pool = tuple(
            pool_by_task_state[(task_id, AnswerState.CORRECT.value)]
        )
        target_pool = tuple(
            pool_by_task_state[(task_id, AnswerState.TARGET_ERROR.value)]
        )
        for replicate in range(TOKEN_MATCHED_REPLICATES):
            long_ids, short_ids, long_tokens, short_tokens = _token_matched_pair(
                correct_ids=correct_pool,
                target_ids=target_pool,
                token_lengths=token_lengths,
                seed_parts=(
                    EXPERIMENT_VERSION,
                    task_id,
                    "token_matched",
                    replicate,
                ),
            )
            generation_seed = stable_seed(
                EXPERIMENT_VERSION,
                task_id,
                "token_matched",
                replicate,
                "generation",
            )
            pair_id = stable_id(
                "token-match-pair", EXPERIMENT_VERSION, task_id, replicate
            )
            for label, peer_ids, peer_tokens in (
                ("four_long", long_ids, long_tokens),
                ("eight_short", short_ids, short_tokens),
            ):
                request_id = stable_id(
                    "token-match-request",
                    EXPERIMENT_VERSION,
                    task_id,
                    replicate,
                    label,
                )
                rows.append(
                    {
                        "request_id": request_id,
                        "request_kind": "token_matched",
                        "token_match_pair_id": pair_id,
                        "token_match_condition": label,
                        "task_id": task_id,
                        "scenario": "attack_adoption",
                        "previous_state": AnswerState.CORRECT.value,
                        "error_state": AnswerState.TARGET_ERROR.value,
                        "primary_state": AnswerState.TARGET_ERROR.value,
                        "ratio_id": "c50_t50_token_matched",
                        "correct_share": 0.5,
                        "error_share": 0.5,
                        "correct_count": len(peer_ids) // 2,
                        "error_count": len(peer_ids) // 2,
                        "incoming_degree": len(peer_ids),
                        "replicate": replicate,
                        "generation_seed": generation_seed,
                        "previous_mode": "omit",
                        "previous_stimulus_id": None,
                        "peer_stimulus_ids": list(peer_ids),
                        "peer_message_tokens": peer_tokens,
                        "peer_set_fingerprint": content_fingerprint(
                            {"stimulus_id": item} for item in sorted(peer_ids)
                        ),
                    }
                )
    return rows
