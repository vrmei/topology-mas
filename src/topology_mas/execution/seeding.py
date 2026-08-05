"""Stable experiment seeds and identifiers independent of Python hash randomization."""

from __future__ import annotations

import hashlib


def stable_integer(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


def online_replica_round_seed(
    *,
    experiment_seed: int,
    task_id: str,
    replica_slot: int,
    round_index: int,
) -> int:
    """Online sampling stream that moves with an assigned anonymous replica."""

    return stable_integer(
        "online-replica-round",
        experiment_seed,
        task_id,
        replica_slot,
        round_index,
    )


def round_zero_replica_seed(*, experiment_seed: int, task_id: str, replica_slot: int) -> int:
    """Graph-independent sampling seed for one cached initial-state replica."""

    return stable_integer(
        "round-zero-replica",
        experiment_seed,
        task_id,
        replica_slot,
    )


def anonymous_message_order_key(
    *,
    order_seed: int,
    task_id: str,
    round_index: int,
    raw_text: str,
) -> tuple[int, str]:
    """Label-invariant pseudorandom order based only on visible message content."""

    return (
        stable_integer("anonymous-message-order", order_seed, task_id, round_index, raw_text),
        raw_text,
    )
