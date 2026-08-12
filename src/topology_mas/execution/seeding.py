"""Stable experiment seeds and identifiers independent of Python hash randomization."""

from __future__ import annotations

import hashlib


def stable_integer(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


def stable_fingerprint(*parts: object) -> str:
    """Full SHA-256 identity for persisted experimental content."""

    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_replica_round_seed(
    *,
    experiment_seed: int,
    task_id: str,
    replica_slot: int,
    round_index: int,
) -> int:
    """Post-Round-zero inference stream attached to an anonymous replica."""

    # Preserve the persisted seed namespace used by existing artifacts. Here "online"
    # is a legacy hash salt, not a claim that inference uses an external API.
    return stable_integer(
        "online-replica-round",
        experiment_seed,
        task_id,
        replica_slot,
        round_index,
    )


def independent_run_round_seed(
    *,
    run_id: str,
    node_id: int,
    round_index: int,
) -> int:
    """Independent stream bound to one graph/condition/attacker run.

    Unlike the historical anonymous-replica seed, this identity deliberately includes
    the full run ID. Two topologies or conditions therefore cannot receive the same
    stochastic stream merely because their task and structural node number agree.
    """

    return stable_integer("independent-run-round", run_id, node_id, round_index)


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
