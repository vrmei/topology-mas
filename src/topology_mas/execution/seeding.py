"""Stable experiment seeds and identifiers independent of Python hash randomization."""

from __future__ import annotations

import hashlib


def stable_integer(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


def node_round_seed(*, experiment_seed: int, task_id: str, node_id: int, round_index: int) -> int:
    """Common random number key shared by graph and attack conditions."""

    return stable_integer(experiment_seed, task_id, node_id, round_index)
