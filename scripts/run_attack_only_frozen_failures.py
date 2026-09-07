#!/usr/bin/env python3
"""Run only a frozen set of attack run-spec IDs using the normal batch stack."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from topology_mas.execution import batch_cli
from topology_mas.execution.batch import BatchExecutionRunner as Original


def parse_wrapper_args() -> tuple[set[str], list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--frozen-run-spec-ids", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    ids = {
        line.strip()
        for line in known.frozen_run_spec_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not ids:
        raise ValueError("frozen run-spec set is empty")
    return ids, [sys.argv[0], *remaining]


TARGET_IDS, FORWARDED_ARGV = parse_wrapper_args()
TIMING_LOCK = threading.Lock()


class FrozenFailureRunner(Original):
    def _build_plan(self, *args, **kwargs):
        original = super()._build_plan(*args, **kwargs)
        selected = tuple(spec for spec in original if spec.run_spec_id in TARGET_IDS)
        selected_ids = {spec.run_spec_id for spec in selected}
        missing = TARGET_IDS - selected_ids
        if missing:
            raise RuntimeError(f"frozen IDs absent from regenerated plan: {sorted(missing)}")
        if any(str(getattr(spec.condition, "value", spec.condition)) != "attack" for spec in selected):
            raise RuntimeError("frozen recovery set contains a non-attack spec")
        print(
            f"[frozen-failure-recovery] source_plan={len(original)} selected={len(selected)}",
            flush=True,
        )
        return selected

    def _execute_one(self, *, spec, **kwargs):
        """Record actual per-cell wall time separately from provider latency."""

        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        status = "success"
        error_type = None
        try:
            return super()._execute_one(spec=spec, **kwargs)
        except Exception as exc:
            status = "failure"
            error_type = type(exc).__name__
            raise
        finally:
            record = {
                "run_spec_id": spec.run_spec_id,
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "run_wall_seconds": time.perf_counter() - started,
                "status": status,
                "error_type": error_type,
            }
            timing_path = self.store.root / "recovery_run_wall.jsonl"
            with TIMING_LOCK:
                with timing_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")


batch_cli.BatchExecutionRunner = FrozenFailureRunner


if __name__ == "__main__":
    sys.argv = FORWARDED_ARGV
    batch_cli.main()
