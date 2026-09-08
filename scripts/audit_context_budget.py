#!/usr/bin/env python3
"""Find completed solve calls whose output budget was reduced by the provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--affected-ids", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.trace_dir.glob("*.json")):
        stored = json.loads(path.read_text(encoding="utf-8"))
        for turn in stored.get("trace", {}).get("turns", []):
            metadata = turn.get("metadata") or {}
            adjustment = (metadata.get("full_provider_metadata") or {}).get(
                "context_window_adjustment"
            )
            if adjustment is None:
                continue
            rows.append(
                {
                    "run_spec_id": path.stem,
                    "node_id": turn.get("node_id"),
                    "round_index": turn.get("round_index"),
                    "finish_reason": metadata.get("full_finish_reason"),
                    "answer_state": turn.get("answer_state"),
                    "final_answer_state": stored.get("trace", {}).get("final_answer_state"),
                    "full_input_tokens": metadata.get("full_input_tokens"),
                    "full_output_tokens": metadata.get("full_output_tokens"),
                    "requested_max_output_tokens": adjustment.get(
                        "requested_max_output_tokens"
                    ),
                    "effective_max_output_tokens": adjustment.get(
                        "effective_max_output_tokens"
                    ),
                    "requires_recovery": metadata.get("full_finish_reason") == "length",
                }
            )

    affected_ids = sorted({row["run_spec_id"] for row in rows if row["requires_recovery"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    args.affected_ids.write_text("".join(f"{item}\n" for item in affected_ids), encoding="utf-8")
    print(
        json.dumps(
            {
                "adjusted_turns": len(rows),
                "adjusted_cells": len({row["run_spec_id"] for row in rows}),
                "recovery_turns": sum(row["requires_recovery"] for row in rows),
                "recovery_cells": len(affected_ids),
            }
        )
    )


if __name__ == "__main__":
    main()
