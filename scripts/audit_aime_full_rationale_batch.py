#!/usr/bin/env python3
"""Verify one-stage verbatim communication in a complete AIME clean batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from topology_mas.analysis.loader import load_complete_batch
from topology_mas.execution.aime import AIME_FULL_RATIONALE_PROMPT_VERSION

AUDIT_VERSION = "aime-full-rationale-audit-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def turn_key(turn: Any) -> tuple[int, int]:
    return turn.round_index, turn.node_id


def main() -> None:
    args = parse_args()
    batch = load_complete_batch(args.batch_dir)
    manifest = batch.manifest
    if manifest.prompt_version != AIME_FULL_RATIONALE_PROMPT_VERSION:
        raise ValueError("batch is not the frozen full-rationale protocol")
    if manifest.execution_settings.generation_pipeline != "single-pass":
        raise ValueError("full-rationale batch is not single-pass")

    failures: list[dict[str, Any]] = []
    total_turns = 0
    total_messages = 0
    model_calls = 0
    backend_calls = 0
    example = None
    for stored in batch.runs:
        trace = stored.trace
        turns = {turn_key(turn): turn for turn in trace.turns}
        model_calls += trace.total_model_calls
        backend_calls += trace.total_backend_calls or 0
        total_turns += len(trace.turns)
        total_messages += len(trace.messages)
        for turn in trace.turns:
            audit = turn.metadata.get("communication_audit")
            if not isinstance(audit, dict):
                failures.append(
                    {
                        "run_id": trace.run_id,
                        "turn": turn_key(turn),
                        "error": "missing_audit",
                    }
                )
                continue
            for flag in (
                "summarization",
                "message_compression",
                "context_overflow",
                "context_truncation",
            ):
                if audit.get(flag) is not False:
                    failures.append(
                        {"run_id": trace.run_id, "turn": turn_key(turn), "error": flag}
                    )
            visible = "\n".join(message["content"] for message in turn.prompt_messages)
            if turn.previous_raw_output is not None and turn.previous_raw_output not in visible:
                failures.append(
                    {
                        "run_id": trace.run_id,
                        "turn": turn_key(turn),
                        "error": "previous_not_verbatim",
                    }
                )
            incoming = [
                message
                for message in trace.messages
                if message.message_id in turn.incoming_message_ids
            ]
            for message in incoming:
                if message.raw_text not in visible:
                    failures.append(
                        {
                            "run_id": trace.run_id,
                            "turn": turn_key(turn),
                            "error": "incoming_not_verbatim",
                        }
                    )
            if example is None and turn.round_index in (1, 2) and incoming:
                example = {
                    "run_id": trace.run_id,
                    "task_id": trace.task_id,
                    "graph_id": trace.graph_id,
                    "receiver_id": turn.node_id,
                    "round": turn.round_index,
                    "own_previous_response": turn.previous_raw_output,
                    "incoming_responses": [
                        {
                            "sender_id": message.sender,
                            "message_id": message.message_id,
                            "raw_response": message.raw_text,
                            "tokens": message.output_tokens,
                            "sha256": sha256(message.raw_text),
                        }
                        for message in incoming
                    ],
                    "actual_prompt_messages": list(turn.prompt_messages),
                    "generated_raw_response": turn.raw_output,
                    "communication_audit": audit,
                }
        for message in trace.messages:
            source = turns[(message.round_index, message.sender)]
            if message.raw_text != source.raw_output:
                failures.append(
                    {
                        "run_id": trace.run_id,
                        "message_id": message.message_id,
                        "error": "broadcast_not_raw",
                    }
                )
            if message.metadata.get("public_message_equals_raw_output") is not True:
                failures.append(
                    {
                        "run_id": trace.run_id,
                        "message_id": message.message_id,
                        "error": "identity_flag_false",
                    }
                )

    report = {
        "audit_version": AUDIT_VERSION,
        "batch_dir": str(args.batch_dir.resolve()),
        "prompt_version": manifest.prompt_version,
        "generation_pipeline": manifest.execution_settings.generation_pipeline,
        "runs": len(batch.runs),
        "turns": total_turns,
        "messages": total_messages,
        "trace_model_calls": model_calls,
        "trace_backend_calls": backend_calls,
        "one_physical_call_per_node_update": (
            model_calls == backend_calls == total_turns
        ),
        "no_summarization": True,
        "no_message_compression": True,
        "full_raw_peer_responses": True,
        "failure_count": len(failures),
        "failures": failures,
        "example_file": "prompt_audit_example.json" if example is not None else None,
    }
    if failures or not report["one_physical_call_per_node_update"]:
        report["no_summarization"] = False
        report["no_message_compression"] = False
        report["full_raw_peer_responses"] = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if example is not None:
        (args.output_dir / "prompt_audit_example.json").write_text(
            json.dumps(example, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failures or not report["one_physical_call_per_node_update"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
