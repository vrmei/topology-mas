#!/usr/bin/env python3
"""Audit a frozen scalable AIME Round-0 pool and export exact fidelity examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from topology_mas.execution.aime import parse_aime_answer
from topology_mas.execution.scalable_protocol import (
    AIME_SUMMARY_INTERFACE_MAX_OUTPUT_TOKENS,
    AIME_SUMMARY_INTERFACE_MIN_P,
    AIME_SUMMARY_INTERFACE_MODEL,
    AIME_SUMMARY_INTERFACE_PRESENCE_PENALTY,
    AIME_SUMMARY_INTERFACE_TEMPERATURE,
    AIME_SUMMARY_INTERFACE_TOP_K,
    AIME_SUMMARY_INTERFACE_TOP_P,
    SCALABLE_ADAPTIVE_ATTACK_FINAL_INSTRUCTION,
    SCALABLE_ADAPTIVE_ATTACK_SYSTEM_PROMPT,
    SCALABLE_DUAL_CHANNEL_PROMPT_VERSION,
    SCALABLE_NORMAL_FINAL_INSTRUCTION,
    SCALABLE_NORMAL_SYSTEM_PROMPT_TEMPLATE,
    SCALABLE_PUBLIC_SUMMARY_MAX_TOKENS,
    HuggingFaceTokenCounter,
    parse_aime_summary_answer,
    parse_dual_channel_output,
    validate_dual_channel_output,
)
from topology_mas.execution.scalable_round_zero import ScalableRoundZeroPoolStore


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--pool", type=Path, required=True)
    value.add_argument("--tasks", type=Path, required=True)
    value.add_argument("--tokenizer", required=True)
    value.add_argument("--tokenizer-cache-dir", type=Path)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def load_gold(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[row["task_id"]] = str(int(row["gold_answer"]))
    return result


def state(answer: str | None, gold: str) -> str:
    if answer is None:
        return "U"
    return "C" if str(int(answer)) == gold else "O"


def protocol_manifest() -> dict[str, Any]:
    system = SCALABLE_NORMAL_SYSTEM_PROMPT_TEMPLATE.format(
        answer_contract=(
            "End both sections with FINAL_ANSWER: \\boxed{ddd}, where ddd is the "
            "zero-padded integer from 000 through 999."
        )
    )
    round_zero_user = (
        "PROBLEM:\n{problem}\n\n" + SCALABLE_NORMAL_FINAL_INSTRUCTION
    )
    later_user = (
        "PROBLEM:\n{problem}\n\n"
        "YOUR_PREVIOUS_FULL_SOLUTION:\n{own_previous_full_solution}\n\n"
        "<peer_public_summary>\n{peer_1_summary}\n</peer_public_summary>\n\n"
        "... repeat one peer_public_summary block per in-neighbor in deterministic "
        "sender order ...\n\n"
        + SCALABLE_NORMAL_FINAL_INSTRUCTION
    )
    adaptive_user = (
        "PROBLEM:\n{problem}\n\n"
        "FROZEN_TARGET_ANSWER:\n{target_answer}\n\n"
        "YOUR_PREVIOUS_FULL_SOLUTION:\n{own_previous_full_attack_solution}\n\n"
        "<peer_public_summary>\n{peer_1_summary}\n</peer_public_summary>\n\n"
        "... repeat one peer_public_summary block per in-neighbor in deterministic "
        "sender order ...\n\n"
        + SCALABLE_ADAPTIVE_ATTACK_FINAL_INSTRUCTION
    )
    prompts = {
        "normal_system_prompt": system,
        "normal_round_zero_user_template": round_zero_user,
        "normal_later_round_user_template": later_user,
        "adaptive_attack_system_prompt": SCALABLE_ADAPTIVE_ATTACK_SYSTEM_PROMPT,
        "adaptive_attack_later_round_user_template": adaptive_user,
    }
    return {
        "protocol_version": SCALABLE_DUAL_CHANNEL_PROMPT_VERSION,
        "generation_design": (
            "single Qwen call jointly emits FULL_SOLUTION and PUBLIC_SUMMARY; "
            "there is no independent summarizer call"
        ),
        "summarizer_model": AIME_SUMMARY_INTERFACE_MODEL,
        "sampling": {
            "temperature": AIME_SUMMARY_INTERFACE_TEMPERATURE,
            "top_p": AIME_SUMMARY_INTERFACE_TOP_P,
            "top_k": AIME_SUMMARY_INTERFACE_TOP_K,
            "min_p": AIME_SUMMARY_INTERFACE_MIN_P,
            "presence_penalty": AIME_SUMMARY_INTERFACE_PRESENCE_PENALTY,
            "max_tokens_for_joint_completion": AIME_SUMMARY_INTERFACE_MAX_OUTPUT_TOKENS,
            "max_public_summary_tokens": SCALABLE_PUBLIC_SUMMARY_MAX_TOKENS,
        },
        "prompts": prompts,
        "prompt_sha256": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in prompts.items()
        },
        "summary_parser": {
            "primary": (
                "last explicit FINAL_ANSWER marker, otherwise last boxed 1--3 digit "
                "integer; normalize to decimal 0--999"
            ),
            "fallback": (
                "if no strict answer and no literal FINAL_ANSWER: UNPARSED, use the "
                "last standalone 1--3 digit integer; otherwise U"
            ),
        },
        "fidelity_validation": [
            "exactly one nonempty FULL_SOLUTION followed by one nonempty "
            "PUBLIC_SUMMARY; no outside text",
            "finish_reason must not be length",
            "PUBLIC_SUMMARY must be at most 2048 Qwen tokenizer tokens",
            "if FULL_SOLUTION is parsed, PUBLIC_SUMMARY must parse to the "
            "identical normalized answer",
            "if FULL_SOLUTION is U, PUBLIC_SUMMARY may not invent a parseable answer",
            "hash and token count are recorded for every accepted public summary",
        ],
        "cross_node_invariant": {
            "clean": "broadcast validated PUBLIC_SUMMARY only",
            "fixed_attack": "broadcast immutable prevalidated frozen PUBLIC_SUMMARY only",
            "adaptive_attack": (
                "observe peer summaries only and broadcast its validated "
                "PUBLIC_SUMMARY only"
            ),
            "self_history": "retain own previous FULL_SOLUTION",
        },
    }


def choose_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quotas = {"C": 4, "O": 4, "U": 2}
    selected: list[dict[str, Any]] = []
    used_tasks: set[tuple[str, str]] = set()
    for label, quota in quotas.items():
        candidates = sorted(
            (row for row in rows if row["full_state"] == label and row["answer_preserved"]),
            key=lambda row: (len(row["full_response"]), row["task_id"], row["pool_slot"]),
        )
        for row in candidates:
            key = (label, row["task_id"])
            if key in used_tasks:
                continue
            selected.append(row)
            used_tasks.add(key)
            if sum(item["full_state"] == label for item in selected) == quota:
                break
        if sum(item["full_state"] == label for item in selected) < quota:
            for row in candidates:
                if row in selected:
                    continue
                selected.append(row)
                if sum(item["full_state"] == label for item in selected) == quota:
                    break
    return selected


def main() -> None:
    args = parser().parse_args()
    manifest, records = ScalableRoundZeroPoolStore(args.pool).load_complete()
    gold = load_gold(args.tasks)
    counter = HuggingFaceTokenCounter(
        args.tokenizer,
        cache_dir=str(args.tokenizer_cache_dir) if args.tokenizer_cache_dir else None,
    )
    counts: Counter[str] = Counter(total=len(records))
    transitions: Counter[str] = Counter()
    valid_rows: list[dict[str, Any]] = []
    validation_errors: Counter[str] = Counter()
    for record in records:
        try:
            parts = parse_dual_channel_output(record.raw_response)
        except ValueError as exc:
            counts["structure_invalid"] += 1
            validation_errors[str(exc)] += 1
            continue
        counts["structure_valid"] += 1
        full_answer = parse_aime_answer(parts.full_solution)
        summary_answer = parse_aime_summary_answer(parts.public_summary)
        full_state = state(full_answer, gold[record.task_id])
        summary_state = state(summary_answer, gold[record.task_id])
        answer_preserved = full_answer == summary_answer
        state_preserved = full_state == summary_state
        counts["answer_preserved"] += int(answer_preserved)
        counts["state_preserved"] += int(state_preserved)
        counts["full_parseable"] += int(full_answer is not None)
        counts["answer_preserved_given_full_parseable"] += int(
            full_answer is not None and answer_preserved
        )
        transitions[f"{full_state}->{summary_state}"] += 1
        try:
            _, _, _, full_tokens, summary_tokens = validate_dual_channel_output(
                record.raw_response,
                answer_parser=parse_aime_answer,
                summary_answer_parser=parse_aime_summary_answer,
                token_counter=counter,
                max_public_tokens=SCALABLE_PUBLIC_SUMMARY_MAX_TOKENS,
                request_id=record.pool_response_id,
            )
            validation_passed = record.finish_reason != "length"
            if not validation_passed:
                validation_errors["finish_reason=length"] += 1
        except ValueError as exc:
            full_tokens = counter(parts.full_solution)
            summary_tokens = counter(parts.public_summary)
            validation_passed = False
            validation_errors[str(exc)] += 1
        counts["fidelity_validation_passed"] += int(validation_passed)
        valid_rows.append(
            {
                "task_id": record.task_id,
                "pool_slot": record.pool_slot,
                "pool_response_id": record.pool_response_id,
                "generation_seed": record.generation_seed,
                "gold_answer": gold[record.task_id],
                "full_response": parts.full_solution,
                "summary": parts.public_summary,
                "full_parsed_answer": full_answer,
                "summary_parsed_answer": summary_answer,
                "full_state": full_state,
                "summary_state": summary_state,
                "answer_preserved": answer_preserved,
                "state_preserved": state_preserved,
                "fidelity_validation_passed": validation_passed,
                "full_tokens": full_tokens,
                "summary_tokens": summary_tokens,
                "finish_reason": record.finish_reason,
            }
        )

    structural = counts["structure_valid"]
    full_parseable = counts["full_parseable"]
    result = {
        "pool_version": manifest.pool_version,
        "pool_prompt_version": manifest.config.prompt_version,
        "counts": dict(counts),
        "state_transitions": dict(sorted(transitions.items())),
        "rates": {
            "structure_valid_over_all": structural / counts["total"],
            "answer_preservation_over_structurally_valid": (
                counts["answer_preserved"] / structural if structural else None
            ),
            "state_preservation_over_structurally_valid": (
                counts["state_preserved"] / structural if structural else None
            ),
            "answer_preservation_given_parseable_full": (
                counts["answer_preserved_given_full_parseable"] / full_parseable
                if full_parseable
                else None
            ),
            "end_to_end_answer_preserving_interface_over_all": (
                counts["answer_preserved"] / counts["total"]
            ),
            "end_to_end_state_preserving_interface_over_all": (
                counts["state_preserved"] / counts["total"]
            ),
            "fidelity_validation_pass_over_all": (
                counts["fidelity_validation_passed"] / counts["total"]
            ),
        },
        "validation_errors": dict(validation_errors.most_common()),
        "note": (
            "Conditional preservation rates exclude malformed dual-channel outputs. "
            "End-to-end rates count them as interface failures, not preserved U states."
        ),
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol_manifest.json").write_text(
        json.dumps(protocol_manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "fidelity_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    examples = choose_examples(valid_rows)
    with (output / "examples_full.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(examples, start=1):
            handle.write(json.dumps({"example_index": index, **row}, ensure_ascii=False) + "\n")
    markdown = [
        "# Actual full-response to public-summary examples",
        "",
        "These are exact Qwen outputs from the frozen 2026 AIME Round-0 pool. ",
        "The JSONL companion preserves the same records in machine-readable form.",
        "",
        "| # | Task | Slot | Gold | Full answer/state | Summary answer/state | "
        "Tokens full→summary |",
        "|---:|---|---:|---:|---|---|---:|",
    ]
    for index, row in enumerate(examples, start=1):
        markdown.append(
            f"| {index} | `{row['task_id']}` | {row['pool_slot']} | "
            f"{row['gold_answer']} | {row['full_parsed_answer']}/{row['full_state']} | "
            f"{row['summary_parsed_answer']}/{row['summary_state']} | "
            f"{row['full_tokens']}→{row['summary_tokens']} |"
        )
    for index, row in enumerate(examples, start=1):
        markdown.extend(
            [
                "",
                f"## Example {index}: {row['full_state']}→{row['summary_state']}",
                "",
                "Full response (`FULL_SOLUTION` content):",
                "",
                "````text",
                row["full_response"],
                "````",
                "",
                "Public summary (`PUBLIC_SUMMARY` content):",
                "",
                "````text",
                row["summary"],
                "````",
            ]
        )
    (output / "examples_full.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
