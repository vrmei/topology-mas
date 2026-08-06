"""Preflight checks for a self-hosted OpenAI-compatible text server."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from topology_mas.execution.answers import parse_numeric_answer
from topology_mas.execution.generation import TextGenerator
from topology_mas.execution.schemas import ChatMessage, TextGenerationRequest

SERVER_PROBE_VERSION = "openai-compatible-server-probe-v1"


class ServerProbeConfig(BaseModel):
    """Frozen request used to calibrate one serving endpoint."""

    model_config = ConfigDict(frozen=True)

    requested_model: str = Field(min_length=1)
    expected_returned_model: str | None = None
    repetitions: int = Field(default=3, ge=2)
    seed: int = 17
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=128, ge=1)
    prompt: str = Field(
        default=(
            "Solve 37 + 5. End with one plain final line in exactly this format: "
            "FINAL_ANSWER: <number>"
        ),
        min_length=1,
    )


class ServerProbeAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt_index: int = Field(ge=0)
    returned_model: str | None = None
    finish_reason: str | None = None
    raw_output: str
    raw_output_sha256: str = Field(min_length=64, max_length=64)
    parsed_answer: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    provider_request_id: str | None = None


class ServerProbeReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    probe_version: str = SERVER_PROBE_VERSION
    base_url: str = Field(min_length=1)
    config: ServerProbeConfig
    attempts: tuple[ServerProbeAttempt, ...]
    exact_repeat_observed: bool
    returned_model_stable: bool
    returned_model_matches_expectation: bool | None = None
    all_outputs_parseable: bool
    token_usage_complete: bool


def run_server_probe(
    generator: TextGenerator,
    *,
    base_url: str,
    config: ServerProbeConfig,
) -> ServerProbeReport:
    """Repeat one identical request; this observes repeatability but does not prove it."""

    messages = (ChatMessage(role="user", content=config.prompt),)
    attempts: list[ServerProbeAttempt] = []
    for attempt_index in range(config.repetitions):
        result = generator.generate(
            TextGenerationRequest(
                request_id=f"server-probe-{attempt_index}",
                messages=messages,
                seed=config.seed,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            )
        )
        attempts.append(
            ServerProbeAttempt(
                attempt_index=attempt_index,
                returned_model=result.model_name,
                finish_reason=result.finish_reason,
                raw_output=result.raw_text,
                raw_output_sha256=hashlib.sha256(
                    result.raw_text.encode("utf-8")
                ).hexdigest(),
                parsed_answer=parse_numeric_answer(result.raw_text),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                provider_request_id=result.metadata.get("provider_request_id"),
            )
        )

    output_hashes = {attempt.raw_output_sha256 for attempt in attempts}
    returned_models = {attempt.returned_model for attempt in attempts}
    expected_match = (
        None
        if config.expected_returned_model is None
        else returned_models == {config.expected_returned_model}
    )
    return ServerProbeReport(
        base_url=base_url.rstrip("/"),
        config=config,
        attempts=tuple(attempts),
        exact_repeat_observed=len(output_hashes) == 1,
        returned_model_stable=len(returned_models) == 1,
        returned_model_matches_expectation=expected_match,
        all_outputs_parseable=all(
            attempt.parsed_answer is not None for attempt in attempts
        ),
        token_usage_complete=all(
            attempt.input_tokens is not None and attempt.output_tokens is not None
            for attempt in attempts
        ),
    )


def write_server_probe_report(path: str | Path, report: ServerProbeReport) -> Path:
    """Persist the calibration report atomically without credentials."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise ValueError(f"existing server probe report differs at {destination}")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
