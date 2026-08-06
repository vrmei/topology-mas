from pathlib import Path

import pytest

from topology_mas.execution import (
    ServerProbeConfig,
    TextGenerationResult,
    run_server_probe,
    write_server_probe_report,
)


class StubGenerator:
    def __init__(self, outputs: tuple[str, ...]) -> None:
        self.outputs = outputs
        self.calls = 0

    def generate(self, request: object) -> TextGenerationResult:
        output = self.outputs[self.calls]
        self.calls += 1
        return TextGenerationResult(
            raw_text=output,
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            finish_reason="stop",
            input_tokens=20,
            output_tokens=8,
            latency_ms=12.5,
            metadata={"provider_request_id": f"req-{self.calls}"},
        )


def config() -> ServerProbeConfig:
    return ServerProbeConfig(
        requested_model="meta-llama/Llama-3.1-8B-Instruct",
        expected_returned_model="meta-llama/Llama-3.1-8B-Instruct",
        repetitions=3,
    )


def test_server_probe_records_exact_repeat_and_required_fields() -> None:
    report = run_server_probe(
        StubGenerator(("FINAL_ANSWER: 42",) * 3),
        base_url="http://server:8000/v1/",
        config=config(),
    )

    assert report.base_url == "http://server:8000/v1"
    assert report.exact_repeat_observed is True
    assert report.returned_model_stable is True
    assert report.returned_model_matches_expectation is True
    assert report.all_outputs_parseable is True
    assert report.token_usage_complete is True
    assert {attempt.parsed_answer for attempt in report.attempts} == {"42"}
    assert len({attempt.raw_output_sha256 for attempt in report.attempts}) == 1


def test_server_probe_detects_nonidentical_outputs_without_overclaiming() -> None:
    report = run_server_probe(
        StubGenerator(
            (
                "Reasoning A\nFINAL_ANSWER: 42",
                "Reasoning B\nFINAL_ANSWER: 42",
                "Reasoning C\nFINAL_ANSWER: 42",
            )
        ),
        base_url="http://server:8000/v1",
        config=config(),
    )

    assert report.exact_repeat_observed is False
    assert report.all_outputs_parseable is True


def test_server_probe_report_is_conflict_safe(tmp_path: Path) -> None:
    report = run_server_probe(
        StubGenerator(("FINAL_ANSWER: 42",) * 3),
        base_url="http://server:8000/v1",
        config=config(),
    )
    path = tmp_path / "probe.json"

    assert write_server_probe_report(path, report) == path
    assert write_server_probe_report(path, report) == path

    changed = report.model_copy(update={"exact_repeat_observed": False})
    with pytest.raises(ValueError, match="existing server probe report differs"):
        write_server_probe_report(path, changed)
