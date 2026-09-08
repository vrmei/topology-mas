import pytest

from topology_mas.execution.schemas import (
    TextGenerationRequest,
    TextGenerationResult,
)
from topology_mas.execution.summary_protocol_v3 import (
    SUMMARY_PROTOCOL_V3,
    SolveThenSummarizeGeneratorV3,
    SummaryProtocolV3Cache,
    SummaryProtocolV3Error,
    parse_summary_envelope_v3,
    sanitize_retry_summary_body_v3,
    serialize_public_summary_v3,
    summary_protocol_v3,
    validate_public_summary_v3,
    validate_summary_body_v3,
)
from topology_mas.models import TaskInstance


def count_words(text: str) -> int:
    return len(text.split())


def task() -> TaskInstance:
    return TaskInstance(
        task_id="aime-test",
        dataset="aime",
        split="test",
        prompt="Find the requested integer.",
        reference_answer="42",
        oracle_type="aime_integer",
    )


def request() -> TextGenerationRequest:
    protocol = summary_protocol_v3(count_words)
    return TextGenerationRequest(
        request_id="request-v3",
        messages=protocol.build_messages(
            task(),
            previous_output=None,
            incoming_messages=(),
        ),
        seed=7,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_output_tokens=16384,
    )


class ScriptedBackend:
    def __init__(
        self,
        *,
        full_text=(
            "Complete private reasoning.\n"
            "FINAL_ANSWER: \\boxed{042}"
        ),
        full_finish_reason="stop",
        summary_body="Use the decisive identity and verify the integer.",
    ):
        self.requests = []
        self.full_text = full_text
        self.full_finish_reason = full_finish_reason
        self.summary_body = summary_body

    def generate(self, req):
        self.requests.append(req)

        if "summary-v3" in req.request_id:
            raw = self.summary_body
            finish = "stop"
        else:
            raw = self.full_text
            finish = self.full_finish_reason

        return TextGenerationResult(
            raw_text=raw,
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            finish_reason=finish,
            input_tokens=10,
            output_tokens=count_words(raw),
            latency_ms=1.0,
        )


def test_python_serializer_zero_pads_answer():
    text = serialize_public_summary_v3(
        "Use the decisive identity.",
        full_answer="12",
    )

    assert text == (
        "SOLUTION_SUMMARY:\n"
        "Use the decisive identity.\n"
        "FINAL_ANSWER: \\boxed{012}"
    )

    validated = validate_public_summary_v3(
        text,
        full_answer="12",
        finish_reason="stop",
        token_counter=count_words,
    )

    assert validated.parsed_answer == "12"


def test_unparsed_state_is_python_owned_even_if_body_mentions_candidate():
    body = (
        "The truncated derivation repeatedly considers "
        "\\boxed{243} as a candidate value."
    )

    validated_body = validate_summary_body_v3(
        body,
        finish_reason="stop",
        token_counter=count_words,
    )

    text = serialize_public_summary_v3(
        validated_body.text,
        full_answer=None,
    )

    assert "\\boxed{243}" in text
    assert text.endswith("FINAL_ANSWER: UNPARSED")

    validated = validate_public_summary_v3(
        text,
        full_answer=None,
        finish_reason="stop",
        token_counter=count_words,
    )

    assert validated.parsed_answer is None


@pytest.mark.parametrize(
    "bad",
    [
        "SOLUTION_SUMMARY:\nBody",
        "Body\nFINAL_ANSWER: \\boxed{042}",
        "EXTRACTED_FULL_ANSWER: 042",
        "FROZEN_FULL_PARSER_STATE: PARSED(042)",
        "```text\nBody\n```",
    ],
)
def test_model_body_cannot_emit_protocol_or_metadata_markers(bad):
    with pytest.raises(ValueError):
        validate_summary_body_v3(
            bad,
            finish_reason="stop",
            token_counter=count_words,
        )


def test_generator_model_emits_body_python_emits_public_protocol(tmp_path):
    backend = ScriptedBackend()

    result = SolveThenSummarizeGeneratorV3(
        backend,
        cache=SummaryProtocolV3Cache(tmp_path),
        token_counter=count_words,
    ).generate(request())

    envelope = parse_summary_envelope_v3(result.raw_text)

    assert len(backend.requests) == 2

    # Physical summary generation is body-only.
    assert backend.requests[1].request_id.endswith("summary-v3-a1")
    assert "FINAL_ANSWER:" not in backend.summary_body
    assert "SOLUTION_SUMMARY:" not in backend.summary_body

    # Broadcast is canonical Python serialization.
    assert envelope.public_summary == (
        "SOLUTION_SUMMARY:\n"
        "Use the decisive identity and verify the integer.\n"
        "FINAL_ANSWER: \\boxed{042}"
    )

    assert result.metadata["generation_pipeline"] == SUMMARY_PROTOCOL_V3
    assert result.metadata["public_parsed_answer"] == "42"
    assert result.metadata["summary_validation_passed"] is True
    assert result.metadata["summary_answer_matches_raw"] is True


def test_length_full_remains_unparsed_despite_existing_boxed_candidate(tmp_path):
    backend = ScriptedBackend(
        full_text=(
            "Long truncated work. Candidate already present: "
            "\\boxed{243}. More repeated reasoning..."
        ),
        full_finish_reason="length",
        summary_body=(
            "The derivation contains 243 as a candidate result "
            "before the truncation."
        ),
    )

    protocol = summary_protocol_v3(count_words)

    result = SolveThenSummarizeGeneratorV3(
        backend,
        cache=SummaryProtocolV3Cache(tmp_path),
        token_counter=count_words,
    ).generate(request())

    envelope = parse_summary_envelope_v3(result.raw_text)

    assert result.metadata["full_parsed_answer"] is None
    assert result.metadata["public_parsed_answer"] is None

    assert "243" in envelope.public_summary
    assert envelope.public_summary.endswith(
        "FINAL_ANSWER: UNPARSED"
    )

    assert protocol.parse_answer(
        result.raw_text,
        finish_reason=result.finish_reason,
    ) is None

    assert (
        "FROZEN_FULL_PARSER_STATE: UNPARSED"
        in backend.requests[1].messages[-1].content
    )


def test_v3_cache_reuses_solve_and_body_transform(tmp_path):
    backend = ScriptedBackend()
    cache = SummaryProtocolV3Cache(tmp_path)

    generator = SolveThenSummarizeGeneratorV3(
        backend,
        cache=cache,
        token_counter=count_words,
    )

    first = generator.generate(request())
    calls = len(backend.requests)

    second = SolveThenSummarizeGeneratorV3(
        backend,
        cache=cache,
        token_counter=count_words,
    ).generate(request())

    assert len(backend.requests) == calls
    assert first.raw_text == second.raw_text
    assert second.metadata["full_cache_hit"] is True
    assert second.metadata["summary_cache_hit"] is True
    assert second.metadata["backend_call_count"] == 0


def test_summary_body_length_stop_is_still_a_technical_failure():
    with pytest.raises(
        ValueError,
        match="stopped at the output limit",
    ):
        validate_summary_body_v3(
            "Incomplete body",
            finish_reason="length",
            token_counter=count_words,
        )


class RecoveryBackend:
    def __init__(self, summaries):
        self.requests = []
        self.summaries = iter(summaries)

    def generate(self, req):
        self.requests.append(req)
        if "summary-v3" in req.request_id:
            raw, finish = next(self.summaries)
        else:
            raw = "Complete private reasoning.\nFINAL_ANSWER: \\boxed{042}"
            finish = "stop"
        return TextGenerationResult(
            raw_text=raw,
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            finish_reason=finish,
            input_tokens=10,
            output_tokens=count_words(raw),
            latency_ms=1.0,
        )


def test_single_summary_length_stop_remains_a_technical_failure(tmp_path):
    backend = RecoveryBackend([("An unfinished overlong summary", "length")])
    with pytest.raises(SummaryProtocolV3Error, match="failed after 1 summary attempts"):
        SolveThenSummarizeGeneratorV3(
            backend,
            cache=SummaryProtocolV3Cache(tmp_path),
            token_counter=count_words,
        ).generate(request())

    assert len(backend.requests) == 2
    summary_prompt = backend.requests[1].messages[0].content
    assert "target at most 1800 model tokens" in summary_prompt


def test_single_call_sanitizer_removes_only_protocol_owned_labels(tmp_path):
    body, markers = sanitize_retry_summary_body_v3(
        "The derivation gives 42.\nFINAL_ANSWER: \\boxed{042}"
    )
    assert markers == ("final_answer:",)
    assert "FINAL_ANSWER:" not in body
    assert "Concluding value:" in body

    backend = RecoveryBackend([
        ("The derivation gives 42.\nFINAL_ANSWER: \\boxed{042}", "stop"),
    ])
    result = SolveThenSummarizeGeneratorV3(
        backend,
        cache=SummaryProtocolV3Cache(tmp_path),
        token_counter=count_words,
    ).generate(request())
    envelope = parse_summary_envelope_v3(result.raw_text)

    assert envelope.public_summary.count("FINAL_ANSWER:") == 1
    assert "Concluding value:" in envelope.public_summary
    assert len(backend.requests) == 2
    assert result.metadata["summary_retry_count"] == 0
    assert result.metadata["summary_attempts"][0]["sanitized_markers"] == [
        "final_answer:"
    ]
