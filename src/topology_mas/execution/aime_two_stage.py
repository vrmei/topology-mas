"""Private-solve/public-summary generation pipeline for bounded AIME messages."""

from __future__ import annotations

import hashlib

from topology_mas.execution.aime import (
    AIME_PUBLIC_SUMMARY_SYSTEM_PROMPT,
    parse_aime_answer,
)
from topology_mas.execution.generation import TextGenerator
from topology_mas.execution.schemas import (
    ChatMessage,
    TextGenerationRequest,
    TextGenerationResult,
)

AIME_TWO_STAGE_PIPELINE_VERSION = "aime-private-solve-public-summary-v1"


def _sum_optional(left: int | None, right: int | None) -> int | None:
    return left + right if left is not None and right is not None else None


def _summary_seed(seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0public-summary".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _safe_provider_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Avoid duplicating completion text already stored in the trace."""

    return {key: value for key, value in metadata.items() if key != "raw_response"}


class AIMETwoStageTextGenerator:
    """Run a long private solve, then compress it into the broadcast state."""

    def __init__(
        self,
        backend: TextGenerator,
        *,
        private_max_output_tokens: int,
        summary_temperature: float,
    ) -> None:
        if private_max_output_tokens < 1:
            raise ValueError("private_max_output_tokens must be positive")
        if not 0.0 <= summary_temperature <= 2.0:
            raise ValueError("summary_temperature must be between zero and two")
        self._backend = backend
        self.private_max_output_tokens = private_max_output_tokens
        self.summary_temperature = summary_temperature

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        private_request = request.model_copy(
            update={
                "request_id": f"{request.request_id}-private",
                "max_output_tokens": self.private_max_output_tokens,
            }
        )
        private = self._backend.generate(private_request)
        private_answer = (
            None
            if private.finish_reason == "length"
            else parse_aime_answer(private.raw_text)
        )
        answer_instruction = (
            f"EXTRACTED_PRIVATE_FINAL_ANSWER: {int(private_answer):03d}"
            if private_answer is not None
            else "EXTRACTED_PRIVATE_FINAL_ANSWER: UNPARSED"
        )
        summary_request = TextGenerationRequest(
            request_id=f"{request.request_id}-summary",
            messages=(
                ChatMessage(role="system", content=AIME_PUBLIC_SUMMARY_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=(
                        f"PRIVATE_SOLUTION_DRAFT:\n{private.raw_text}\n\n"
                        f"{answer_instruction}"
                    ),
                ),
            ),
            seed=_summary_seed(request.seed),
            temperature=self.summary_temperature,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            presence_penalty=0.0,
            max_output_tokens=request.max_output_tokens,
        )
        public = self._backend.generate(summary_request)
        public_raw_text = (
            public.raw_text
            if private_answer is not None
            else (
                "SOLUTION_SUMMARY:\nThe private solve did not produce a complete, "
                "parseable answer.\nFINAL_ANSWER: UNPARSED"
            )
        )
        return TextGenerationResult(
            raw_text=public_raw_text,
            model_name=public.model_name,
            finish_reason=public.finish_reason,
            input_tokens=_sum_optional(private.input_tokens, public.input_tokens),
            output_tokens=_sum_optional(private.output_tokens, public.output_tokens),
            latency_ms=(private.latency_ms or 0.0) + (public.latency_ms or 0.0),
            metadata={
                "generation_pipeline": AIME_TWO_STAGE_PIPELINE_VERSION,
                "backend_call_count": 2,
                "private_request_id": private_request.request_id,
                "private_raw_output": private.raw_text,
                "private_finish_reason": private.finish_reason,
                "private_parsed_answer": private_answer,
                "private_input_tokens": private.input_tokens,
                "private_output_tokens": private.output_tokens,
                "private_latency_ms": private.latency_ms,
                "private_provider_metadata": _safe_provider_metadata(private.metadata),
                "public_request_id": summary_request.request_id,
                "public_input_tokens": public.input_tokens,
                "public_output_tokens": public.output_tokens,
                "public_latency_ms": public.latency_ms,
                "public_provider_metadata": _safe_provider_metadata(public.metadata),
                "public_model_raw_output": public.raw_text,
                "summary_answer_matches_private": (
                    private_answer is not None
                    and public.finish_reason != "length"
                    and parse_aime_answer(public.raw_text) == private_answer
                ),
            },
        )
