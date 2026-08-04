"""Auditable OpenAI-compatible text adapter for vLLM and API gateways."""

from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

from topology_mas.execution.schemas import TextGenerationRequest, TextGenerationResult


class InvalidTextCompletionError(ValueError):
    """The server returned HTTP success but not a usable text completion."""

    def __init__(self, raw_response: dict[str, Any]) -> None:
        super().__init__("provider response did not contain a valid text completion")
        self.raw_response = raw_response


class UnexpectedReturnedModelError(ValueError):
    """A successful response came from a model other than the pinned expectation."""

    def __init__(self, *, expected: str, returned: object) -> None:
        super().__init__(f"returned model {returned!r} does not match expected {expected!r}")
        self.expected = expected
        self.returned = returned


class OpenAICompatibleTextGenerator:
    """Synchronous adapter with bounded retries and no model fallback."""

    _RETRYABLE_STATUSES = {408, 409, 429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        api_key: str | None = None,
        expected_returned_model: str | None = None,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        retry_base_seconds: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key or os.getenv(api_key_env)
        if not key:
            raise RuntimeError(f"API key environment variable {api_key_env!r} is not set")
        if not model:
            raise ValueError("model cannot be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_base_seconds < 0:
            raise ValueError("retry_base_seconds cannot be negative")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.expected_returned_model = expected_returned_model
        self._api_key = key
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatibleTextGenerator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in request.messages],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "seed": request.seed,
            "stream": False,
        }
        started = time.perf_counter()
        response, attempts = self._post_with_retry(payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            raw = response.json()
        except ValueError as exc:
            raise InvalidTextCompletionError(
                {"http_status": response.status_code, "non_json_body": response.text}
            ) from exc
        if not isinstance(raw, dict):
            raise InvalidTextCompletionError({"response": raw})

        returned_model = raw.get("model")
        if (
            self.expected_returned_model is not None
            and returned_model != self.expected_returned_model
        ):
            raise UnexpectedReturnedModelError(
                expected=self.expected_returned_model,
                returned=returned_model,
            )

        try:
            choice = raw["choices"][0]
            raw_text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidTextCompletionError(raw) from exc
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise InvalidTextCompletionError(raw)

        usage = raw.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        return TextGenerationResult(
            raw_text=raw_text,
            model_name=returned_model,
            finish_reason=choice.get("finish_reason"),
            input_tokens=self._optional_nonnegative_int(usage.get("prompt_tokens")),
            output_tokens=self._optional_nonnegative_int(usage.get("completion_tokens")),
            latency_ms=latency_ms,
            metadata={
                "provider_request_id": raw.get("id"),
                "requested_model": self.model,
                "returned_model": returned_model,
                "http_attempts": attempts,
                "raw_response": raw,
            },
        )

    def _post_with_retry(
        self, payload: dict[str, Any]
    ) -> tuple[httpx.Response, int]:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code not in self._RETRYABLE_STATUSES:
                    response.raise_for_status()
                    return response, attempt
                last_error = httpx.HTTPStatusError(
                    f"retryable provider status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc

            if attempt < self._max_attempts:
                delay = self._retry_base_seconds * (2 ** (attempt - 1))
                delay += random.uniform(0.0, delay * 0.1)
                time.sleep(delay)

        assert last_error is not None
        raise last_error

    @staticmethod
    def _optional_nonnegative_int(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value
