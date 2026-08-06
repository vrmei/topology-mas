"""Minimal OpenAI-compatible JSON client with auditable request behavior."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field


class ChatUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class JSONCompletion(BaseModel):
    """Provider response after extracting and parsing a JSON object."""

    model_config = ConfigDict(frozen=True)

    request_id: str | None = None
    requested_model: str
    returned_model: str | None = None
    finish_reason: str | None = None
    content: dict[str, Any]
    raw_content: str
    usage: ChatUsage = Field(default_factory=ChatUsage)
    raw_response: dict[str, Any]
    raw_attempts: tuple[dict[str, Any], ...] = ()


class InvalidJSONCompletionError(ValueError):
    """All provider responses were successful HTTP calls but invalid JSON completions."""

    def __init__(self, raw_attempts: list[dict[str, Any]]) -> None:
        super().__init__("provider did not return a valid JSON object")
        self.raw_attempts = tuple(raw_attempts)


class JSONChatClient(Protocol):
    """Narrow interface used by mutation components and test doubles."""

    def complete_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_output_tokens: int,
    ) -> JSONCompletion: ...


@dataclass(frozen=True)
class ModelRequestProfile:
    """Known request differences that are not normalized by every gateway."""

    output_token_parameter: str = "max_tokens"
    temperature: float | None = 0.0


_MODEL_PROFILES: dict[str, ModelRequestProfile] = {
    # Verified against OhMyGPT on 2026-08-03.
    "gpt-5.6-sol": ModelRequestProfile(
        output_token_parameter="max_completion_tokens",
        temperature=None,
    ),
}


class OpenAICompatibleJSONClient:
    """Synchronous client that retries the same model without silent fallback."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        max_json_attempts: int = 2,
        retry_base_seconds: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key or os.getenv(api_key_env)
        if not key:
            raise RuntimeError(f"API key environment variable {api_key_env!r} is not set")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if max_json_attempts < 1:
            raise ValueError("max_json_attempts must be at least 1")

        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._max_attempts = max_attempts
        self._max_json_attempts = max_json_attempts
        self._retry_base_seconds = retry_base_seconds
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatibleJSONClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def complete_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_output_tokens: int,
    ) -> JSONCompletion:
        profile = _MODEL_PROFILES.get(model, ModelRequestProfile())
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            profile.output_token_parameter: max_output_tokens,
        }
        if profile.temperature is not None:
            payload["temperature"] = profile.temperature

        raw_attempts: list[dict[str, Any]] = []
        current_max_output_tokens = max_output_tokens
        for _ in range(self._max_json_attempts):
            payload[profile.output_token_parameter] = current_max_output_tokens
            response = self._post_with_retry(payload)
            raw = response.json()
            raw_attempts.append(raw)
            try:
                choice = raw["choices"][0]
                raw_content = choice["message"]["content"]
                content = json.loads(raw_content)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                try:
                    finish_reason = raw["choices"][0].get("finish_reason")
                except (KeyError, IndexError, TypeError):
                    finish_reason = None
                if finish_reason == "length":
                    current_max_output_tokens *= 2
                continue
            if not isinstance(content, dict):
                continue

            usage = raw.get("usage") or {}
            return JSONCompletion(
                request_id=raw.get("id"),
                requested_model=model,
                returned_model=raw.get("model"),
                finish_reason=choice.get("finish_reason"),
                content=content,
                raw_content=raw_content,
                usage=ChatUsage(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                ),
                raw_response=raw,
                raw_attempts=tuple(raw_attempts),
            )
        raise InvalidJSONCompletionError(raw_attempts)

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code not in retryable_statuses:
                    response.raise_for_status()
                    return response
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
