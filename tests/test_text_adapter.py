import json

import httpx
import pytest

from topology_mas.execution import (
    ChatMessage,
    InvalidTextCompletionError,
    OpenAICompatibleTextGenerator,
    TextGenerationRequest,
    UnexpectedReturnedModelError,
)


def request() -> TextGenerationRequest:
    return TextGenerationRequest(
        request_id="req-1",
        messages=(ChatMessage(role="user", content="Return FINAL_ANSWER: 42"),),
        seed=17,
        temperature=0.25,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_output_tokens=64,
    )


def response_payload() -> dict[str, object]:
    return {
        "id": "provider-id",
        "model": "returned-snapshot",
        "choices": [
            {
                "message": {"role": "assistant", "content": "FINAL_ANSWER: 42"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17},
    }


def test_adapter_sends_openai_compatible_payload_and_normalizes_response() -> None:
    captured: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.append(http_request)
        return httpx.Response(200, json=response_payload())

    with OpenAICompatibleTextGenerator(
        model="requested-alias",
        base_url="https://example.test/v1/",
        api_key_env="UNUSED",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    ) as generator:
        result = generator.generate(request())

    body = json.loads(captured[0].content)
    assert captured[0].url == "https://example.test/v1/chat/completions"
    assert captured[0].headers["authorization"] == "Bearer secret"
    assert body == {
        "model": "requested-alias",
        "messages": [{"role": "user", "content": "Return FINAL_ANSWER: 42"}],
        "max_tokens": 64,
        "temperature": 0.25,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "seed": 17,
        "stream": False,
    }
    assert result.raw_text == "FINAL_ANSWER: 42"
    assert result.model_name == "returned-snapshot"
    assert result.input_tokens == 11
    assert result.output_tokens == 6
    assert result.metadata["requested_model"] == "requested-alias"
    assert result.metadata["http_attempts"] == 1


def test_adapter_retries_only_retryable_statuses() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=response_payload())

    with OpenAICompatibleTextGenerator(
        model="model",
        base_url="https://example.test/v1",
        api_key_env="UNUSED",
        api_key="secret",
        max_attempts=2,
        retry_base_seconds=0,
        transport=httpx.MockTransport(handler),
    ) as generator:
        result = generator.generate(request())

    assert calls == 2
    assert result.metadata["http_attempts"] == 2


def test_adapter_does_not_retry_nonretryable_client_error() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "unsupported parameter"})

    with OpenAICompatibleTextGenerator(
        model="model",
        base_url="https://example.test/v1",
        api_key_env="UNUSED",
        api_key="secret",
        max_attempts=3,
        retry_base_seconds=0,
        transport=httpx.MockTransport(handler),
    ) as generator, pytest.raises(httpx.HTTPStatusError):
        generator.generate(request())

    assert calls == 1


def test_adapter_retries_context_overflow_with_safe_output_limit() -> None:
    payloads: list[dict[str, object]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        payloads.append(payload)
        if len(payloads) == 1:
            return httpx.Response(
                400,
                json={
                    "message": (
                        "'max_tokens' is too large: 64. This model's maximum "
                        "context length is 80 tokens and your request has 40 "
                        "input tokens (64 > 80 - 40)."
                    )
                },
            )
        return httpx.Response(200, json=response_payload())

    with OpenAICompatibleTextGenerator(
        model="model",
        base_url="https://example.test/v1",
        api_key_env="UNUSED",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    ) as generator:
        result = generator.generate(request())

    assert [payload["max_tokens"] for payload in payloads] == [64, 40]
    assert result.metadata["http_attempts"] == 2
    assert result.metadata["context_window_adjustment"] == {
        "requested_max_output_tokens": 64,
        "effective_max_output_tokens": 40,
    }


def test_adapter_parses_nested_vllm_context_error() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "This model's maximum context length is 80 tokens and "
                            "your request has 40 input tokens."
                        )
                    }
                },
            )
        return httpx.Response(200, json=response_payload())

    with OpenAICompatibleTextGenerator(
        model="model",
        base_url="https://example.test/v1",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    ) as generator:
        result = generator.generate(request())

    assert calls == 2
    assert result.metadata["context_window_adjustment"] == {
        "requested_max_output_tokens": 64,
        "effective_max_output_tokens": 40,
    }


def test_adapter_rejects_success_without_text() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "id": "x",
                "model": "model",
                "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
            },
        )
    )
    with OpenAICompatibleTextGenerator(
        model="model",
        base_url="https://example.test/v1",
        api_key_env="UNUSED",
        api_key="secret",
        transport=transport,
    ) as generator, pytest.raises(InvalidTextCompletionError):
        generator.generate(request())


def test_adapter_requires_a_key_without_exposing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_TEST_KEY"):
        OpenAICompatibleTextGenerator(
            model="model",
            base_url="https://example.test/v1",
            api_key_env="MISSING_TEST_KEY",
        )


def test_adapter_can_connect_to_an_unauthenticated_local_server() -> None:
    captured: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.append(http_request)
        return httpx.Response(200, json=response_payload())

    with OpenAICompatibleTextGenerator(
        model="local-model",
        base_url="http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(handler),
    ) as generator:
        generator.generate(request())

    assert "authorization" not in captured[0].headers
    assert captured[0].headers["content-type"] == "application/json"


def test_adapter_can_fail_closed_on_returned_model_mismatch() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json=response_payload())
    )
    with OpenAICompatibleTextGenerator(
        model="requested-alias",
        expected_returned_model="pinned-snapshot",
        base_url="https://example.test/v1",
        api_key_env="UNUSED",
        api_key="secret",
        transport=transport,
    ) as generator, pytest.raises(UnexpectedReturnedModelError, match="returned-snapshot"):
        generator.generate(request())
