import json

import httpx

from topology_mas.providers import OpenAICompatibleJSONClient


def test_gpt_56_profile_uses_verified_parameters() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "request-1",
                "model": "gpt-5.6-sol-2026-07-09",
                "choices": [
                    {
                        "message": {"content": '{"ok": true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 4,
                    "total_tokens": 9,
                },
            },
        )

    client = OpenAICompatibleJSONClient(
        base_url="https://example.test/v1",
        api_key_env="UNUSED",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )
    try:
        completion = client.complete_json(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "test"}],
            max_output_tokens=100,
        )
    finally:
        client.close()

    assert observed["max_completion_tokens"] == 100
    assert "max_tokens" not in observed
    assert "temperature" not in observed
    assert completion.returned_model == "gpt-5.6-sol-2026-07-09"
    assert completion.content == {"ok": True}


def test_generic_profile_uses_json_mode_and_temperature_zero() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-pro",
                "choices": [
                    {"message": {"content": '{"plausible": true}'}, "finish_reason": "stop"}
                ],
            },
        )

    with OpenAICompatibleJSONClient(
        base_url="https://example.test/v1",
        api_key_env="UNUSED",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        client.complete_json(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "test"}],
            max_output_tokens=200,
        )

    assert observed["max_tokens"] == 200
    assert observed["temperature"] == 0.0
    assert observed["response_format"] == {"type": "json_object"}


def test_invalid_json_retries_same_model_and_preserves_attempts() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "" if calls == 1 else '{"ok": true}'
        return httpx.Response(
            200,
            json={
                "id": f"request-{calls}",
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            },
        )

    with OpenAICompatibleJSONClient(
        base_url="https://example.test/v1",
        api_key_env="UNUSED",
        api_key="secret",
        max_json_attempts=2,
        transport=httpx.MockTransport(handler),
    ) as client:
        completion = client.complete_json(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "test"}],
            max_output_tokens=200,
        )

    assert calls == 2
    assert completion.requested_model == "deepseek-chat"
    assert len(completion.raw_attempts) == 2


def test_length_truncated_json_retry_doubles_output_budget() -> None:
    observed_limits: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        observed_limits.append(payload["max_tokens"])
        content = "" if len(observed_limits) == 1 else '{"ok": true}'
        finish_reason = "length" if len(observed_limits) == 1 else "stop"
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash",
                "choices": [
                    {"message": {"content": content}, "finish_reason": finish_reason}
                ],
            },
        )

    with OpenAICompatibleJSONClient(
        base_url="https://example.test/v1",
        api_key_env="UNUSED",
        api_key="secret",
        max_json_attempts=2,
        transport=httpx.MockTransport(handler),
    ) as client:
        completion = client.complete_json(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "test"}],
            max_output_tokens=200,
        )

    assert observed_limits == [200, 400]
    assert completion.content == {"ok": True}
