import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from topology_mas.execution import (
    ChatMessage,
    StateConsistentReplayGenerator,
    StateReplayCacheError,
    TextGenerationRequest,
    TextGenerationResult,
)


class CountingBackend:
    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        self.calls = 0
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        with self._lock:
            self.calls += 1
            call_number = self.calls
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return TextGenerationResult(
            raw_text=f"sample-{call_number}\nFINAL_ANSWER: 42",
            model_name="pinned-model",
            finish_reason="stop",
            input_tokens=11,
            output_tokens=7,
            latency_ms=2.0,
        )


def request(*, request_id: str = "request-1", seed: int = 7) -> TextGenerationRequest:
    return TextGenerationRequest(
        request_id=request_id,
        messages=(ChatMessage(role="user", content="Solve 40 + 2."),),
        seed=seed,
        temperature=0.3,
        max_output_tokens=128,
    )


def replay(
    tmp_path: Path, backend: CountingBackend, *, fingerprint: str = "a" * 64
) -> StateConsistentReplayGenerator:
    return StateConsistentReplayGenerator(
        backend,
        cache_dir=tmp_path,
        requested_model="model-alias",
        expected_returned_model="pinned-model",
        model_fingerprint=fingerprint,
        namespace="test-v1",
    )


def test_exact_request_is_generated_once_and_replayed_verbatim(tmp_path: Path) -> None:
    backend = CountingBackend()
    generator = replay(tmp_path, backend)

    first = generator.generate(request(request_id="clean"))
    second = generator.generate(request(request_id="attack"))

    assert backend.calls == 1
    assert first.raw_text == second.raw_text == "sample-1\nFINAL_ANSWER: 42"
    assert first.metadata["state_replay_cache_hit"] is False
    assert first.metadata["backend_called"] is True
    assert second.metadata["state_replay_cache_hit"] is True
    assert second.metadata["backend_called"] is False
    assert generator.stats.logical_requests == 2
    assert generator.stats.backend_calls == 1
    assert generator.stats.cache_hits == 1


def test_seed_is_part_of_replay_identity(tmp_path: Path) -> None:
    backend = CountingBackend()
    generator = replay(tmp_path, backend)

    first = generator.generate(request(seed=7))
    second = generator.generate(request(seed=8))

    assert backend.calls == 2
    assert first.raw_text != second.raw_text
    assert generator.stats.cache_hits == 0


def test_concurrent_identical_requests_use_single_flight(tmp_path: Path) -> None:
    backend = CountingBackend(delay_seconds=0.05)
    generator = replay(tmp_path, backend)

    with ThreadPoolExecutor(max_workers=16) as executor:
        outputs = tuple(
            executor.map(
                lambda index: generator.generate(request(request_id=f"request-{index}")),
                range(32),
            )
        )

    assert backend.calls == 1
    assert {output.raw_text for output in outputs} == {"sample-1\nFINAL_ANSWER: 42"}
    assert sum(output.metadata["state_replay_cache_hit"] for output in outputs) == 31
    assert generator.stats.logical_requests == 32
    assert generator.stats.backend_calls == 1
    assert generator.stats.cache_hits == 31


def test_persistent_entry_is_reused_by_a_new_generator(tmp_path: Path) -> None:
    first_backend = CountingBackend()
    first = replay(tmp_path, first_backend)
    original = first.generate(request())

    second_backend = CountingBackend()
    second = replay(tmp_path, second_backend)
    restored = second.generate(request())

    assert first_backend.calls == 1
    assert second_backend.calls == 0
    assert restored.raw_text == original.raw_text
    assert restored.metadata["state_replay_cache_hit"] is True


def test_cache_manifest_rejects_a_different_model_fingerprint(tmp_path: Path) -> None:
    replay(tmp_path, CountingBackend(), fingerprint="a" * 64)

    with pytest.raises(StateReplayCacheError, match="manifest differs"):
        replay(tmp_path, CountingBackend(), fingerprint="b" * 64)


def test_tampered_entry_fails_closed(tmp_path: Path) -> None:
    generator = replay(tmp_path, CountingBackend())
    generator.generate(request())
    entry_path = next((tmp_path / "entries").glob("*/*.json"))
    stored = json.loads(entry_path.read_text(encoding="utf-8"))
    stored["result"]["raw_text"] = "tampered"
    entry_path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(StateReplayCacheError, match="result fingerprint mismatch"):
        generator.generate(request())
