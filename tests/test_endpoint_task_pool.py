from __future__ import annotations

from topology_mas.execution.endpoint_task_pool import EndpointTaskPoolTextGenerator
from topology_mas.execution.schemas import ChatMessage, TextGenerationRequest, TextGenerationResult


class FakeBackend:
    def __init__(self, index: int) -> None:
        self.index = index

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        return TextGenerationResult(raw_text=str(self.index), metadata={"source": self.index})


def test_endpoint_slots_are_reused_and_audited() -> None:
    pool = EndpointTaskPoolTextGenerator(
        (FakeBackend(0), FakeBackend(1)),
        slots_per_backend=1,
    )
    request = TextGenerationRequest(
        request_id="x",
        messages=(ChatMessage(role="user", content="x"),),
        seed=1,
        temperature=0.0,
        max_output_tokens=1,
    )
    first = pool.generate(request)
    second = pool.generate(request)
    assert {first.raw_text, second.raw_text} == {"0", "1"}
    snapshot = pool.snapshot()
    assert snapshot["active_by_backend"] == [0, 0]
    assert snapshot["completed_by_backend"] == [1, 1]
    assert snapshot["available_slots"] == 2
