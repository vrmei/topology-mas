"""Dynamic endpoint-slot scheduling for equivalent inference replicas."""

from __future__ import annotations

import threading
from queue import Queue

from topology_mas.execution.generation import TextGenerator
from topology_mas.execution.schemas import TextGenerationRequest, TextGenerationResult


class EndpointTaskPoolTextGenerator:
    """Route each call to the next available capacity slot.

    Repeating an endpoint index ``slots_per_backend`` times bounds the number of
    concurrent HTTP requests admitted to that replica.  A slot is returned as
    soon as its request finishes, so a fast GPU can immediately accept more work
    instead of waiting for a statically paired slow job on another GPU.
    """

    def __init__(
        self,
        backends: tuple[TextGenerator, ...],
        *,
        slots_per_backend: int,
    ) -> None:
        if not backends:
            raise ValueError("at least one backend is required")
        if slots_per_backend < 1:
            raise ValueError("slots_per_backend must be positive")
        self.backends = backends
        self.slots_per_backend = slots_per_backend
        self._slots: Queue[int] = Queue()
        for backend_index in range(len(backends)):
            for _ in range(slots_per_backend):
                self._slots.put(backend_index)
        self._lock = threading.Lock()
        self._active = [0 for _ in backends]
        self._completed = [0 for _ in backends]

    @property
    def total_slots(self) -> int:
        return len(self.backends) * self.slots_per_backend

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "backend_count": len(self.backends),
                "slots_per_backend": self.slots_per_backend,
                "total_slots": self.total_slots,
                "active_by_backend": list(self._active),
                "completed_by_backend": list(self._completed),
                "available_slots": self._slots.qsize(),
            }

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        backend_index = self._slots.get()
        with self._lock:
            self._active[backend_index] += 1
        try:
            result = self.backends[backend_index].generate(request)
            metadata = {
                **result.metadata,
                "endpoint_task_pool_backend_index": backend_index,
                "endpoint_task_pool_slots_per_backend": self.slots_per_backend,
            }
            return result.model_copy(update={"metadata": metadata})
        finally:
            with self._lock:
                self._active[backend_index] -= 1
                self._completed[backend_index] += 1
            self._slots.put(backend_index)
