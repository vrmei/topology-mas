"""Provider-neutral text-generation boundary used by the execution engine."""

from __future__ import annotations

from typing import Protocol

from topology_mas.execution.schemas import TextGenerationRequest, TextGenerationResult


class TextGenerator(Protocol):
    def generate(self, request: TextGenerationRequest) -> TextGenerationResult: ...
