"""Exact state-consistent replay for paired counterfactual execution."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from topology_mas.execution.generation import TextGenerator
from topology_mas.execution.prompts import PROMPT_VERSION
from topology_mas.execution.schemas import TextGenerationRequest, TextGenerationResult

STATE_REPLAY_CACHE_VERSION = "state-consistent-replay-v1"


class StateReplayCacheError(RuntimeError):
    """A replay artifact is missing, corrupt, or incompatible."""


class StateReplayManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_version: Literal["state-consistent-replay-v1"] = STATE_REPLAY_CACHE_VERSION
    requested_model: str = Field(min_length=1)
    expected_returned_model: str | None = None
    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: str = PROMPT_VERSION
    namespace: str = Field(default="default", min_length=1)


class StateReplayRequestIdentity(BaseModel):
    """Every semantic and stochastic field used to identify one node transition."""

    model_config = ConfigDict(frozen=True)

    cache_version: Literal["state-consistent-replay-v1"] = STATE_REPLAY_CACHE_VERSION
    requested_model: str
    expected_returned_model: str | None
    model_fingerprint: str
    prompt_version: str
    namespace: str
    messages: tuple[dict[str, str], ...]
    generation_seed: int
    temperature: float
    max_output_tokens: int


class StateReplayEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_version: Literal["state-consistent-replay-v1"] = STATE_REPLAY_CACHE_VERSION
    request_fingerprint: str = Field(min_length=64, max_length=64)
    request: StateReplayRequestIdentity
    result_fingerprint: str = Field(min_length=64, max_length=64)
    result: TextGenerationResult


class StateReplayStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_requests: int = Field(ge=0)
    backend_calls: int = Field(ge=0)
    cache_hits: int = Field(ge=0)


def _canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_first_complete(path: Path, content: str) -> None:
    """Publish one complete immutable file; an existing first writer always wins."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        with suppress(FileExistsError):
            os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class StateConsistentReplayGenerator:
    """Memoize an exact stochastic transition and replay it across paired conditions.

    A request ID is intentionally excluded from identity. Complete messages, the
    stochastic seed, decoding settings, prompt version, and pinned model identity
    are included. Different experiment seeds therefore never share an entry.
    """

    def __init__(
        self,
        backend: TextGenerator,
        *,
        cache_dir: str | Path,
        requested_model: str,
        expected_returned_model: str | None,
        model_fingerprint: str,
        namespace: str = "default",
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self._backend = backend
        self.root = Path(cache_dir)
        self.entries_dir = self.root / "entries"
        self.manifest = StateReplayManifest(
            requested_model=requested_model,
            expected_returned_model=expected_returned_model,
            model_fingerprint=model_fingerprint,
            prompt_version=prompt_version,
            namespace=namespace,
        )
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self._logical_requests = 0
        self._backend_calls = 0
        self._cache_hits = 0
        self._initialize()

    @property
    def stats(self) -> StateReplayStats:
        with self._lock:
            return StateReplayStats(
                logical_requests=self._logical_requests,
                backend_calls=self._backend_calls,
                cache_hits=self._cache_hits,
            )

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        identity = self._request_identity(request)
        request_fingerprint = _fingerprint(identity)
        with self._lock:
            self._logical_requests += 1

        while True:
            cached = self._load(request_fingerprint, expected_request=identity)
            if cached is not None:
                with self._lock:
                    self._cache_hits += 1
                return self._annotate(cached.result, request_fingerprint, cache_hit=True)

            with self._lock:
                event = self._inflight.get(request_fingerprint)
                if event is None:
                    event = threading.Event()
                    self._inflight[request_fingerprint] = event
                    owner = True
                else:
                    owner = False
            if not owner:
                event.wait()
                continue

            try:
                result = self._backend.generate(request)
                with self._lock:
                    self._backend_calls += 1
                published = self._publish(
                    request_fingerprint=request_fingerprint,
                    request=identity,
                    result=result,
                )
                return self._annotate(
                    published.result,
                    request_fingerprint,
                    cache_hit=False,
                )
            finally:
                with self._lock:
                    completed = self._inflight.pop(request_fingerprint)
                    completed.set()

    def _initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "manifest.json"
        _write_first_complete(manifest_path, self.manifest.model_dump_json(indent=2) + "\n")
        try:
            existing = StateReplayManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            raise StateReplayCacheError("state replay manifest is invalid") from exc
        if existing != self.manifest:
            raise StateReplayCacheError(
                "state replay manifest differs; use a new cache directory"
            )
        self.entries_dir.mkdir(parents=True, exist_ok=True)

    def _request_identity(
        self, request: TextGenerationRequest
    ) -> StateReplayRequestIdentity:
        return StateReplayRequestIdentity(
            requested_model=self.manifest.requested_model,
            expected_returned_model=self.manifest.expected_returned_model,
            model_fingerprint=self.manifest.model_fingerprint,
            prompt_version=self.manifest.prompt_version,
            namespace=self.manifest.namespace,
            messages=tuple(message.model_dump(mode="json") for message in request.messages),
            generation_seed=request.seed,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )

    def _entry_path(self, request_fingerprint: str) -> Path:
        return self.entries_dir / request_fingerprint[:2] / f"{request_fingerprint}.json"

    def _load(
        self,
        request_fingerprint: str,
        *,
        expected_request: StateReplayRequestIdentity,
    ) -> StateReplayEntry | None:
        path = self._entry_path(request_fingerprint)
        if not path.exists():
            return None
        try:
            entry = StateReplayEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise StateReplayCacheError(f"invalid state replay entry: {path}") from exc
        if entry.request_fingerprint != request_fingerprint or entry.request != expected_request:
            raise StateReplayCacheError(f"state replay request identity mismatch: {path}")
        if entry.result_fingerprint != _fingerprint(entry.result):
            raise StateReplayCacheError(f"state replay result fingerprint mismatch: {path}")
        return entry

    def _publish(
        self,
        *,
        request_fingerprint: str,
        request: StateReplayRequestIdentity,
        result: TextGenerationResult,
    ) -> StateReplayEntry:
        candidate = StateReplayEntry(
            request_fingerprint=request_fingerprint,
            request=request,
            result_fingerprint=_fingerprint(result),
            result=result,
        )
        path = self._entry_path(request_fingerprint)
        _write_first_complete(path, candidate.model_dump_json() + "\n")
        published = self._load(request_fingerprint, expected_request=request)
        if published is None:  # pragma: no cover - defensive filesystem guard
            raise StateReplayCacheError(f"state replay entry was not published: {path}")
        return published

    @staticmethod
    def _annotate(
        result: TextGenerationResult,
        request_fingerprint: str,
        *,
        cache_hit: bool,
    ) -> TextGenerationResult:
        metadata = {
            **result.metadata,
            "state_replay_cache_version": STATE_REPLAY_CACHE_VERSION,
            "state_replay_request_fingerprint": request_fingerprint,
            "state_replay_cache_hit": cache_hit,
            "backend_called": not cache_hit,
        }
        return result.model_copy(update={"metadata": metadata})
