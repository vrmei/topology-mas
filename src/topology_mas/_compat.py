"""Small standard-library compatibility helpers."""

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - exercised on Python 3.10 only
    from datetime import timezone

    UTC = timezone.utc

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised on Python 3.10 only
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10-compatible subset of :class:`enum.StrEnum`."""

        def __str__(self) -> str:
            return self.value


__all__ = ["UTC", "StrEnum"]
