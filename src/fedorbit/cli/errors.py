from __future__ import annotations

from fedorbit.execution.errors import NotReadyError


class CliUsageError(ValueError):
    pass


__all__ = ["CliUsageError", "NotReadyError"]
