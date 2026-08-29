from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from fedorbit.domain.enums import ArtifactState


@dataclass(frozen=True, slots=True)
class ExecutionLogEvent:
    occurred_at: datetime
    cell_coordinates: str
    artifact_id: str | None
    state: ArtifactState


class ExecutionLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def record(self, event: ExecutionLogEvent) -> None:
        self._logger.info(
            "execution_event",
            extra={
                "occurred_at": event.occurred_at.isoformat(),
                "cell_coordinates": event.cell_coordinates,
                "artifact_id": event.artifact_id,
                "state": event.state.value,
            },
        )


def execution_logger() -> ExecutionLogger:
    return ExecutionLogger(logging.getLogger("fedorbit.execution"))
