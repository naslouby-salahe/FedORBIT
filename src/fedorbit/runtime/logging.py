from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

from fedorbit.domain.enums import ArtifactState
from fedorbit.domain.records import ArtifactIdentifier, SemanticCoordinates


@dataclass(frozen=True, slots=True)
class ExecutionLogEvent:
    occurred_at: datetime
    cell_coordinates: SemanticCoordinates
    artifact_id: ArtifactIdentifier | None
    state: ArtifactState


class ExecutionLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def record(self, event: ExecutionLogEvent) -> None:
        self._logger.info(
            "execution_event",
            extra=OrderedDict(
                occurred_at=event.occurred_at.isoformat(),
                cell_coordinates=event.cell_coordinates.value,
                artifact_id=event.artifact_id.value if event.artifact_id is not None else None,
                state=event.state.value,
            ),
        )


def execution_logger() -> ExecutionLogger:
    return ExecutionLogger(logging.getLogger("fedorbit.execution"))
