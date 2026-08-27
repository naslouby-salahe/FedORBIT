from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import TextIO

from fedorbit.domain.records import SemanticCell


@dataclass(frozen=True, slots=True)
class ExecutionLogRecord:
    event: str
    semantic_cell: SemanticCell | None
    message: str


class StructuredExecutionLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def emit(self, record: ExecutionLogRecord) -> None:
        payload = asdict(record)
        if record.semantic_cell is not None:
            payload["semantic_cell"] = record.semantic_cell.identity_json(
                frozenset(
                    {
                        "experiment",
                        "dataset",
                        "source_client",
                        "target_client",
                        "directed_pair",
                        "method",
                        "condition",
                        "support",
                        "seed",
                    }
                )
            )
        self._logger.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def configure_execution_logging(stream: TextIO) -> StructuredExecutionLogger:
    logger = logging.getLogger("fedorbit.execution")
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return StructuredExecutionLogger(logger)
