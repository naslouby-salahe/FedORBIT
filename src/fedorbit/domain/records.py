from __future__ import annotations

import json
from dataclasses import dataclass

from fedorbit.domain.enums import DatasetId, ExperimentName, TransferMethod


@dataclass(frozen=True, slots=True)
class DirectedPair:
    source: DatasetId
    target: DatasetId

    @property
    def direction(self) -> str:
        return f"{self.source.value} -> {self.target.value}"


@dataclass(frozen=True, slots=True)
class SemanticCell:
    experiment: ExperimentName
    dataset: DatasetId | None = None
    source_client: DatasetId | None = None
    target_client: DatasetId | None = None
    directed_pair: DirectedPair | None = None
    method: TransferMethod | None = None
    condition: str | None = None
    support: int | None = None
    seed: int | None = None

    def coordinates_json(self) -> str:
        payload: dict[str, str | int] = {"experiment": self.experiment.value}
        if self.dataset is not None:
            payload["dataset"] = self.dataset.value
        if self.source_client is not None:
            payload["source_client"] = self.source_client.value
        if self.target_client is not None:
            payload["target_client"] = self.target_client.value
        if self.directed_pair is not None:
            payload["directed_pair"] = self.directed_pair.direction
        if self.method is not None:
            payload["method"] = self.method.value
        if self.condition is not None:
            payload["condition"] = self.condition
        if self.support is not None:
            payload["support"] = self.support
        if self.seed is not None:
            payload["seed"] = self.seed
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
