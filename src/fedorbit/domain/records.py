from __future__ import annotations

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

    def coordinates(self) -> dict[str, str | int]:
        result: dict[str, str | int] = {"experiment": self.experiment.value}
        if self.dataset is not None:
            result["dataset"] = self.dataset.value
        if self.source_client is not None:
            result["source_client"] = self.source_client.value
        if self.target_client is not None:
            result["target_client"] = self.target_client.value
        if self.directed_pair is not None:
            result["directed_pair"] = self.directed_pair.direction
        if self.method is not None:
            result["method"] = self.method.value
        if self.condition is not None:
            result["condition"] = self.condition
        if self.support is not None:
            result["support"] = self.support
        if self.seed is not None:
            result["seed"] = self.seed
        return result
