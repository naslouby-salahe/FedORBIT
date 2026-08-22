from __future__ import annotations

from dataclasses import dataclass

from fedorbit.domain.canonical import canonical_json
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

    def identity_json(self, relevance: frozenset[str]) -> str:
        values: dict[str, object] = {"experiment": self.experiment.value}
        if "dataset" in relevance and self.dataset is not None:
            values["dataset"] = self.dataset.value
        if "source_client" in relevance and self.source_client is not None:
            values["source_client"] = self.source_client.value
        if "target_client" in relevance and self.target_client is not None:
            values["target_client"] = self.target_client.value
        if "directed_pair" in relevance and self.directed_pair is not None:
            values["directed_pair"] = [
                self.directed_pair.source.value,
                self.directed_pair.target.value,
            ]
        if "method" in relevance and self.method is not None:
            values["method"] = self.method.value
        if "condition" in relevance and self.condition is not None:
            values["condition"] = self.condition
        if "support" in relevance and self.support is not None:
            values["support"] = self.support
        if "seed" in relevance and self.seed is not None:
            values["seed"] = self.seed
        return canonical_json(values)
