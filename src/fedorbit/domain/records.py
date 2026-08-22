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
        present: dict[str, str | int | float | list[str] | None] = {
            "dataset": self.dataset.value if self.dataset is not None else None,
            "source_client": self.source_client.value if self.source_client is not None else None,
            "target_client": self.target_client.value if self.target_client is not None else None,
            "method": self.method.value if self.method is not None else None,
            "condition": self.condition,
            "support": self.support,
            "seed": self.seed,
        }
        if self.directed_pair is not None:
            present["directed_pair"] = [
                self.directed_pair.source.value,
                self.directed_pair.target.value,
            ]
        values: dict[str, str | int | float | list[str] | None] = {
            "experiment": self.experiment.value
        }
        for coordinate in relevance:
            value = present.get(coordinate)
            if value is not None:
                values[coordinate] = value
        return canonical_json(values)
