from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from fedorbit.domain.enums import (
    DatasetId,
    ExperimentName,
    SemanticCoordinate,
    TransferMethod,
)
from fedorbit.domain.serialization import stable_json


@dataclass(frozen=True, slots=True)
class ArtifactPath:
    value: Path

    def __post_init__(self) -> None:
        if not self.value.is_absolute():
            raise ValueError("artifact paths must be absolute")


@dataclass(frozen=True, slots=True)
class ArtifactIdentifier:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("artifact identifier must not be empty")


@dataclass(frozen=True, slots=True)
class ArtifactFingerprint:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("artifact fingerprint must not be empty")


@dataclass(frozen=True, slots=True)
class SemanticCoordinates:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("semantic coordinates must not be empty")


@dataclass(frozen=True, slots=True)
class ExperimentCondition:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("experiment condition must not be empty")


@dataclass(frozen=True, slots=True)
class SupportSize:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError("support size must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentSeed:
    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value < 2**32:
            raise ValueError("experiment seed must be in the unsigned 32-bit range")


@dataclass(frozen=True, slots=True)
class ExecutionCell:
    coordinates: SemanticCoordinates
    artifact_identifier: ArtifactIdentifier
    dependency_fingerprint: ArtifactFingerprint


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
    condition: ExperimentCondition | None = None
    support: SupportSize | None = None
    seed: ExperimentSeed | None = None

    def identity_json(self, relevance: frozenset[SemanticCoordinate]) -> str:
        present: OrderedDict[str, str | int | float | list[str] | None] = OrderedDict(
            dataset=self.dataset.value if self.dataset is not None else None,
            source_client=self.source_client.value if self.source_client is not None else None,
            target_client=self.target_client.value if self.target_client is not None else None,
            method=self.method.value if self.method is not None else None,
            condition=self.condition.value if self.condition is not None else None,
            support=self.support.value if self.support is not None else None,
            seed=self.seed.value if self.seed is not None else None,
        )
        if self.directed_pair is not None:
            present["directed_pair"] = [
                self.directed_pair.source.value,
                self.directed_pair.target.value,
            ]
        values: OrderedDict[str, str | int | float | list[str] | None] = OrderedDict(
            experiment=self.experiment.value
        )
        for coordinate in relevance:
            value = present.get(coordinate.value)
            if value is not None:
                values[coordinate.value] = value
        return stable_json(values)
