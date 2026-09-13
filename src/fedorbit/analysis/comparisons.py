from __future__ import annotations

import math
from dataclasses import dataclass

from fedorbit.types import (
    ArtifactIdentifier,
    Coefficient,
    ConsumedBudget,
    DirectedPairName,
    Index,
    RandomSeed,
    RelativeGain,
    SampleCount,
    Score,
    Sha256Digest,
    SignificanceLevel,
    Split,
    StrictResourceValidity,
    TransferMethod,
    is_sha256_digest,
)

type ScoreSeries = tuple[Score, ...]
type RankSeries = tuple[Score, ...]
type PairSeedIdentity = tuple[DirectedPairName, RandomSeed]
type IndexedPairedObservations = tuple[tuple[PairSeedIdentity, PairedObservation], ...]


class PairingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PairingLineage:
    raw_dataset_lineage_sha256: Sha256Digest
    directed_pair: DirectedPairName
    seed: RandomSeed
    split: Split
    target_pre_transfer_checkpoint_artifact_id: ArtifactIdentifier
    target_importance_artifact_id: ArtifactIdentifier
    source_packet_artifact_id: ArtifactIdentifier | None
    action_budget: ConsumedBudget
    support_budget: Index
    confirmation_budget: Index
    environment_lineage_sha256: Sha256Digest

    def __post_init__(self) -> None:
        if not is_sha256_digest(self.raw_dataset_lineage_sha256):
            raise PairingError("raw dataset lineage must be lowercase SHA-256 hex")
        if not is_sha256_digest(self.environment_lineage_sha256):
            raise PairingError("environment lineage must be lowercase SHA-256 hex")
        if not self.directed_pair:
            raise PairingError("directed pair must be non-empty")
        if not self.target_pre_transfer_checkpoint_artifact_id.value:
            raise PairingError("target checkpoint identity must be non-empty")
        if not self.target_importance_artifact_id.value:
            raise PairingError("target importance identity must be non-empty")
        if not math.isfinite(self.action_budget) or self.action_budget < 0.0:
            raise PairingError("action budget must be finite and nonnegative")
        if self.support_budget < 0 or self.confirmation_budget < 0:
            raise PairingError("support and confirmation budgets must be nonnegative")


@dataclass(frozen=True, slots=True)
class PairedObservation:
    method: TransferMethod
    value: Score
    lineage: PairingLineage

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise PairingError("paired observation value must be finite")


@dataclass(frozen=True, slots=True)
class PairedValues:
    directed_pair: DirectedPairName
    seeds: tuple[RandomSeed, ...]
    method_values: ScoreSeries
    reference_values: ScoreSeries


@dataclass(frozen=True, slots=True)
class PairContrastEvidence:
    directed_pair: DirectedPairName
    mean_gain: RelativeGain | None
    holm_p: SignificanceLevel | None
    bca_lower: RelativeGain | None
    strict_resource_valid: StrictResourceValidity
    valid_seed_count: Index


@dataclass(frozen=True, slots=True)
class PairContrastEvidenceSet:
    entries: tuple[PairContrastEvidence, ...]

    def __post_init__(self) -> None:
        names = tuple(entry.directed_pair for entry in self.entries)
        if len(set(names)) != len(names):
            raise ValueError("pair-contrast evidence contains duplicate directed pairs")


class SpearmanError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SpearmanReport:
    rho: Coefficient
    point_count: SampleCount
    pair: DirectedPairName
