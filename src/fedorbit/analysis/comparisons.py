from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from fedorbit.types import (
    ArtifactIdentifier,
    ConsumedBudget,
    DirectedPairName,
    Index,
    RandomSeed,
    RelativeGain,
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


class PairingField(StrEnum):
    RAW_DATASET_LINEAGE = "raw dataset lineage"
    DIRECTED_PAIR = "directed pair"
    SEED = "seed"
    SPLIT = "split"
    TARGET_PRE_TRANSFER_CHECKPOINT = "target pre-transfer checkpoint"
    TARGET_IMPORTANCE = "target importance"
    SOURCE_PACKET = "source packet"
    ACTION_BUDGET = "action budget"
    SUPPORT_BUDGET = "support budget"
    CONFIRMATION_BUDGET = "confirmation budget"
    ENVIRONMENT_LINEAGE = "environment lineage"


class PairingMismatchError(PairingError):
    field: PairingField

    def __init__(self, field: PairingField) -> None:
        self.field = field
        super().__init__(f"paired comparison cells disagree on {field.value}")


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


def require_matching_lineage(first: PairedObservation, second: PairedObservation) -> None:
    left = first.lineage
    right = second.lineage
    if left.raw_dataset_lineage_sha256 != right.raw_dataset_lineage_sha256:
        raise PairingMismatchError(PairingField.RAW_DATASET_LINEAGE)
    if left.directed_pair != right.directed_pair:
        raise PairingMismatchError(PairingField.DIRECTED_PAIR)
    if left.seed != right.seed:
        raise PairingMismatchError(PairingField.SEED)
    if left.split != right.split:
        raise PairingMismatchError(PairingField.SPLIT)
    if left.target_pre_transfer_checkpoint_artifact_id != (
        right.target_pre_transfer_checkpoint_artifact_id
    ):
        raise PairingMismatchError(PairingField.TARGET_PRE_TRANSFER_CHECKPOINT)
    if left.target_importance_artifact_id != right.target_importance_artifact_id:
        raise PairingMismatchError(PairingField.TARGET_IMPORTANCE)
    if left.source_packet_artifact_id != right.source_packet_artifact_id:
        raise PairingMismatchError(PairingField.SOURCE_PACKET)
    if left.action_budget != right.action_budget:
        raise PairingMismatchError(PairingField.ACTION_BUDGET)
    if left.support_budget != right.support_budget:
        raise PairingMismatchError(PairingField.SUPPORT_BUDGET)
    if left.confirmation_budget != right.confirmation_budget:
        raise PairingMismatchError(PairingField.CONFIRMATION_BUDGET)
    if left.environment_lineage_sha256 != right.environment_lineage_sha256:
        raise PairingMismatchError(PairingField.ENVIRONMENT_LINEAGE)


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
