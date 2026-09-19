from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from pydantic import JsonValue

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.environment import environment_snapshot
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
    StableJsonPayload,
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

    def payload(self) -> StableJsonPayload:
        packet = self.source_packet_artifact_id
        return cast(
            StableJsonPayload,
            OrderedDict(
                raw_dataset_lineage_sha256=self.raw_dataset_lineage_sha256,
                directed_pair=self.directed_pair,
                seed=self.seed,
                split=self.split.value,
                target_pre_transfer_checkpoint_artifact_id=(
                    self.target_pre_transfer_checkpoint_artifact_id
                ),
                target_importance_artifact_id=self.target_importance_artifact_id,
                source_packet_artifact_id=None if packet is None else packet,
                action_budget=self.action_budget,
                support_budget=self.support_budget,
                confirmation_budget=self.confirmation_budget,
                environment_lineage_sha256=self.environment_lineage_sha256,
            ),
        )

    @staticmethod
    def from_payload(payload: Mapping[str, JsonValue]) -> PairingLineage:
        packet = payload.get("source_packet_artifact_id")
        return PairingLineage(
            raw_dataset_lineage_sha256=Sha256Digest(str(payload["raw_dataset_lineage_sha256"])),
            directed_pair=DirectedPairName(str(payload["directed_pair"])),
            seed=_payload_int(payload["seed"]),
            split=Split(str(payload["split"])),
            target_pre_transfer_checkpoint_artifact_id=ArtifactIdentifier(
                str(payload["target_pre_transfer_checkpoint_artifact_id"])
            ),
            target_importance_artifact_id=ArtifactIdentifier(
                str(payload["target_importance_artifact_id"])
            ),
            source_packet_artifact_id=(None if packet is None else ArtifactIdentifier(str(packet))),
            action_budget=_payload_float(payload["action_budget"]),
            support_budget=_payload_int(payload["support_budget"]),
            confirmation_budget=_payload_int(payload["confirmation_budget"]),
            environment_lineage_sha256=Sha256Digest(str(payload["environment_lineage_sha256"])),
        )


def _payload_int(value: JsonValue) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PairingError("pairing lineage integer field is invalid")
    return value


def _payload_float(value: JsonValue) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PairingError("pairing lineage float field is invalid")
    return float(value)


def pair_seed_resource_identity(
    directed_pair: DirectedPairName,
    seed: RandomSeed,
    resource: str,
) -> ArtifactIdentifier:
    digest = hashlib.sha256(f"{directed_pair}|{seed}|{resource}".encode()).hexdigest()
    return ArtifactIdentifier(digest)


def pair_seed_pairing_lineage(
    directed_pair: DirectedPairName,
    seed: RandomSeed,
    checkpoint_artifact_id: ArtifactIdentifier,
    source_packet_artifact_id: ArtifactIdentifier | None = None,
    target_importance_artifact_id: ArtifactIdentifier | None = None,
    raw_dataset_lineage_sha256: Sha256Digest | None = None,
    environment_lineage_sha256: Sha256Digest | None = None,
) -> PairingLineage:
    action = active_config().scientific.action
    confirmation = active_config().scientific.confirmation
    raw_lineage = raw_dataset_lineage_sha256
    if raw_lineage is None:
        raw_lineage = Sha256Digest(hashlib.sha256(directed_pair.encode("utf-8")).hexdigest())
    environment = environment_lineage_sha256
    if environment is None:
        environment = environment_snapshot().fingerprint_sha256
    packet = source_packet_artifact_id
    if packet is None:
        packet = pair_seed_resource_identity(directed_pair, seed, "packet")
    importance = target_importance_artifact_id
    if importance is None:
        importance = pair_seed_resource_identity(directed_pair, seed, "importance")
    return PairingLineage(
        raw_dataset_lineage_sha256=raw_lineage,
        directed_pair=directed_pair,
        seed=seed,
        split=Split.TEST,
        target_pre_transfer_checkpoint_artifact_id=checkpoint_artifact_id,
        target_importance_artifact_id=importance,
        source_packet_artifact_id=packet,
        action_budget=action.total_curriculum_budget,
        support_budget=action.principal_sparse_support,
        confirmation_budget=confirmation.optimizer_steps_per_shadow,
        environment_lineage_sha256=environment,
    )


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
