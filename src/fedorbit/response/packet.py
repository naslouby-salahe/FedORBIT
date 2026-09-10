from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
from pydantic import ValidationError

from fedorbit.config.loading import active_config
from fedorbit.config.models import FrozenModel
from fedorbit.datasets.ontology import TRANSFER_ONTOLOGY
from fedorbit.interface import (
    AnonymityCoordinate,
    AnonymityCoordinateEntry,
    anonymous_node_order,
    validate_anonymous_node_ids,
    validate_exact_fields,
    validate_rfc3339_utc,
    validate_sha256,
    validate_static_leakage_scan,
)
from fedorbit.learning.pilot import create_classifier
from fedorbit.learning.training import BaseCheckpoint, ClassWeights
from fedorbit.response.estimation import ShadowSettings
from fedorbit.response.pilot import PilotData, ResponseCandidate
from fedorbit.response.uncertainty import (
    FinalResponseEntry,
    FinalResponseEstimate,
    NativeClassSets,
    estimate_final_response,
)
from fedorbit.types import (
    AnonymousNodeDisplayId,
    ClientRole,
    CoarseGroup,
    DatasetId,
    Estimate,
    ExposedCoarseGroupId,
    FeatureCount,
    Index,
    RandomSeed,
    ReplicateCount,
    Rfc3339UtcTimestamp,
    SerializedPacket,
    Sha256Digest,
    stable_json,
)

type NodeDisplayIds = tuple[AnonymousNodeDisplayId, ...]
type ResponseBounds = tuple[Estimate, ...]
type PerNodeSupport = tuple[Index, ...]
type PerNodeReplicateCounts = tuple[ReplicateCount, ...]
type ArrayShape = tuple[Index, ...]
type Float64ArrayData = tuple[Estimate, ...]


class ResponsePacketSchema(StrEnum):
    V1 = "source-response-packet/v1"


class PacketField(StrEnum):
    ANONYMOUS_FINE_NODE_IDS = "anonymous_fine_node_ids"
    EXPOSED_COARSE_GROUP_ID = "exposed_coarse_group_id"
    LOWER_BOUNDS = "L"
    UPPER_BOUNDS = "U"
    PER_NODE_TRAIN_SUPPORT = "per_node_train_support"
    PER_NODE_META_SUPPORT = "per_node_meta_support"
    PER_NODE_EFFECTIVE_REPLICATE_COUNT = "per_node_effective_replicate_count"
    SCHEMA_METADATA = "packet_schema_metadata"
    SOURCE_CHECKPOINT_SHA256 = "source_checkpoint_sha256"
    RESPONSE_CONFIGURATION_SHA256 = "response_configuration_sha256"
    INTEGRITY_SHA256 = "packet_integrity_sha256"
    VALIDITY_STATE = "packet_validity_state"
    TECHNICAL_CREATION_TIMESTAMP = "technical_creation_timestamp"


class PacketArrayDtype(StrEnum):
    FLOAT64 = "float64"


class PacketArrayOrder(StrEnum):
    C = "C"


class PacketValidityState(StrEnum):
    STABLE = "stable"
    UNSTABLE = "unstable"


RESPONSE_PACKET_SCHEMA = ResponsePacketSchema.V1
PACKET_PERMITTED_FIELDS = frozenset(field.value for field in PacketField)


def _fine_semantic_label_terms() -> frozenset[str]:
    terms: set[str] = set()
    for _, edge_labels, ton_labels in TRANSFER_ONTOLOGY.values():
        terms.update(str(label) for label in edge_labels)
        terms.update(str(label) for label in ton_labels)
    return frozenset(terms)


FINE_SEMANTIC_LABEL_TERMS = _fine_semantic_label_terms()


class PacketError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Float64ArrayPayload:
    dtype: PacketArrayDtype
    order: PacketArrayOrder
    shape: ArrayShape
    data: Float64ArrayData


class PacketArrayDocument(FrozenModel):
    dtype: PacketArrayDtype
    order: PacketArrayOrder
    shape: ArrayShape
    data: Float64ArrayData


class PacketDocument(FrozenModel):
    anonymous_fine_node_ids: NodeDisplayIds
    exposed_coarse_group_id: ExposedCoarseGroupId
    L: PacketArrayDocument
    U: PacketArrayDocument
    per_node_train_support: PacketArrayDocument
    per_node_meta_support: PacketArrayDocument
    per_node_effective_replicate_count: PacketArrayDocument
    packet_schema_metadata: ResponsePacketSchema
    source_checkpoint_sha256: Sha256Digest
    response_configuration_sha256: Sha256Digest
    packet_integrity_sha256: Sha256Digest
    packet_validity_state: PacketValidityState
    technical_creation_timestamp: Rfc3339UtcTimestamp


@dataclass(frozen=True, slots=True)
class PacketIntegrityPayload:
    anonymous_fine_node_ids: NodeDisplayIds
    exposed_coarse_group_id: ExposedCoarseGroupId
    L: Float64ArrayPayload
    U: Float64ArrayPayload
    per_node_train_support: Float64ArrayPayload
    per_node_meta_support: Float64ArrayPayload
    per_node_effective_replicate_count: Float64ArrayPayload
    packet_schema_metadata: ResponsePacketSchema
    source_checkpoint_sha256: Sha256Digest
    response_configuration_sha256: Sha256Digest
    packet_validity_state: PacketValidityState


@dataclass(frozen=True, slots=True)
class PacketWirePayload:
    anonymous_fine_node_ids: NodeDisplayIds
    exposed_coarse_group_id: ExposedCoarseGroupId
    L: Float64ArrayPayload
    U: Float64ArrayPayload
    per_node_train_support: Float64ArrayPayload
    per_node_meta_support: Float64ArrayPayload
    per_node_effective_replicate_count: Float64ArrayPayload
    packet_schema_metadata: ResponsePacketSchema
    source_checkpoint_sha256: Sha256Digest
    response_configuration_sha256: Sha256Digest
    packet_integrity_sha256: Sha256Digest
    packet_validity_state: PacketValidityState
    technical_creation_timestamp: Rfc3339UtcTimestamp


@dataclass(frozen=True, slots=True)
class SourcePacket:
    anonymous_fine_node_ids: NodeDisplayIds
    exposed_coarse_group_id: ExposedCoarseGroupId
    L: ResponseBounds
    U: ResponseBounds
    per_node_train_support: PerNodeSupport
    per_node_meta_support: PerNodeSupport
    per_node_effective_replicate_count: PerNodeReplicateCounts
    packet_schema_metadata: ResponsePacketSchema
    source_checkpoint_sha256: Sha256Digest
    response_configuration_sha256: Sha256Digest
    packet_integrity_sha256: Sha256Digest
    packet_validity_state: PacketValidityState
    technical_creation_timestamp: Rfc3339UtcTimestamp

    def integrity_payload(self) -> SerializedPacket:
        return SerializedPacket(stable_json(self._integrity_payload()))

    def serialized(self) -> SerializedPacket:
        return SerializedPacket(stable_json(self._wire_payload()))

    @classmethod
    def from_serialized(cls, payload: SerializedPacket) -> SourcePacket:
        try:
            document = PacketDocument.model_validate_json(payload)
        except ValidationError as error:
            raise PacketError("malformed source-response packet serialization") from error
        arrays = (
            document.L,
            document.U,
            document.per_node_train_support,
            document.per_node_meta_support,
            document.per_node_effective_replicate_count,
        )
        if any(
            array.dtype != "float64"
            or array.order != "C"
            or len(array.shape) != 1
            or array.shape[0] != len(array.data)
            for array in arrays
        ):
            raise PacketError("packet arrays must be one-dimensional C-order float64")
        integer_arrays = (
            document.per_node_train_support,
            document.per_node_meta_support,
            document.per_node_effective_replicate_count,
        )
        if any(
            any(not float(value).is_integer() for value in array.data) for array in integer_arrays
        ):
            raise PacketError("packet support arrays must contain integer values")
        packet = cls(
            anonymous_fine_node_ids=document.anonymous_fine_node_ids,
            exposed_coarse_group_id=document.exposed_coarse_group_id,
            L=document.L.data,
            U=document.U.data,
            per_node_train_support=tuple(
                int(value) for value in document.per_node_train_support.data
            ),
            per_node_meta_support=tuple(
                int(value) for value in document.per_node_meta_support.data
            ),
            per_node_effective_replicate_count=tuple(
                int(value) for value in document.per_node_effective_replicate_count.data
            ),
            packet_schema_metadata=document.packet_schema_metadata,
            source_checkpoint_sha256=document.source_checkpoint_sha256,
            response_configuration_sha256=document.response_configuration_sha256,
            packet_integrity_sha256=document.packet_integrity_sha256,
            packet_validity_state=document.packet_validity_state,
            technical_creation_timestamp=document.technical_creation_timestamp,
        )
        packet.validate()
        return packet

    def compute_integrity_sha256(self) -> Sha256Digest:
        return Sha256Digest(hashlib.sha256(self.integrity_payload().encode("utf-8")).hexdigest())

    def payload_sha256(self) -> Sha256Digest:
        return Sha256Digest(hashlib.sha256(self.serialized().encode("utf-8")).hexdigest())

    def lower_matrix(self) -> np.ndarray:
        return self._response_matrix(self.L)

    def upper_matrix(self) -> np.ndarray:
        return self._response_matrix(self.U)

    def validate(self) -> None:
        validate_exact_fields(
            frozenset(field.name for field in fields(self)),
            PACKET_PERMITTED_FIELDS,
        )
        validate_anonymous_node_ids(self.anonymous_fine_node_ids)
        validate_rfc3339_utc(self.technical_creation_timestamp)
        validate_sha256(self.source_checkpoint_sha256, "source checkpoint SHA-256")
        validate_sha256(
            self.response_configuration_sha256,
            "response configuration SHA-256",
        )
        validate_sha256(self.packet_integrity_sha256, "packet integrity SHA-256")
        if self.packet_schema_metadata is not RESPONSE_PACKET_SCHEMA:
            raise PacketError("unrecognized source-response packet schema")
        validate_static_leakage_scan(self.serialized().encode("utf-8"), FINE_SEMANTIC_LABEL_TERMS)
        node_count = len(self.anonymous_fine_node_ids)
        if not all(
            len(values) == node_count
            for values in (
                self.per_node_train_support,
                self.per_node_meta_support,
                self.per_node_effective_replicate_count,
            )
        ):
            raise PacketError("per-node packet arrays do not match anonymous node count")
        if len(self.L) != len(self.U) or not self.L:
            raise PacketError("response interval arrays must be non-empty and equal length")
        if len(self.L) != node_count * node_count:
            raise PacketError("response interval arrays do not form a square node matrix")
        if any(not math.isfinite(value) for value in (*self.L, *self.U)):
            raise PacketError("response interval contains a non-finite value")
        if any(value < 0 for value in (*self.per_node_train_support, *self.per_node_meta_support)):
            raise PacketError("per-node support must be nonnegative")
        if any(value <= 0 for value in self.per_node_effective_replicate_count):
            raise PacketError("effective replicate counts must be positive")
        if self.packet_validity_state not in set(PacketValidityState):
            raise PacketError("invalid packet validity state")
        if self.packet_integrity_sha256 != self.compute_integrity_sha256():
            raise PacketError("packet integrity SHA-256 mismatch")

    def _integrity_payload(self) -> PacketIntegrityPayload:
        return PacketIntegrityPayload(
            anonymous_fine_node_ids=self.anonymous_fine_node_ids,
            exposed_coarse_group_id=self.exposed_coarse_group_id,
            L=_float64_array(self.L),
            U=_float64_array(self.U),
            per_node_train_support=_float64_array(self.per_node_train_support),
            per_node_meta_support=_float64_array(self.per_node_meta_support),
            per_node_effective_replicate_count=_float64_array(
                self.per_node_effective_replicate_count
            ),
            packet_schema_metadata=self.packet_schema_metadata,
            source_checkpoint_sha256=self.source_checkpoint_sha256,
            response_configuration_sha256=self.response_configuration_sha256,
            packet_validity_state=self.packet_validity_state,
        )

    def _wire_payload(self) -> PacketWirePayload:
        scientific = self._integrity_payload()
        return PacketWirePayload(
            anonymous_fine_node_ids=scientific.anonymous_fine_node_ids,
            exposed_coarse_group_id=scientific.exposed_coarse_group_id,
            L=scientific.L,
            U=scientific.U,
            per_node_train_support=scientific.per_node_train_support,
            per_node_meta_support=scientific.per_node_meta_support,
            per_node_effective_replicate_count=scientific.per_node_effective_replicate_count,
            packet_schema_metadata=scientific.packet_schema_metadata,
            source_checkpoint_sha256=scientific.source_checkpoint_sha256,
            response_configuration_sha256=scientific.response_configuration_sha256,
            packet_integrity_sha256=self.packet_integrity_sha256,
            packet_validity_state=scientific.packet_validity_state,
            technical_creation_timestamp=self.technical_creation_timestamp,
        )

    def _response_matrix(self, entries: ResponseBounds) -> np.ndarray:
        node_count = len(self.anonymous_fine_node_ids)
        if len(entries) != node_count * node_count:
            raise PacketError("packet response entries do not form a square node matrix")
        return np.asarray(entries, dtype=np.float64).reshape((node_count, node_count), order="C")


@dataclass(frozen=True, slots=True)
class PacketConstructionContext:
    dataset: DatasetId
    input_dimension: FeatureCount
    n_classes: Index
    coarse_group_id: CoarseGroup
    fine_node_order: NodeDisplayIds
    per_node_train_support: PerNodeSupport
    per_node_meta_support: PerNodeSupport
    source_checkpoint_sha256: Sha256Digest
    response_configuration_sha256: Sha256Digest
    seed: RandomSeed


@dataclass(frozen=True, slots=True)
class ConstructedPacket:
    packet: SourcePacket
    estimate: FinalResponseEstimate


class PacketConstructionError(ValueError):
    pass


def load_source_packet(source: Path) -> SourcePacket:
    if not source.is_file():
        raise PacketError(f"source-response packet does not exist: {source}")
    return SourcePacket.from_serialized(SerializedPacket(source.read_text(encoding="utf-8")))


def construct_source_packet(
    context: PacketConstructionContext,
    checkpoint: BaseCheckpoint,
    model: torch.nn.Module | None,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    meta_features: torch.Tensor,
    meta_targets: torch.Tensor,
    intervention_classes: NativeClassSets,
    outcome_native_class_sets: NativeClassSets,
    base_class_weights: ClassWeights,
    selected_configuration: ResponseCandidate,
    creation_timestamp: Rfc3339UtcTimestamp,
) -> ConstructedPacket:
    _validate_context(context)
    hyperparameters = checkpoint.selected_hyperparameters
    if model is None:
        model = create_classifier(
            context.dataset,
            context.input_dimension,
            context.n_classes,
            hyperparameters.dropout_probability,
            context.seed,
        )
    data = PilotData(
        train_features,
        train_targets,
        meta_features,
        meta_targets,
        outcome_native_class_sets,
        base_class_weights,
        hyperparameters.learning_rate,
        hyperparameters.weight_decay,
    )
    settings = ShadowSettings(
        selected_configuration.intervention_magnitude,
        selected_configuration.optimizer_step_horizon,
        hyperparameters.learning_rate,
        hyperparameters.weight_decay,
    )
    estimate = estimate_final_response(
        model,
        checkpoint,
        data,
        intervention_classes,
        settings,
        context.seed,
    )
    complete_estimate = pad_absent_transfer_nodes(
        estimate,
        len(outcome_native_class_sets),
        len(intervention_classes),
    )
    ordering = anonymous_node_order(
        len(context.fine_node_order),
        context.seed,
        ClientRole.SOURCE,
        context.coarse_group_id,
        AnonymityCoordinate(
            (
                AnonymityCoordinateEntry(
                    "source_checkpoint_sha256",
                    context.source_checkpoint_sha256,
                ),
                AnonymityCoordinateEntry(
                    "response_configuration_sha256",
                    context.response_configuration_sha256,
                ),
            )
        ),
    )
    replicate_count = (
        active_config().scientific.source_response_final.paired_replicates_per_intervention
    )
    packet = build_source_packet(
        complete_estimate,
        anonymous_fine_node_ids=ordering.display_ids,
        exposed_coarse_group_id=ExposedCoarseGroupId(context.coarse_group_id.value),
        per_node_train_support=ordering.reorder(context.per_node_train_support),
        per_node_meta_support=ordering.reorder(context.per_node_meta_support),
        per_node_effective_replicate_count=ordering.reorder(
            tuple(replicate_count for _ in context.fine_node_order)
        ),
        source_checkpoint_sha256=context.source_checkpoint_sha256,
        response_configuration_sha256=context.response_configuration_sha256,
        creation_timestamp=creation_timestamp,
    )
    packet.validate()
    return ConstructedPacket(packet, complete_estimate)


def build_source_packet(
    estimate: FinalResponseEstimate,
    anonymous_fine_node_ids: NodeDisplayIds,
    exposed_coarse_group_id: ExposedCoarseGroupId,
    per_node_train_support: PerNodeSupport,
    per_node_meta_support: PerNodeSupport,
    per_node_effective_replicate_count: PerNodeReplicateCounts,
    source_checkpoint_sha256: Sha256Digest,
    response_configuration_sha256: Sha256Digest,
    creation_timestamp: Rfc3339UtcTimestamp,
) -> SourcePacket:
    def create(integrity: Sha256Digest) -> SourcePacket:
        return SourcePacket(
            anonymous_fine_node_ids=anonymous_fine_node_ids,
            exposed_coarse_group_id=exposed_coarse_group_id,
            L=tuple(entry.lower for entry in estimate.entries),
            U=tuple(entry.upper for entry in estimate.entries),
            per_node_train_support=per_node_train_support,
            per_node_meta_support=per_node_meta_support,
            per_node_effective_replicate_count=per_node_effective_replicate_count,
            packet_schema_metadata=RESPONSE_PACKET_SCHEMA,
            source_checkpoint_sha256=source_checkpoint_sha256,
            response_configuration_sha256=response_configuration_sha256,
            packet_integrity_sha256=integrity,
            packet_validity_state=(
                PacketValidityState.STABLE
                if estimate.stability_rule_passed
                else PacketValidityState.UNSTABLE
            ),
            technical_creation_timestamp=creation_timestamp,
        )

    packet = create(Sha256Digest(""))
    return create(packet.compute_integrity_sha256())


def pad_absent_transfer_nodes(
    estimate: FinalResponseEstimate,
    outcome_count: Index,
    intervention_count: Index,
) -> FinalResponseEstimate:
    present = {(entry.outcome_index, entry.intervention_index) for entry in estimate.entries}
    padded = list(estimate.entries)
    for outcome in range(outcome_count):
        for intervention in range(intervention_count):
            if (outcome, intervention) not in present:
                padded.append(
                    FinalResponseEntry(
                        outcome,
                        intervention,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        False,
                    )
                )
    padded.sort(key=lambda entry: (entry.outcome_index, entry.intervention_index))
    return FinalResponseEstimate(
        tuple(padded),
        estimate.critical_value,
        estimate.useful_intervention_columns,
        estimate.median_band_width_ratio,
        estimate.stability_rule_passed,
    )


def _validate_context(context: PacketConstructionContext) -> None:
    node_count = len(context.fine_node_order)
    if node_count == 0:
        raise PacketConstructionError("source packet requires at least one fine node")
    if len(set(context.fine_node_order)) != node_count:
        raise PacketConstructionError("source fine-node order contains duplicates")
    if len(context.per_node_train_support) != node_count:
        raise PacketConstructionError("TRAIN support count differs from source fine-node count")
    if len(context.per_node_meta_support) != node_count:
        raise PacketConstructionError("META support count differs from source fine-node count")


def _float64_array(values: tuple[Index | Estimate, ...]) -> Float64ArrayPayload:
    array = np.asarray(values, dtype=np.float64, order="C")
    return Float64ArrayPayload(
        dtype=PacketArrayDtype.FLOAT64,
        order=PacketArrayOrder.C,
        shape=tuple(int(size) for size in array.shape),
        data=tuple(float(value) for value in array.ravel(order="C")),
    )
