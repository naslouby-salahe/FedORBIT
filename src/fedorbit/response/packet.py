from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields

import numpy as np
import torch

from fedorbit.config.context import active_config
from fedorbit.domain.enums import ClientRole, CoarseGroup, DatasetId
from fedorbit.domain.serialization import stable_json
from fedorbit.response.estimation import ShadowSettings
from fedorbit.response.pilot import PilotData, ResponseCandidate
from fedorbit.response.uncertainty import (
    FinalResponseEntry,
    FinalResponseEstimate,
    estimate_final_response,
)
from fedorbit.strict_interface.anonymity import (
    AnonymityCoordinate,
    AnonymityCoordinateEntry,
    anonymous_node_order,
)
from fedorbit.strict_interface.validation import (
    validate_anonymous_node_ids,
    validate_exact_fields,
    validate_rfc3339_utc,
    validate_sha256,
)
from fedorbit.training.losses import ClassWeights
from fedorbit.training.pilot import create_classifier
from fedorbit.training.trainer import BaseCheckpoint

RESPONSE_PACKET_SCHEMA = "source-response-packet/v1"
PACKET_PERMITTED_FIELDS = frozenset(
    {
        "anonymous_fine_node_ids",
        "exposed_coarse_group_id",
        "L",
        "U",
        "per_node_train_support",
        "per_node_meta_support",
        "per_node_effective_replicate_count",
        "packet_schema_metadata",
        "source_checkpoint_sha256",
        "response_configuration_sha256",
        "packet_integrity_sha256",
        "packet_validity_state",
        "technical_creation_timestamp",
    }
)


class PacketError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Float64ArrayPayload:
    dtype: str
    order: str
    shape: tuple[int, ...]
    data: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PacketIntegrityPayload:
    anonymous_fine_node_ids: tuple[str, ...]
    exposed_coarse_group_id: str
    L: Float64ArrayPayload
    U: Float64ArrayPayload
    per_node_train_support: Float64ArrayPayload
    per_node_meta_support: Float64ArrayPayload
    per_node_effective_replicate_count: Float64ArrayPayload
    packet_schema_metadata: str
    source_checkpoint_sha256: str
    response_configuration_sha256: str
    packet_validity_state: str


@dataclass(frozen=True, slots=True)
class PacketWirePayload:
    anonymous_fine_node_ids: tuple[str, ...]
    exposed_coarse_group_id: str
    L: Float64ArrayPayload
    U: Float64ArrayPayload
    per_node_train_support: Float64ArrayPayload
    per_node_meta_support: Float64ArrayPayload
    per_node_effective_replicate_count: Float64ArrayPayload
    packet_schema_metadata: str
    source_checkpoint_sha256: str
    response_configuration_sha256: str
    packet_integrity_sha256: str
    packet_validity_state: str
    technical_creation_timestamp: str


@dataclass(frozen=True, slots=True)
class SourcePacket:
    anonymous_fine_node_ids: tuple[str, ...]
    exposed_coarse_group_id: str
    L: tuple[float, ...]
    U: tuple[float, ...]
    per_node_train_support: tuple[int, ...]
    per_node_meta_support: tuple[int, ...]
    per_node_effective_replicate_count: tuple[int, ...]
    packet_schema_metadata: str
    source_checkpoint_sha256: str
    response_configuration_sha256: str
    packet_integrity_sha256: str
    packet_validity_state: str
    technical_creation_timestamp: str

    def integrity_payload(self) -> str:
        return stable_json(self._integrity_payload())

    def serialized(self) -> str:
        return stable_json(self._wire_payload())

    def compute_integrity_sha256(self) -> str:
        return hashlib.sha256(self.integrity_payload().encode("utf-8")).hexdigest()

    def payload_sha256(self) -> str:
        return hashlib.sha256(self.serialized().encode("utf-8")).hexdigest()

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
        if self.packet_schema_metadata != RESPONSE_PACKET_SCHEMA:
            raise PacketError("unrecognized source-response packet schema")
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
        if any(not math.isfinite(value) for value in (*self.L, *self.U)):
            raise PacketError("response interval contains a non-finite value")
        if any(value < 0 for value in (*self.per_node_train_support, *self.per_node_meta_support)):
            raise PacketError("per-node support must be nonnegative")
        if any(value <= 0 for value in self.per_node_effective_replicate_count):
            raise PacketError("effective replicate counts must be positive")
        if self.packet_validity_state not in {"stable", "unstable"}:
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


@dataclass(frozen=True, slots=True)
class PacketConstructionContext:
    dataset: DatasetId
    input_dimension: int
    n_classes: int
    coarse_group_id: CoarseGroup
    fine_node_order: tuple[str, ...]
    per_node_train_support: tuple[int, ...]
    per_node_meta_support: tuple[int, ...]
    source_checkpoint_sha256: str
    response_configuration_sha256: str
    seed: int


@dataclass(frozen=True, slots=True)
class ConstructedPacket:
    packet: SourcePacket
    estimate: FinalResponseEstimate


class PacketConstructionError(ValueError):
    pass


def construct_source_packet(
    context: PacketConstructionContext,
    checkpoint: BaseCheckpoint,
    model: torch.nn.Module | None,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    meta_features: torch.Tensor,
    meta_targets: torch.Tensor,
    intervention_classes: tuple[tuple[int, ...], ...],
    outcome_native_class_sets: tuple[tuple[int, ...], ...],
    base_class_weights: ClassWeights,
    selected_configuration: ResponseCandidate,
    creation_timestamp: str,
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
        exposed_coarse_group_id=context.coarse_group_id.value,
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
    anonymous_fine_node_ids: tuple[str, ...],
    exposed_coarse_group_id: str,
    per_node_train_support: tuple[int, ...],
    per_node_meta_support: tuple[int, ...],
    per_node_effective_replicate_count: tuple[int, ...],
    source_checkpoint_sha256: str,
    response_configuration_sha256: str,
    creation_timestamp: str,
) -> SourcePacket:
    def create(integrity: str) -> SourcePacket:
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
            packet_validity_state="stable" if estimate.stability_rule_passed else "unstable",
            technical_creation_timestamp=creation_timestamp,
        )

    packet = create("")
    return create(packet.compute_integrity_sha256())


def pad_absent_transfer_nodes(
    estimate: FinalResponseEstimate,
    outcome_count: int,
    intervention_count: int,
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


def _float64_array(values: tuple[int | float, ...]) -> Float64ArrayPayload:
    array = np.asarray(values, dtype=np.float64, order="C")
    return Float64ArrayPayload(
        dtype="float64",
        order="C",
        shape=tuple(int(size) for size in array.shape),
        data=tuple(float(value) for value in array.ravel(order="C")),
    )
