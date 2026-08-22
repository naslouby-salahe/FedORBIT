from __future__ import annotations

from dataclasses import dataclass

import torch

from fedorbit.config.models import FedorbitConfig
from fedorbit.models.architectures import classifier_for_modality
from fedorbit.models.training import BaseCheckpoint
from fedorbit.response.final import (
    FinalResponseEntry,
    FinalResponseEstimate,
    build_source_packet,
    estimate_final_response,
)
from fedorbit.response.pilot import ResponseCandidate
from fedorbit.runtime.seeds import RngNamespace, derive_seed32
from fedorbit.strict_interface.packet import SourcePacket


@dataclass(frozen=True, slots=True)
class PacketConstructionContext:
    modality: str
    input_dimension: int
    n_classes: int
    coarse_group_id: str
    fine_node_order: tuple[str, ...]
    per_node_train_support: tuple[int, ...]
    per_node_meta_support: tuple[int, ...]
    source_checkpoint_sha256: str
    preprocessing_state_sha256: str
    transfer_node_manifest_sha256: str
    response_configuration_sha256: str
    seed: int
    learning_rate: float
    weight_decay: float
    dropout_probability: float


@dataclass(frozen=True, slots=True)
class ConstructedPacket:
    packet: SourcePacket
    estimate: FinalResponseEstimate


class PacketConstructionError(ValueError):
    pass


def construct_source_packet(
    config: FedorbitConfig,
    context: PacketConstructionContext,
    checkpoint: BaseCheckpoint,
    model: torch.nn.Module | None,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    meta_features: torch.Tensor,
    meta_targets: torch.Tensor,
    intervention_classes: tuple[tuple[int, ...], ...],
    outcome_native_class_sets: tuple[tuple[int, ...], ...],
    base_class_weights: torch.Tensor,
    selected_configuration: ResponseCandidate,
    creation_timestamp: str,
) -> ConstructedPacket:
    if model is None:
        model = classifier_for_modality(
            context.modality,
            context.input_dimension,
            context.n_classes,
            context.dropout_probability,
        )
    estimate = estimate_final_response(
        config,
        model,
        checkpoint,
        train_features,
        train_targets,
        meta_features,
        meta_targets,
        intervention_classes,
        outcome_native_class_sets,
        base_class_weights,
        selected_configuration.intervention_magnitude,
        selected_configuration.optimizer_step_horizon,
        context.learning_rate,
        context.weight_decay,
        context.seed,
    )
    absent_estimate = pad_absent_transfer_nodes(
        estimate, len(outcome_native_class_sets), len(intervention_classes)
    )
    anonymous_order = anonymize_node_order(context)
    per_node_replicates = tuple(
        config.scientific.source_response_final.paired_replicates_per_intervention
        for _ in context.fine_node_order
    )
    packet = build_source_packet(
        absent_estimate,
        anonymous_fine_node_ids=anonymous_order,
        exposed_coarse_group_id=context.coarse_group_id,
        per_node_train_support=context.per_node_train_support,
        per_node_meta_support=context.per_node_meta_support,
        per_node_effective_replicate_count=per_node_replicates,
        source_checkpoint_sha256=context.source_checkpoint_sha256,
        response_configuration_sha256=context.response_configuration_sha256,
        creation_timestamp=creation_timestamp,
        preprocessing_state_sha256=context.preprocessing_state_sha256,
        transfer_node_manifest_sha256=context.transfer_node_manifest_sha256,
        response_seed=context.seed,
    )
    return ConstructedPacket(packet, absent_estimate)


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
                padded.append(zero_entry(outcome, intervention))
    return FinalResponseEstimate(
        tuple(padded),
        estimate.critical_value,
        estimate.useful_intervention_columns,
        estimate.median_band_width_ratio,
        estimate.stability_rule_passed,
    )


def zero_entry(outcome_index: int, intervention_index: int) -> FinalResponseEntry:
    return FinalResponseEntry(
        outcome_index,
        intervention_index,
        0.0,
        0.0,
        0.0,
        0.0,
        False,
    )


def anonymize_node_order(context: PacketConstructionContext) -> tuple[str, ...]:
    if len(context.fine_node_order) != len(context.per_node_train_support):
        raise PacketConstructionError("fine-node order and support lengths differ")
    rng = torch.Generator().manual_seed(
        derive_seed32(context.seed, RngNamespace.ANONYMOUS_NODE_ORDER, "source-packet")
    )
    permutation = tuple(
        int(torch.randint(0, len(context.fine_node_order), (1,), generator=rng)[0])
        for _ in range(len(context.fine_node_order))
    )
    shuffled = tuple(context.fine_node_order[index] for index in permutation)
    if tuple(sorted(shuffled)) != tuple(sorted(context.fine_node_order)):
        raise PacketConstructionError("anonymization permutation is not a bijection")
    return shuffled
