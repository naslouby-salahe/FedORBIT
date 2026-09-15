from __future__ import annotations

import contextlib
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fedorbit.analysis.records import (
    MetricDirection,
)
from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.datasets.materialization import (
    MaterializationError,
    MaterializedClient,
)
from fedorbit.experiments.catalogue import ExperimentExecutionRequest
from fedorbit.experiments.scoring import (
    assemble_cross_client_response_matrix,
    assemble_target_response_matrix,
    common_eligible_groups,
    cross_client_padded_blocks,
    load_dataset_source_packet,
    persist_primary_transfer_metric,
)
from fedorbit.experiments.solvers import (
    registered_transfer_method,
)
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
    ExecutionError,
)
from fedorbit.infrastructure.preparation import (
    load_or_materialize_client,
)
from fedorbit.infrastructure.runtime import (
    RandomSeed,
)
from fedorbit.infrastructure.storage import atomic_write_json
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    experiment_workspace,
)
from fedorbit.methods.map_availability_audit import (
    MapAvailabilityAuditFileName,
    MapAvailabilityAuditPairOutcome,
    MapAvailabilityAuditSubmission,
    accepted_map_availability_submissions,
    blank_audit_template,
    decide_map_availability_outcome,
    documented_public_labels,
    oracle_correspondence_accuracy,
    required_concept_count,
    submission_is_complete_one_to_one,
    submission_sha256,
    summarise_map_availability_outcomes,
)
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
)
from fedorbit.optimization.exact_qap import (
    QapSeparatorResult,
    generic_exact_qap_correspondence,
    point_correspondence_commitment,
)
from fedorbit.response.packet import (
    SourcePacket,
)
from fedorbit.types import (
    ArtifactDirectorySegment,
    ArtifactIdentifier,
    CoarseGroup,
    DatasetId,
    DirectedPair,
    ExperimentName,
    HumanAuditFileName,
    Index,
    InvalidReason,
    MethodName,
    MetricId,
    MetricUnit,
    ResearcherIdentifier,
    StableJsonPayload,
    StorageLayoutSegment,
    TransferMethod,
    directed_pair_name,
)


class PacketOnlyRecoveryUnavailability(StrEnum):
    PAIR_ENDPOINT_UNAVAILABLE = "directed-pair endpoint client is not materialized"
    NO_SHARED_ELIGIBLE_COARSE_GROUP = "no shared eligible coarse group"
    NO_ANONYMOUS_SOURCE_PACKET = (
        "no eligible anonymous source packet for the directed pair and seed"
    )
    UNCERTIFIED_STRUCTURAL_QAP = "structural QAP correspondence is not certified"


type StructuralQapRecovery = Callable[
    [NDArray[np.float64], NDArray[np.float64], PaddedBlockStructure], QapSeparatorResult
]

RECOVERY_METHOD_ATTEMPTS: Mapping[TransferMethod, StructuralQapRecovery] = OrderedDict(
    (
        (TransferMethod.POINT_CORRESPONDENCE_COMMITMENT, point_correspondence_commitment),
        (TransferMethod.GENERIC_EXACT_QAP, generic_exact_qap_correspondence),
    )
)


@dataclass(frozen=True, slots=True)
class PacketOnlyRecoveryAttempt:
    unavailability: PacketOnlyRecoveryUnavailability | None
    recovered: bool
    input_artifact_ids: tuple[ArtifactIdentifier, ...]


def registered_recovery_method(method: MethodName) -> TransferMethod:
    transfer_method = registered_transfer_method(method)
    if transfer_method not in RECOVERY_METHOD_ATTEMPTS:
        raise ExecutionError(
            f"registered packet-only recovery method has no certified implementation: {method}"
        )
    return transfer_method


def human_audit_researcher_id(
    index: Index,
) -> ResearcherIdentifier:
    return ResearcherIdentifier(f"researcher-{index + 1}")


def human_audit_pair_directory(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    source: DatasetId,
    target: DatasetId,
) -> Path:
    return (
        experiment_workspace(layout, experiment)
        / StorageLayoutSegment.ARTIFACTS
        / ArtifactDirectorySegment.FITTED
        / ArtifactDirectorySegment.HUMAN_AUDIT
        / f"{source.value}-to-{target.value}"
    )


def human_audit_researcher_directory(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    source: DatasetId,
    target: DatasetId,
    researcher_id: ResearcherIdentifier,
) -> Path:
    return human_audit_pair_directory(layout, experiment, source, target) / researcher_id


def completed_human_audit_outcomes(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
) -> tuple[MapAvailabilityAuditPairOutcome, ...]:
    outcomes: list[MapAvailabilityAuditPairOutcome] = []
    for pair in active_config().scientific.datasets.primary_directed_pairs:
        path = (
            human_audit_pair_directory(layout, experiment, pair.source, pair.target)
            / MapAvailabilityAuditFileName.PAIR_OUTCOME
        )
        if not path.is_file():
            continue
        outcomes.append(
            MapAvailabilityAuditPairOutcome.model_validate_json(path.read_text(encoding="utf-8"))
        )
    return tuple(outcomes)


def _unavailable_attempt(
    unavailability: PacketOnlyRecoveryUnavailability,
    input_artifact_ids: tuple[ArtifactIdentifier, ...],
) -> PacketOnlyRecoveryAttempt:
    return PacketOnlyRecoveryAttempt(unavailability, False, input_artifact_ids)


def _score_packet_only_recovery_attempt(
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient | None,
    target_materialized: MaterializedClient | None,
    seed: RandomSeed,
    method: TransferMethod,
) -> PacketOnlyRecoveryAttempt:
    if source_materialized is None or target_materialized is None:
        return _unavailable_attempt(
            PacketOnlyRecoveryUnavailability.PAIR_ENDPOINT_UNAVAILABLE,
            (ArtifactIdentifier("ineligible-cell"),),
        )
    common = common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return _unavailable_attempt(
            PacketOnlyRecoveryUnavailability.NO_SHARED_ELIGIBLE_COARSE_GROUP,
            (ArtifactIdentifier("ineligible-cell"),),
        )
    source_eligible, target_eligible = common
    common_coarse = tuple(target_eligible)
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    source_packets: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    target_packets: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in common_coarse:
        source_packet = load_dataset_source_packet(layout, source, seed, coarse_group)
        if source_packet is not None:
            source_packets[coarse_group] = source_packet
        target_packet = load_dataset_source_packet(layout, target, seed, coarse_group)
        if target_packet is not None:
            target_packets[coarse_group] = target_packet
    if not source_packets or not target_packets:
        return _unavailable_attempt(
            PacketOnlyRecoveryUnavailability.NO_ANONYMOUS_SOURCE_PACKET,
            (ArtifactIdentifier("ineligible-cell"),),
        )
    source_matrix = assemble_cross_client_response_matrix(
        blocks, source_packets, SourcePacket.lower_matrix
    )
    target_matrix = assemble_target_response_matrix(
        blocks, target_packets, SourcePacket.lower_matrix
    )
    input_artifact_ids = tuple(
        ArtifactIdentifier(packet.packet_integrity_sha256)
        for packet in (*source_packets.values(), *target_packets.values())
    )
    outcome = RECOVERY_METHOD_ATTEMPTS[method](source_matrix, target_matrix, blocks)
    if not outcome.certified or outcome.correspondence is None:
        return _unavailable_attempt(
            PacketOnlyRecoveryUnavailability.UNCERTIFIED_STRUCTURAL_QAP, input_artifact_ids
        )
    oracle_correspondence = BlockCorrespondence.lexicographically_smallest(blocks)
    return PacketOnlyRecoveryAttempt(
        None,
        outcome.correspondence.images == oracle_correspondence.images,
        input_artifact_ids,
    )


def execute_map_availability_applicability_audit(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.map_availability_applicability_audit
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    public_labels = documented_public_labels()
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        domain_pair = DirectedPair(source=source, target=target)
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        pair_direction = directed_pair_name(source, target)
        for seed in confirmatory_seeds:
            for method_name in config.packet_only_recovery_methods:
                method = registered_recovery_method(method_name)
                attempt = _score_packet_only_recovery_attempt(
                    layout,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    method,
                )
                unavailability = attempt.unavailability
                persist_primary_transfer_metric(
                    store,
                    layout,
                    request.experiment,
                    pair_direction,
                    source,
                    target,
                    method,
                    seed,
                    MetricId.PACKET_ONLY_RECOVERY_ACCURACY,
                    None if unavailability is not None else (1.0 if attempt.recovered else 0.0),
                    MetricUnit.BOOLEAN,
                    MetricDirection.HIGHER_IS_BETTER,
                    attempt.input_artifact_ids,
                    request.overwrite_policy,
                    valid=unavailability is None,
                    invalid_reason=(
                        None if unavailability is None else InvalidReason(unavailability.value)
                    ),
                )
        submissions: list[MapAvailabilityAuditSubmission] = []
        for researcher_index in range(config.independent_researchers):
            index: Index = researcher_index
            researcher_id = human_audit_researcher_id(index)
            directory = human_audit_researcher_directory(
                layout, request.experiment, source, target, researcher_id
            )
            submission_path = directory / HumanAuditFileName.SUBMISSION
            if not submission_path.is_file():
                template = blank_audit_template(researcher_id, domain_pair)
                atomic_write_json(
                    directory / HumanAuditFileName.TEMPLATE,
                    cast(StableJsonPayload, template.model_dump(mode="json")),
                )
                continue
            submission = MapAvailabilityAuditSubmission.model_validate_json(
                submission_path.read_text(encoding="utf-8")
            )
            submissions.append(submission)
        accepted = accepted_map_availability_submissions(
            tuple(submissions),
            config.minutes_per_researcher_per_pair,
            public_labels,
        )
        endpoint_directory = human_audit_pair_directory(layout, request.experiment, source, target)
        if accepted:
            atomic_write_json(
                endpoint_directory / MapAvailabilityAuditFileName.RECORDED_SUBMISSION_SHA256,
                cast(
                    StableJsonPayload,
                    OrderedDict(
                        source=source.value,
                        target=target.value,
                        recorded_submission_sha256=OrderedDict(
                            (
                                submission.researcher_id,
                                submission_sha256(submission),
                            )
                            for submission in accepted
                        ),
                    ),
                ),
            )
        decision = decide_map_availability_outcome(
            domain_pair,
            accepted,
            config.independent_researchers,
            config.minutes_per_researcher_per_pair,
            public_labels,
        )
        atomic_write_json(
            endpoint_directory / MapAvailabilityAuditFileName.PAIR_OUTCOME,
            cast(StableJsonPayload, decision.record().model_dump(mode="json")),
        )
        for submission in accepted:
            atomic_write_json(
                human_audit_researcher_directory(
                    layout, request.experiment, source, target, submission.researcher_id
                )
                / HumanAuditFileName.VALIDATED_SHA256,
                cast(
                    StableJsonPayload,
                    OrderedDict(
                        sha256=submission_sha256(submission),
                        complete_one_to_one=submission_is_complete_one_to_one(
                            submission, required_concept_count()
                        ),
                        oracle_correspondence_accuracy=oracle_correspondence_accuracy(submission),
                    ),
                ),
            )
    atomic_write_json(
        experiment_workspace(layout, request.experiment)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
        / MapAvailabilityAuditFileName.EXPERIMENT_SUMMARY,
        cast(
            StableJsonPayload,
            summarise_map_availability_outcomes(
                config.independent_researchers,
                completed_human_audit_outcomes(layout, request.experiment),
            ).model_dump(mode="json"),
        ),
    )
