from __future__ import annotations

import contextlib
from collections import OrderedDict
from pathlib import Path
from typing import cast

from fedorbit.analysis.records import (
    MetricDirection,
)
from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.datasets.materialization import (
    MaterializationError,
    MaterializedClient,
)
from fedorbit.experiments.protocol import ExperimentExecutionRequest
from fedorbit.experiments.scoring import (
    assemble_cross_client_response_matrix,
    assemble_target_response_matrix,
    common_eligible_groups,
    cross_client_padded_blocks,
    load_dataset_source_packet,
    persist_primary_transfer_metric,
)
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
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
    MapAvailabilityAuditSubmission,
    blank_audit_template,
    distinct_researcher_ids,
    documented_public_labels,
    submission_sha256,
    validate_submission,
)
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
)
from fedorbit.optimization.exact_qap import (
    point_correspondence_commitment,
)
from fedorbit.response.packet import (
    SourcePacket,
)
from fedorbit.types import (
    ArtifactIdentifier,
    CoarseGroup,
    DatasetId,
    DirectedPair,
    DirectedPairName,
    ExperimentName,
    Index,
    MetricId,
    MetricUnit,
    ProducerModuleName,
    StableJsonPayload,
    TransferMethod,
)

_MODULE_NAME = ProducerModuleName("fedorbit.experiments.audit")


def _human_audit_researcher_id(index: Index) -> str:
    return f"researcher-{index + 1}"


def _human_audit_directory(
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    source: DatasetId,
    target: DatasetId,
    researcher_id: str,
) -> Path:
    return (
        experiment_workspace(layout, experiment)
        / "artifacts"
        / "fitted"
        / "human_audit"
        / f"{source.value}-to-{target.value}"
        / researcher_id
    )


def _score_packet_only_recovery_attempt(
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
) -> tuple[bool, tuple[ArtifactIdentifier, ...]] | None:
    common = common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return None
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
        return None
    source_matrix = assemble_cross_client_response_matrix(
        blocks, source_packets, SourcePacket.lower_matrix
    )
    target_matrix = assemble_target_response_matrix(
        blocks, target_packets, SourcePacket.lower_matrix
    )
    qap_result = point_correspondence_commitment(source_matrix, target_matrix, blocks)
    if not qap_result.certified or qap_result.correspondence is None:
        return None
    oracle_correspondence = BlockCorrespondence.lexicographically_smallest(blocks)
    recovered = qap_result.correspondence.images == oracle_correspondence.images
    input_artifact_ids = tuple(
        ArtifactIdentifier(packet.packet_integrity_sha256)
        for packet in (*source_packets.values(), *target_packets.values())
    )
    return recovered, input_artifact_ids


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
        if source_materialized is not None and target_materialized is not None:
            pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
            for seed in confirmatory_seeds:
                attempt = _score_packet_only_recovery_attempt(
                    layout, source, target, source_materialized, target_materialized, seed
                )
                if attempt is None:
                    continue
                recovered, input_artifact_ids = attempt
                persist_primary_transfer_metric(
                    store,
                    layout,
                    request.experiment,
                    pair_direction,
                    source,
                    target,
                    TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
                    seed,
                    MetricId.PACKET_ONLY_RECOVERY_ACCURACY,
                    1.0 if recovered else 0.0,
                    MetricUnit("boolean"),
                    MetricDirection.HIGHER_IS_BETTER,
                    input_artifact_ids,
                    request.overwrite_policy,
                )
        submissions: list[MapAvailabilityAuditSubmission] = []
        for researcher_index in range(config.independent_researchers):
            index: Index = researcher_index
            researcher_id = _human_audit_researcher_id(index)
            directory = _human_audit_directory(
                layout, request.experiment, source, target, researcher_id
            )
            submission_path = directory / "submission.json"
            if not submission_path.is_file():
                template = blank_audit_template(researcher_id, domain_pair)
                atomic_write_json(
                    directory / "template.json",
                    cast(StableJsonPayload, template.model_dump(mode="json")),
                )
                continue
            submission = MapAvailabilityAuditSubmission.model_validate_json(
                submission_path.read_text(encoding="utf-8")
            )
            failures = validate_submission(
                submission, config.minutes_per_researcher_per_pair, public_labels
            )
            if failures:
                continue
            submissions.append(submission)
        if len(submissions) != config.independent_researchers or not distinct_researcher_ids(
            tuple(submissions)
        ):
            continue
        for submission in submissions:
            atomic_write_json(
                _human_audit_directory(
                    layout, request.experiment, source, target, submission.researcher_id
                )
                / "validated.sha256.json",
                cast(
                    StableJsonPayload,
                    OrderedDict(sha256=submission_sha256(submission)),
                ),
            )
