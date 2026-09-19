from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fedorbit.analysis.comparisons import PairingLineage, pair_seed_pairing_lineage
from fedorbit.analysis.metrics import (
    ClassF1,
    ClassF1Set,
    ClassRecall,
    ClassRecallSet,
    balanced_accuracy,
    confusion_counts,
    f1_from_counts,
    macro_f1,
    recall_from_counts,
)
from fedorbit.analysis.records import (
    DiagnosticMetricRecord,
    DiagnosticMetricRecordCollection,
    MetricDirection,
    MetricRecord,
    MetricRecordCollection,
    validate_diagnostic_metric_records,
    validate_metric_records,
)
from fedorbit.experiments.catalogue import ExperimentExecutionRequest
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
    StagedPayload,
)
from fedorbit.infrastructure.environment import environment_snapshot
from fedorbit.infrastructure.manifests import (
    CompletionManifest,
    ReusableArtifactManifest,
    SemanticCellManifest,
    artifact_id,
    build_artifact_completion,
)
from fedorbit.infrastructure.provenance import (
    configuration_subset_digest,
    implementation_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.infrastructure.runtime import (
    RandomSeed,
    current_code_revision,
)
from fedorbit.infrastructure.storage import (
    payload_sha256 as digest_payload_bytes,
)
from fedorbit.infrastructure.storage import (
    serialized_payload_bytes,
    stage_bytes,
)
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    experiment_workspace,
)
from fedorbit.learning.scoring import ScoreArtifact
from fedorbit.optimization.objective import (
    CurriculumAction,
)
from fedorbit.types import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactPath,
    ArtifactSchemaVersion,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    ClassCount,
    ClassIndex,
    ConfigurationSection,
    DatasetId,
    DirectedPair,
    DirectedPairName,
    Estimate,
    EvaluationConditionName,
    ExperimentName,
    ExperimentSeed,
    FieldDescription,
    ImplementationIdentity,
    InvalidReason,
    MetricId,
    MetricUnit,
    OverwritePolicy,
    PairSeedIneligibilityReason,
    Score,
    SemanticCell,
    SemanticCoordinateText,
    SerializedPacket,
    Sha256Digest,
    StableJsonPayload,
    StorageLayoutSegment,
    TransferMethod,
    stable_json,
)


def build_completion_manifest(
    coordinates: SemanticCoordinateText,
    fingerprint: Sha256Digest,
    payload_path: ArtifactPath,
    payload_sha256: Sha256Digest,
    configuration_sha256: Sha256Digest,
    code_sha256: Sha256Digest,
    stage: ArtifactStage = ArtifactStage.EVALUATION,
    upstream_artifact_ids: ArtifactIdentifiers = (),
    has_no_upstream_inputs: bool = False,
) -> CompletionManifest:
    return build_artifact_completion(
        coordinates,
        fingerprint,
        payload_path,
        payload_sha256,
        configuration_sha256,
        code_sha256,
        stage,
        upstream_artifact_ids=upstream_artifact_ids,
        has_no_upstream_inputs=has_no_upstream_inputs,
    )


def latest_completed_manifest(
    store: ArtifactStore, experiment: ExperimentName
) -> ReusableArtifactManifest | None:
    return store.current_completed_manifest(experiment.value)


_PRIMARY_TRANSFER_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.MODELS, ConfigurationSection.METRICS}
)


def class_metric_sets(
    score: ScoreArtifact, n_classes: ClassCount
) -> tuple[ClassF1Set, ClassRecallSet]:
    predicted = tuple(ClassIndex(row.predicted_class.value) for row in score.rows)
    actual = tuple(ClassIndex(row.target.value) for row in score.rows)
    f1_values: list[ClassF1] = []
    recall_values: list[ClassRecall] = []
    for class_index in range(n_classes):
        counts = confusion_counts(predicted, actual, ClassIndex(class_index))
        recall_values.append(
            ClassRecall(recall_from_counts(counts.true_positives, counts.false_negatives))
        )
        f1_values.append(
            ClassF1(
                f1_from_counts(
                    counts.true_positives, counts.false_positives, counts.false_negatives
                )
            )
        )
    return ClassF1Set(tuple(f1_values)), ClassRecallSet(tuple(recall_values))


def persist_primary_transfer_metric(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    pair_direction: DirectedPairName,
    directed_pair_source: DatasetId,
    directed_pair_target: DatasetId,
    method: TransferMethod,
    seed: RandomSeed,
    metric_name: MetricId,
    metric_value: Estimate | None,
    metric_unit: MetricUnit,
    direction: MetricDirection,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
    condition: EvaluationConditionName,
    valid: bool = True,
    invalid_reason: InvalidReason | None = None,
    pairing_lineage: PairingLineage | None = None,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        directed_pair=DirectedPair(source=directed_pair_source, target=directed_pair_target),
        method=method,
        condition=condition,
        seed=ExperimentSeed(seed),
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.EVALUATION,
            cell,
            relevance,
            input_artifact_ids,
            _PRIMARY_TRANSFER_CONFIGURATION_SECTIONS,
            ImplementationIdentity.SCORING_V1,
            metric_name=metric_name,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    metric = MetricRecord(
        experiment=experiment,
        pair=pair_direction,
        method=method,
        condition=condition,
        seed=seed,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        direction=direction,
        evaluation_class_set_sha256=Sha256Digest(
            hashlib.sha256(coordinates.encode("utf-8")).hexdigest()
        ),
        input_artifact_ids=tuple(input_artifact_ids),
        dependency_fingerprint_sha256=fingerprint,
        valid=valid,
        invalid_reason=invalid_reason,
    )
    validate_metric_records(MetricRecordCollection((metric,)))
    payload_path = (
        experiment_workspace(layout, experiment)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
        / (
            f"metric.{directed_pair_source.value}-{directed_pair_target.value}"
            f".{method.value}.{condition}.{seed}.{metric_name.value}.json"
        )
    )
    lineage = pairing_lineage
    if lineage is None and input_artifact_ids:
        lineage = pair_seed_pairing_lineage(
            pair_direction,
            seed,
            input_artifact_ids[0],
        )
    payload_entries: list[tuple[str, StableJsonPayload]] = [
        ("metric_record", cast(StableJsonPayload, metric.model_dump(mode="json"))),
    ]
    if lineage is not None:
        payload_entries.append(("pairing_lineage", lineage.payload()))
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_PRIMARY_TRANSFER_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(ImplementationIdentity.SCORING_V1))
    environment = environment_snapshot()
    payload_entries.append(
        (
            "semantic_cell",
            cast(
                StableJsonPayload,
                SemanticCellManifest(
                    experiment=experiment,
                    dataset=directed_pair_target,
                    source_client=directed_pair_source,
                    target_client=directed_pair_target,
                    directed_pair=pair_direction,
                    method=method,
                    condition=condition,
                    seed=ExperimentSeed(seed),
                    scientific_configuration_sha256=configuration_sha256,
                    dependency_fingerprint_sha256=fingerprint,
                    producer_stage=ArtifactStage.EVALUATION,
                    upstream_artifact_ids=tuple(input_artifact_ids),
                    relevant_code_sha256=code_sha256,
                    material_runtime_sha256=environment.fingerprint_sha256,
                    git_commit=current_code_revision().commit,
                    environment_sha256=environment.fingerprint_sha256,
                    state=ArtifactState.COMPLETED if valid else ArtifactState.INVALID,
                    state_reason=(
                        None if invalid_reason is None else FieldDescription(invalid_reason)
                    ),
                ).model_dump(mode="json"),
            ),
        )
    )
    payload = cast(StableJsonPayload, OrderedDict(payload_entries))
    payload_bytes = serialized_payload_bytes(SerializedPacket(stable_json(payload)))
    payload_digest = digest_payload_bytes(payload_bytes)
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_digest,
        configuration_sha256,
        code_sha256,
        stage=ArtifactStage.EVALUATION,
        upstream_artifact_ids=tuple(input_artifact_ids),
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(ArtifactType.PREDICTION, payload, Sha256Digest(fingerprint)),
            artifact_type=ArtifactType.PREDICTION,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.EVALUATION,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=tuple(input_artifact_ids),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_digest,
            schema_version=ArtifactSchemaVersion.V1,
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment.fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    staged = stage_bytes(payload_path, payload_bytes, store.staging_area(manifest.artifact_id))
    store.promote_artifact(
        manifest,
        completion,
        staged_payloads=(StagedPayload(staged, payload_path),),
    )
    return manifest


def persist_diagnostic_metric(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    dataset: DatasetId,
    condition: EvaluationConditionName,
    seed: RandomSeed,
    metric_name: MetricId,
    metric_value: Estimate | None,
    metric_unit: MetricUnit,
    direction: MetricDirection,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
    valid: bool = True,
    invalid_reason: InvalidReason | None = None,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        dataset=dataset,
        condition=condition,
        seed=ExperimentSeed(seed),
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.EVALUATION,
            cell,
            relevance,
            input_artifact_ids,
            _PRIMARY_TRANSFER_CONFIGURATION_SECTIONS,
            ImplementationIdentity.SCORING_V1,
            metric_name=metric_name,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    metric = DiagnosticMetricRecord(
        experiment=experiment,
        dataset=dataset,
        condition=condition,
        seed=seed,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        direction=direction,
        evaluation_class_set_sha256=Sha256Digest(
            hashlib.sha256(coordinates.encode("utf-8")).hexdigest()
        ),
        input_artifact_ids=tuple(input_artifact_ids),
        dependency_fingerprint_sha256=fingerprint,
        valid=valid,
        invalid_reason=invalid_reason,
    )
    validate_diagnostic_metric_records(DiagnosticMetricRecordCollection((metric,)))
    payload_path = (
        experiment_workspace(layout, experiment)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
        / f"diagnostic.{dataset.value}.{condition}.{seed}.{metric_name.value}.json"
    )
    payload = cast(
        StableJsonPayload, OrderedDict(diagnostic_metric_record=metric.model_dump(mode="json"))
    )
    payload_bytes = serialized_payload_bytes(SerializedPacket(stable_json(payload)))
    payload_digest = digest_payload_bytes(payload_bytes)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_PRIMARY_TRANSFER_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(ImplementationIdentity.SCORING_V1))
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_digest,
        configuration_sha256,
        code_sha256,
        stage=ArtifactStage.EVALUATION,
        upstream_artifact_ids=tuple(input_artifact_ids),
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(ArtifactType.PREDICTION, payload, Sha256Digest(fingerprint)),
            artifact_type=ArtifactType.PREDICTION,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.EVALUATION,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=tuple(input_artifact_ids),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_digest,
            schema_version=ArtifactSchemaVersion.V1,
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    staged = stage_bytes(payload_path, payload_bytes, store.staging_area(manifest.artifact_id))
    store.promote_artifact(
        manifest,
        completion,
        staged_payloads=(StagedPayload(staged, payload_path),),
    )
    return manifest


def persist_ineligible_transfer_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    pair_direction: DirectedPairName,
    directed_pair_source: DatasetId,
    directed_pair_target: DatasetId,
    method: TransferMethod,
    seed: RandomSeed,
    overwrite_policy: OverwritePolicy,
    condition: EvaluationConditionName,
    pair_seed_ineligibility_reason: PairSeedIneligibilityReason | None = None,
) -> ReusableArtifactManifest | None:
    return persist_primary_transfer_metric(
        store,
        layout,
        experiment,
        pair_direction,
        directed_pair_source,
        directed_pair_target,
        method,
        seed,
        MetricId.ABSTENTION_INDICATOR,
        None,
        MetricUnit.BOOLEAN,
        MetricDirection.DESCRIPTIVE,
        (ArtifactIdentifier("ineligible-cell"),),
        overwrite_policy,
        condition,
        valid=False,
        invalid_reason=(
            InvalidReason("INELIGIBLE/ABSTAIN")
            if pair_seed_ineligibility_reason is None
            else InvalidReason(pair_seed_ineligibility_reason.value)
        ),
    )


def persist_primary_transfer_cell_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    method: TransferMethod,
    seed: RandomSeed,
    score: ScoreArtifact,
    n_classes: ClassCount,
    input_artifact_ids: tuple[ArtifactIdentifier, ...],
    condition: EvaluationConditionName,
    pairing_lineage: PairingLineage | None = None,
) -> None:
    f1_set, recall_set = class_metric_sets(score, n_classes)
    for metric_name, metric_value, metric_unit, direction in (
        (
            MetricId.MACRO_CROSS_ENTROPY,
            float(score.macro_cross_entropy.value),
            MetricUnit.NATS,
            MetricDirection.LOWER_IS_BETTER,
        ),
        (
            MetricId.MACRO_F1,
            float(macro_f1(f1_set).value),
            MetricUnit.FRACTION,
            MetricDirection.HIGHER_IS_BETTER,
        ),
        (
            MetricId.BALANCED_ACCURACY,
            float(balanced_accuracy(recall_set).value),
            MetricUnit.FRACTION,
            MetricDirection.HIGHER_IS_BETTER,
        ),
    ):
        persist_primary_transfer_metric(
            store,
            layout,
            request.experiment,
            pair_direction,
            source,
            target,
            method,
            seed,
            metric_name,
            metric_value,
            metric_unit,
            direction,
            input_artifact_ids,
            request.overwrite_policy,
            condition,
            pairing_lineage=pairing_lineage,
        )


def persist_boundary_diagnostic_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    method: TransferMethod,
    seed: RandomSeed,
    input_artifact_ids: tuple[ArtifactIdentifier, ...],
    condition: EvaluationConditionName,
    certified_value: Score | None,
    action: CurriculumAction | None,
    confirmation_accepted: bool | None,
) -> None:
    diagnostics: list[tuple[MetricId, float, MetricUnit, MetricDirection]] = []
    if certified_value is not None:
        diagnostics.append(
            (
                MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE,
                float(certified_value),
                MetricUnit.SCORE,
                MetricDirection.DESCRIPTIVE,
            )
        )
    if action is not None:
        abstained = 1.0 if bool(np.all(action.coordinates == 0.0)) else 0.0
        diagnostics.append(
            (
                MetricId.ABSTENTION_INDICATOR,
                abstained,
                MetricUnit.BOOLEAN,
                MetricDirection.DESCRIPTIVE,
            )
        )
        zero_cap_mask: NDArray[np.bool_] = action.problem.coordinate_caps == 0.0
        null_node_count = float(np.sum(zero_cap_mask))
        diagnostics.append(
            (
                MetricId.NULL_NODE_COUNT,
                null_node_count,
                MetricUnit.COUNT,
                MetricDirection.DESCRIPTIVE,
            )
        )
        diagnostics.append(
            (
                MetricId.ORBIT_SIZE,
                float(action.problem.blocks.orbit_size),
                MetricUnit.COUNT,
                MetricDirection.DESCRIPTIVE,
            )
        )
    if confirmation_accepted is not None:
        diagnostics.append(
            (
                MetricId.PROPOSAL_ACCEPTANCE_RATE,
                1.0 if confirmation_accepted else 0.0,
                MetricUnit.FRACTION,
                MetricDirection.HIGHER_IS_BETTER,
            )
        )
    for metric_name, metric_value, metric_unit, direction in diagnostics:
        persist_primary_transfer_metric(
            store,
            layout,
            request.experiment,
            pair_direction,
            source,
            target,
            method,
            seed,
            metric_name,
            metric_value,
            metric_unit,
            direction,
            input_artifact_ids,
            request.overwrite_policy,
            condition,
        )
