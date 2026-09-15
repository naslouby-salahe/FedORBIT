from __future__ import annotations

import contextlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
import torch
from pydantic import JsonValue
from torch import nn

from fedorbit.analysis.records import (
    MetricDirection,
)
from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.config.models import DirectedPairSpec
from fedorbit.datasets.common import (
    DatasetInspectionRequest,
    FieldRole,
    file_sha256,
    inspect_dataset,
)
from fedorbit.datasets.materialization import (
    MaterializationError,
    MaterializedClient,
    transfer_concept_groups,
)
from fedorbit.experiments.catalogue import (
    CatalogueMethod,
    ExperimentExecutionRequest,
    build_catalogue,
    method_resource_manifest,
)
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.experiments.scoring import (
    build_completion_manifest,
    persist_primary_transfer_metric,
)
from fedorbit.experiments.solvers import (
    persist_synthetic_diagnostic_metric,
    registered_transfer_method,
    solver_benchmark_reference_truth,
    synthetic_solver_instance,
)
from fedorbit.experiments.synthesis import transfer_ontology_null_padding_rows
from fedorbit.experiments.synthetic import (
    CouplingGenerationError,
    CouplingInstanceRequest,
    ExactSeparatorInstanceRequest,
    MechanismGenerationError,
    UnresolvedMapWorldKind,
    UnresolvedMapWorldRequest,
    eligible_coupling_support_sizes,
    generate_coupling_instance,
    generate_exact_separator_instance,
    generate_unresolved_map_world,
)
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
    ExecutionError,
)
from fedorbit.infrastructure.environment import environment_snapshot
from fedorbit.infrastructure.manifests import (
    ReusableArtifactManifest,
    artifact_id,
)
from fedorbit.infrastructure.preparation import (
    load_or_materialize_client,
)
from fedorbit.infrastructure.provenance import (
    configuration_subset_digest,
    implementation_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.infrastructure.runtime import (
    ExecutionLogEvent,
    RandomSeed,
    current_code_revision,
    execution_logger,
)
from fedorbit.infrastructure.storage import atomic_write_json
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    experiment_workspace,
)
from fedorbit.interface import (
    ResourceKind,
    StrictResourcePolicy,
    validate_disjoint_feature_namespaces,
    validate_no_cross_client_entity_ids,
    validate_no_cross_client_timestamp_pairing,
    validate_oracle_acl_isolation,
)
from fedorbit.learning.scoring import LocalClassCount, ScoreArtifact, ScoringRequest, score_model
from fedorbit.optimization.assignment import solve_minimum_cost_assignment
from fedorbit.optimization.certificates import (
    build_rectangular_hull,
    verify_correspondence_certificate,
    verify_exactness_certificate,
)
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
    build_padded_block_structure,
    enumerate_active_image_maps,
    enumerate_block_permutations,
)
from fedorbit.optimization.diagnostics import (
    analytic_orbit_mean,
    analytic_rectangular_hull_bounds,
    fixed_action_rectangularization_gap,
)
from fedorbit.optimization.exact_qap import (
    fixed_action_worst_correspondence_qap,
    point_correspondence_commitment,
)
from fedorbit.optimization.exact_sparse import (
    fixed_action_worst_correspondence,
)
from fedorbit.optimization.objective import (
    CurriculumAction,
    build_robust_action_problem,
    evaluate_objective,
)
from fedorbit.oracle import ORACLE_METHOD
from fedorbit.types import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactName,
    ArtifactPath,
    ArtifactSchemaVersion,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    ClientRole,
    CoarseGroup,
    Coefficient,
    ConceptCount,
    ConfigurationSection,
    CouplingCompatibility,
    DatasetId,
    EvaluationConditionName,
    ExperimentName,
    ExperimentSeed,
    FieldDescription,
    ImplementationIdentity,
    Index,
    InvalidReason,
    MethodName,
    MetricId,
    MetricUnit,
    OverwritePolicy,
    ScalabilityBlockPattern,
    SemanticCell,
    SemanticCoordinates,
    SemanticCoordinateText,
    Sha256Digest,
    StableJsonPayload,
    StorageLayoutSegment,
    StrictResourceViolationError,
    SupportCount,
    SupportSize,
    TabularColumnName,
    Threshold,
    Tolerance,
    TransferMethod,
    ValidationReason,
    directed_pair_name,
    stable_json,
)


class DatasetValidationInvalidity(StrEnum):
    CLIENT_NOT_MATERIALIZED = "registered primary client is not materialized"
    PAIR_ENDPOINT_NOT_MATERIALIZED = "directed-pair endpoint client is not materialized"
    SHARED_FEATURE_NAMESPACE = "source and target feature namespaces are not disjoint"
    CROSS_CLIENT_ENTITY_ID_SHARED = "cross-client entity identifiers are shared"
    CROSS_CLIENT_TIMESTAMP_PAIRING = "cross-client timestamps enable row pairing"


@dataclass(frozen=True, slots=True)
class StrictResourceValidity:
    valid: bool
    reason: InvalidReason | None


def pair_strict_resource_validity(
    source_materialized: MaterializedClient | None,
    target_materialized: MaterializedClient | None,
) -> StrictResourceValidity:
    if source_materialized is None or target_materialized is None:
        return StrictResourceValidity(
            False, InvalidReason(DatasetValidationInvalidity.PAIR_ENDPOINT_NOT_MATERIALIZED.value)
        )
    try:
        validate_disjoint_feature_namespaces(
            frozenset(source_materialized.feature_names),
            frozenset(target_materialized.feature_names),
        )
    except StrictResourceViolationError:
        return StrictResourceValidity(
            False, InvalidReason(DatasetValidationInvalidity.SHARED_FEATURE_NAMESPACE.value)
        )
    try:
        validate_no_cross_client_entity_ids(
            _forbidden_identity_columns(source_materialized),
            _forbidden_identity_columns(target_materialized),
        )
    except StrictResourceViolationError:
        return StrictResourceValidity(
            False, InvalidReason(DatasetValidationInvalidity.CROSS_CLIENT_ENTITY_ID_SHARED.value)
        )
    try:
        validate_no_cross_client_timestamp_pairing(
            _timestamp_columns(source_materialized),
            _timestamp_columns(target_materialized),
        )
    except StrictResourceViolationError:
        return StrictResourceValidity(
            False, InvalidReason(DatasetValidationInvalidity.CROSS_CLIENT_TIMESTAMP_PAIRING.value)
        )
    return StrictResourceValidity(True, None)


type DatasetClientPairSeedUnit = Mapping[str, JsonValue]


def dataset_client_pair_seed_units(
    materialized: Mapping[DatasetId, MaterializedClient | None],
    primary_clients: tuple[DatasetId, ...],
    pairs: tuple[DirectedPairSpec, ...],
    seeds: tuple[RandomSeed, ...],
) -> tuple[DatasetClientPairSeedUnit, ...]:
    pair_validity: OrderedDict[tuple[DatasetId, DatasetId], StrictResourceValidity] = OrderedDict()
    for pair in pairs:
        pair_validity[(pair.source, pair.target)] = pair_strict_resource_validity(
            materialized.get(pair.source), materialized.get(pair.target)
        )
    units: list[DatasetClientPairSeedUnit] = []
    for client in primary_clients:
        client_materialized = materialized.get(client)
        for pair in pairs:
            is_source_endpoint = pair.source == client
            is_target_endpoint = pair.target == client
            pair_resource = pair_validity[(pair.source, pair.target)]
            unavailable: bool | None = None
            reason: InvalidReason | None = None
            if client_materialized is None:
                unavailable, reason = (
                    True,
                    InvalidReason(DatasetValidationInvalidity.CLIENT_NOT_MATERIALIZED.value),
                )
            elif is_source_endpoint or is_target_endpoint:
                unavailable, reason = (not pair_resource.valid, pair_resource.reason)
            for seed in seeds:
                units.append(
                    cast(
                        DatasetClientPairSeedUnit,
                        OrderedDict(
                            client=client.value,
                            source=pair.source.value,
                            target=pair.target.value,
                            seed=seed,
                            is_source_endpoint=is_source_endpoint,
                            is_target_endpoint=is_target_endpoint,
                            state=(
                                ArtifactState.INVALID.value
                                if unavailable
                                else ArtifactState.COMPLETED.value
                            ),
                            invalid_reason=None if reason is None else reason,
                            strict_resource_valid=pair_resource.valid,
                            shared_transfer_concept_count=(
                                None
                                if client_materialized is None
                                else len(transfer_concept_groups(client, client_materialized))
                            ),
                            feature_count=(
                                None
                                if client_materialized is None
                                else len(client_materialized.feature_names)
                            ),
                            local_class_count=(
                                None
                                if client_materialized is None
                                else client_materialized.class_manifest.class_count
                            ),
                        ),
                    )
                )
    return tuple(units)


def execute_dataset_client_and_resource_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    blocked = _chronology_block_reasons()
    if blocked:
        _persist_blocked_experiment(layout, request, blocked)
    raw_root = raw_dataset_root()
    config = active_config()
    datasets: list[StableJsonPayload] = []
    materialized: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()
    registered_clients = config.scientific.datasets.clients
    for dataset in registered_clients:
        try:
            client = load_or_materialize_client(dataset, raw_root, layout)
        except MaterializationError as error:
            datasets.append(
                cast(
                    StableJsonPayload,
                    OrderedDict(
                        dataset=dataset.value,
                        state=ArtifactState.INVALID.value,
                        reason=str(error),
                    ),
                )
            )
            continue
        materialized[dataset] = client
        datasets.append(
            cast(
                StableJsonPayload,
                OrderedDict(
                    dataset=dataset.value,
                    state=ArtifactState.COMPLETED.value,
                    local_class_count=client.class_manifest.class_count,
                    feature_count=len(client.feature_names),
                    transfer_candidates=len(transfer_concept_groups(dataset, client)),
                ),
            )
        )
    pairs: list[StableJsonPayload] = []
    for pair in config.scientific.datasets.primary_directed_pairs:
        source = materialized.get(pair.source)
        target = materialized.get(pair.target)
        if source is None or target is None:
            pairs.append(
                cast(
                    StableJsonPayload,
                    OrderedDict(
                        source=pair.source.value,
                        target=pair.target.value,
                        state=ArtifactState.INVALID.value,
                    ),
                )
            )
            continue
        source_groups = {group.concept for group in transfer_concept_groups(pair.source, source)}
        target_groups = {group.concept for group in transfer_concept_groups(pair.target, target)}
        pairs.append(
            cast(
                StableJsonPayload,
                OrderedDict(
                    source=pair.source.value,
                    target=pair.target.value,
                    state=ArtifactState.COMPLETED.value,
                    shared_transfer_concept_count=len(source_groups & target_groups),
                    transfer_ontology=transfer_ontology_null_padding_rows(
                        pair.direction,
                        source,
                        target,
                    ),
                ),
            )
        )
    primary_clients = tuple(
        dataset
        for dataset, client in registered_clients.items()
        if client.role == ClientRole.PRIMARY
    )
    units = dataset_client_pair_seed_units(
        materialized,
        primary_clients,
        config.scientific.datasets.primary_directed_pairs,
        config.scientific.randomness.confirmatory_seeds,
    )
    seed = ExperimentSeed(config.scientific.randomness.confirmatory_seeds[0])
    return persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        seed,
        lambda fingerprint: cast(
            StableJsonPayload,
            OrderedDict(
                experiment=request.experiment.value,
                dependency_fingerprint_sha256=fingerprint,
                datasets=tuple(datasets),
                primary_pairs=tuple(pairs),
                unit_count=len(units),
                units=units,
            ),
        ),
        frozenset(),
        ImplementationIdentity.VALIDATION_V1,
        ArtifactName("dataset-client-resource-validation"),
    )


def _chronology_block_reasons() -> OrderedDict[DatasetId, ValidationReason]:
    raw_root = raw_dataset_root()
    primary_datasets = tuple(
        dataset
        for dataset, client in active_config().scientific.datasets.clients.items()
        if client.role == ClientRole.PRIMARY
    )
    observations = tuple(
        inspect_dataset(DatasetInspectionRequest(dataset, raw_root)) for dataset in primary_datasets
    )
    return OrderedDict(
        (observation.dataset, observation.event_time.reason)
        for observation in observations
        if not observation.valid_for_chronological_preprocessing
    )


def _persist_blocked_experiment(
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    reasons: OrderedDict[DatasetId, ValidationReason],
) -> None:
    destination = (
        experiment_workspace(layout, request.experiment)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
    )
    payload = cast(
        StableJsonPayload,
        OrderedDict(
            experiment=request.experiment.value,
            state=ArtifactState.BLOCKED.value,
            reason="chronological preprocessing prerequisite is unsatisfied",
            blocked_datasets=reasons,
        ),
    )
    atomic_write_json(
        destination / StorageLayoutSegment.BLOCKED_JSON,
        payload,
    )


def persist_synthetic_experiment_payload(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    seed: ExperimentSeed,
    payload_builder: Callable[[Sha256Digest], StableJsonPayload],
    configuration_sections: frozenset[ConfigurationSection],
    implementation_identity: ImplementationIdentity,
    artifact_name: ArtifactName,
    condition: EvaluationConditionName | None = None,
    support: SupportSize | None = None,
) -> ReusableArtifactManifest:
    cell = SemanticCell(
        experiment=request.experiment,
        seed=seed,
        condition=condition,
        support=support,
    )
    relevance = experiment_relevance(request.experiment)
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.EVALUATION,
            cell,
            relevance,
            (),
            configuration_sections,
            implementation_identity,
        )
    )
    if request.overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    payload = payload_builder(fingerprint)
    payload_path = (
        experiment_workspace(layout, request.experiment)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
        / f"{artifact_name}.{fingerprint[:16]}.json"
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(configuration_subset_digest(configuration_sections))
    code_sha256 = Sha256Digest(implementation_fingerprint(implementation_identity))
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(ArtifactType.OTHER, payload, Sha256Digest(fingerprint)),
            artifact_type=ArtifactType.OTHER,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.EVALUATION,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=(),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version=ArtifactSchemaVersion.V1,
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    store.write_completed(manifest, completion)
    return manifest


_THEOREM_VALIDATION_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.ACTION, ConfigurationSection.GENERATORS, ConfigurationSection.SOLVERS}
)


def execute_exact_sparse_theorem_exhaustive_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    generator_config = active_config().generators.exact_separator_theorem
    solver_config = active_config().solvers.exact_sparse
    seeds = active_config().scientific.randomness.confirmatory_seeds
    instances_per_seed = generator_config.generated_instances_per_block_pattern_support_seed_cell
    summary_seed = ExperimentSeed(request.definition.seeds[0])
    instance_artifact_ids: list[ArtifactIdentifier] = []
    for pattern in generator_config.block_patterns:
        total_nodes = sum(pattern)
        groups = tuple(CoarseGroup)[: len(pattern)]
        counts = OrderedDict(zip(groups, pattern, strict=True))
        blocks = build_padded_block_structure(groups, counts, counts)
        orbit = tuple(enumerate_block_permutations(blocks))
        pattern_key = "-".join(str(size) for size in pattern)
        for support in generator_config.supports:
            if support > total_nodes:
                continue
            for seed in seeds:
                for instance_index in range(instances_per_seed):

                    def _instance_payload(
                        fingerprint: Sha256Digest,
                        cell_pattern: tuple[ConceptCount, ...] = pattern,
                        cell_support: SupportCount = support,
                        cell_seed: RandomSeed = seed,
                        cell_instance_index: Index = instance_index,
                        cell_blocks: PaddedBlockStructure = blocks,
                        cell_orbit: tuple[BlockCorrespondence, ...] = orbit,
                    ) -> StableJsonPayload:
                        cell = theorem_exhaustive_validation_instance(
                            cell_pattern,
                            cell_support,
                            cell_seed,
                            cell_instance_index,
                            cell_blocks,
                            cell_orbit,
                            solver_config.lap_objective_tie_tolerance,
                            solver_config.action_tie_tolerance,
                            solver_config.exact_validation_absolute_tolerance,
                        )
                        return cast(
                            StableJsonPayload,
                            OrderedDict(
                                experiment=request.experiment.value,
                                dependency_fingerprint_sha256=fingerprint,
                                cell=cell,
                            ),
                        )

                    manifest = persist_synthetic_experiment_payload(
                        store,
                        layout,
                        request,
                        summary_seed,
                        _instance_payload,
                        _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
                        ImplementationIdentity.VALIDATION_V1,
                        ArtifactName(
                            f"theorem-exhaustive.{pattern_key}.s{support}.seed{seed}.i{instance_index}"
                        ),
                        EvaluationConditionName(
                            f"pattern-{pattern_key}-seed{seed}-instance{instance_index}"
                        ),
                        SupportSize(support),
                    )
                    instance_artifact_ids.append(manifest.artifact_id)
    return persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        summary_seed,
        lambda fingerprint: cast(
            StableJsonPayload,
            OrderedDict(
                experiment=request.experiment.value,
                dependency_fingerprint_sha256=fingerprint,
                total_instances=len(instance_artifact_ids),
                instance_artifact_ids=[identifier.value for identifier in instance_artifact_ids],
            ),
        ),
        _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
        ImplementationIdentity.VALIDATION_V1,
        ArtifactName("theorem-exhaustive-validation"),
    )


def theorem_exhaustive_validation_instance(
    pattern: tuple[ConceptCount, ...],
    support: SupportCount,
    seed: RandomSeed,
    instance_index: Index,
    blocks: PaddedBlockStructure,
    orbit: tuple[BlockCorrespondence, ...],
    lap_objective_tie_tolerance: Tolerance,
    action_tie_tolerance: Tolerance,
    exact_validation_absolute_tolerance: Tolerance,
) -> StableJsonPayload:
    total_nodes = sum(pattern)
    instance = generate_exact_separator_instance(
        ExactSeparatorInstanceRequest(pattern, seed, support, instance_index)
    )
    problem = build_robust_action_problem(
        blocks,
        instance.lower_response_matrix,
        instance.upper_response_matrix,
        instance.target_importance / instance.target_importance.sum(),
        tuple(range(total_nodes)),
    )
    action = CurriculumAction(problem=problem, coordinates=instance.active_action)
    outcome = fixed_action_worst_correspondence(
        problem, action, lap_objective_tie_tolerance, action_tie_tolerance
    )
    exhaustive_truth = min(evaluate_objective(action, correspondence) for correspondence in orbit)
    error = abs(outcome.separator_objective - exhaustive_truth)
    exact_minima = verify_exactness_certificate(
        outcome.separator_objective, exhaustive_truth, exact_validation_absolute_tolerance
    )
    valid_certificate = verify_correspondence_certificate(
        outcome.worst_correspondence,
        outcome.separator_objective,
        action,
        exact_validation_absolute_tolerance,
    )
    return cast(
        StableJsonPayload,
        OrderedDict(
            block_pattern=list(pattern),
            support=support,
            seed=seed,
            instance_index=instance_index,
            absolute_objective_error=error,
            exact_minima=exact_minima,
            valid_certificate=valid_certificate,
        ),
    )


_COUPLING_VALIDATION_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.ACTION, ConfigurationSection.GENERATORS, ConfigurationSection.SOLVERS}
)


def execute_coupling_and_map_bound_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> ReusableArtifactManifest:
    coupling_config = active_config().generators.coupling_structure
    seeds = active_config().scientific.randomness.confirmatory_seeds
    incompatible_gap_threshold = coupling_config.incompatible_fixed_action_gap_strictly_greater_than
    summary_seed = ExperimentSeed(request.definition.seeds[0])
    logger = execution_logger()
    cell_artifact_ids: list[ArtifactIdentifier] = []
    total_generated = 0
    total_generation_failures = 0
    total_incompatible_gap_failures = 0
    started_at = time.monotonic()
    for compatibility in coupling_config.compatibility:
        eligible_supports = eligible_coupling_support_sizes(compatibility, coupling_config.supports)
        for support in eligible_supports:
            for heterogeneity in coupling_config.response_heterogeneity:
                for asymmetry in coupling_config.directed_asymmetry:
                    for sparsity in coupling_config.response_sparsity:
                        for block_pattern in coupling_config.block_patterns:
                            condition_label = (
                                f"{compatibility.value}-{heterogeneity}-{asymmetry}-"
                                f"{sparsity}-{'x'.join(str(size) for size in block_pattern)}"
                            )

                            def _cell_payload(
                                fingerprint: Sha256Digest,
                                cell_compatibility: CouplingCompatibility = compatibility,
                                cell_heterogeneity: Coefficient = heterogeneity,
                                cell_asymmetry: Coefficient = asymmetry,
                                cell_sparsity: Coefficient = sparsity,
                                cell_pattern: tuple[ConceptCount, ...] = block_pattern,
                                cell_support: SupportCount = support,
                            ) -> StableJsonPayload:
                                generated = 0
                                generation_failures = 0
                                gap_failures = 0
                                for instance_seed in seeds:
                                    coupling_outcome = _coupling_validation_instance(
                                        cell_compatibility,
                                        cell_heterogeneity,
                                        cell_asymmetry,
                                        cell_sparsity,
                                        cell_pattern,
                                        cell_support,
                                        instance_seed,
                                        incompatible_gap_threshold,
                                    )
                                    generated += 1
                                    if not coupling_outcome.produced:
                                        generation_failures += 1
                                    elif coupling_outcome.incompatible_gap_failed:
                                        gap_failures += 1
                                cell_payload = cast(
                                    StableJsonPayload,
                                    OrderedDict(
                                        compatibility=cell_compatibility.value,
                                        support=cell_support,
                                        response_heterogeneity=cell_heterogeneity,
                                        directed_asymmetry=cell_asymmetry,
                                        response_sparsity=cell_sparsity,
                                        block_pattern=list(cell_pattern),
                                        generated=generated,
                                        generation_failures=generation_failures,
                                        incompatible_gap_failures=gap_failures,
                                    ),
                                )
                                return cast(
                                    StableJsonPayload,
                                    OrderedDict(
                                        experiment=request.experiment.value,
                                        dependency_fingerprint_sha256=fingerprint,
                                        cell=cell_payload,
                                    ),
                                )

                            manifest = persist_synthetic_experiment_payload(
                                store,
                                layout,
                                request,
                                summary_seed,
                                _cell_payload,
                                _COUPLING_VALIDATION_CONFIGURATION_SECTIONS,
                                ImplementationIdentity.VALIDATION_V1,
                                ArtifactName(f"coupling-validation.{condition_label}.s{support}"),
                                EvaluationConditionName(condition_label),
                                SupportSize(support),
                            )
                            cell_artifact_ids.append(manifest.artifact_id)
                            payload = json.loads(Path(manifest.payload_paths[0]).read_text())
                            cell = payload["cell"]
                            total_generated += cast(int, cell["generated"])
                            total_generation_failures += cast(int, cell["generation_failures"])
                            total_incompatible_gap_failures += cast(
                                int, cell["incompatible_gap_failures"]
                            )
                            logger.record(
                                ExecutionLogEvent(
                                    occurred_at=datetime.now(UTC),
                                    cell_coordinates=SemanticCoordinates(condition_label),
                                    artifact_id=manifest.artifact_id,
                                    state=ArtifactState.COMPLETED,
                                    stage=ArtifactStage.EVALUATION,
                                    experiment=request.experiment,
                                    elapsed_seconds=time.monotonic() - started_at,
                                    reuse_note=FieldDescription(
                                        f"{len(cell_artifact_ids)} cells reused or executed"
                                    ),
                                )
                            )
    map_bound_results = _map_bound_fixture_results(seeds)
    return persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        summary_seed,
        lambda fingerprint: cast(
            StableJsonPayload,
            OrderedDict(
                experiment=request.experiment.value,
                dependency_fingerprint_sha256=fingerprint,
                total_instances=total_generated,
                total_generation_failures=total_generation_failures,
                total_incompatible_gap_failures=total_incompatible_gap_failures,
                cell_artifact_ids=tuple(cell_artifact_ids),
                map_bound_fixtures=map_bound_results,
            ),
        ),
        _COUPLING_VALIDATION_CONFIGURATION_SECTIONS,
        ImplementationIdentity.VALIDATION_V1,
        ArtifactName("coupling-and-map-bound-validation"),
    )


@dataclass(frozen=True, slots=True)
class CouplingValidationOutcome:
    produced: bool
    incompatible_gap_failed: bool


def _coupling_validation_instance(
    compatibility: CouplingCompatibility,
    heterogeneity: Coefficient,
    asymmetry: Coefficient,
    sparsity: Coefficient,
    block_pattern: tuple[ConceptCount, ...],
    support: SupportCount,
    seed: RandomSeed,
    incompatible_gap_threshold: Threshold,
) -> CouplingValidationOutcome:
    request = CouplingInstanceRequest(
        compatibility=compatibility,
        response_heterogeneity=heterogeneity,
        directed_asymmetry=asymmetry,
        response_sparsity=sparsity,
        block_pattern=tuple(block_pattern),
        support_size=support,
        seed=seed,
        instance_index=0,
    )
    try:
        instance = generate_coupling_instance(request)
    except CouplingGenerationError:
        return CouplingValidationOutcome(False, False)
    if compatibility != CouplingCompatibility.INCOMPATIBLE:
        return CouplingValidationOutcome(True, False)
    groups = tuple(CoarseGroup)[: len(instance.block_pattern)]
    counts = OrderedDict(zip(groups, instance.block_pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    orbit = tuple(enumerate_block_permutations(blocks))
    problem = build_robust_action_problem(
        blocks,
        instance.lower_response_matrix,
        instance.lower_response_matrix,
        instance.target_importance,
        tuple(range(sum(instance.block_pattern))),
    )
    alpha = CurriculumAction(problem=problem, coordinates=instance.active_action)
    hull = build_rectangular_hull(
        blocks, instance.lower_response_matrix, instance.lower_response_matrix
    )
    gap = fixed_action_rectangularization_gap(alpha, orbit, hull.lower_bounds)
    return CouplingValidationOutcome(True, not gap > incompatible_gap_threshold)


def _map_bound_fixture_results(seeds: tuple[RandomSeed, ...]) -> StableJsonPayload:
    zero_map_value_failures = 0
    high_map_value_failures = 0
    for seed in seeds:
        try:
            generate_unresolved_map_world(
                UnresolvedMapWorldRequest(UnresolvedMapWorldKind.COMMON_ACTION, seed)
            )
        except MechanismGenerationError:
            zero_map_value_failures += 1
        try:
            generate_unresolved_map_world(
                UnresolvedMapWorldRequest(UnresolvedMapWorldKind.MAP_DEPENDENT, seed)
            )
        except MechanismGenerationError:
            high_map_value_failures += 1
    return cast(
        StableJsonPayload,
        OrderedDict(
            zero_map_value_fixtures=len(seeds),
            zero_map_value_failures=zero_map_value_failures,
            high_map_value_fixtures=len(seeds),
            high_map_value_failures=high_map_value_failures,
        ),
    )


def _deterministic_replay_consistent(
    node_count: ConceptCount,
    pattern: ScalabilityBlockPattern,
    support: SupportCount,
    seed: RandomSeed,
) -> bool:
    instance_a = synthetic_solver_instance(node_count, pattern, support, seed)
    instance_b = synthetic_solver_instance(node_count, pattern, support, seed)
    solver_config = active_config().solvers.exact_sparse
    outcome_a = fixed_action_worst_correspondence(
        instance_a.problem,
        instance_a.action,
        solver_config.lap_objective_tie_tolerance,
        solver_config.action_tie_tolerance,
    )
    outcome_b = fixed_action_worst_correspondence(
        instance_b.problem,
        instance_b.action,
        solver_config.lap_objective_tie_tolerance,
        solver_config.action_tie_tolerance,
    )
    return (
        outcome_a.separator_objective == outcome_b.separator_objective
        and outcome_a.worst_correspondence.images == outcome_b.worst_correspondence.images
    )


def _forbidden_identity_columns(
    materialized: MaterializedClient,
) -> frozenset[TabularColumnName]:
    return frozenset(
        column
        for column, role in materialized.schema.roles.items()
        if role == FieldRole.FORBIDDEN_IDENTITY
    )


def _timestamp_columns(
    materialized: MaterializedClient,
) -> frozenset[TabularColumnName]:
    column = materialized.schema.timestamp_column
    return frozenset() if column is None else frozenset({column})


def execute_baseline_and_oracle_correctness_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    validation_seeds = (confirmatory_seeds[0], confirmatory_seeds[4])
    tractable_config = active_config().experiments.exact_sparse_solver_benchmark
    tractable_k: ConceptCount = tractable_config.synthetic_k.minimum
    tractable_pattern = tractable_config.block_patterns[0]
    tractable_support = tractable_config.supports[0]
    for seed in validation_seeds:
        replay_consistent = _deterministic_replay_consistent(
            tractable_k, tractable_pattern, tractable_support, seed
        )
        persist_synthetic_diagnostic_metric(
            store,
            layout,
            request.experiment,
            EvaluationConditionName(f"deterministic-replay-k{tractable_k}"),
            seed,
            MetricId.DETERMINISTIC_REPLAY_CONSISTENCY,
            1.0 if replay_consistent else 0.0,
            MetricUnit.BOOLEAN,
            MetricDirection.HIGHER_IS_BETTER,
            (ArtifactIdentifier("synthetic-generator"),),
            request.overwrite_policy,
        )
        tractable_instance = synthetic_solver_instance(
            tractable_k, tractable_pattern, tractable_support, seed
        )
        exhaustive_truth = solver_benchmark_reference_truth(
            tractable_instance.blocks,
            tractable_instance.action,
            tractable_config.exhaustive_truth_correspondence_count_maximum,
        )
        if exhaustive_truth is not None:
            qap_result = fixed_action_worst_correspondence_qap(
                tractable_instance.problem, tractable_instance.action
            )
            if qap_result.certified and qap_result.objective_value is not None:
                persist_synthetic_diagnostic_metric(
                    store,
                    layout,
                    request.experiment,
                    EvaluationConditionName(f"generic-qap-vs-exhaustive-truth-k{tractable_k}"),
                    seed,
                    MetricId.ABSOLUTE_OBJECTIVE_ERROR,
                    float(abs(qap_result.objective_value - exhaustive_truth)),
                    MetricUnit.SCORE,
                    MetricDirection.LOWER_IS_BETTER,
                    (ArtifactIdentifier("synthetic-generator"),),
                    request.overwrite_policy,
                )
            source_matrix = tractable_instance.problem.lower_response_matrix
            pc_result = point_correspondence_commitment(
                source_matrix, source_matrix, tractable_instance.blocks
            )
            if (
                pc_result.certified
                and pc_result.correspondence is not None
                and pc_result.objective_value is not None
            ):
                pc_truth = min(
                    -float(
                        np.sum(
                            correspondence.permute_response_matrix(source_matrix) * source_matrix
                        )
                    )
                    for correspondence in enumerate_block_permutations(tractable_instance.blocks)
                )
                persist_synthetic_diagnostic_metric(
                    store,
                    layout,
                    request.experiment,
                    EvaluationConditionName(f"point-map-qap-correctness-k{tractable_k}"),
                    seed,
                    MetricId.ABSOLUTE_OBJECTIVE_ERROR,
                    float(abs(pc_result.objective_value - pc_truth)),
                    MetricUnit.SCORE,
                    MetricDirection.LOWER_IS_BETTER,
                    (ArtifactIdentifier("synthetic-generator"),),
                    request.overwrite_policy,
                )
            exhaustive_hull = build_rectangular_hull(
                tractable_instance.blocks, source_matrix, source_matrix
            )
            analytic_lower, analytic_upper = analytic_rectangular_hull_bounds(
                tractable_instance.blocks, source_matrix, source_matrix
            )
            hull_max_error = float(
                max(
                    np.max(np.abs(exhaustive_hull.lower_bounds - analytic_lower)),
                    np.max(np.abs(exhaustive_hull.upper_bounds - analytic_upper)),
                )
            )
            persist_synthetic_diagnostic_metric(
                store,
                layout,
                request.experiment,
                EvaluationConditionName(f"rectangular-baseline-vs-analytical-k{tractable_k}"),
                seed,
                MetricId.ABSOLUTE_OBJECTIVE_ERROR,
                hull_max_error,
                MetricUnit.SCORE,
                MetricDirection.LOWER_IS_BETTER,
                (ArtifactIdentifier("synthetic-generator"),),
                request.overwrite_policy,
            )
    persist_real_pair_baseline_validation(store, layout, request, validation_seeds)


class BaselineValidationUnavailability(StrEnum):
    PAIR_ENDPOINT_NOT_MATERIALIZED = "directed-pair endpoint client is not materialized"
    METHOD_RESOURCE_OUTSIDE_STRICT_WHITELIST = (
        "registered method resource is outside the strict whitelist"
    )


def _resource_permitted_for_either_role(
    policy: StrictResourcePolicy,
    resource: ResourceKind,
) -> bool:
    for role in (ClientRole.SOURCE, ClientRole.TARGET):
        try:
            policy.assert_role_allowed(role, resource, transfer_finalized=True)
        except StrictResourceViolationError:
            continue
        return True
    return False


def registered_method_resource_validity(
    method: TransferMethod,
) -> StrictResourceValidity:
    policy = StrictResourcePolicy()
    for resource in sorted(method_resource_manifest(method), key=lambda kind: kind.value):
        if not _resource_permitted_for_either_role(policy, resource):
            return StrictResourceValidity(
                False,
                InvalidReason(
                    f"{BaselineValidationUnavailability.METHOD_RESOURCE_OUTSIDE_STRICT_WHITELIST.value}"
                    f": {resource.value}"
                ),
            )
    return StrictResourceValidity(True, None)


def oracle_validation_context(
    experiment: ExperimentName,
    registered_methods: tuple[CatalogueMethod, ...],
) -> bool:
    return (
        experiment is ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION
        and ORACLE_METHOD in registered_methods
    )


def oracle_information_accessed(method: MethodName) -> bool:
    return method == ORACLE_METHOD


def persist_real_pair_baseline_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    validation_seeds: tuple[RandomSeed, ...],
) -> None:
    raw_root = raw_dataset_root()
    registered_methods = request.definition.methods
    oracle_context = oracle_validation_context(request.experiment, registered_methods)
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    for directed_pair in active_config().scientific.datasets.primary_directed_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        pair_direction = directed_pair_name(source, target)
        pair_resource = pair_strict_resource_validity(source_materialized, target_materialized)
        endpoint_available = source_materialized is not None and target_materialized is not None
        for seed in validation_seeds:
            for method_name in registered_methods:
                if not isinstance(method_name, TransferMethod):
                    raise ExecutionError(
                        f"registered baseline method is not a transfer method: {method_name}"
                    )
                method = registered_transfer_method(method_name)
                method_resource = registered_method_resource_validity(method)
                try:
                    validate_oracle_acl_isolation(
                        oracle_context, oracle_information_accessed(method)
                    )
                except StrictResourceViolationError:
                    acl_valid = False
                else:
                    acl_valid = True
                unavailable_reason: InvalidReason | None = None
                if not endpoint_available:
                    unavailable_reason = InvalidReason(
                        BaselineValidationUnavailability.PAIR_ENDPOINT_NOT_MATERIALIZED.value
                    )
                elif not method_resource.valid:
                    unavailable_reason = method_resource.reason
                persist_primary_transfer_metric(
                    store,
                    layout,
                    request.experiment,
                    pair_direction,
                    source,
                    target,
                    method,
                    seed,
                    MetricId.STRICT_RESOURCE_VALIDITY,
                    (
                        None
                        if unavailable_reason is not None
                        else (1.0 if pair_resource.valid and acl_valid else 0.0)
                    ),
                    MetricUnit.BOOLEAN,
                    MetricDirection.HIGHER_IS_BETTER,
                    (ArtifactIdentifier("materialized-client-schema"),),
                    request.overwrite_policy,
                    valid=unavailable_reason is None,
                    invalid_reason=unavailable_reason,
                )


class PrimitiveValidationError(ValueError):
    pass


_EXPERIMENT = ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION
_STAGE = ArtifactStage.EVALUATION
_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.ACTION, ConfigurationSection.GENERATORS, ConfigurationSection.METRICS}
)


def execute_primitive_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    overwrite_policy: OverwritePolicy = OverwritePolicy.REUSE,
) -> ReusableArtifactManifest:
    configuration = active_config()
    definition = build_catalogue().definition(_EXPERIMENT)
    hand_fixture_seed = (
        configuration.experiments.mathematical_primitive_validation.hand_fixture_seed
    )
    property_check_seeds = configuration.scientific.randomness.confirmatory_seeds
    seed = ExperimentSeed(hand_fixture_seed)
    cell = SemanticCell(experiment=_EXPERIMENT, seed=seed)
    relevance = experiment_relevance(_EXPERIMENT)
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            _STAGE,
            cell,
            relevance,
            (),
            _CONFIGURATION_SECTIONS,
            ImplementationIdentity.VALIDATION_V1,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    payload_path = _payload_path(layout, fingerprint)
    block_patterns = configuration.generators.exact_separator_theorem.block_patterns
    cells = tuple(
        _validation_payload(block_pattern, hand_fixture_seed, PrimitiveFixtureKind.HAND_FIXTURE)
        for block_pattern in block_patterns
    ) + tuple(
        _validation_payload(block_pattern, seed, PrimitiveFixtureKind.PROPERTY_CHECK)
        for block_pattern in block_patterns
        for seed in property_check_seeds
    )
    payload = cast(
        StableJsonPayload,
        OrderedDict(
            registered_planned_cells=definition.derived_planned_cells,
            registered_seeds=list(definition.seeds),
            hand_fixture_seed=hand_fixture_seed,
            property_check_seeds=list(property_check_seeds),
            realized_cell_count=len(cells),
            cells=list(cells),
        ),
    )
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(configuration_subset_digest(_CONFIGURATION_SECTIONS))
    code_sha256 = Sha256Digest(implementation_fingerprint(ImplementationIdentity.VALIDATION_V1))
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(ArtifactType.OTHER, payload, Sha256Digest(fingerprint)),
            artifact_type=ArtifactType.OTHER,
            semantic_producer_coordinates=coordinates,
            producer_stage=_STAGE,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=(),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version=ArtifactSchemaVersion.V1,
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    store.write_completed(manifest, completion)
    return manifest


def _payload_path(layout: WorkspaceLayout, fingerprint: Sha256Digest) -> Path:
    return (
        experiment_workspace(layout, _EXPERIMENT)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
        / f"primitive-validation.{fingerprint[:16]}.json"
    )


class PrimitiveFixtureKind(StrEnum):
    HAND_FIXTURE = "hand fixture"
    PROPERTY_CHECK = "property check"


def _validation_payload(
    block_pattern: tuple[ConceptCount, ...],
    source_seed: RandomSeed,
    fixture_kind: PrimitiveFixtureKind,
) -> StableJsonPayload:
    configuration = active_config()
    fixture_tolerance = (
        configuration.experiments.mathematical_primitive_validation.fixture_error_tolerance
    )
    instance = generate_exact_separator_instance(
        ExactSeparatorInstanceRequest(block_pattern, source_seed)
    )
    if instance.lower_response_matrix.shape != instance.upper_response_matrix.shape:
        raise PrimitiveValidationError("response bounds have different shapes")
    if not np.all(instance.lower_response_matrix <= instance.upper_response_matrix):
        raise PrimitiveValidationError("response lower bounds exceed upper bounds")
    assignment = solve_minimum_cost_assignment(
        instance.lower_response_matrix,
        configuration.solvers.exact_sparse.lap_objective_tie_tolerance,
    )
    if len(set(assignment.column_for_row)) != len(assignment.column_for_row):
        raise PrimitiveValidationError("assignment is not bijective")
    if not np.isfinite(assignment.objective_value):
        raise PrimitiveValidationError("assignment objective is not finite")
    groups = tuple(CoarseGroup)[: len(block_pattern)]
    equal_counts = OrderedDict(zip(groups, block_pattern, strict=True))
    equal_blocks = build_padded_block_structure(groups, equal_counts, equal_counts)
    padding_target_counts = OrderedDict((group, count + 1) for group, count in equal_counts.items())
    padded_blocks = build_padded_block_structure(groups, equal_counts, padding_target_counts)
    null_padded = any(
        padded > source
        for padded, source in zip(
            padded_blocks.padded_size_tuple, padded_blocks.source_real_counts, strict=True
        )
    )
    orbit_mean = analytic_orbit_mean(equal_blocks, instance.lower_response_matrix)
    if not np.all(np.isfinite(orbit_mean)):
        raise PrimitiveValidationError("orbit mean is not finite")
    lower_hull, upper_hull = analytic_rectangular_hull_bounds(
        equal_blocks, instance.lower_response_matrix, instance.upper_response_matrix
    )
    if not np.all(lower_hull <= upper_hull):
        raise PrimitiveValidationError("rectangular hull lower bound exceeds upper bound")
    serialized_assignment = stable_json(
        cast(
            StableJsonPayload,
            OrderedDict(
                column_for_row=list(assignment.column_for_row),
                objective=float(assignment.objective_value),
            ),
        )
    )
    deserialized = json.loads(serialized_assignment)
    if deserialized["column_for_row"] != list(assignment.column_for_row):
        raise PrimitiveValidationError("assignment serialization round-trip failed")
    total_nodes = sum(block_pattern)
    orbit = tuple(enumerate_block_permutations(equal_blocks))
    if len(orbit) != equal_blocks.orbit_size:
        raise PrimitiveValidationError("orbit enumeration length disagrees with orbit size")
    active_maps = tuple(enumerate_active_image_maps(equal_blocks, tuple(range(total_nodes))))
    if not active_maps:
        raise PrimitiveValidationError("active-image enumeration is empty")
    problem = build_robust_action_problem(
        equal_blocks,
        instance.lower_response_matrix,
        instance.upper_response_matrix,
        instance.target_importance / instance.target_importance.sum(),
        tuple(range(total_nodes)),
    )
    action = CurriculumAction(problem=problem, coordinates=instance.active_action)
    outcome = fixed_action_worst_correspondence(
        problem,
        action,
        configuration.solvers.exact_sparse.lap_objective_tie_tolerance,
        configuration.solvers.exact_sparse.action_tie_tolerance,
    )
    exhaustive_truth = min(evaluate_objective(action, correspondence) for correspondence in orbit)
    fixture_error = abs(outcome.separator_objective - exhaustive_truth)
    if fixture_error > fixture_tolerance:
        raise PrimitiveValidationError("fixture error exceeds registered tolerance")
    scoring = _score_deterministic_validation_batch()
    return cast(
        StableJsonPayload,
        OrderedDict(
            cell_kind=fixture_kind.value,
            state=ArtifactState.COMPLETED.value,
            invalid_reason=None,
            block_pattern=list(block_pattern),
            seed=source_seed,
            response_shape=list(instance.lower_response_matrix.shape),
            lower_bound_not_above_upper_bound=True,
            assignment_is_bijective=True,
            assignment_objective=assignment.objective_value,
            scoring_row_count=len(scoring.rows),
            scoring_macro_cross_entropy=scoring.macro_cross_entropy.value,
            null_padding_present=null_padded,
            orbit_mean_finite=True,
            rectangular_hull_ordered=True,
            assignment_serialization_round_trip=True,
            fixture_error=fixture_error,
            fixture_error_within_tolerance=True,
            orbit_size=equal_blocks.orbit_size,
            active_image_map_count=len(active_maps),
        ),
    )


def _score_deterministic_validation_batch() -> ScoreArtifact:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(((1.0, -1.0), (-1.0, 1.0))))
    return score_model(
        ScoringRequest(
            model=model,
            features=torch.tensor(((1.0, 0.0), (0.0, 1.0))),
            targets=torch.tensor((0, 1)),
            local_class_count=LocalClassCount(2),
        )
    )
