from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import cast

from fedorbit.analysis.records import (
    MetricDirection,
    MetricRecord,
    MetricRecordCollection,
    validate_metric_records,
)
from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.datasets.common import (
    file_sha256,
)
from fedorbit.datasets.materialization import (
    MaterializationError,
)
from fedorbit.experiments.catalogue import ExperimentExecutionRequest
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.experiments.scoring import (
    assemble_principal_action,
    build_completion_manifest,
    solve_fedorbit_exact_sparse_action,
)
from fedorbit.experiments.synthetic import (
    CouplingGenerationError,
    CouplingInstance,
    CouplingInstanceRequest,
    MechanismGenerationError,
    ScalabilityGenerationError,
    ScalabilityInstanceRequest,
    UnresolvedMapWorld,
    UnresolvedMapWorldKind,
    UnresolvedMapWorldRequest,
    eligible_coupling_support_sizes,
    generate_coupling_instance,
    generate_scalability_instance,
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
    EfficiencyMeasurement,
    RandomSeed,
    SeedDerivationRequest,
    current_code_revision,
    derive_seed32,
    execution_device,
    execution_logger,
    measure_efficiency,
)
from fedorbit.infrastructure.storage import atomic_write_json
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
    experiment_workspace,
)
from fedorbit.methods.baselines import (
    coupling_destroyed_matrices,
    optimize_against_fixed_matrix,
)
from fedorbit.optimization.certificates import (
    RectangularHull,
    build_rectangular_hull,
    rectangular_value_over_candidates,
    robust_coupling_gap,
    verify_exactness_certificate,
)
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    BlockNodeCounts,
    PaddedBlockStructure,
    active_image_assignment_count,
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.optimization.dense_ccp import solve_dense_ccp
from fedorbit.optimization.diagnostics import (
    fixed_action_rectangularization_gap,
)
from fedorbit.optimization.exact_qap import (
    fixed_action_worst_correspondence_qap,
)
from fedorbit.optimization.exact_sparse import (
    fixed_action_worst_correspondence,
    solve_robust_action,
)
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    build_robust_action_problem,
    evaluate_objective,
    zero_action,
)
from fedorbit.types import (
    SYNTHETIC_DIRECTED_PAIR,
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactName,
    ArtifactPath,
    ArtifactSchemaVersion,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    CoarseGroup,
    ConceptCount,
    ConfigurationSection,
    ContrastCoordinates,
    Estimate,
    EvaluationCondition,
    EvaluationConditionKind,
    EvaluationConditionName,
    ExecutionEventName,
    ExperimentLocalMethod,
    ExperimentName,
    ExperimentSeed,
    ImplementationIdentity,
    Index,
    InvalidReason,
    MethodName,
    MetricId,
    MetricUnit,
    OverwritePolicy,
    ReplicateCount,
    RngNamespace,
    ScalabilityBlockPattern,
    Score,
    SemanticCell,
    SemanticCoordinates,
    SemanticCoordinateText,
    Sha256Digest,
    StableJsonPayload,
    StepCount,
    StorageLayoutSegment,
    SupportCount,
    SupportSize,
    TerminalState,
    Tolerance,
    TransferMethod,
)

_THEOREM_VALIDATION_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.ACTION, ConfigurationSection.GENERATORS, ConfigurationSection.SOLVERS}
)


def persist_synthetic_benchmark_metric(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportSize | None,
    method: TransferMethod,
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
        method=method,
        condition=condition,
        support=support,
        seed=ExperimentSeed(seed),
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.EVALUATION,
            cell,
            relevance,
            input_artifact_ids,
            _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
            ImplementationIdentity.SOLVERS_V1,
            metric_name=metric_name,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    metric = MetricRecord(
        experiment=experiment,
        pair=SYNTHETIC_DIRECTED_PAIR,
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
    support_label = "uncapped" if support is None else str(support.value)
    payload_path = (
        experiment_workspace(layout, experiment)
        / StorageLayoutSegment.ARTIFACTS
        / StorageLayoutSegment.DERIVED
        / (
            f"metric.{condition}.support-{support_label}.{method.value}."
            f"{seed}.{metric_name.value}.json"
        )
    )
    payload = cast(StableJsonPayload, OrderedDict(metric_record=metric.model_dump(mode="json")))
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_THEOREM_VALIDATION_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(ImplementationIdentity.SOLVERS_V1))
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
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


@dataclass(frozen=True, slots=True)
class ScalabilitySolverInstance:
    problem: RobustActionProblem
    action: CurriculumAction
    blocks: PaddedBlockStructure


@dataclass(frozen=True, slots=True)
class MeasuredKernel[ResultT]:
    result: ResultT
    measurement: EfficiencyMeasurement


@dataclass(frozen=True, slots=True)
class ScalabilityCellCoordinates:
    k: ConceptCount
    block_pattern: ScalabilityBlockPattern


def registered_transfer_method(method: MethodName) -> TransferMethod:
    if not isinstance(method, TransferMethod):
        raise ExecutionError(f"registered method is not a transfer method: {method}")
    return method


def persist_unavailable_registered_methods(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportSize | None,
    seed: RandomSeed,
    methods: tuple[MethodName, ...],
    invalid_reason: InvalidReason,
    overwrite_policy: OverwritePolicy,
) -> None:
    for method in methods:
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            support,
            registered_transfer_method(method),
            seed,
            MetricId.CORRESPONDENCE_CERTIFICATE_VALIDITY,
            None,
            MetricUnit.BOOLEAN,
            MetricDirection.DESCRIPTIVE,
            (ArtifactIdentifier("unavailable-synthetic-cell"),),
            overwrite_policy,
            valid=False,
            invalid_reason=invalid_reason,
        )


def measure_registered_kernel[ResultT](
    operation: Callable[[], ResultT],
) -> MeasuredKernel[ResultT]:
    runtime = active_config().runtime
    for _ in range(runtime.deterministic_kernel_warmups):
        operation()
    results: list[ResultT] = []
    measurements: list[EfficiencyMeasurement] = []
    for _ in range(runtime.deterministic_kernel_timed_repetitions):
        with measure_efficiency() as efficiency:
            results.append(operation())
        measurements.append(efficiency.result)
    combined = EfficiencyMeasurement(
        wall_time_seconds=median(measurement.wall_time_seconds for measurement in measurements),
        peak_host_rss_mib=median(measurement.peak_host_rss_mib for measurement in measurements),
        peak_cuda_allocated_bytes=int(
            median(measurement.peak_cuda_allocated_bytes for measurement in measurements)
        ),
    )
    return MeasuredKernel(results[-1], combined)


def scalability_predicted_work(
    blocks: PaddedBlockStructure,
    action: CurriculumAction,
) -> float:
    per_block = [0] * len(blocks.padded_size_tuple)
    for node in action.active_support_nodes:
        per_block[blocks.block_of_node(node)] += 1
    assignment_count = active_image_assignment_count(
        blocks, BlockNodeCounts(blocks, tuple(per_block))
    )
    return float(assignment_count * sum(size**3 for size in blocks.padded_size_tuple))


def synthetic_solver_instance(
    node_count: ConceptCount,
    block_pattern: ScalabilityBlockPattern,
    support: SupportCount,
    seed: RandomSeed,
    actionable_node_count: ConceptCount | None = None,
) -> ScalabilitySolverInstance:
    instance = generate_scalability_instance(
        ScalabilityInstanceRequest(node_count, block_pattern, support, seed)
    )
    groups = tuple(CoarseGroup)[:2]
    counts = OrderedDict(zip(groups, instance.block_pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    actionable = support if actionable_node_count is None else actionable_node_count
    problem = build_robust_action_problem(
        blocks,
        instance.lower_response_matrix,
        instance.lower_response_matrix,
        instance.target_importance,
        tuple(range(actionable)),
    )
    action = CurriculumAction(problem=problem, coordinates=instance.fixed_action)
    return ScalabilitySolverInstance(problem, action, blocks)


def solver_benchmark_reference_truth(
    blocks: PaddedBlockStructure,
    action: CurriculumAction,
    exhaustive_truth_correspondence_count_maximum: StepCount,
) -> Score | None:
    if blocks.orbit_size > exhaustive_truth_correspondence_count_maximum:
        return None
    return min(
        evaluate_objective(action, correspondence)
        for correspondence in enumerate_block_permutations(blocks)
    )


def _persist_solver_benchmark_error_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportSize | None,
    method: TransferMethod,
    seed: RandomSeed,
    reference_truth: Score | None,
    reported_objective: Score,
    exact_validation_absolute_tolerance: Tolerance,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> None:
    if reference_truth is None:
        return
    absolute_error = abs(reported_objective - reference_truth)
    persist_synthetic_benchmark_metric(
        store,
        layout,
        experiment,
        condition,
        support,
        method,
        seed,
        MetricId.ABSOLUTE_OBJECTIVE_ERROR,
        float(absolute_error),
        MetricUnit.SCORE,
        MetricDirection.LOWER_IS_BETTER,
        input_artifact_ids,
        overwrite_policy,
    )
    relative_error = absolute_error / abs(reference_truth) if reference_truth != 0.0 else 0.0
    persist_synthetic_benchmark_metric(
        store,
        layout,
        experiment,
        condition,
        support,
        method,
        seed,
        MetricId.RELATIVE_OBJECTIVE_ERROR,
        float(relative_error),
        MetricUnit.FRACTION,
        MetricDirection.LOWER_IS_BETTER,
        input_artifact_ids,
        overwrite_policy,
    )
    persist_synthetic_benchmark_metric(
        store,
        layout,
        experiment,
        condition,
        support,
        method,
        seed,
        MetricId.CORRESPONDENCE_CERTIFICATE_VALIDITY,
        1.0
        if verify_exactness_certificate(
            reported_objective, reference_truth, exact_validation_absolute_tolerance
        )
        else 0.0,
        MetricUnit.BOOLEAN,
        MetricDirection.HIGHER_IS_BETTER,
        input_artifact_ids,
        overwrite_policy,
    )


def _persist_solver_benchmark_efficiency_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportSize | None,
    method: TransferMethod,
    seed: RandomSeed,
    measurement: EfficiencyMeasurement,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> None:
    for metric_name, metric_value, metric_unit in (
        (
            MetricId.WALL_TIME,
            measurement.wall_time_seconds,
            MetricUnit.SECONDS,
        ),
        (
            MetricId.PEAK_HOST_RSS,
            measurement.peak_host_rss_mib,
            MetricUnit.MEBIBYTES,
        ),
        (
            MetricId.PEAK_CUDA_ALLOCATED_BYTES,
            float(measurement.peak_cuda_allocated_bytes),
            MetricUnit.BYTES,
        ),
    ):
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            support,
            method,
            seed,
            metric_name,
            float(metric_value),
            metric_unit,
            MetricDirection.LOWER_IS_BETTER,
            input_artifact_ids,
            overwrite_policy,
        )


def score_synthetic_solver_benchmark_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportSize | None,
    seed: RandomSeed,
    problem: RobustActionProblem,
    action: CurriculumAction,
    reference_truth: Score | None,
    methods: tuple[MethodName, ...],
    overwrite_policy: OverwritePolicy,
    repeated_timing_methods: frozenset[TransferMethod] = frozenset(),
) -> None:
    input_artifact_ids = (ArtifactIdentifier("synthetic-generator"),)
    solver_config = active_config().solvers.exact_sparse
    if TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER in methods:
        if TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER in repeated_timing_methods:
            measured = measure_registered_kernel(
                lambda: fixed_action_worst_correspondence(
                    problem,
                    action,
                    solver_config.lap_objective_tie_tolerance,
                    solver_config.action_tie_tolerance,
                )
            )
            outcome = measured.result
            measurement = measured.measurement
        else:
            with measure_efficiency() as efficiency:
                outcome = fixed_action_worst_correspondence(
                    problem,
                    action,
                    solver_config.lap_objective_tie_tolerance,
                    solver_config.action_tie_tolerance,
                )
                measurement = efficiency.result
        _persist_solver_benchmark_efficiency_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            measurement,
            input_artifact_ids,
            overwrite_policy,
        )
        _persist_solver_benchmark_error_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            reference_truth,
            outcome.separator_objective,
            solver_config.exact_validation_absolute_tolerance,
            input_artifact_ids,
            overwrite_policy,
        )
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            MetricId.ACTIVE_IMAGE_CANDIDATES,
            float(outcome.active_image_candidates),
            MetricUnit.COUNT,
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
        )
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            MetricId.LAP_CALLS,
            float(outcome.lap_calls),
            MetricUnit.COUNT,
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
        )
    if TransferMethod.GENERIC_EXACT_QAP in methods:
        with measure_efficiency() as efficiency:
            qap_result = fixed_action_worst_correspondence_qap(problem, action)
        _persist_solver_benchmark_efficiency_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.GENERIC_EXACT_QAP,
            seed,
            efficiency.result,
            input_artifact_ids,
            overwrite_policy,
        )
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.GENERIC_EXACT_QAP,
            seed,
            MetricId.TIMEOUT_INDICATOR,
            1.0 if qap_result.terminal_state == TerminalState.TIME_LIMIT else 0.0,
            MetricUnit.BOOLEAN,
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
        )
        if qap_result.certified and qap_result.objective_value is not None:
            _persist_solver_benchmark_error_metrics(
                store,
                layout,
                experiment,
                condition,
                support,
                TransferMethod.GENERIC_EXACT_QAP,
                seed,
                reference_truth,
                qap_result.objective_value,
                solver_config.exact_validation_absolute_tolerance,
                input_artifact_ids,
                overwrite_policy,
            )
    if TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK in methods:
        with measure_efficiency() as efficiency:
            dense_outcome = solve_dense_ccp(
                problem,
                seed,
                SemanticCoordinates(f"{experiment.value}:{condition}:{support}:{seed}"),
            )
        _persist_solver_benchmark_efficiency_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
            seed,
            efficiency.result,
            input_artifact_ids,
            overwrite_policy,
        )
        for metric_name, metric_value in (
            (MetricId.DENSE_RELAXATION_BOUND, dense_outcome.relaxation_lower_bound),
            (MetricId.DENSE_PROJECTED_OBJECTIVE, dense_outcome.best_projected_response_objective),
            (MetricId.DENSE_BOUND_GAP, dense_outcome.dense_bound_gap),
            (MetricId.DENSE_INTEGRALITY_RESIDUAL, dense_outcome.integrality_residual),
        ):
            persist_synthetic_benchmark_metric(
                store,
                layout,
                experiment,
                condition,
                support,
                TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
                seed,
                metric_name,
                float(metric_value),
                MetricUnit.SCORE,
                MetricDirection.DESCRIPTIVE,
                input_artifact_ids,
                overwrite_policy,
            )
        _persist_solver_benchmark_error_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
            seed,
            reference_truth,
            dense_outcome.best_projected_response_objective,
            solver_config.exact_validation_absolute_tolerance,
            input_artifact_ids,
            overwrite_policy,
        )


def execute_exact_sparse_solver_benchmark(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.exact_sparse_solver_benchmark
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    k_values: tuple[ConceptCount, ...] = tuple(
        range(config.synthetic_k.minimum, config.synthetic_k.maximum + 1)
    )
    for node_count in k_values:
        for pattern in config.block_patterns:
            for support in config.supports:
                condition = scalability_condition_name(node_count, pattern)
                for seed in confirmatory_seeds:
                    try:
                        instance = synthetic_solver_instance(node_count, pattern, support, seed)
                    except ScalabilityGenerationError as error:
                        persist_unavailable_registered_methods(
                            store,
                            layout,
                            request.experiment,
                            condition,
                            SupportSize(support),
                            seed,
                            config.methods,
                            InvalidReason(str(error)),
                            request.overwrite_policy,
                        )
                        continue
                    reference_truth = solver_benchmark_reference_truth(
                        instance.blocks,
                        instance.action,
                        config.exhaustive_truth_correspondence_count_maximum,
                    )
                    score_synthetic_solver_benchmark_cell(
                        store,
                        layout,
                        request.experiment,
                        condition,
                        SupportSize(support),
                        seed,
                        instance.problem,
                        instance.action,
                        reference_truth,
                        config.methods,
                        request.overwrite_policy,
                    )


def execute_scalability_and_efficiency(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.scalability_and_efficiency
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    exact_methods = (
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.GENERIC_EXACT_QAP,
    )
    dense_methods = (TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,)
    logger = execution_logger()
    for node_count in config.k_values:
        for pattern in config.block_patterns:
            for support in config.exact_qap_supports:
                condition = scalability_condition_name(node_count, pattern)
                for seed in confirmatory_seeds:
                    try:
                        instance = synthetic_solver_instance(node_count, pattern, support, seed)
                    except ScalabilityGenerationError:
                        continue
                    work = scalability_predicted_work(instance.blocks, instance.action)
                    logger.event(
                        ExecutionEventName.SOLVER_CELL_START,
                        experiment=request.experiment.value,
                        k=node_count,
                        pattern=pattern.value,
                        support=support,
                        seed=seed,
                        predicted_work=work,
                    )
                    persist_synthetic_diagnostic_metric(
                        store,
                        layout,
                        request.experiment,
                        condition,
                        seed,
                        MetricId.PREDICTED_WORK_COORDINATE,
                        work,
                        MetricUnit.COUNT,
                        MetricDirection.DESCRIPTIVE,
                        (ArtifactIdentifier("synthetic-generator"),),
                        request.overwrite_policy,
                    )
                    score_synthetic_solver_benchmark_cell(
                        store,
                        layout,
                        request.experiment,
                        condition,
                        SupportSize(support),
                        seed,
                        instance.problem,
                        instance.action,
                        None,
                        exact_methods,
                        request.overwrite_policy,
                        REPEATED_TIMING_METHODS,
                    )
            dense_condition = EvaluationConditionName(f"k{node_count}-{pattern.value}-dense")
            dense_support = config.exact_qap_supports[0]
            for seed in confirmatory_seeds:
                try:
                    dense_instance = synthetic_solver_instance(
                        node_count, pattern, dense_support, seed, node_count
                    )
                except ScalabilityGenerationError:
                    continue
                score_synthetic_solver_benchmark_cell(
                    store,
                    layout,
                    request.experiment,
                    dense_condition,
                    None,
                    seed,
                    dense_instance.problem,
                    dense_instance.action,
                    None,
                    dense_methods,
                    request.overwrite_policy,
                )
    raw_root = raw_dataset_root()
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    real_methods = active_config().experiments.scalability_and_efficiency.real_timing_methods
    device = execution_device()
    principal_support = active_config().scientific.action.principal_sparse_support
    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        try:
            source_materialized = load_or_materialize_client(source, raw_root, layout)
            target_materialized = load_or_materialize_client(target, raw_root, layout)
        except MaterializationError:
            continue
        for seed in confirmatory_seeds:
            assembly = assemble_principal_action(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
                solve_fedorbit_exact_sparse_action,
            )
            if assembly is None:
                continue
            condition = EvaluationConditionName(f"real-{source.value}-to-{target.value}")
            logger.event(
                ExecutionEventName.REAL_TIMING_CELL,
                experiment=request.experiment.value,
                source=source.value,
                target=target.value,
                seed=seed,
            )
            supported_methods = tuple(
                method
                for method in real_methods
                if method is not TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK
            )
            if supported_methods:
                score_synthetic_solver_benchmark_cell(
                    store,
                    layout,
                    request.experiment,
                    condition,
                    SupportSize(principal_support),
                    seed,
                    assembly.problem,
                    assembly.action,
                    None,
                    supported_methods,
                    request.overwrite_policy,
                    REPEATED_TIMING_METHODS,
                )
            if TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK in real_methods:
                dense_problem = build_robust_action_problem(
                    assembly.blocks,
                    assembly.problem.lower_response_matrix,
                    assembly.problem.upper_response_matrix,
                    assembly.problem.target_importance,
                    tuple(range(assembly.blocks.total_padded_nodes)),
                )
                dense_action = CurriculumAction(assembly.problem, assembly.action.coordinates)
                score_synthetic_solver_benchmark_cell(
                    store,
                    layout,
                    request.experiment,
                    condition,
                    None,
                    seed,
                    dense_problem,
                    dense_action,
                    None,
                    dense_methods,
                    request.overwrite_policy,
                )
    persist_work_structure_trend(store, layout, request)


REPEATED_TIMING_METHODS: frozenset[TransferMethod] = frozenset(
    {TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER}
)


@dataclass(frozen=True, slots=True)
class CompletedMetricArtifact:
    artifact_id: ArtifactIdentifier
    record: MetricRecord
    support: SupportSize | None


@dataclass(frozen=True, slots=True)
class WorkStructurePoint:
    k: ConceptCount
    predicted_work: float
    median_runtime_seconds: float


@dataclass(frozen=True, slots=True)
class WorkStructureStratum:
    block_pattern: ScalabilityBlockPattern
    support: SupportSize | None
    points: tuple[WorkStructurePoint, ...]
    correlation: float | None


type ScalabilityCellKey = tuple[EvaluationConditionName, RandomSeed, SupportSize | None]


class WorkStructureTrendState(StrEnum):
    INSUFFICIENT_TREND_EVIDENCE = "Insufficient Trend Evidence"


def scalability_condition_name(
    node_count: ConceptCount, pattern: ScalabilityBlockPattern
) -> EvaluationConditionName:
    return EvaluationConditionName(f"k{node_count}-{pattern.value}")


def scalability_condition_coordinates(
    condition: EvaluationConditionName,
) -> ScalabilityCellCoordinates | None:
    prefix, separator, remainder = condition.partition("-")
    if not separator or not prefix.startswith("k") or not prefix[1:].isdigit():
        return None
    if remainder not in {pattern.value for pattern in ScalabilityBlockPattern}:
        return None
    k: ConceptCount = int(prefix[1:])
    return ScalabilityCellCoordinates(k, ScalabilityBlockPattern(remainder))


def completed_scalability_metric_artifacts(
    store: ArtifactStore,
) -> tuple[CompletedMetricArtifact, ...]:
    artifacts: list[CompletedMetricArtifact] = []
    for manifest in store.all_manifests():
        if (
            ExperimentName.SCALABILITY_AND_EFFICIENCY.value
            not in manifest.semantic_producer_coordinates
        ):
            continue
        try:
            resolved = store.resolve(manifest.artifact_id)
        except ValueError:
            continue
        if resolved.state is not ArtifactState.COMPLETED:
            continue
        support_value = json.loads(resolved.semantic_producer_coordinates).get("support")
        for payload_path in resolved.payload_paths:
            path = Path(payload_path)
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            record_payload = payload.get("metric_record")
            if record_payload is None:
                continue
            artifacts.append(
                CompletedMetricArtifact(
                    resolved.artifact_id,
                    MetricRecord.model_validate(record_payload),
                    None if support_value is None else SupportSize(support_value),
                )
            )
    return tuple(artifacts)


def work_structure_strata(
    artifacts: tuple[CompletedMetricArtifact, ...],
) -> tuple[WorkStructureStratum, ...]:
    predicted_work: OrderedDict[ScalabilityCellKey, float] = OrderedDict()
    runtime: OrderedDict[ScalabilityCellKey, float] = OrderedDict()
    timed_out: set[ScalabilityCellKey] = set()
    for artifact in artifacts:
        record = artifact.record
        if (
            record.method is not TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
            or not record.valid
            or record.metric_value is None
        ):
            continue
        key: ScalabilityCellKey = (record.condition, record.seed, artifact.support)
        if record.metric_name is MetricId.PREDICTED_WORK_COORDINATE:
            predicted_work[key] = record.metric_value
        elif record.metric_name is MetricId.WALL_TIME:
            runtime[key] = record.metric_value
        elif record.metric_name is MetricId.TIMEOUT_INDICATOR and record.metric_value > 0.0:
            timed_out.add(key)
    grouped: OrderedDict[
        tuple[ScalabilityBlockPattern, SupportSize | None],
        OrderedDict[ConceptCount, list[float]],
    ] = OrderedDict()
    work_per_coordinate: OrderedDict[ScalabilityCellCoordinates, float] = OrderedDict()
    for key, runtime_seconds in runtime.items():
        if key in timed_out or runtime_seconds <= 0.0:
            continue
        coordinates = scalability_condition_coordinates(key[0])
        if coordinates is None:
            continue
        work = predicted_work.get(key)
        if work is None or work <= 0.0:
            continue
        work_per_coordinate[coordinates] = work
        stratum = grouped.setdefault((coordinates.block_pattern, key[2]), OrderedDict())
        stratum.setdefault(coordinates.k, []).append(runtime_seconds)
    minimum_points = active_config().scientific.statistics.spearman_minimum_valid_points
    strata: list[WorkStructureStratum] = []
    for (block_pattern, support), runtimes_by_k in grouped.items():
        points = tuple(
            WorkStructurePoint(
                k,
                work_per_coordinate[ScalabilityCellCoordinates(k, block_pattern)],
                median(runtimes),
            )
            for k, runtimes in sorted(runtimes_by_k.items())
        )
        correlation = _work_structure_correlation(points) if len(points) >= minimum_points else None
        strata.append(WorkStructureStratum(block_pattern, support, points, correlation))
    return tuple(strata)


def _work_structure_correlation(points: tuple[WorkStructurePoint, ...]) -> float:
    from math import log

    from scipy.stats import spearmanr

    statistic = spearmanr(
        [log(point.predicted_work) for point in points],
        [log(point.median_runtime_seconds) for point in points],
    ).statistic
    return float(statistic)


def _persist_work_structure_summary(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    strata: tuple[WorkStructureStratum, ...],
    input_artifact_ids: tuple[ArtifactIdentifier, ...],
) -> None:
    from fedorbit.experiments.validation import persist_synthetic_experiment_payload

    seed = ExperimentSeed(active_config().scientific.randomness.confirmatory_seeds[0])
    persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        seed,
        lambda fingerprint: cast(
            StableJsonPayload,
            OrderedDict(
                experiment=request.experiment.value,
                dependency_fingerprint_sha256=fingerprint,
                predicted_work_coordinate=(
                    "active-image assignment count multiplied by the sum of "
                    "padded block sizes cubed"
                ),
                strata=tuple(
                    cast(
                        StableJsonPayload,
                        OrderedDict(
                            block_pattern=stratum.block_pattern.value,
                            support=None if stratum.support is None else stratum.support,
                            distinct_k_count=len(stratum.points),
                            correlation=stratum.correlation,
                            points=tuple(
                                cast(
                                    StableJsonPayload,
                                    OrderedDict(
                                        k=point.k,
                                        predicted_work=point.predicted_work,
                                        median_runtime_seconds=point.median_runtime_seconds,
                                    ),
                                )
                                for point in stratum.points
                            ),
                        ),
                    )
                    for stratum in strata
                ),
                input_artifact_ids=tuple(identifier.value for identifier in input_artifact_ids),
            ),
        ),
        _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
        ImplementationIdentity.SOLVERS_V1,
        ArtifactName("work-structure-trend"),
        EvaluationCondition(EvaluationConditionKind.WORK_STRUCTURE).name,
    )


def persist_work_structure_trend(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    artifacts = completed_scalability_metric_artifacts(store)
    strata = work_structure_strata(artifacts)
    correlations = tuple(
        stratum.correlation for stratum in strata if stratum.correlation is not None
    )
    condition = EvaluationCondition(EvaluationConditionKind.WORK_STRUCTURE).name
    seed = active_config().scientific.randomness.confirmatory_seeds[0]
    input_artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
    if not correlations:
        persist_synthetic_diagnostic_metric(
            store,
            layout,
            request.experiment,
            condition,
            seed,
            MetricId.WORK_STRUCTURE_SPEARMAN,
            None,
            MetricUnit.CORRELATION,
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids or (ArtifactIdentifier("ineligible-cell"),),
            request.overwrite_policy,
            valid=False,
            invalid_reason=InvalidReason(WorkStructureTrendState.INSUFFICIENT_TREND_EVIDENCE.value),
        )
    else:
        persist_synthetic_diagnostic_metric(
            store,
            layout,
            request.experiment,
            condition,
            seed,
            MetricId.WORK_STRUCTURE_SPEARMAN,
            min(correlations),
            MetricUnit.CORRELATION,
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            request.overwrite_policy,
        )
    _persist_work_structure_summary(store, layout, request, strata, input_artifact_ids)


def persist_synthetic_diagnostic_metric(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
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
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
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
            _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
            ImplementationIdentity.SOLVERS_V1,
            metric_name=metric_name,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    metric = MetricRecord(
        experiment=experiment,
        pair=SYNTHETIC_DIRECTED_PAIR,
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
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
        / f"metric.{condition}.{seed}.{metric_name.value}.json"
    )
    payload = cast(StableJsonPayload, OrderedDict(metric_record=metric.model_dump(mode="json")))
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_THEOREM_VALIDATION_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(ImplementationIdentity.SOLVERS_V1))
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
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


def _fixture_seed(
    base_seed: RandomSeed, world_kind: UnresolvedMapWorldKind, fixture_index: Index
) -> RandomSeed:
    derived: RandomSeed = derive_seed32(
        SeedDerivationRequest(
            base_seed,
            RngNamespace.SYNTHETIC_INSTANCE,
            cast(
                StableJsonPayload,
                OrderedDict(world_kind=world_kind.value, fixture_index=fixture_index),
            ),
        )
    )
    return derived


class UnresolvedMapWorldUnavailability(StrEnum):
    FIXTURE_NOT_CONSTRUCTIBLE = "registered unresolved-map fixture is not constructible"


def persist_unavailable_map_world_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    world_kind: UnresolvedMapWorldKind,
    fixture_index: Index,
    seed: RandomSeed,
    invalid_reason: InvalidReason,
    overwrite_policy: OverwritePolicy,
) -> None:
    condition = EvaluationConditionName(f"{world_kind.value}-{fixture_index}")
    for metric_name in (
        MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE,
        MetricId.EXACT_MAP_ACTION_VALUE,
        MetricId.ORBIT_RADIUS_MAP_BOUND,
    ):
        persist_synthetic_diagnostic_metric(
            store,
            layout,
            experiment,
            condition,
            seed,
            metric_name,
            None,
            MetricUnit.SCORE,
            MetricDirection.DESCRIPTIVE,
            (ArtifactIdentifier("unavailable-synthetic-cell"),),
            overwrite_policy,
            valid=False,
            invalid_reason=invalid_reason,
        )


def persist_map_world_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    world_kind: UnresolvedMapWorldKind,
    fixture_index: Index,
    seed: RandomSeed,
    world: UnresolvedMapWorld,
    overwrite_policy: OverwritePolicy,
) -> None:
    condition = EvaluationConditionName(f"{world_kind.value}-{fixture_index}")
    input_artifact_ids = (ArtifactIdentifier("synthetic-generator"),)
    for metric_name, metric_value in (
        (MetricId.CERTIFIED_ROBUST_PREDICTED_VALUE, world.certified_robust_value),
        (MetricId.EXACT_MAP_ACTION_VALUE, world.diagnostics.exact_map_action_value),
        (MetricId.ORBIT_RADIUS_MAP_BOUND, world.diagnostics.bound),
    ):
        persist_synthetic_diagnostic_metric(
            store,
            layout,
            experiment,
            condition,
            seed,
            metric_name,
            float(metric_value),
            MetricUnit.SCORE,
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
        )


def _run_unresolved_map_fixtures(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    world_kind: UnresolvedMapWorldKind,
    fixtures_per_seed: ReplicateCount,
) -> None:
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    for seed in confirmatory_seeds:
        for fixture_index in range(fixtures_per_seed):
            index: Index = fixture_index
            fixture_seed = _fixture_seed(seed, world_kind, index)
            try:
                world = generate_unresolved_map_world(
                    UnresolvedMapWorldRequest(world_kind, fixture_seed)
                )
            except MechanismGenerationError as error:
                persist_unavailable_map_world_metrics(
                    store,
                    layout,
                    request.experiment,
                    world_kind,
                    index,
                    seed,
                    InvalidReason(
                        f"{UnresolvedMapWorldUnavailability.FIXTURE_NOT_CONSTRUCTIBLE.value}"
                        f": {error}"
                    ),
                    request.overwrite_policy,
                )
                continue
            persist_map_world_metrics(
                store,
                layout,
                request.experiment,
                world_kind,
                index,
                seed,
                world,
                request.overwrite_policy,
            )


def execute_common_action_under_unidentified_map(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.common_action_under_unidentified_map
    _run_unresolved_map_fixtures(
        store, layout, request, UnresolvedMapWorldKind.COMMON_ACTION, config.fixtures_per_seed
    )


def execute_robust_compromise_under_unidentified_map(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.robust_compromise_under_unidentified_map
    _run_unresolved_map_fixtures(
        store, layout, request, UnresolvedMapWorldKind.ROBUST_COMPROMISE, config.fixtures_per_seed
    )


def execute_map_dependent_action_boundary(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.map_dependent_action_boundary
    _run_unresolved_map_fixtures(
        store, layout, request, UnresolvedMapWorldKind.MAP_DEPENDENT, config.fixtures_per_seed
    )


def execute_exact_map_value_bound_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    config = active_config().experiments.exact_map_value_bound_validation
    _run_unresolved_map_fixtures(
        store,
        layout,
        request,
        UnresolvedMapWorldKind.COMMON_ACTION,
        config.zero_map_value_fixtures_per_seed,
    )
    _run_unresolved_map_fixtures(
        store,
        layout,
        request,
        UnresolvedMapWorldKind.MAP_DEPENDENT,
        config.high_map_value_fixtures_per_seed,
    )


def synthetic_coupling_problem(
    instance: CouplingInstance,
    support: SupportCount,
) -> tuple[RobustActionProblem, tuple[BlockCorrespondence, ...], CurriculumAction, RectangularHull]:
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
    problem = replace(problem, principal_support=support)
    alpha = CurriculumAction(problem=problem, coordinates=instance.active_action)
    hull = build_rectangular_hull(
        blocks, instance.lower_response_matrix, instance.lower_response_matrix
    )
    return problem, orbit, alpha, hull


def persist_coupling_mechanism_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportCount,
    method: TransferMethod,
    seed: RandomSeed,
    problem: RobustActionProblem,
    orbit: tuple[BlockCorrespondence, ...],
    alpha: CurriculumAction,
    hull: RectangularHull,
    overwrite_policy: OverwritePolicy,
) -> None:
    input_artifact_ids = (ArtifactIdentifier("synthetic-generator"),)
    exact_action = solve_robust_action(problem).selected_action
    rectangular_action = optimize_against_fixed_matrix(problem, hull.lower_bounds).selected_action
    candidates = (exact_action, rectangular_action, zero_action(problem))
    for metric_name, metric_value in (
        (
            MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP,
            fixed_action_rectangularization_gap(alpha, orbit, hull.lower_bounds),
        ),
        (
            MetricId.ROBUST_COUPLING_VALUE_GAP,
            robust_coupling_gap(candidates, problem, orbit, hull),
        ),
        (
            MetricId.COUPLING_UPPER_BOUND_DIAGNOSTIC,
            rectangular_value_over_candidates(candidates, problem, hull),
        ),
        (
            MetricId.COUPLING_ACTION_SET_SUPPORT,
            float(problem.principal_support),
        ),
    ):
        metric_unit = (
            MetricUnit.COUNT
            if metric_name is MetricId.COUPLING_ACTION_SET_SUPPORT
            else MetricUnit.SCORE
        )
        persist_synthetic_benchmark_metric(
            store,
            layout,
            experiment,
            condition,
            SupportSize(support),
            method,
            seed,
            metric_name,
            float(metric_value),
            metric_unit,
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
        )


def persist_exact_orbit_coupling_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    condition: EvaluationConditionName,
    support: SupportCount,
    seed: RandomSeed,
    problem: RobustActionProblem,
    orbit: tuple[BlockCorrespondence, ...],
    alpha: CurriculumAction,
    hull: RectangularHull,
) -> None:
    from fedorbit.experiments.validation import persist_synthetic_experiment_payload

    exact_action = solve_robust_action(problem).selected_action
    rectangular_action = optimize_against_fixed_matrix(problem, hull.lower_bounds).selected_action
    candidates = (exact_action, rectangular_action, zero_action(problem))
    persist_synthetic_experiment_payload(
        store,
        layout,
        request,
        ExperimentSeed(seed),
        lambda fingerprint: cast(
            StableJsonPayload,
            OrderedDict(
                experiment=request.experiment.value,
                dependency_fingerprint_sha256=fingerprint,
                method=ExperimentLocalMethod.EXACT_ORBIT.value,
                condition=condition,
                support=support,
                seed=seed,
                state=ArtifactState.COMPLETED.value,
                orbit_size=len(orbit),
                metrics=OrderedDict(
                    (
                        (
                            MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP.value,
                            float(
                                fixed_action_rectangularization_gap(alpha, orbit, hull.lower_bounds)
                            ),
                        ),
                        (
                            MetricId.ROBUST_COUPLING_VALUE_GAP.value,
                            float(robust_coupling_gap(candidates, problem, orbit, hull)),
                        ),
                        (
                            MetricId.COUPLING_UPPER_BOUND_DIAGNOSTIC.value,
                            float(rectangular_value_over_candidates(candidates, problem, hull)),
                        ),
                        (
                            MetricId.COUPLING_ACTION_SET_SUPPORT.value,
                            float(problem.principal_support),
                        ),
                    )
                ),
                input_artifact_ids=[ArtifactIdentifier("synthetic-generator").value],
            ),
        ),
        _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
        ImplementationIdentity.SOLVERS_V1,
        ArtifactName(f"exact-orbit.{condition}.support-{support}.{seed}"),
        condition,
        SupportSize(support),
    )


def execute_synthetic_coupling_mechanism_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    coupling_config = active_config().generators.coupling_structure
    seeds = active_config().scientific.randomness.confirmatory_seeds
    for compatibility in coupling_config.compatibility:
        eligible_supports = eligible_coupling_support_sizes(compatibility, coupling_config.supports)
        for support in eligible_supports:
            for heterogeneity in coupling_config.response_heterogeneity:
                for asymmetry in coupling_config.directed_asymmetry:
                    for sparsity in coupling_config.response_sparsity:
                        for block_pattern in coupling_config.block_patterns:
                            condition = EvaluationConditionName(
                                f"{compatibility.value}-{heterogeneity}-{asymmetry}-"
                                f"{sparsity}-{'x'.join(str(size) for size in block_pattern)}"
                            )
                            for seed in seeds:
                                coupling_request = CouplingInstanceRequest(
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
                                    instance = generate_coupling_instance(coupling_request)
                                except CouplingGenerationError:
                                    continue
                                problem, orbit, alpha, hull = synthetic_coupling_problem(
                                    instance, support
                                )
                                persist_exact_orbit_coupling_cell(
                                    store,
                                    layout,
                                    request,
                                    condition,
                                    support,
                                    seed,
                                    problem,
                                    orbit,
                                    alpha,
                                    hull,
                                )
                                persist_coupling_mechanism_metrics(
                                    store,
                                    layout,
                                    request.experiment,
                                    condition,
                                    support,
                                    TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                                    seed,
                                    problem,
                                    orbit,
                                    alpha,
                                    hull,
                                    request.overwrite_policy,
                                )
                                destroyed = coupling_destroyed_matrices(
                                    problem.blocks,
                                    instance.lower_response_matrix,
                                    instance.lower_response_matrix,
                                    seed,
                                    ContrastCoordinates(condition),
                                )
                                destroyed_problem = build_robust_action_problem(
                                    problem.blocks,
                                    destroyed.lower_response_matrix,
                                    destroyed.lower_response_matrix,
                                    instance.target_importance,
                                    tuple(range(sum(instance.block_pattern))),
                                )
                                destroyed_problem = replace(
                                    destroyed_problem, principal_support=support
                                )
                                destroyed_alpha = CurriculumAction(
                                    problem=destroyed_problem, coordinates=instance.active_action
                                )
                                destroyed_hull = build_rectangular_hull(
                                    problem.blocks,
                                    destroyed.lower_response_matrix,
                                    destroyed.lower_response_matrix,
                                )
                                persist_coupling_mechanism_metrics(
                                    store,
                                    layout,
                                    request.experiment,
                                    condition,
                                    support,
                                    TransferMethod.COUPLING_DESTROYED_FEDORBIT,
                                    seed,
                                    destroyed_problem,
                                    orbit,
                                    destroyed_alpha,
                                    destroyed_hull,
                                    request.overwrite_policy,
                                )
