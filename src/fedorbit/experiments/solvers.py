from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import cast

import torch

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
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.experiments.protocol import ExperimentExecutionRequest
from fedorbit.experiments.scoring import (
    assemble_principal_action,
    build_completion_manifest,
    solve_fedorbit_exact_sparse_action,
)
from fedorbit.experiments.synthetic import (
    CouplingGenerationError,
    CouplingInstance,
    CouplingInstanceRequest,
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
    runtime_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.infrastructure.runtime import (
    EfficiencyMeasurement,
    RandomSeed,
    SeedDerivationRequest,
    current_code_revision,
    derive_seed32,
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
    PaddedBlockStructure,
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
)
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    build_robust_action_problem,
    evaluate_objective,
    zero_action,
)
from fedorbit.types import (
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactPath,
    ArtifactStage,
    ArtifactState,
    ArtifactType,
    ArtifactTypeName,
    CoarseGroup,
    ConceptCount,
    ConfigurationSection,
    ContrastCoordinates,
    DirectedPairName,
    EvaluationConditionName,
    ExperimentCondition,
    ExperimentName,
    ExperimentSeed,
    Index,
    MethodName,
    MetricId,
    MetricUnit,
    OverwritePolicy,
    ProducerModuleName,
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
    SupportCount,
    SupportSize,
    TerminalState,
    Tolerance,
    TransferMethod,
)

_MODULE_NAME = ProducerModuleName("fedorbit.experiments.solvers")
_THEOREM_VALIDATION_CONFIGURATION_SECTIONS = frozenset(
    {ConfigurationSection.ACTION, ConfigurationSection.GENERATORS, ConfigurationSection.SOLVERS}
)


def persist_synthetic_benchmark_metric(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportCount,
    method: TransferMethod,
    seed: RandomSeed,
    metric_name: MetricId,
    metric_value: float, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    metric_unit: MetricUnit,
    direction: MetricDirection,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        method=method,
        condition=ExperimentCondition(condition),
        support=SupportSize(support),
        seed=ExperimentSeed(seed),
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.EVALUATION,
            cell,
            relevance,
            (*(identifier.value for identifier in input_artifact_ids), metric_name.value),
            _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    metric = MetricRecord(
        experiment=experiment,
        pair=DirectedPairName("synthetic"), #TODO: should be enum instead of hardcoded string
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
        valid=True,
        invalid_reason=None,
    )
    validate_metric_records(MetricRecordCollection((metric,)))
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts" #TODO: use enum, not hardcoded string
        / "derived" #TODO: use enum, not hardcoded string
        / f"metric.{condition}.support-{support}.{method.value}.{seed}.{metric_name.value}.json" #TODO: should be enums not hardcoded strings
    )
    payload = cast(StableJsonPayload, OrderedDict(metric_record=metric.model_dump(mode="json")))
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_THEOREM_VALIDATION_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.EVALUATION).sha256)
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.EVALUATION,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.PREDICTION.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.PREDICTION,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.EVALUATION,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=(),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version="1.0", #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
            created_git_commit=current_code_revision().commit,
            created_environment_sha256=environment_snapshot().fingerprint_sha256,
            state=ArtifactState.COMPLETED,
            completion_required=True,
            completion_manifest_sha256=completion.completion_manifest_sha256,
        )
    )
    store.write_completed(manifest, completion)
    return manifest


def synthetic_solver_instance(
    node_count: ConceptCount,
    block_pattern: ScalabilityBlockPattern,
    support: SupportCount,
    seed: RandomSeed,
) -> tuple[RobustActionProblem, CurriculumAction, PaddedBlockStructure]:
    instance = generate_scalability_instance(
        ScalabilityInstanceRequest(node_count, block_pattern, support, seed)
    )
    groups = tuple(CoarseGroup)[:2]
    counts = OrderedDict(zip(groups, instance.block_pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    problem = build_robust_action_problem(
        blocks,
        instance.lower_response_matrix,
        instance.lower_response_matrix,
        instance.target_importance,
        tuple(range(support)),
    )
    action = CurriculumAction(problem=problem, coordinates=instance.fixed_action)
    return problem, action, blocks


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
    support: SupportCount,
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
        MetricUnit("score"), #TODO: should be enum instead of hardcoded string
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
        MetricUnit("fraction"), #TODO: should be enum instead of hardcoded string
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
        MetricUnit("boolean"), #TODO: should be enum instead of hardcoded string
        MetricDirection.HIGHER_IS_BETTER,
        input_artifact_ids,
        overwrite_policy,
    )


def _persist_solver_benchmark_efficiency_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportCount,
    method: TransferMethod,
    seed: RandomSeed,
    measurement: EfficiencyMeasurement,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> None:
    for metric_name, metric_value, metric_unit in (
        (MetricId.WALL_TIME, measurement.wall_time_seconds, MetricUnit("seconds")), #TODO: should be enum instead of hardcoded string
        (MetricId.PEAK_HOST_RSS, measurement.peak_host_rss_mib, MetricUnit("mib")), #TODO: should be enum instead of hardcoded string
        (
            MetricId.PEAK_CUDA_ALLOCATED_BYTES,
            float(measurement.peak_cuda_allocated_bytes),
            MetricUnit("bytes"), #TODO: should be enum instead of hardcoded string
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


def _score_exact_sparse_solver_benchmark_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    support: SupportCount,
    seed: RandomSeed,
    problem: RobustActionProblem,
    action: CurriculumAction,
    reference_truth: Score | None,
    methods: tuple[MethodName, ...],
    overwrite_policy: OverwritePolicy,
) -> None:
    input_artifact_ids = (ArtifactIdentifier("synthetic-generator"),) #TODO: should be enum instead of hardcoded string
    solver_config = active_config().solvers.exact_sparse
    if TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER in methods:
        with measure_efficiency() as efficiency:
            outcome = fixed_action_worst_correspondence(
                problem,
                action,
                solver_config.lap_objective_tie_tolerance,
                solver_config.action_tie_tolerance,
            )
        _persist_solver_benchmark_efficiency_metrics(
            store,
            layout,
            experiment,
            condition,
            support,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            efficiency.result,
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
            MetricUnit("count"), #TODO: should be enum instead of hardcoded string
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
            MetricUnit("count"), #TODO: should be enum instead of hardcoded string
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
            MetricUnit("boolean"), #TODO: should be enum instead of hardcoded string
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
                MetricUnit("score"), #TODO: should be enum instead of hardcoded string
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
                condition = EvaluationConditionName(f"k{node_count}-{pattern.value}") #TODO: should be enum instead of hardcoded string
                for seed in confirmatory_seeds:
                    try:
                        problem, action, blocks = synthetic_solver_instance(
                            node_count, pattern, support, seed
                        )
                    except ScalabilityGenerationError:
                        continue
                    reference_truth = solver_benchmark_reference_truth(
                        blocks, action, config.exhaustive_truth_correspondence_count_maximum
                    )
                    _score_exact_sparse_solver_benchmark_cell(
                        store,
                        layout,
                        request.experiment,
                        condition,
                        support,
                        seed,
                        problem,
                        action,
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
                condition = EvaluationConditionName(f"k{node_count}-{pattern.value}") #TODO: should be enum instead of hardcoded string
                for seed in confirmatory_seeds:
                    try:
                        problem, action, blocks = synthetic_solver_instance(
                            node_count, pattern, support, seed
                        )
                    except ScalabilityGenerationError:
                        continue
                    work = float(
                        sum(size**3 for size in blocks.padded_size_tuple)
                        * max(blocks.total_padded_nodes, 1)
                    )
                    logger.event(
                        "solver_cell_start", #TODO: should be enum not hardcoded string
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
                        MetricUnit("count"), #TODO: should be enum instead of hardcoded string
                        MetricDirection.DESCRIPTIVE,
                        (),
                        request.overwrite_policy,
                    )
                    _score_exact_sparse_solver_benchmark_cell(
                        store,
                        layout,
                        request.experiment,
                        condition,
                        support,
                        seed,
                        problem,
                        action,
                        None,
                        exact_methods,
                        request.overwrite_policy,
                    )
            dense_condition = EvaluationConditionName(f"k{node_count}-{pattern.value}-dense") #TODO: should be enum instead of hardcoded string
            dense_support = config.exact_qap_supports[0]
            for seed in confirmatory_seeds:
                try:
                    problem, action, _ = synthetic_solver_instance(
                        node_count, pattern, dense_support, seed
                    )
                except ScalabilityGenerationError:
                    continue
                _score_exact_sparse_solver_benchmark_cell(
                    store,
                    layout,
                    request.experiment,
                    dense_condition,
                    dense_support,
                    seed,
                    problem,
                    action,
                    None,
                    dense_methods,
                    request.overwrite_policy,
                )
    raw_root = raw_dataset_root()
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    real_methods = (
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.GENERIC_EXACT_QAP,
        TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
    )
    device = torch.device("cuda" #TODO: should be enum instead of hardcoded string
                          if torch.cuda.is_available() else
                          "cpu") #TODO: should be enum instead of hardcoded string
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
            condition = EvaluationConditionName(f"real-{source.value}-to-{target.value}") #TODO: should be enum instead of hardcoded string
            logger.event(
                "real_timing_cell", #TODO: should be enum not hardcoded string
                experiment=request.experiment.value,
                source=source.value,
                target=target.value,
                seed=seed,
            )
            _score_exact_sparse_solver_benchmark_cell(
                store,
                layout,
                request.experiment,
                condition,
                principal_support,
                seed,
                assembly.problem,
                assembly.action,
                None,
                real_methods,
                request.overwrite_policy,
            )
    _persist_work_structure_spearman(store, layout, request)


def _persist_work_structure_spearman(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    from math import log

    from scipy.stats import spearmanr

    from fedorbit.experiments.synthesis import completed_experiment_metric_records_with_support

    records = completed_experiment_metric_records_with_support(store, request.experiment)
    grouped: OrderedDict[tuple[str, int | None], list[tuple[float, float]]] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    work_by_key: OrderedDict[tuple[str, RandomSeed], float] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    runtime_by_key: OrderedDict[tuple[str, RandomSeed], float] = OrderedDict() #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    for record, support in records:
        if record.method != TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER:
            continue
        if not record.valid or record.metric_value is None:
            continue
        key = (record.condition, record.seed)
        if record.metric_name == MetricId.PREDICTED_WORK_COORDINATE:
            work_by_key[key] = record.metric_value
        if record.metric_name == MetricId.WALL_TIME:
            runtime_by_key[key] = record.metric_value
        grouped.setdefault((record.condition.rsplit("-", 1)[-1], support), [])
    points: list[tuple[float, float]] = []
    for key, work in work_by_key.items():
        runtime = runtime_by_key.get(key)
        if runtime is None or work <= 0.0 or runtime <= 0.0:
            continue
        points.append((log(work), log(runtime)))
    minimum = active_config().scientific.statistics.spearman_minimum_valid_points
    if len(points) < minimum:
        return
    correlation = float(
        spearmanr([point[0] for point in points], [point[1] for point in points]).statistic
    )
    persist_synthetic_diagnostic_metric(
        store,
        layout,
        request.experiment,
        EvaluationConditionName("work-structure"), #TODO: should be enum instead of hardcoded string
        active_config().scientific.randomness.confirmatory_seeds[0],
        MetricId.WORK_STRUCTURE_SPEARMAN,
        correlation,
        MetricUnit("correlation"), #TODO: should be enum instead of hardcoded string
        MetricDirection.DESCRIPTIVE,
        (),
        request.overwrite_policy,
    )


def persist_synthetic_diagnostic_metric(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    condition: EvaluationConditionName,
    seed: RandomSeed,
    metric_name: MetricId,
    metric_value: float, #TODO: do not use primitives. Fix by introducing a proper error type or message class and identify and fix why architecture tests didn't catch this
    metric_unit: MetricUnit,
    direction: MetricDirection,
    input_artifact_ids: ArtifactIdentifiers,
    overwrite_policy: OverwritePolicy,
) -> ReusableArtifactManifest | None:
    relevance = experiment_relevance(experiment)
    cell = SemanticCell(
        experiment=experiment,
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        condition=ExperimentCondition(condition),
        seed=ExperimentSeed(seed),
    )
    coordinates = SemanticCoordinateText(cell.identity_json(relevance))
    fingerprint = Sha256Digest(
        stage_dependency_fingerprint(
            ArtifactStage.EVALUATION,
            cell,
            relevance,
            (*(identifier.value for identifier in input_artifact_ids), metric_name.value),
            _THEOREM_VALIDATION_CONFIGURATION_SECTIONS,
            _MODULE_NAME,
        )
    )
    if overwrite_policy == OverwritePolicy.REUSE:
        existing = store.find_by_fingerprint(ArtifactFingerprint(fingerprint))
        if existing is not None:
            return existing
    metric = MetricRecord(
        experiment=experiment,
        pair=DirectedPairName("synthetic"), #TODO: should be enum instead of hardcoded string
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
        valid=True,
        invalid_reason=None,
    )
    validate_metric_records(MetricRecordCollection((metric,)))
    payload_path = (
        experiment_workspace(layout, experiment)
        / "artifacts" #TODO: should be enum instead of hardcoded string
        / "derived" #TODO: should be enum instead of hardcoded string
        / f"metric.{condition}.{seed}.{metric_name.value}.json" #TODO: should be enums not hardcoded strings
    )
    payload = cast(StableJsonPayload, OrderedDict(metric_record=metric.model_dump(mode="json")))
    atomic_write_json(payload_path, payload)
    payload_sha256 = file_sha256(payload_path)
    configuration_sha256 = Sha256Digest(
        configuration_subset_digest(_THEOREM_VALIDATION_CONFIGURATION_SECTIONS)
    )
    code_sha256 = Sha256Digest(implementation_fingerprint(_MODULE_NAME))
    runtime_sha256 = Sha256Digest(runtime_fingerprint(ArtifactStage.EVALUATION).sha256)
    completion = build_completion_manifest(
        coordinates,
        fingerprint,
        ArtifactPath(payload_path),
        payload_sha256,
        configuration_sha256,
        code_sha256,
        runtime_sha256,
        stage=ArtifactStage.EVALUATION,
    )
    manifest = ReusableArtifactManifest.model_validate(
        OrderedDict(
            artifact_id=artifact_id(
                ArtifactTypeName(ArtifactType.PREDICTION.value), payload, Sha256Digest(fingerprint)
            ),
            artifact_type=ArtifactType.PREDICTION,
            semantic_producer_coordinates=coordinates,
            producer_stage=ArtifactStage.EVALUATION,
            dependency_fingerprint_sha256=fingerprint,
            upstream_artifact_ids=(),
            applicable_configuration_sha256=configuration_sha256,
            relevant_code_sha256=code_sha256,
            material_runtime_sha256=runtime_sha256,
            payload_paths=(str(payload_path),),
            payload_sha256=payload_sha256,
            schema_version="1.0", #TODO: should be retrieved from yml and accessed through config. Identify any similar issues and fix it
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


def _persist_map_world_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    world_kind: UnresolvedMapWorldKind,
    fixture_index: Index,
    seed: RandomSeed,
    world: UnresolvedMapWorld,
    overwrite_policy: OverwritePolicy,
) -> None:
    condition = EvaluationConditionName(f"{world_kind.value}-{fixture_index}") #TODO: should be enum instead of hardcoded string
    input_artifact_ids = (ArtifactIdentifier("synthetic-generator"),) #TODO: should be enum instead of hardcoded string
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
            MetricUnit("score"), #TODO: should be enum instead of hardcoded string
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
            world = generate_unresolved_map_world(
                UnresolvedMapWorldRequest(world_kind, fixture_seed)
            )
            _persist_map_world_metrics(
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


def _synthetic_coupling_problem(
    instance: CouplingInstance,
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
    alpha = CurriculumAction(problem=problem, coordinates=instance.active_action)
    hull = build_rectangular_hull(
        blocks, instance.lower_response_matrix, instance.lower_response_matrix
    )
    return problem, orbit, alpha, hull


def _persist_coupling_mechanism_metrics(
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
    input_artifact_ids = (ArtifactIdentifier("synthetic-generator"),) #TODO: should be enum instead of hardcoded string
    candidates = (alpha, zero_action(problem))
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
            MetricUnit("score"), #TODO: should be enum instead of hardcoded string
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            overwrite_policy,
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
                                problem, orbit, alpha, hull = _synthetic_coupling_problem(instance)
                                _persist_coupling_mechanism_metrics(
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
                                destroyed_alpha = CurriculumAction(
                                    problem=destroyed_problem, coordinates=instance.active_action
                                )
                                destroyed_hull = build_rectangular_hull(
                                    problem.blocks,
                                    destroyed.lower_response_matrix,
                                    destroyed.lower_response_matrix,
                                )
                                _persist_coupling_mechanism_metrics(
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
