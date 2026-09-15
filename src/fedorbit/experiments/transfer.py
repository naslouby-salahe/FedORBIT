from __future__ import annotations

import contextlib
import functools
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import cast

import torch

from fedorbit.analysis.metrics import (
    CrossEntropy,
    InvalidEvaluationDataError,
    RelativeMacroCeGain,
    harm_indicator,
    relative_macro_ce_gain,
)
from fedorbit.analysis.records import (
    MetricDirection,
)
from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.datasets.materialization import (
    MaterializationError,
    MaterializedClient,
    subsampled_materialized_client,
)
from fedorbit.experiments.catalogue import ExperimentExecutionRequest
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.experiments.scoring import (
    PrincipalActionAssembly,
    WeakSignalPerturbation,
    action_sha256,
    assemble_principal_action,
    assess_pair_seed_structure,
    ci_half_width_perturbation,
    common_eligible_groups,
    curriculum_multipliers_from_action,
    persist_boundary_diagnostic_metrics,
    persist_ineligible_transfer_cell,
    persist_primary_transfer_cell_metrics,
    persist_primary_transfer_metric,
    response_heterogeneity_perturbation,
    response_scale_perturbation,
    score_coarse_block_mean_cell,
    score_coarse_block_min_cell,
    score_coupling_destroyed_fedorbit_cell,
    score_exact_map_oracle_cell,
    score_fedorbit_exact_sparse_solver_cell,
    score_fedorbit_with_confirmation_verdict_cell,
    score_fedorbit_without_confirmation_cell,
    score_generic_exact_qap_cell,
    score_local_only_cell,
    score_local_only_cell_adapter,
    score_local_sir_cell,
    score_local_sir_cell_adapter,
    score_matched_resource_rectangular_cell,
    score_orbit_mean_cell,
    score_point_correspondence_commitment_cell,
    score_robust_action_cell,
    semantic_partition_bucket_of,
    semantic_partition_label,
    solve_dense_ccp_fallback_action,
    solve_exact_map_oracle_action,
    solve_fedorbit_exact_sparse_action,
    solve_fedorbit_exact_sparse_action_at_support,
    solve_matched_resource_rectangular_action,
)
from fedorbit.experiments.solvers import persist_synthetic_diagnostic_metric
from fedorbit.experiments.synthesis import completed_experiment_metric_records
from fedorbit.experiments.training import execute_client_base_model_pilot
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
    ExecutionError,
)
from fedorbit.infrastructure.preparation import (
    load_or_materialize_client,
)
from fedorbit.infrastructure.runtime import (
    RandomSeed,
    SeedDerivationRequest,
    derive_seed32,
    execution_device,
    execution_logger,
)
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
)
from fedorbit.learning.scoring import ScoreArtifact
from fedorbit.learning.training import (
    make_adamw,
)
from fedorbit.methods.assimilation import (
    AssimilationCoordinates,
    ConfirmationRequest,
    ConfirmationVerdict,
    apply_accepted_assimilation,
    capture_pre_confirm_pair,
    run_proposal_confirmation,
    settle_rejected_proposal,
)
from fedorbit.methods.target import (
    SourceProposal,
    TargetOptimizerStepLedger,
    assert_target_diagnostic_reserve,
    rank_source_proposals,
    select_source_sequentially,
)
from fedorbit.optimization.certificates import (
    build_rectangular_hull,
    rectangular_value_over_candidates,
    robust_coupling_gap,
)
from fedorbit.optimization.correspondence import (
    enumerate_block_permutations,
)
from fedorbit.optimization.diagnostics import (
    fixed_action_rectangularization_gap,
)
from fedorbit.optimization.exact_sparse import (
    solve_robust_action,
)
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    zero_action,
)
from fedorbit.oracle import authorize_oracle_access
from fedorbit.types import (
    PRINCIPAL_EVALUATION_CONDITION,
    ArtifactIdentifier,
    CellUnavailabilityReason,
    ClassCount,
    ClientRole,
    ConceptCount,
    ContrastCoordinates,
    DatasetId,
    DirectedPairName,
    EvaluationCondition,
    EvaluationConditionKind,
    EvaluationConditionName,
    ExperimentLocalMethod,
    ExperimentName,
    FilesystemSlug,
    InvalidReason,
    MetricId,
    MetricUnit,
    MutableCell,
    PairSeedIneligibilityReason,
    RngNamespace,
    ScientificAlgorithmicFailureError,
    Score,
    SemanticPartitionId,
    SourceClientName,
    StableJsonPayload,
    SupportCount,
    SupportSize,
    TransferMethod,
    WeakSignalBoundaryDimension,
    directed_pair_name,
)


def _persist_ineligible_methods(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    seed: RandomSeed,
    methods: tuple[TransferMethod, ...],
    condition: EvaluationConditionName = PRINCIPAL_EVALUATION_CONDITION.name,
    pair_seed_ineligibility_reason: PairSeedIneligibilityReason | None = None,
) -> None:
    for method in methods:
        persist_ineligible_transfer_cell(
            store,
            layout,
            request.experiment,
            pair_direction,
            source,
            target,
            method,
            seed,
            request.overwrite_policy,
            condition,
            pair_seed_ineligibility_reason,
        )


def _assess_or_persist_cross_pair_ineligibility(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    methods: tuple[TransferMethod, ...],
    condition: EvaluationConditionName = PRINCIPAL_EVALUATION_CONDITION.name,
) -> bool:
    structure = assess_pair_seed_structure(
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
    )
    if structure.is_eligible:
        return True
    _persist_ineligible_methods(
        store,
        layout,
        request,
        pair_direction,
        source,
        target,
        seed,
        methods,
        condition,
        structure.ineligibility_reason,
    )
    return False


def _reuse_principal_transfer_metrics(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    method: TransferMethod,
    seed: RandomSeed,
    condition: EvaluationConditionName,
) -> bool:
    from fedorbit.experiments.synthesis import completed_experiment_metric_records

    records = completed_experiment_metric_records(
        store, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER
    )
    matched = tuple(
        record
        for record in records
        if record.pair == pair_direction
        and record.method == method
        and record.seed == seed
        and record.condition == PRINCIPAL_EVALUATION_CONDITION.name
        and record.valid
        and record.metric_value is not None
    )
    if not matched:
        return False
    for record in matched:
        persist_primary_transfer_metric(
            store,
            layout,
            request.experiment,
            pair_direction,
            source,
            target,
            method,
            seed,
            record.metric_name,
            record.metric_value,
            record.metric_unit,
            record.direction,
            record.input_artifact_ids,
            request.overwrite_policy,
            condition,
        )
    return True


def _persist_unavailable_method_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    method: TransferMethod,
    seed: RandomSeed,
    reason: InvalidReason,
    category: CellUnavailabilityReason,
) -> None:
    persist_primary_transfer_metric(
        store,
        layout,
        request.experiment,
        pair_direction,
        source,
        target,
        method,
        seed,
        MetricId.ABSTENTION_INDICATOR,
        None,
        MetricUnit.BOOLEAN,
        MetricDirection.DESCRIPTIVE,
        (ArtifactIdentifier(category.value),),
        request.overwrite_policy,
        PRINCIPAL_EVALUATION_CONDITION.name,
        valid=False,
        invalid_reason=reason,
    )


def execute_primary_strict_cross_telemetry_transfer(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = execution_device()
    authorize_oracle_access(
        request.experiment,
        tuple(
            method
            for method in request.definition.methods
            if isinstance(method, (TransferMethod, ExperimentLocalMethod))
        ),
    )
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_target: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()
    for directed_pair in primary_pairs:
        target = directed_pair.target
        if target not in materialized_by_target:
            try:
                materialized_by_target[target] = load_or_materialize_client(
                    target, raw_root, layout
                )
            except MaterializationError:
                pair_direction = DirectedPairName(
                    f"{directed_pair.source.value} -> {directed_pair.target.value}"
                )
                for seed in confirmatory_seeds:
                    _persist_ineligible_methods(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        seed,
                        (
                            TransferMethod.LOCAL_ONLY,
                            TransferMethod.LOCAL_SIR,
                            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                        ),
                    )
                continue
        materialized = materialized_by_target[target]
        source = directed_pair.source
        if source not in materialized_by_target:
            with contextlib.suppress(MaterializationError):
                materialized_by_target[source] = load_or_materialize_client(
                    source, raw_root, layout
                )
        source_materialized = materialized_by_target.get(source)
        pair_direction = DirectedPairName(
            f"{directed_pair.source.value} -> {directed_pair.target.value}"
        )
        for seed in confirmatory_seeds:
            local_only = score_local_only_cell(store, layout, target, materialized, seed, device)
            if local_only is not None:
                score, n_classes, checkpoint_artifact_id = local_only
                persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    directed_pair.source,
                    directed_pair.target,
                    TransferMethod.LOCAL_ONLY,
                    seed,
                    score,
                    n_classes,
                    (checkpoint_artifact_id,),
                )
            local_sir = score_local_sir_cell(store, layout, target, materialized, seed, device)
            if local_sir is not None:
                score, n_classes, input_artifact_ids = local_sir
                persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    directed_pair.source,
                    directed_pair.target,
                    TransferMethod.LOCAL_SIR,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                )
            if source_materialized is not None:
                pair_seed_structure = assess_pair_seed_structure(
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                )
                if not pair_seed_structure.is_eligible:
                    _persist_ineligible_methods(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        seed,
                        (
                            TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                            TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
                            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                            TransferMethod.GENERIC_EXACT_QAP,
                            TransferMethod.EXACT_MAP_ORACLE,
                        ),
                        pair_seed_ineligibility_reason=pair_seed_structure.ineligibility_reason,
                    )
                    continue
                matched_resource_rectangular = score_matched_resource_rectangular_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                    device,
                )
                if matched_resource_rectangular is not None:
                    score, n_classes, input_artifact_ids = matched_resource_rectangular
                    persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                    )
                point_correspondence = score_point_correspondence_commitment_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                    device,
                )
                if point_correspondence is not None:
                    score, n_classes, input_artifact_ids = point_correspondence
                    persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                    )
                fedorbit_exact_sparse = score_fedorbit_exact_sparse_solver_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                    device,
                )
                if fedorbit_exact_sparse is not None:
                    score, n_classes, input_artifact_ids = fedorbit_exact_sparse
                    persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                    )
                generic_exact_qap = score_generic_exact_qap_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                    device,
                )
                if generic_exact_qap is not None:
                    score, n_classes, input_artifact_ids = generic_exact_qap
                    persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        TransferMethod.GENERIC_EXACT_QAP,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                    )
                exact_map_oracle = score_exact_map_oracle_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    materialized,
                    seed,
                    device,
                )
                if exact_map_oracle is not None:
                    score, n_classes, input_artifact_ids = exact_map_oracle
                    persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        directed_pair.source,
                        directed_pair.target,
                        TransferMethod.EXACT_MAP_ORACLE,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                    )


def execute_mechanism_ablations(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = execution_device()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    scorers: tuple[
        tuple[
            TransferMethod,
            Callable[
                [
                    ArtifactStore,
                    WorkspaceLayout,
                    DatasetId,
                    DatasetId,
                    MaterializedClient,
                    MaterializedClient,
                    RandomSeed,
                    torch.device,
                ],
                tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None,
            ],
        ],
        ...,
    ] = (
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, score_fedorbit_exact_sparse_solver_cell),
        (TransferMethod.MATCHED_RESOURCE_RECTANGULAR, score_matched_resource_rectangular_cell),
        (
            TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
            score_point_correspondence_commitment_cell,
        ),
        (TransferMethod.COUPLING_DESTROYED_FEDORBIT, score_coupling_destroyed_fedorbit_cell),
        (TransferMethod.COARSE_BLOCK_MEAN, score_coarse_block_mean_cell),
        (TransferMethod.COARSE_BLOCK_MIN, score_coarse_block_min_cell),
        (TransferMethod.ORBIT_MEAN, score_orbit_mean_cell),
        (TransferMethod.LOCAL_SIR, score_local_sir_cell_adapter),
    )
    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        pair_direction = directed_pair_name(source, target)
        if source_materialized is None or target_materialized is None:
            for seed in confirmatory_seeds:
                _persist_ineligible_methods(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    seed,
                    tuple(method for method, _scorer in scorers),
                )
            continue
        for seed in confirmatory_seeds:
            cross_pair_eligible = _assess_or_persist_cross_pair_ineligibility(
                store,
                layout,
                request,
                pair_direction,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                tuple(method for method, _scorer in scorers if method != TransferMethod.LOCAL_SIR),
            )
            for method, scorer in scorers:
                if method != TransferMethod.LOCAL_SIR and not cross_pair_eligible:
                    continue
                if _reuse_principal_transfer_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    method,
                    seed,
                    PRINCIPAL_EVALUATION_CONDITION.name,
                ):
                    continue
                try:
                    scored = scorer(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        target_materialized,
                        seed,
                        device,
                    )
                except InvalidEvaluationDataError as error:
                    _persist_unavailable_method_cell(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        error.reason,
                        CellUnavailabilityReason.INVALID_EVALUATION_DATA,
                    )
                    continue
                except ScientificAlgorithmicFailureError as error:
                    _persist_unavailable_method_cell(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        InvalidReason(f"{type(error).__name__}: {error}"),
                        CellUnavailabilityReason.SCIENTIFIC_ALGORITHMIC_FAILURE,
                    )
                    continue
                if scored is None:
                    continue
                score, n_classes, input_artifact_ids = scored
                persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    method,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                )


def _principal_local_only_macro_ce(
    store: ArtifactStore,
    pair_direction: DirectedPairName,
    seed: RandomSeed,
) -> float | None:
    records = completed_experiment_metric_records(
        store, ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER
    )
    values = tuple(
        record.metric_value
        for record in records
        if record.pair == pair_direction
        and record.seed == seed
        and record.method == TransferMethod.LOCAL_ONLY
        and record.metric_name == MetricId.MACRO_CROSS_ENTROPY
        and record.valid
        and record.metric_value is not None
    )
    return values[0] if values else None


def _persist_confirmation_safety_indicators(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    seed: RandomSeed,
    local_ce: CrossEntropy,
    with_score: ScoreArtifact,
    without_score: ScoreArtifact | None,
    verdicts: list[ConfirmationVerdict],
    input_artifact_ids: tuple[ArtifactIdentifier, ...],
) -> None:
    materiality = active_config().scientific.materiality
    if not verdicts:
        no_proposal = InvalidReason(
            "no proposal-eligible target decision: confirmation coverage and safety rates are "
            "reported separately and are not defined for this seed"
        )
        for metric_name in (
            MetricId.COVERAGE_CONFIRM,
            MetricId.HARM_RATE_CONFIRM,
            MetricId.HARMFUL_ACCEPTED_RATE,
            MetricId.USEFUL_ACCEPTED_RATE,
            MetricId.BENEFICIAL_REJECTED_RATE,
        ):
            persist_primary_transfer_metric(
                store,
                layout,
                request.experiment,
                pair_direction,
                source,
                target,
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                seed,
                metric_name,
                None,
                MetricUnit.FRACTION,
                MetricDirection.DESCRIPTIVE,
                input_artifact_ids,
                request.overwrite_policy,
                valid=False,
                invalid_reason=no_proposal,
            )
        return
    accepted = bool(verdicts[-1].accepted)
    with_gain = relative_macro_ce_gain(local_ce, with_score.macro_cross_entropy)
    unavailable = InvalidReason(
        "relative macro-CE gain is unavailable: reference macro-CE is below the registered "
        "denominator floor"
    )
    _persist_relative_gain_metric(
        store,
        layout,
        request,
        pair_direction,
        source,
        target,
        seed,
        with_gain,
        unavailable,
        input_artifact_ids,
    )
    indicators: list[tuple[MetricId, float]] = []
    if with_gain.relative is None:
        for metric_name in (
            MetricId.HARMFUL_ACCEPTED_RATE,
            MetricId.USEFUL_ACCEPTED_RATE,
            MetricId.BENEFICIAL_REJECTED_RATE,
            MetricId.HARM_RATE_CONFIRM,
            MetricId.COVERAGE_CONFIRM,
        ):
            persist_primary_transfer_metric(
                store,
                layout,
                request.experiment,
                pair_direction,
                source,
                target,
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                seed,
                metric_name,
                None,
                MetricUnit.FRACTION,
                MetricDirection.DESCRIPTIVE,
                input_artifact_ids,
                request.overwrite_policy,
                valid=False,
                invalid_reason=unavailable,
            )
        return
    harmful = harm_indicator(
        with_gain.relative, materiality.harmful_transfer_relative_macro_ce_gain
    )
    useful = with_gain.relative >= materiality.useful_transfer_relative_macro_ce_gain
    indicators.extend(
        (
            (MetricId.HARMFUL_ACCEPTED_RATE, 1.0 if accepted and harmful else 0.0),
            (MetricId.USEFUL_ACCEPTED_RATE, 1.0 if accepted and useful else 0.0),
            (MetricId.BENEFICIAL_REJECTED_RATE, 1.0 if (not accepted) and useful else 0.0),
            (MetricId.COVERAGE_CONFIRM, 1.0 if accepted else 0.0),
            (MetricId.HARM_RATE_CONFIRM, 1.0 if harmful else 0.0),
        )
    )
    if without_score is not None:
        without_gain = relative_macro_ce_gain(local_ce, without_score.macro_cross_entropy)
        if without_gain.relative is None:
            persist_primary_transfer_metric(
                store,
                layout,
                request.experiment,
                pair_direction,
                source,
                target,
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                seed,
                MetricId.HARM_RATE_NO_CONFIRM,
                None,
                MetricUnit.FRACTION,
                MetricDirection.DESCRIPTIVE,
                input_artifact_ids,
                request.overwrite_policy,
                valid=False,
                invalid_reason=unavailable,
            )
        else:
            without_harmful = harm_indicator(
                without_gain.relative, materiality.harmful_transfer_relative_macro_ce_gain
            )
            harm_confirm = 1.0 if harmful else 0.0
            harm_no = 1.0 if without_harmful else 0.0
            indicators.append((MetricId.HARM_RATE_NO_CONFIRM, harm_no))
            indicators.append((MetricId.ABSOLUTE_RISK_REDUCTION, harm_no - harm_confirm))
            if harm_no > 0.0:
                indicators.append(
                    (MetricId.RELATIVE_RISK_REDUCTION, (harm_no - harm_confirm) / harm_no)
                )
    for metric_name, metric_value in indicators:
        persist_primary_transfer_metric(
            store,
            layout,
            request.experiment,
            pair_direction,
            source,
            target,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            metric_name,
            metric_value,
            MetricUnit.FRACTION,
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            request.overwrite_policy,
        )


def _persist_relative_gain_metric(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    seed: RandomSeed,
    gain: RelativeMacroCeGain,
    unavailable: InvalidReason,
    input_artifact_ids: tuple[ArtifactIdentifier, ...],
) -> None:
    if gain.relative is None:
        persist_primary_transfer_metric(
            store,
            layout,
            request.experiment,
            pair_direction,
            source,
            target,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            seed,
            MetricId.RELATIVE_MACRO_CE_GAIN,
            None,
            MetricUnit.FRACTION,
            MetricDirection.HIGHER_IS_BETTER,
            input_artifact_ids,
            request.overwrite_policy,
            valid=False,
            invalid_reason=unavailable,
        )
        return
    persist_primary_transfer_metric(
        store,
        layout,
        request.experiment,
        pair_direction,
        source,
        target,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        seed,
        MetricId.RELATIVE_MACRO_CE_GAIN,
        gain.relative,
        MetricUnit.FRACTION,
        MetricDirection.HIGHER_IS_BETTER,
        input_artifact_ids,
        request.overwrite_policy,
    )


def execute_target_confirmation_and_portability(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = execution_device()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    directed_pairs = (
        *active_config().scientific.datasets.primary_directed_pairs,
        *active_config().scientific.datasets.secondary_directed_pairs,
    )
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    for directed_pair in directed_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        eligible = common_eligible_groups(source, target, source_materialized, target_materialized)
        if eligible is not None:
            target_eligible = eligible[1]
            target_concepts: ConceptCount = sum(
                len(concept_groups) for concept_groups in target_eligible.values()
            )
            assert_target_diagnostic_reserve(target_concepts)
        pair_direction = directed_pair_name(source, target)
        for seed in confirmatory_seeds:
            local_only_ce = _principal_local_only_macro_ce(store, pair_direction, seed)
            if local_only_ce is None:
                _persist_unavailable_method_cell(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                    seed,
                    InvalidReason("principal Local-Only reference unavailable for this pair-seed"),
                    CellUnavailabilityReason.INELIGIBLE_OR_ABSTAIN,
                )
                continue
            if not _assess_or_persist_cross_pair_ineligibility(
                store,
                layout,
                request,
                pair_direction,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                (
                    TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                    TransferMethod.FEDORBIT_WITHOUT_CONFIRMATION,
                ),
            ):
                continue
            verdict_sink: MutableCell[ConfirmationVerdict] = MutableCell()
            with_confirmation = score_fedorbit_with_confirmation_verdict_cell(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
                verdict_sink,
            )
            verdicts = [] if verdict_sink.value is None else [verdict_sink.value]
            if with_confirmation is not None:
                score, n_classes, input_artifact_ids = with_confirmation
                persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                )
                for verdict in verdicts:
                    persist_primary_transfer_metric(
                        store,
                        layout,
                        request.experiment,
                        pair_direction,
                        source,
                        target,
                        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                        seed,
                        MetricId.PROPOSAL_ACCEPTANCE_RATE,
                        1.0 if verdict.accepted else 0.0,
                        MetricUnit.FRACTION,
                        MetricDirection.DESCRIPTIVE,
                        input_artifact_ids,
                        request.overwrite_policy,
                    )
            without_confirmation = score_fedorbit_without_confirmation_cell(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
            )
            if without_confirmation is not None:
                score, n_classes, input_artifact_ids = without_confirmation
                persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    TransferMethod.FEDORBIT_WITHOUT_CONFIRMATION,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                )
            if with_confirmation is not None:
                _persist_confirmation_safety_indicators(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    seed,
                    CrossEntropy(local_only_ce),
                    with_confirmation[0],
                    None if without_confirmation is None else without_confirmation[0],
                    verdicts,
                    with_confirmation[2],
                )


def execute_secondary_cross_modality_generalization(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = execution_device()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    secondary_pairs = active_config().scientific.datasets.secondary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    scorers = (
        (TransferMethod.LOCAL_ONLY, score_local_only_cell_adapter),
        (TransferMethod.LOCAL_SIR, score_local_sir_cell_adapter),
        (TransferMethod.MATCHED_RESOURCE_RECTANGULAR, score_matched_resource_rectangular_cell),
        (
            TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
            score_point_correspondence_commitment_cell,
        ),
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, score_fedorbit_exact_sparse_solver_cell),
    )
    for directed_pair in secondary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = directed_pair_name(source, target)
        for seed in confirmatory_seeds:
            cross_pair_eligible = _assess_or_persist_cross_pair_ineligibility(
                store,
                layout,
                request,
                pair_direction,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                tuple(
                    method
                    for method, _scorer in scorers
                    if method not in (TransferMethod.LOCAL_ONLY, TransferMethod.LOCAL_SIR)
                ),
            )
            for method, scorer in scorers:
                if (
                    method
                    not in (
                        TransferMethod.LOCAL_ONLY,
                        TransferMethod.LOCAL_SIR,
                    )
                    and not cross_pair_eligible
                ):
                    continue
                try:
                    scored = scorer(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        target_materialized,
                        seed,
                        device,
                    )
                except InvalidEvaluationDataError as error:
                    _persist_unavailable_method_cell(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        error.reason,
                        CellUnavailabilityReason.INVALID_EVALUATION_DATA,
                    )
                    continue
                except ScientificAlgorithmicFailureError as error:
                    _persist_unavailable_method_cell(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        InvalidReason(f"{type(error).__name__}: {error}"),
                        CellUnavailabilityReason.SCIENTIFIC_ALGORITHMIC_FAILURE,
                    )
                    continue
                if scored is None:
                    continue
                score, n_classes, input_artifact_ids = scored
                persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    method,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                )


def execute_sparsity_and_dense_fallback(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = execution_device()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    conditions: tuple[
        tuple[
            EvaluationConditionName,
            TransferMethod,
            Callable[[RobustActionProblem, RandomSeed], CurriculumAction | None],
        ],
        ...,
    ] = (
        (
            EvaluationCondition.exact_sparse(SupportSize(1)).name,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            solve_fedorbit_exact_sparse_action_at_support(1),
        ),
        (
            EvaluationCondition.exact_sparse(SupportSize(2)).name,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            solve_fedorbit_exact_sparse_action_at_support(2),
        ),
        (
            EvaluationCondition.exact_sparse(SupportSize(3)).name,
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            solve_fedorbit_exact_sparse_action_at_support(3),
        ),
        (
            EvaluationCondition(EvaluationConditionKind.DENSE_CCP).name,
            TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
            solve_dense_ccp_fallback_action,
        ),
    )
    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        pair_direction = directed_pair.direction
        if source_materialized is None or target_materialized is None:
            for seed in confirmatory_seeds:
                _persist_ineligible_methods(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    seed,
                    (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,),
                    EvaluationCondition.exact_sparse(SupportSize(2)).name,
                )
            continue
        principal_support = active_config().scientific.action.principal_sparse_support
        principal_condition = EvaluationCondition.exact_sparse(SupportSize(principal_support)).name
        for seed in confirmatory_seeds:
            for condition_name, method, solve_action in conditions:
                if not _assess_or_persist_cross_pair_ineligibility(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    (method,),
                    condition_name,
                ):
                    continue
                if (
                    method == TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER
                    and condition_name == principal_condition
                    and _reuse_principal_transfer_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        condition_name,
                    )
                ):
                    continue
                scored = score_robust_action_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    device,
                    FilesystemSlug("sparsity-and-dense-fallback"),
                    method,
                    solve_action,
                )
                if scored is None:
                    continue
                score, n_classes, input_artifact_ids = scored
                persist_primary_transfer_cell_metrics(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    method,
                    seed,
                    score,
                    n_classes,
                    input_artifact_ids,
                    condition_name,
                )


def execute_real_packet_coupling_mechanism_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = execution_device()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
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
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        pair_direction = directed_pair_name(source, target)
        if source_materialized is None or target_materialized is None:
            for seed in confirmatory_seeds:
                _persist_ineligible_methods(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    seed,
                    (TransferMethod.MATCHED_RESOURCE_RECTANGULAR,),
                )
            continue
        for seed in confirmatory_seeds:
            if not _assess_or_persist_cross_pair_ineligibility(
                store,
                layout,
                request,
                pair_direction,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                (TransferMethod.MATCHED_RESOURCE_RECTANGULAR,),
            ):
                continue
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
            orbit = tuple(enumerate_block_permutations(assembly.blocks))
            hull = build_rectangular_hull(
                assembly.blocks,
                assembly.problem.lower_response_matrix,
                assembly.problem.upper_response_matrix,
            )
            rectangular_action = solve_matched_resource_rectangular_action(assembly.problem, seed)
            if rectangular_action is None:
                continue
            candidates = (assembly.action, rectangular_action, zero_action(assembly.problem))
            for metric_name, metric_value in (
                (
                    MetricId.FIXED_ACTION_RECTANGULARIZATION_GAP,
                    fixed_action_rectangularization_gap(assembly.action, orbit, hull.lower_bounds),
                ),
                (
                    MetricId.ROBUST_COUPLING_VALUE_GAP,
                    robust_coupling_gap(candidates, assembly.problem, orbit, hull),
                ),
                (
                    MetricId.COUPLING_UPPER_BOUND_DIAGNOSTIC,
                    rectangular_value_over_candidates(candidates, assembly.problem, hull),
                ),
                (
                    MetricId.COUPLING_ACTION_SET_SUPPORT,
                    float(assembly.problem.principal_support),
                ),
            ):
                metric_unit = (
                    MetricUnit.COUNT
                    if metric_name is MetricId.COUPLING_ACTION_SET_SUPPORT
                    else MetricUnit.SCORE
                )
                persist_primary_transfer_metric(
                    store,
                    layout,
                    request.experiment,
                    pair_direction,
                    source,
                    target,
                    TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                    seed,
                    metric_name,
                    float(metric_value),
                    metric_unit,
                    MetricDirection.DESCRIPTIVE,
                    assembly.input_artifact_ids,
                    request.overwrite_policy,
                )


def execute_multi_source_selection_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = execution_device()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.multi_source_selection_validation
    all_clients = active_config().scientific.datasets.clients
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    for target in config.targets:
        target_materialized = materialized(target)
        if target_materialized is None:
            continue
        candidate_sources = tuple(
            client
            for client, specification in all_clients.items()
            if client != target and specification.role is ClientRole.PRIMARY
        )
        for seed in confirmatory_seeds:
            assemblies_by_source: OrderedDict[DatasetId, PrincipalActionAssembly] = OrderedDict()
            proposals: list[SourceProposal] = []
            for source in candidate_sources:
                source_materialized = materialized(source)
                if source_materialized is None:
                    continue
                pair_direction = directed_pair_name(source, target)
                if not _assess_or_persist_cross_pair_ineligibility(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,),
                ):
                    continue
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
                solution = solve_robust_action(assembly.problem)
                assemblies_by_source[source] = assembly
                proposals.append(
                    SourceProposal(SourceClientName(source.value), solution.certified_robust_value)
                )
            if not proposals:
                continue
            ranked = rank_source_proposals(proposals)

            def confirmation_decision(
                proposal: SourceProposal,
                *,
                target: DatasetId = target,
                seed: RandomSeed = seed,
                assemblies_by_source: Mapping[
                    DatasetId, PrincipalActionAssembly
                ] = assemblies_by_source,
            ) -> bool:
                source = DatasetId(proposal.source_client_name)
                assembly = assemblies_by_source[source]
                multipliers = curriculum_multipliers_from_action(
                    assembly.action, assembly.blocks, assembly.target_eligible, assembly.n_classes
                )
                optimizer = make_adamw(
                    assembly.model,
                    assembly.checkpoint.selected_hyperparameters.learning_rate,
                    assembly.checkpoint.selected_hyperparameters.weight_decay,
                )
                pre_confirm = capture_pre_confirm_pair(assembly.model, optimizer)
                contrast_coordinates = ContrastCoordinates(
                    f"multi-source-selection:{source.value}-to-{target.value}:{seed}"
                )
                optimizer_step_ledger = TargetOptimizerStepLedger(
                    TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                    directed_pair_name(source, target),
                    seed,
                )
                verdict = run_proposal_confirmation(
                    ConfirmationRequest(
                        assembly.model,
                        pre_confirm.baseline,
                        pre_confirm.curriculum,
                        assembly.train.features,
                        assembly.train.targets,
                        assembly.confirm.features,
                        assembly.confirm.targets,
                        assembly.checkpoint.train_class_weights,
                        multipliers,
                        assembly.checkpoint.selected_hyperparameters,
                        seed,
                        contrast_coordinates,
                    ),
                    optimizer_step_ledger,
                )
                if verdict.accepted:
                    apply_accepted_assimilation(
                        assembly.model,
                        optimizer,
                        pre_confirm.curriculum,
                        assembly.train.features,
                        assembly.train.targets,
                        assembly.checkpoint.train_class_weights,
                        multipliers,
                        seed,
                        AssimilationCoordinates(
                            target_client=SourceClientName(target.value),
                            directed_pair=directed_pair_name(source, target),
                            condition=PRINCIPAL_EVALUATION_CONDITION.name,
                            seed=seed,
                            clean_pretransfer_checkpoint_artifact_id=assembly.checkpoint_artifact_id,
                            source_packet_artifact_id=assembly.first_packet_artifact_id,
                            action_artifact_sha256=action_sha256(assembly.action),
                        ),
                        optimizer_step_ledger,
                    )
                else:
                    settle_rejected_proposal(assembly.model, optimizer, pre_confirm.baseline)
                return verdict.accepted

            decision = select_source_sequentially(ranked, confirmation_decision)
            accepted_source = (
                assemblies_by_source[DatasetId(decision.accepted_proposal.source_client_name)]
                if decision.accepted_proposal is not None
                else None
            )
            input_artifact_ids = (
                accepted_source.input_artifact_ids
                if accepted_source is not None
                else next(iter(assemblies_by_source.values())).input_artifact_ids
            )
            persist_synthetic_diagnostic_metric(
                store,
                layout,
                request.experiment,
                EvaluationConditionName(target.value),
                seed,
                MetricId.PROPOSAL_ACCEPTANCE_RATE,
                1.0 if decision.accepted_proposal is not None else 0.0,
                MetricUnit.FRACTION,
                MetricDirection.DESCRIPTIVE,
                input_artifact_ids,
                request.overwrite_policy,
            )


def execute_semantic_sufficiency_frontier(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = execution_device()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.semantic_sufficiency_frontier
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    authorize_oracle_access(
        request.experiment,
        tuple(
            method
            for method in request.definition.methods
            if isinstance(method, (TransferMethod, ExperimentLocalMethod))
        ),
    )
    scorers = (
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, solve_fedorbit_exact_sparse_action),
        (TransferMethod.MATCHED_RESOURCE_RECTANGULAR, solve_matched_resource_rectangular_action),
        (TransferMethod.EXACT_MAP_ORACLE, solve_exact_map_oracle_action),
    )
    for partition in config.partitions:
        fine_singleton = partition == SemanticPartitionId.ORACLE_FINE_SINGLETON_GROUPS
        bucket_of = None if fine_singleton else semantic_partition_bucket_of(partition)
        if not fine_singleton and bucket_of is None:
            raise ExecutionError(f"semantic sufficiency partition is not registered: {partition!r}")
        condition = semantic_partition_label(partition)
        for directed_pair in primary_pairs:
            source = directed_pair.source
            target = directed_pair.target
            source_materialized = materialized(source)
            target_materialized = materialized(target)
            if source_materialized is None or target_materialized is None:
                continue
            pair_direction = directed_pair_name(source, target)
            for seed in confirmatory_seeds:
                if not _assess_or_persist_cross_pair_ineligibility(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    tuple(method for method, _solve_action in scorers),
                    condition,
                ):
                    continue
                for method, solve_action in scorers:
                    certified_value_sink: MutableCell[Score] = MutableCell()
                    action_sink: MutableCell[CurriculumAction] = MutableCell()
                    confirmation_verdict_sink: MutableCell[bool] = MutableCell()
                    scored = score_robust_action_cell(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        target_materialized,
                        seed,
                        device,
                        FilesystemSlug("semantic-sufficiency-frontier"),
                        method,
                        functools.partial(solve_action, certified_value_sink=certified_value_sink),
                        None,
                        bucket_of,
                        action_sink=action_sink,
                        confirmation_verdict_sink=confirmation_verdict_sink,
                        fine_singleton=fine_singleton,
                    )
                    if scored is None:
                        continue
                    score, n_classes, input_artifact_ids = scored
                    persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                        condition,
                    )
                    persist_boundary_diagnostic_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        input_artifact_ids,
                        condition,
                        certified_value_sink.value,
                        action_sink.value,
                        confirmation_verdict_sink.value,
                    )


def execute_weak_signal_support_and_heterogeneity_boundaries(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = execution_device()
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.weak_signal_support_and_heterogeneity_boundaries
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    conditions: list[
        tuple[EvaluationConditionName, WeakSignalPerturbation | None, SupportCount | None]
    ] = [(PRINCIPAL_EVALUATION_CONDITION.name, None, None)]
    for scale in config.response_scales:
        if scale == config.baseline_response_scale:
            continue
        conditions.append(
            (
                EvaluationConditionName(f"{WeakSignalBoundaryDimension.RESPONSE_SCALE}-{scale}"),
                response_scale_perturbation(scale),
                None,
            )
        )
    for multiplier in config.ci_half_width_multipliers:
        if multiplier == config.baseline_ci_half_width_multiplier:
            continue
        conditions.append(
            (
                EvaluationConditionName(
                    f"{WeakSignalBoundaryDimension.CI_HALF_WIDTH}-{multiplier}"
                ),
                ci_half_width_perturbation(multiplier),
                None,
            )
        )
    for multiplier in config.response_heterogeneity_multipliers:
        if multiplier == config.baseline_response_heterogeneity_multiplier:
            continue
        conditions.append(
            (
                EvaluationConditionName(
                    f"{WeakSignalBoundaryDimension.RESPONSE_HETEROGENEITY}-{multiplier}"
                ),
                response_heterogeneity_perturbation(multiplier),
                None,
            )
        )
    for support in config.support_budgets:
        if support == config.baseline_support_budget:
            continue
        conditions.append(
            (
                EvaluationConditionName(f"{WeakSignalBoundaryDimension.SUPPORT_BUDGET}-{support}"),
                None,
                support,
            )
        )
    target_support_condition_count = sum(
        fraction != config.baseline_target_usable_support_fraction
        for fraction in config.target_usable_support_fractions
    )
    if len(conditions) + target_support_condition_count != config.distinct_condition_count():
        raise ExecutionError("weak-signal condition grid does not match its registered contract")

    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = directed_pair.direction
        for seed in confirmatory_seeds:
            local_only = score_local_only_cell_adapter(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
            )
            for condition_name, perturb, support_limit in conditions:
                exact_sparse_solve = (
                    solve_fedorbit_exact_sparse_action_at_support(support_limit)
                    if support_limit is not None
                    else solve_fedorbit_exact_sparse_action
                )
                condition = condition_name
                cross_pair_eligible = _assess_or_persist_cross_pair_ineligibility(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    (
                        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                    ),
                    condition,
                )
                for method, solve_action in (
                    (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, exact_sparse_solve),
                    (
                        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                        solve_matched_resource_rectangular_action,
                    ),
                ):
                    if not cross_pair_eligible:
                        continue
                    certified_value_sink: MutableCell[Score] = MutableCell()
                    action_sink: MutableCell[CurriculumAction] = MutableCell()
                    confirmation_verdict_sink: MutableCell[bool] = MutableCell()
                    scored = score_robust_action_cell(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        target_materialized,
                        seed,
                        device,
                        FilesystemSlug("weak-signal-boundaries"),
                        method,
                        functools.partial(solve_action, certified_value_sink=certified_value_sink),
                        None,
                        None,
                        perturb,
                        action_sink=action_sink,
                        confirmation_verdict_sink=confirmation_verdict_sink,
                    )
                    if scored is None:
                        persist_ineligible_transfer_cell(
                            store,
                            layout,
                            request.experiment,
                            pair_direction,
                            source,
                            target,
                            method,
                            seed,
                            request.overwrite_policy,
                            condition,
                        )
                        continue
                    score, n_classes, input_artifact_ids = scored
                    persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                        condition,
                    )
                    persist_boundary_diagnostic_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        input_artifact_ids,
                        condition,
                        certified_value_sink.value,
                        action_sink.value,
                        confirmation_verdict_sink.value,
                    )
                if local_only is not None:
                    score, n_classes, input_artifact_ids = local_only
                    persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        TransferMethod.LOCAL_ONLY,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                        condition,
                    )

    for fraction in config.target_usable_support_fractions:
        if fraction == config.baseline_target_usable_support_fraction:
            continue
        condition = EvaluationConditionName(
            f"{WeakSignalBoundaryDimension.TARGET_USABLE_SUPPORT_FRACTION}-{fraction}"
        )
        for directed_pair in primary_pairs:
            source = directed_pair.source
            target = directed_pair.target
            source_materialized = materialized(source)
            target_materialized = materialized(target)
            if source_materialized is None or target_materialized is None:
                continue
            pair_direction = directed_pair_name(source, target)
            subsample_seed: RandomSeed = derive_seed32(
                SeedDerivationRequest(
                    confirmatory_seeds[0],
                    RngNamespace.SYNTHETIC_INSTANCE,
                    cast(
                        StableJsonPayload,
                        OrderedDict(target=target.value, fraction=fraction),
                    ),
                )
            )
            subsampled_target = subsampled_materialized_client(
                target_materialized, fraction, subsample_seed
            )
            execute_client_base_model_pilot(
                store,
                layout,
                request.experiment,
                experiment_relevance(request.experiment),
                target,
                subsampled_target,
                confirmatory_seeds,
                request.overwrite_policy,
                device,
                execution_logger(),
            )
            for seed in confirmatory_seeds:
                local_only = score_local_only_cell(
                    store, layout, target, subsampled_target, seed, device, request.experiment
                )
                cross_pair_eligible = _assess_or_persist_cross_pair_ineligibility(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    source_materialized,
                    subsampled_target,
                    seed,
                    (
                        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                    ),
                    condition,
                )
                for method, solve_action in (
                    (
                        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                        solve_fedorbit_exact_sparse_action,
                    ),
                    (
                        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                        solve_matched_resource_rectangular_action,
                    ),
                ):
                    if not cross_pair_eligible:
                        continue
                    certified_value_sink: MutableCell[Score] = MutableCell()
                    action_sink: MutableCell[CurriculumAction] = MutableCell()
                    confirmation_verdict_sink: MutableCell[bool] = MutableCell()
                    scored = score_robust_action_cell(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        subsampled_target,
                        seed,
                        device,
                        FilesystemSlug("weak-signal-boundaries"),
                        method,
                        functools.partial(solve_action, certified_value_sink=certified_value_sink),
                        None,
                        None,
                        None,
                        request.experiment,
                        action_sink=action_sink,
                        confirmation_verdict_sink=confirmation_verdict_sink,
                    )
                    if scored is None:
                        continue
                    score, n_classes, input_artifact_ids = scored
                    persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        score,
                        n_classes,
                        input_artifact_ids,
                        condition,
                    )
                    persist_boundary_diagnostic_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        method,
                        seed,
                        input_artifact_ids,
                        condition,
                        certified_value_sink.value,
                        action_sink.value,
                        confirmation_verdict_sink.value,
                    )
                if local_only is not None:
                    score, n_classes, artifact_id = local_only
                    persist_primary_transfer_cell_metrics(
                        store,
                        layout,
                        request,
                        pair_direction,
                        source,
                        target,
                        TransferMethod.LOCAL_ONLY,
                        seed,
                        score,
                        n_classes,
                        (artifact_id,),
                        condition,
                    )
