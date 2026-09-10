from __future__ import annotations

import contextlib
import functools
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import cast

import torch

from fedorbit.analysis.metrics import harm_indicator
from fedorbit.analysis.records import (
    MetricDirection,
)
from fedorbit.config.loading import active_config, raw_dataset_root
from fedorbit.datasets.materialization import (
    MaterializationError,
    MaterializedClient,
    subsampled_materialized_client,
)
from fedorbit.experiments.cells import experiment_relevance
from fedorbit.experiments.protocol import ExperimentExecutionRequest
from fedorbit.experiments.scoring import (
    PrincipalActionAssembly,
    _action_sha256,
    _assemble_principal_action,
    _ci_half_width_perturbation,
    _persist_boundary_diagnostic_metrics,
    _persist_primary_transfer_cell_metrics,
    _response_heterogeneity_perturbation,
    _response_scale_perturbation,
    _score_coarse_block_mean_cell,
    _score_coarse_block_min_cell,
    _score_coupling_destroyed_fedorbit_cell,
    _score_exact_map_oracle_cell,
    _score_fedorbit_exact_sparse_solver_cell,
    _score_fedorbit_with_confirmation_verdict_cell,
    _score_fedorbit_without_confirmation_cell,
    _score_generic_exact_qap_cell,
    _score_local_only_cell,
    _score_local_only_cell_adapter,
    _score_local_sir_cell,
    _score_local_sir_cell_adapter,
    _score_matched_resource_rectangular_cell,
    _score_orbit_mean_cell,
    _score_point_correspondence_commitment_cell,
    _score_robust_action_cell,
    _semantic_partition_bucket_of,
    _semantic_partition_label,
    _solve_dense_ccp_fallback_action,
    _solve_exact_map_oracle_action,
    _solve_fedorbit_exact_sparse_action,
    _solve_fedorbit_exact_sparse_action_at_support,
    _solve_matched_resource_rectangular_action,
    _WeakSignalPerturbation,
    curriculum_multipliers_from_action,
    persist_ineligible_transfer_cell,
    persist_primary_transfer_metric,
)
from fedorbit.experiments.solvers import persist_synthetic_diagnostic_metric
from fedorbit.experiments.training import _execute_client_base_model_pilot
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
    ExecutionError,
)
from fedorbit.infrastructure.preparation import (
    _load_or_materialize_client,
)
from fedorbit.infrastructure.runtime import (
    RandomSeed,
    SeedDerivationRequest,
    derive_seed32,
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
from fedorbit.types import (
    ArtifactIdentifier,
    ClassCount,
    ContrastCoordinates,
    DatasetId,
    DirectedPairName,
    EvaluationConditionName,
    MetricId,
    MetricUnit,
    ProducerModuleName,
    RngNamespace,
    Score,
    SemanticPartitionId,
    SourceClientName,
    StableJsonPayload,
    SupportCount,
    TransferMethod,
)

_MODULE_NAME = ProducerModuleName("fedorbit.experiments.transfer")
_PRINCIPAL_CONDITION = EvaluationConditionName("principal")


def execute_primary_strict_cross_telemetry_transfer(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_target: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()
    for directed_pair in primary_pairs:
        target = directed_pair.target
        if target not in materialized_by_target:
            try:
                materialized_by_target[target] = _load_or_materialize_client(
                    target, raw_root, layout
                )
            except MaterializationError:
                continue
        materialized = materialized_by_target[target]
        source = directed_pair.source
        if source not in materialized_by_target:
            with contextlib.suppress(MaterializationError):
                materialized_by_target[source] = _load_or_materialize_client(
                    source, raw_root, layout
                )
        source_materialized = materialized_by_target.get(source)
        pair_direction = DirectedPairName(
            f"{directed_pair.source.value} -> {directed_pair.target.value}"
        )
        for seed in confirmatory_seeds:
            local_only = _score_local_only_cell(store, layout, target, materialized, seed, device)
            if local_only is not None:
                score, n_classes, checkpoint_artifact_id = local_only
                _persist_primary_transfer_cell_metrics(
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
            local_sir = _score_local_sir_cell(store, layout, target, materialized, seed, device)
            if local_sir is not None:
                score, n_classes, input_artifact_ids = local_sir
                _persist_primary_transfer_cell_metrics(
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
                matched_resource_rectangular = _score_matched_resource_rectangular_cell(
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
                    _persist_primary_transfer_cell_metrics(
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
                point_correspondence = _score_point_correspondence_commitment_cell(
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
                    _persist_primary_transfer_cell_metrics(
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
                fedorbit_exact_sparse = _score_fedorbit_exact_sparse_solver_cell(
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
                    _persist_primary_transfer_cell_metrics(
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
                generic_exact_qap = _score_generic_exact_qap_cell(
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
                    _persist_primary_transfer_cell_metrics(
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
                exact_map_oracle = _score_exact_map_oracle_cell(
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
                    _persist_primary_transfer_cell_metrics(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = _load_or_materialize_client(
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
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, _score_fedorbit_exact_sparse_solver_cell),
        (TransferMethod.MATCHED_RESOURCE_RECTANGULAR, _score_matched_resource_rectangular_cell),
        (
            TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
            _score_point_correspondence_commitment_cell,
        ),
        (TransferMethod.COUPLING_DESTROYED_FEDORBIT, _score_coupling_destroyed_fedorbit_cell),
        (TransferMethod.COARSE_BLOCK_MEAN, _score_coarse_block_mean_cell),
        (TransferMethod.COARSE_BLOCK_MIN, _score_coarse_block_min_cell),
        (TransferMethod.ORBIT_MEAN, _score_orbit_mean_cell),
        (TransferMethod.LOCAL_SIR, _score_local_sir_cell_adapter),
    )
    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            for method, scorer in scorers:
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
                if scored is None:
                    continue
                score, n_classes, input_artifact_ids = scored
                _persist_primary_transfer_cell_metrics(
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


def _persist_confirmation_safety_indicators(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
    pair_direction: DirectedPairName,
    source: DatasetId,
    target: DatasetId,
    seed: RandomSeed,
    local_score: ScoreArtifact,
    with_score: ScoreArtifact,
    without_score: ScoreArtifact | None,
    verdicts: list[ConfirmationVerdict],
    input_artifact_ids: tuple[ArtifactIdentifier, ...],
) -> None:
    materiality = active_config().scientific.materiality
    local_ce = float(local_score.macro_cross_entropy.value)
    with_ce = float(with_score.macro_cross_entropy.value)
    with_gain = 0.0 if local_ce == 0.0 else (local_ce - with_ce) / local_ce
    accepted = bool(verdicts and verdicts[-1].accepted)
    harmful = harm_indicator(with_gain, materiality.harmful_transfer_relative_macro_ce_gain)
    useful = with_gain >= materiality.useful_transfer_relative_macro_ce_gain
    indicators: list[tuple[MetricId, float]] = [
        (MetricId.HARMFUL_ACCEPTED_RATE, 1.0 if accepted and harmful else 0.0),
        (MetricId.USEFUL_ACCEPTED_RATE, 1.0 if accepted and useful else 0.0),
        (MetricId.BENEFICIAL_REJECTED_RATE, 1.0 if (not accepted) and useful else 0.0),
        (MetricId.COVERAGE_CONFIRM, 1.0 if accepted else 0.0),
        (MetricId.HARM_RATE_CONFIRM, 1.0 if harmful else 0.0),
    ]
    if without_score is not None:
        without_ce = float(without_score.macro_cross_entropy.value)
        without_gain = 0.0 if local_ce == 0.0 else (local_ce - without_ce) / local_ce
        without_harmful = harm_indicator(
            without_gain, materiality.harmful_transfer_relative_macro_ce_gain
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
            MetricUnit("fraction"),
            MetricDirection.DESCRIPTIVE,
            input_artifact_ids,
            request.overwrite_policy,
        )


def execute_target_confirmation_and_portability(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    directed_pairs = (
        *active_config().scientific.datasets.primary_directed_pairs,
        *active_config().scientific.datasets.secondary_directed_pairs,
    )
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = _load_or_materialize_client(
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
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            local_only = _score_local_only_cell(
                store, layout, target, target_materialized, seed, device
            )
            if local_only is not None:
                score, n_classes, checkpoint_id = local_only
                _persist_primary_transfer_cell_metrics(
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
                    (checkpoint_id,),
                )
            verdicts: list[ConfirmationVerdict] = []
            with_confirmation = _score_fedorbit_with_confirmation_verdict_cell(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
                verdicts,
            )
            if with_confirmation is not None:
                score, n_classes, input_artifact_ids = with_confirmation
                _persist_primary_transfer_cell_metrics(
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
                        MetricUnit("fraction"),
                        MetricDirection.DESCRIPTIVE,
                        input_artifact_ids,
                        request.overwrite_policy,
                    )
            without_confirmation = _score_fedorbit_without_confirmation_cell(
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
                _persist_primary_transfer_cell_metrics(
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
            if local_only is not None and with_confirmation is not None:
                _persist_confirmation_safety_indicators(
                    store,
                    layout,
                    request,
                    pair_direction,
                    source,
                    target,
                    seed,
                    local_only[0],
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    secondary_pairs = active_config().scientific.datasets.secondary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = _load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    scorers = (
        (TransferMethod.LOCAL_ONLY, _score_local_only_cell_adapter),
        (TransferMethod.LOCAL_SIR, _score_local_sir_cell_adapter),
        (TransferMethod.MATCHED_RESOURCE_RECTANGULAR, _score_matched_resource_rectangular_cell),
        (
            TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
            _score_point_correspondence_commitment_cell,
        ),
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, _score_fedorbit_exact_sparse_solver_cell),
    )
    for directed_pair in secondary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            for method, scorer in scorers:
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
                if scored is None:
                    continue
                score, n_classes, input_artifact_ids = scored
                _persist_primary_transfer_cell_metrics(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = _load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    conditions: tuple[
        tuple[
            str,
            TransferMethod,
            Callable[[RobustActionProblem, RandomSeed], CurriculumAction | None],
        ],
        ...,
    ] = (
        (
            "exact sparse s=1",
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            _solve_fedorbit_exact_sparse_action_at_support(1),
        ),
        (
            "exact sparse s=2",
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            _solve_fedorbit_exact_sparse_action_at_support(2),
        ),
        (
            "exact sparse s=3",
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            _solve_fedorbit_exact_sparse_action_at_support(3),
        ),
        ("dense CCP", TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK, _solve_dense_ccp_fallback_action),
    )
    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            for condition_label, method, solve_action in conditions:
                scored = _score_robust_action_cell(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    device,
                    "sparsity-and-dense-fallback",
                    solve_action,
                )
                if scored is None:
                    continue
                score, n_classes, input_artifact_ids = scored
                _persist_primary_transfer_cell_metrics(
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
                    EvaluationConditionName(condition_label),
                )


def execute_real_packet_coupling_mechanism_validation(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = _load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            assembly = _assemble_principal_action(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
                _solve_fedorbit_exact_sparse_action,
            )
            if assembly is None:
                continue
            orbit = tuple(enumerate_block_permutations(assembly.blocks))
            hull = build_rectangular_hull(
                assembly.blocks,
                assembly.problem.lower_response_matrix,
                assembly.problem.upper_response_matrix,
            )
            candidates = (assembly.action, zero_action(assembly.problem))
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
            ):
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
                    MetricUnit("score"),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.multi_source_selection_validation
    all_clients = active_config().scientific.datasets.clients
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = _load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    for target in config.targets:
        target_materialized = materialized(target)
        if target_materialized is None:
            continue
        candidate_sources = tuple(client for client in all_clients if client != target)
        for seed in confirmatory_seeds:
            assemblies_by_source: OrderedDict[DatasetId, PrincipalActionAssembly] = OrderedDict()
            proposals: list[SourceProposal] = []
            for source in candidate_sources:
                source_materialized = materialized(source)
                if source_materialized is None:
                    continue
                assembly = _assemble_principal_action(
                    store,
                    layout,
                    source,
                    target,
                    source_materialized,
                    target_materialized,
                    seed,
                    device,
                    _solve_fedorbit_exact_sparse_action,
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
                    )
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
                            directed_pair=DirectedPairName(f"{source.value} -> {target.value}"),
                            condition=EvaluationConditionName("principal"),
                            seed=seed,
                            clean_pretransfer_checkpoint_artifact_id=assembly.checkpoint_artifact_id,
                            source_packet_artifact_id=assembly.first_packet_artifact_id,
                            action_artifact_sha256=_action_sha256(assembly.action),
                        ),
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
                MetricUnit("fraction"),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.semantic_sufficiency_frontier
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = _load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    scorers = (
        (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, _solve_fedorbit_exact_sparse_action),
        (TransferMethod.MATCHED_RESOURCE_RECTANGULAR, _solve_matched_resource_rectangular_action),
        (TransferMethod.EXACT_MAP_ORACLE, _solve_exact_map_oracle_action),
    )
    for partition in config.partitions:
        fine_singleton = partition == SemanticPartitionId.ORACLE_FINE_SINGLETON_GROUPS
        bucket_of = None if fine_singleton else _semantic_partition_bucket_of(partition)
        if not fine_singleton and bucket_of is None:
            raise ExecutionError(f"semantic sufficiency partition is not registered: {partition!r}")
        condition = EvaluationConditionName(_semantic_partition_label(partition))
        for directed_pair in primary_pairs:
            source = directed_pair.source
            target = directed_pair.target
            source_materialized = materialized(source)
            target_materialized = materialized(target)
            if source_materialized is None or target_materialized is None:
                continue
            pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
            for seed in confirmatory_seeds:
                for method, solve_action in scorers:
                    certified_value_sink: list[Score] = []
                    action_sink: list[CurriculumAction] = []
                    confirmation_verdict_sink: list[bool] = []
                    scored = _score_robust_action_cell(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        target_materialized,
                        seed,
                        device,
                        "semantic-sufficiency-frontier",
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
                    _persist_primary_transfer_cell_metrics(
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
                    _persist_boundary_diagnostic_metrics(
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
                        certified_value_sink[0] if certified_value_sink else None,
                        action_sink[0] if action_sink else None,
                        confirmation_verdict_sink[0] if confirmation_verdict_sink else None,
                    )


def execute_weak_signal_support_and_heterogeneity_boundaries(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    request: ExperimentExecutionRequest,
) -> None:
    raw_root = raw_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmatory_seeds = active_config().scientific.randomness.confirmatory_seeds
    config = active_config().experiments.weak_signal_support_and_heterogeneity_boundaries
    primary_pairs = active_config().scientific.datasets.primary_directed_pairs
    materialized_by_dataset: OrderedDict[DatasetId, MaterializedClient] = OrderedDict()

    def materialized(dataset: DatasetId) -> MaterializedClient | None:
        if dataset not in materialized_by_dataset:
            with contextlib.suppress(MaterializationError):
                materialized_by_dataset[dataset] = _load_or_materialize_client(
                    dataset, raw_root, layout
                )
        return materialized_by_dataset.get(dataset)

    conditions: list[tuple[str, _WeakSignalPerturbation | None, SupportCount | None]] = []
    for scale in config.response_scales:
        conditions.append((f"response-scale-{scale}", _response_scale_perturbation(scale), None))
    for multiplier in config.ci_half_width_multipliers:
        conditions.append(
            (f"ci-half-width-{multiplier}", _ci_half_width_perturbation(multiplier), None)
        )
    for multiplier in config.response_heterogeneity_multipliers:
        conditions.append(
            (
                f"response-heterogeneity-{multiplier}",
                _response_heterogeneity_perturbation(multiplier),
                None,
            )
        )
    for support in config.support_budgets:
        conditions.append((f"support-budget-{support}", None, support))

    for directed_pair in primary_pairs:
        source = directed_pair.source
        target = directed_pair.target
        source_materialized = materialized(source)
        target_materialized = materialized(target)
        if source_materialized is None or target_materialized is None:
            continue
        pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
        for seed in confirmatory_seeds:
            local_only = _score_local_only_cell_adapter(
                store,
                layout,
                source,
                target,
                source_materialized,
                target_materialized,
                seed,
                device,
            )
            for condition_label, perturb, support_limit in conditions:
                exact_sparse_solve = (
                    _solve_fedorbit_exact_sparse_action_at_support(support_limit)
                    if support_limit is not None
                    else _solve_fedorbit_exact_sparse_action
                )
                condition = EvaluationConditionName(condition_label)
                for method, solve_action in (
                    (TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, exact_sparse_solve),
                    (
                        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                        _solve_matched_resource_rectangular_action,
                    ),
                ):
                    certified_value_sink: list[Score] = []
                    action_sink: list[CurriculumAction] = []
                    confirmation_verdict_sink: list[bool] = []
                    scored = _score_robust_action_cell(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        target_materialized,
                        seed,
                        device,
                        "weak-signal-boundaries",
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
                    _persist_primary_transfer_cell_metrics(
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
                    _persist_boundary_diagnostic_metrics(
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
                        certified_value_sink[0] if certified_value_sink else None,
                        action_sink[0] if action_sink else None,
                        confirmation_verdict_sink[0] if confirmation_verdict_sink else None,
                    )
                if local_only is not None:
                    score, n_classes, input_artifact_ids = local_only
                    _persist_primary_transfer_cell_metrics(
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
        condition = EvaluationConditionName(f"target-usable-support-fraction-{fraction}")
        for directed_pair in primary_pairs:
            source = directed_pair.source
            target = directed_pair.target
            source_materialized = materialized(source)
            target_materialized = materialized(target)
            if source_materialized is None or target_materialized is None:
                continue
            pair_direction = DirectedPairName(f"{source.value} -> {target.value}")
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
            _execute_client_base_model_pilot(
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
                local_only = _score_local_only_cell(
                    store, layout, target, subsampled_target, seed, device, request.experiment
                )
                for method, solve_action in (
                    (
                        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                        _solve_fedorbit_exact_sparse_action,
                    ),
                    (
                        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
                        _solve_matched_resource_rectangular_action,
                    ),
                ):
                    certified_value_sink: list[Score] = []
                    action_sink: list[CurriculumAction] = []
                    confirmation_verdict_sink: list[bool] = []
                    scored = _score_robust_action_cell(
                        store,
                        layout,
                        source,
                        target,
                        source_materialized,
                        subsampled_target,
                        seed,
                        device,
                        "weak-signal-boundaries",
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
                    _persist_primary_transfer_cell_metrics(
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
                    _persist_boundary_diagnostic_metrics(
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
                        certified_value_sink[0] if certified_value_sink else None,
                        action_sink[0] if action_sink else None,
                        confirmation_verdict_sink[0] if confirmation_verdict_sink else None,
                    )
                if local_only is not None:
                    score, n_classes, artifact_id = local_only
                    _persist_primary_transfer_cell_metrics(
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
