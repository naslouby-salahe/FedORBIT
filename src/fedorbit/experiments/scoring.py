from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Mapping

import torch

from fedorbit.analysis.records import (
    MetricDirection,
)
from fedorbit.config.loading import active_config
from fedorbit.datasets.materialization import (
    MaterializedClient,
    SplitTensors,
)
from fedorbit.experiments.assembly import (
    assemble_cross_client_response_matrix,
    assemble_principal_action,
    assemble_self_response_matrix,
    assemble_target_response_matrix,
    common_eligible_groups,
    cross_client_padded_blocks,
    curriculum_multipliers_from_action,
    eligible_groups_by_coarse,
    load_dataset_source_packet,
    pair_seed_structure,
    persist_or_reuse_target_importance,
    resolve_checkpoint_artifact_id,
    self_padded_blocks,
    strict_pair_resource_validity,
    target_confirmatory_checkpoint_path,
    target_node_risks,
)
from fedorbit.experiments.catalogue import method_resource_manifest
from fedorbit.experiments.metric_persistence import (
    persist_primary_transfer_metric,
)
from fedorbit.infrastructure.artifacts import (
    ArtifactStore,
)
from fedorbit.infrastructure.runtime import (
    RandomSeed,
    execution_logger,
)
from fedorbit.infrastructure.workspace import (
    WorkspaceLayout,
)
from fedorbit.interface import (
    AccessLogger,
    ResourceKind,
    validate_dynamic_access_log_scan,
)
from fedorbit.learning.checkpoints import load_base_checkpoint
from fedorbit.learning.pilot import (
    create_classifier,
)
from fedorbit.learning.scoring import LocalClassCount, ScoreArtifact, ScoringRequest, score_model
from fedorbit.learning.training import (
    BaseCheckpoint,
    make_adamw,
)
from fedorbit.methods.assimilation import (
    AssimilationCoordinates,
    ConfirmationRequest,
    ConfirmationVerdict,
    PreTestLifecycle,
    PreTestPhase,
    apply_accepted_assimilation,
    capture_pre_confirm_pair,
    run_proposal_confirmation,
    settle_rejected_proposal,
)
from fedorbit.methods.baselines import (
    coarse_block_mean_matrix,
    coarse_block_min_matrix,
    coupling_destroyed_matrices,
    local_sir_action,
    optimize_against_fixed_matrix,
    orbit_mean_matrix,
)
from fedorbit.methods.target import (
    CurriculumMultipliers,
    TargetOptimizerBudgetCategory,
    TargetOptimizerStepLedger,
)
from fedorbit.optimization.certificates import (
    build_rectangular_hull,
)
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
    ResponseMatrix,
)
from fedorbit.optimization.dense_ccp import solve_dense_ccp
from fedorbit.optimization.exact_qap import (
    point_correspondence_commitment,
    solve_robust_action_qap,
)
from fedorbit.optimization.exact_sparse import (
    solve_robust_action,
)
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    build_robust_action_problem,
)
from fedorbit.response.packet import (
    SourcePacket,
)
from fedorbit.types import (
    PRINCIPAL_EVALUATION_CONDITION,
    ArtifactIdentifier,
    ClassCount,
    ClientRole,
    CoarseGroup,
    ContrastCoordinates,
    DatasetId,
    ExecutionEventName,
    ExperimentName,
    FilesystemSlug,
    MetricId,
    MetricUnit,
    MutableCell,
    OverwritePolicy,
    Score,
    SemanticCoordinates,
    Sha256Digest,
    SourceClientName,
    Split,
    SupportCount,
    TransferMethod,
    directed_pair_name,
)


def score_local_only_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    target: DatasetId,
    materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
    checkpoint_source_experiment: ExperimentName = ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT,
) -> tuple[ScoreArtifact, ClassCount, ArtifactIdentifier] | None:
    checkpoint_path = target_confirmatory_checkpoint_path(
        layout, target, seed, checkpoint_source_experiment
    )
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = resolve_checkpoint_artifact_id(store, checkpoint_path)
    if checkpoint_artifact_id is None:
        return None
    checkpoint = load_base_checkpoint(checkpoint_path)
    test = materialized.splits[Split.TEST]
    n_classes = materialized.class_manifest.class_count
    model = create_classifier(
        target,
        materialized.feature_count,
        n_classes,
        checkpoint.selected_hyperparameters.dropout_probability,
        seed,
        device,
    )
    checkpoint.state_dict.load_into(model)
    access = AccessLogger()
    access.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=True)
    validate_dynamic_access_log_scan(
        access.trace(), method_resource_manifest(TransferMethod.LOCAL_ONLY)
    )
    execution_logger().event(
        ExecutionEventName.SCORING_CELL,
        method=TransferMethod.LOCAL_ONLY.value,
        dataset=target.value,
        seed=seed,
    )
    score = score_model(
        ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
    )
    return score, n_classes, checkpoint_artifact_id


def _confirm_assimilate_and_score(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    checkpoint: BaseCheckpoint,
    train: SplitTensors,
    confirm: SplitTensors,
    test: SplitTensors,
    multipliers: CurriculumMultipliers,
    seed: RandomSeed,
    contrast_coordinates: ContrastCoordinates,
    assimilation_coordinates: AssimilationCoordinates,
    n_classes: ClassCount,
    method: TransferMethod,
    confirmation_verdict_sink: MutableCell[bool] | None = None,
) -> ScoreArtifact:
    access = AccessLogger()
    if method != TransferMethod.LOCAL_SIR:
        access.record(ClientRole.TARGET, ResourceKind.ANONYMOUS_SOURCE_PACKET)
    access.record(ClientRole.TARGET, ResourceKind.TRAIN)
    access.record(ClientRole.TARGET, ResourceKind.META)
    access.record(ClientRole.TARGET, ResourceKind.CONFIRM)
    pre_confirm = capture_pre_confirm_pair(model, optimizer)
    optimizer_step_ledger = TargetOptimizerStepLedger(
        method,
        assimilation_coordinates.directed_pair,
        seed,
    )
    verdict = run_proposal_confirmation(
        ConfirmationRequest(
            model,
            pre_confirm.baseline,
            pre_confirm.curriculum,
            train.features,
            train.targets,
            confirm.features,
            confirm.targets,
            checkpoint.train_class_weights,
            multipliers,
            checkpoint.selected_hyperparameters,
            seed,
            contrast_coordinates,
        ),
        optimizer_step_ledger,
    )
    if confirmation_verdict_sink is not None:
        confirmation_verdict_sink.value = verdict.accepted
    lifecycle = PreTestLifecycle()
    lifecycle.complete_phase(PreTestPhase.SOURCE_SELECTION_FINALIZED)
    lifecycle.complete_phase(PreTestPhase.ACTION_FINALIZED)
    lifecycle.complete_phase(PreTestPhase.CONFIRMATION_DECISION_FINALIZED)
    if verdict.accepted:
        apply_accepted_assimilation(
            model,
            optimizer,
            pre_confirm.curriculum,
            train.features,
            train.targets,
            checkpoint.train_class_weights,
            multipliers,
            seed,
            assimilation_coordinates,
            optimizer_step_ledger,
        )
    else:
        settle_rejected_proposal(model, optimizer, pre_confirm.baseline)
    lifecycle.complete_phase(PreTestPhase.ASSIMILATION_SETTLED)
    lifecycle.complete_phase(PreTestPhase.PRE_TEST_ARTIFACTS_COMMITTED)
    lifecycle.open_test()
    lifecycle.assert_opened()
    access.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=True)
    validate_dynamic_access_log_scan(access.trace(), method_resource_manifest(method))
    return score_model(
        ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
    )


def action_sha256(action: CurriculumAction) -> Sha256Digest:
    return Sha256Digest(
        hashlib.sha256(
            ",".join(f"{value:.17g}" for value in action.coordinates).encode()
        ).hexdigest()
    )


def score_local_sir_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    target: DatasetId,
    materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    checkpoint_path = target_confirmatory_checkpoint_path(layout, target, seed)
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = resolve_checkpoint_artifact_id(store, checkpoint_path)
    if checkpoint_artifact_id is None:
        return None
    eligible_by_coarse = eligible_groups_by_coarse(target, materialized, ClientRole.TARGET)
    if not eligible_by_coarse:
        return None
    blocks = self_padded_blocks(eligible_by_coarse)
    packets_by_coarse: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in eligible_by_coarse:
        packet = load_dataset_source_packet(layout, target, seed, coarse_group)
        if packet is not None:
            packets_by_coarse[coarse_group] = packet
    if not packets_by_coarse:
        return None
    response_matrix = assemble_self_response_matrix(blocks, packets_by_coarse)
    checkpoint = load_base_checkpoint(checkpoint_path)
    n_classes = materialized.class_manifest.class_count
    train = materialized.splits[Split.TRAIN]
    meta = materialized.splits[Split.META]
    confirm = materialized.splits[Split.CONFIRM]
    test = materialized.splits[Split.TEST]
    model = create_classifier(
        target,
        materialized.feature_count,
        n_classes,
        checkpoint.selected_hyperparameters.dropout_probability,
        seed,
        device,
    )
    checkpoint.state_dict.load_into(model)
    meta_score = score_model(
        ScoringRequest(model, meta.features, meta.targets, LocalClassCount(n_classes))
    )
    meta_class_ce = tuple(
        entry.value for entry in meta_score.class_conditional_cross_entropy.values
    )
    node_risks = target_node_risks(blocks, eligible_by_coarse, meta_class_ce)
    target_importance = persist_or_reuse_target_importance(
        store,
        layout,
        target,
        seed,
        node_risks,
        OverwritePolicy.REUSE,
        (checkpoint_artifact_id,),
    )
    if target_importance is None:
        return None
    actionable_nodes = tuple(risk.node_index for risk in node_risks if risk.is_actionable)
    problem = build_robust_action_problem(
        blocks,
        response_matrix,
        response_matrix,
        target_importance.as_vector(blocks.total_padded_nodes),
        actionable_nodes,
    )
    solution = local_sir_action(problem, response_matrix)
    action = solution.selected_action
    multipliers = curriculum_multipliers_from_action(action, blocks, eligible_by_coarse, n_classes)
    optimizer = make_adamw(
        model,
        checkpoint.selected_hyperparameters.learning_rate,
        checkpoint.selected_hyperparameters.weight_decay,
    )
    input_artifact_ids: tuple[ArtifactIdentifier, ...] = (
        checkpoint_artifact_id,
        *(
            ArtifactIdentifier(packet.packet_integrity_sha256)
            for packet in packets_by_coarse.values()
        ),
    )
    first_packet = next(iter(packets_by_coarse.values()))
    score = _confirm_assimilate_and_score(
        model,
        optimizer,
        checkpoint,
        train,
        confirm,
        test,
        multipliers,
        seed,
        ContrastCoordinates(f"local-sir:{target.value}:{seed}"),
        AssimilationCoordinates(
            target_client=SourceClientName(target.value),
            directed_pair=directed_pair_name(target, target),
            condition=PRINCIPAL_EVALUATION_CONDITION.name,
            seed=seed,
            clean_pretransfer_checkpoint_artifact_id=checkpoint_artifact_id,
            source_packet_artifact_id=ArtifactIdentifier(first_packet.packet_integrity_sha256),
            action_artifact_sha256=action_sha256(action),
        ),
        n_classes,
        TransferMethod.LOCAL_SIR,
    )
    return score, n_classes, input_artifact_ids


def score_matched_resource_rectangular_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    checkpoint_path = target_confirmatory_checkpoint_path(layout, target, seed)
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = resolve_checkpoint_artifact_id(store, checkpoint_path)
    if checkpoint_artifact_id is None:
        return None
    common = common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return None
    source_eligible, target_eligible = common
    common_coarse = tuple(target_eligible)
    packets_by_coarse: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in common_coarse:
        packet = load_dataset_source_packet(layout, source, seed, coarse_group)
        if packet is not None:
            packets_by_coarse[coarse_group] = packet
    pair_structure = pair_seed_structure(
        source_eligible,
        target_eligible,
        frozenset(packets_by_coarse),
        strict_pair_resource_validity(source_materialized, target_materialized),
    )
    if not pair_structure.is_eligible:
        return None
    source_eligible = pair_structure.source_eligible
    target_eligible = pair_structure.target_eligible
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    packets_by_coarse = OrderedDict(
        (coarse, packets_by_coarse[coarse]) for coarse in target_eligible
    )
    lower_matrix = assemble_cross_client_response_matrix(
        blocks, packets_by_coarse, SourcePacket.lower_matrix
    )
    upper_matrix = assemble_cross_client_response_matrix(
        blocks, packets_by_coarse, SourcePacket.upper_matrix
    )
    checkpoint = load_base_checkpoint(checkpoint_path)
    n_classes = target_materialized.class_manifest.class_count
    train = target_materialized.splits[Split.TRAIN]
    meta = target_materialized.splits[Split.META]
    confirm = target_materialized.splits[Split.CONFIRM]
    test = target_materialized.splits[Split.TEST]
    model = create_classifier(
        target,
        target_materialized.feature_count,
        n_classes,
        checkpoint.selected_hyperparameters.dropout_probability,
        seed,
        device,
    )
    checkpoint.state_dict.load_into(model)
    meta_score = score_model(
        ScoringRequest(model, meta.features, meta.targets, LocalClassCount(n_classes))
    )
    meta_class_ce = tuple(
        entry.value for entry in meta_score.class_conditional_cross_entropy.values
    )
    node_risks = target_node_risks(blocks, target_eligible, meta_class_ce)
    target_importance = persist_or_reuse_target_importance(
        store,
        layout,
        target,
        seed,
        node_risks,
        OverwritePolicy.REUSE,
        (checkpoint_artifact_id,),
    )
    if target_importance is None:
        return None
    actionable_nodes = tuple(risk.node_index for risk in node_risks if risk.is_actionable)
    problem = build_robust_action_problem(
        blocks,
        lower_matrix,
        upper_matrix,
        target_importance.as_vector(blocks.total_padded_nodes),
        actionable_nodes,
    )
    hull = build_rectangular_hull(blocks, lower_matrix, upper_matrix)
    solution = optimize_against_fixed_matrix(problem, hull.lower_bounds)
    action = solution.selected_action
    multipliers = curriculum_multipliers_from_action(action, blocks, target_eligible, n_classes)
    optimizer = make_adamw(
        model,
        checkpoint.selected_hyperparameters.learning_rate,
        checkpoint.selected_hyperparameters.weight_decay,
    )
    input_artifact_ids: tuple[ArtifactIdentifier, ...] = (
        checkpoint_artifact_id,
        *(
            ArtifactIdentifier(packet.packet_integrity_sha256)
            for packet in packets_by_coarse.values()
        ),
    )
    first_packet = next(iter(packets_by_coarse.values()))
    score = _confirm_assimilate_and_score(
        model,
        optimizer,
        checkpoint,
        train,
        confirm,
        test,
        multipliers,
        seed,
        ContrastCoordinates(
            f"matched-resource-rectangular:{source.value}-to-{target.value}:{seed}"
        ),
        AssimilationCoordinates(
            target_client=SourceClientName(target.value),
            directed_pair=directed_pair_name(source, target),
            condition=PRINCIPAL_EVALUATION_CONDITION.name,
            seed=seed,
            clean_pretransfer_checkpoint_artifact_id=checkpoint_artifact_id,
            source_packet_artifact_id=ArtifactIdentifier(first_packet.packet_integrity_sha256),
            action_artifact_sha256=action_sha256(action),
        ),
        n_classes,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
    )
    return score, n_classes, input_artifact_ids


def score_point_correspondence_commitment_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    checkpoint_path = target_confirmatory_checkpoint_path(layout, target, seed)
    if not checkpoint_path.is_file():
        return None
    checkpoint_artifact_id = resolve_checkpoint_artifact_id(store, checkpoint_path)
    if checkpoint_artifact_id is None:
        return None
    common = common_eligible_groups(source, target, source_materialized, target_materialized)
    if common is None:
        return None
    source_eligible, target_eligible = common
    common_coarse = tuple(target_eligible)
    source_packets: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    target_packets: OrderedDict[CoarseGroup, SourcePacket] = OrderedDict()
    for coarse_group in common_coarse:
        source_packet = load_dataset_source_packet(layout, source, seed, coarse_group)
        if source_packet is not None:
            source_packets[coarse_group] = source_packet
        target_packet = load_dataset_source_packet(layout, target, seed, coarse_group)
        if target_packet is not None:
            target_packets[coarse_group] = target_packet
    pair_structure = pair_seed_structure(
        source_eligible,
        target_eligible,
        frozenset(source_packets) & frozenset(target_packets),
        strict_pair_resource_validity(source_materialized, target_materialized),
    )
    if not pair_structure.is_eligible:
        return None
    source_eligible = pair_structure.source_eligible
    target_eligible = pair_structure.target_eligible
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    source_packets = OrderedDict((coarse, source_packets[coarse]) for coarse in target_eligible)
    target_packets = OrderedDict((coarse, target_packets[coarse]) for coarse in target_eligible)
    source_matrix = assemble_cross_client_response_matrix(
        blocks, source_packets, SourcePacket.lower_matrix
    )
    target_matrix = assemble_target_response_matrix(
        blocks, target_packets, SourcePacket.lower_matrix
    )
    qap_result = point_correspondence_commitment(source_matrix, target_matrix, blocks)
    if not qap_result.certified or qap_result.correspondence is None:
        return None
    committed_matrix = qap_result.correspondence.permute_response_matrix(source_matrix)
    checkpoint = load_base_checkpoint(checkpoint_path)
    n_classes = target_materialized.class_manifest.class_count
    train = target_materialized.splits[Split.TRAIN]
    meta = target_materialized.splits[Split.META]
    confirm = target_materialized.splits[Split.CONFIRM]
    test = target_materialized.splits[Split.TEST]
    model = create_classifier(
        target,
        target_materialized.feature_count,
        n_classes,
        checkpoint.selected_hyperparameters.dropout_probability,
        seed,
        device,
    )
    checkpoint.state_dict.load_into(model)
    meta_score = score_model(
        ScoringRequest(model, meta.features, meta.targets, LocalClassCount(n_classes))
    )
    meta_class_ce = tuple(
        entry.value for entry in meta_score.class_conditional_cross_entropy.values
    )
    node_risks = target_node_risks(blocks, target_eligible, meta_class_ce)
    target_importance = persist_or_reuse_target_importance(
        store,
        layout,
        target,
        seed,
        node_risks,
        OverwritePolicy.REUSE,
        (checkpoint_artifact_id,),
    )
    if target_importance is None:
        return None
    actionable_nodes = tuple(risk.node_index for risk in node_risks if risk.is_actionable)
    problem = build_robust_action_problem(
        blocks,
        committed_matrix,
        committed_matrix,
        target_importance.as_vector(blocks.total_padded_nodes),
        actionable_nodes,
    )
    solution = optimize_against_fixed_matrix(problem, committed_matrix)
    action = solution.selected_action
    multipliers = curriculum_multipliers_from_action(action, blocks, target_eligible, n_classes)
    optimizer = make_adamw(
        model,
        checkpoint.selected_hyperparameters.learning_rate,
        checkpoint.selected_hyperparameters.weight_decay,
    )
    input_artifact_ids: tuple[ArtifactIdentifier, ...] = (
        checkpoint_artifact_id,
        *(
            ArtifactIdentifier(packet.packet_integrity_sha256)
            for packet in (*source_packets.values(), *target_packets.values())
        ),
    )
    first_source_packet = next(iter(source_packets.values()))
    score = _confirm_assimilate_and_score(
        model,
        optimizer,
        checkpoint,
        train,
        confirm,
        test,
        multipliers,
        seed,
        ContrastCoordinates(
            f"point-correspondence-commitment:{source.value}-to-{target.value}:{seed}"
        ),
        AssimilationCoordinates(
            target_client=SourceClientName(target.value),
            directed_pair=directed_pair_name(source, target),
            condition=PRINCIPAL_EVALUATION_CONDITION.name,
            seed=seed,
            clean_pretransfer_checkpoint_artifact_id=checkpoint_artifact_id,
            source_packet_artifact_id=ArtifactIdentifier(
                first_source_packet.packet_integrity_sha256
            ),
            action_artifact_sha256=action_sha256(action),
        ),
        n_classes,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
    )
    return score, n_classes, input_artifact_ids


def score_robust_action_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
    method_slug: FilesystemSlug,
    method: TransferMethod,
    solve_action: Callable[[RobustActionProblem, RandomSeed], CurriculumAction | None],
    settle_and_score: Callable[
        [
            torch.nn.Module,
            torch.optim.AdamW,
            BaseCheckpoint,
            SplitTensors,
            SplitTensors,
            SplitTensors,
            CurriculumMultipliers,
            CurriculumAction,
            RandomSeed,
            ContrastCoordinates,
            AssimilationCoordinates,
            ClassCount,
        ],
        ScoreArtifact,
    ]
    | None = None,
    group_bucket_of: Mapping[CoarseGroup, CoarseGroup] | None = None,
    perturb: Callable[
        [PaddedBlockStructure, ResponseMatrix, ResponseMatrix],
        tuple[ResponseMatrix, ResponseMatrix],
    ]
    | None = None,
    checkpoint_source_experiment: ExperimentName = ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT,
    action_sink: MutableCell[CurriculumAction] | None = None,
    confirmation_verdict_sink: MutableCell[bool] | None = None,
    fine_singleton: bool = False,
    uses_registered_exact_map: bool = False,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    assembly = assemble_principal_action(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        solve_action,
        group_bucket_of,
        perturb,
        checkpoint_source_experiment,
        fine_singleton,
        uses_registered_exact_map,
    )
    if assembly is None:
        return None
    if action_sink is not None:
        action_sink.value = assembly.action
    multipliers = curriculum_multipliers_from_action(
        assembly.action, assembly.blocks, assembly.target_eligible, assembly.n_classes
    )
    optimizer = make_adamw(
        assembly.model,
        assembly.checkpoint.selected_hyperparameters.learning_rate,
        assembly.checkpoint.selected_hyperparameters.weight_decay,
    )
    contrast_coordinates = ContrastCoordinates(
        f"{method_slug}:{source.value}-to-{target.value}:{seed}"
    )
    assimilation_coordinates = AssimilationCoordinates(
        target_client=SourceClientName(target.value),
        directed_pair=directed_pair_name(source, target),
        condition=PRINCIPAL_EVALUATION_CONDITION.name,
        seed=seed,
        clean_pretransfer_checkpoint_artifact_id=assembly.checkpoint_artifact_id,
        source_packet_artifact_id=assembly.first_packet_artifact_id,
        action_artifact_sha256=action_sha256(assembly.action),
    )
    if settle_and_score is None:
        score = _confirm_assimilate_and_score(
            assembly.model,
            optimizer,
            assembly.checkpoint,
            assembly.train,
            assembly.confirm,
            assembly.test,
            multipliers,
            seed,
            contrast_coordinates,
            assimilation_coordinates,
            assembly.n_classes,
            method,
            confirmation_verdict_sink=confirmation_verdict_sink,
        )
    else:
        score = settle_and_score(
            assembly.model,
            optimizer,
            assembly.checkpoint,
            assembly.train,
            assembly.confirm,
            assembly.test,
            multipliers,
            assembly.action,
            seed,
            contrast_coordinates,
            assimilation_coordinates,
            assembly.n_classes,
        )
    return score, assembly.n_classes, assembly.input_artifact_ids


def solve_fedorbit_exact_sparse_action(
    problem: RobustActionProblem,
    seed: RandomSeed,
    certified_value_sink: MutableCell[Score] | None = None,
) -> CurriculumAction | None:
    del seed
    solution = solve_robust_action(problem)
    if certified_value_sink is not None:
        certified_value_sink.value = solution.certified_robust_value
    return solution.selected_action


def _solve_generic_exact_qap_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    outcome = solve_robust_action_qap(problem)
    if outcome.certified_solution is None:
        return None
    return outcome.certified_solution.certified_action


def solve_exact_map_oracle_action(
    problem: RobustActionProblem,
    seed: RandomSeed,
    certified_value_sink: MutableCell[Score] | None = None,
) -> CurriculumAction | None:
    del seed
    identity = BlockCorrespondence.lexicographically_smallest(problem.blocks)
    committed_matrix = identity.permute_response_matrix(problem.lower_response_matrix)
    solution = optimize_against_fixed_matrix(problem, committed_matrix)
    if certified_value_sink is not None:
        certified_value_sink.value = solution.objective_value
    return solution.selected_action


def score_fedorbit_exact_sparse_solver_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("fedorbit-exact-sparse-solver"),
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        solve_fedorbit_exact_sparse_action,
    )


def _settle_without_confirmation_and_score(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    checkpoint: BaseCheckpoint,
    train: SplitTensors,
    confirm: SplitTensors,
    test: SplitTensors,
    multipliers: CurriculumMultipliers,
    action: CurriculumAction,
    seed: RandomSeed,
    contrast_coordinates: ContrastCoordinates,
    assimilation_coordinates: AssimilationCoordinates,
    n_classes: ClassCount,
) -> ScoreArtifact:
    del confirm, contrast_coordinates
    access = AccessLogger()
    access.record(ClientRole.TARGET, ResourceKind.ANONYMOUS_SOURCE_PACKET)
    access.record(ClientRole.TARGET, ResourceKind.TRAIN)
    access.record(ClientRole.TARGET, ResourceKind.META)
    lifecycle = PreTestLifecycle()
    lifecycle.complete_phase(PreTestPhase.SOURCE_SELECTION_FINALIZED)
    lifecycle.complete_phase(PreTestPhase.ACTION_FINALIZED)
    lifecycle.complete_phase(PreTestPhase.CONFIRMATION_DECISION_FINALIZED)
    pre_confirm = capture_pre_confirm_pair(model, optimizer)
    optimizer_step_ledger = TargetOptimizerStepLedger(
        TransferMethod.FEDORBIT_WITHOUT_CONFIRMATION,
        assimilation_coordinates.directed_pair,
        seed,
    )
    if action.realized_support_size > 0:
        apply_accepted_assimilation(
            model,
            optimizer,
            pre_confirm.curriculum,
            train.features,
            train.targets,
            checkpoint.train_class_weights,
            multipliers,
            seed,
            assimilation_coordinates,
            optimizer_step_ledger,
        )
    else:
        settle_rejected_proposal(model, optimizer, pre_confirm.baseline)
    lifecycle.complete_phase(PreTestPhase.ASSIMILATION_SETTLED)
    lifecycle.complete_phase(PreTestPhase.PRE_TEST_ARTIFACTS_COMMITTED)
    lifecycle.open_test()
    lifecycle.assert_opened()
    access.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=True)
    validate_dynamic_access_log_scan(
        access.trace(),
        method_resource_manifest(TransferMethod.FEDORBIT_WITHOUT_CONFIRMATION),
    )
    return score_model(
        ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
    )


def score_fedorbit_without_confirmation_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("fedorbit-without-confirmation"),
        TransferMethod.FEDORBIT_WITHOUT_CONFIRMATION,
        solve_fedorbit_exact_sparse_action,
        _settle_without_confirmation_and_score,
    )


def score_generic_exact_qap_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("generic-exact-qap"),
        TransferMethod.GENERIC_EXACT_QAP,
        _solve_generic_exact_qap_action,
    )


def score_exact_map_oracle_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("exact-map-oracle"),
        TransferMethod.EXACT_MAP_ORACLE,
        solve_exact_map_oracle_action,
        uses_registered_exact_map=True,
    )


def solve_coarse_block_mean_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    summary = coarse_block_mean_matrix(problem.blocks, problem.lower_response_matrix)
    return optimize_against_fixed_matrix(problem, summary.matrix).selected_action


def solve_coarse_block_min_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    summary = coarse_block_min_matrix(problem.blocks, problem.lower_response_matrix)
    return optimize_against_fixed_matrix(problem, summary.matrix).selected_action


def solve_orbit_mean_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    del seed
    summary = orbit_mean_matrix(problem.blocks, problem.lower_response_matrix)
    return optimize_against_fixed_matrix(problem, summary.matrix).selected_action


def solve_coupling_destroyed_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    destroyed = coupling_destroyed_matrices(
        problem.blocks,
        problem.lower_response_matrix,
        problem.upper_response_matrix,
        seed,
        ContrastCoordinates(f"coupling-destroyed:{seed}"),
    )
    destroyed_problem = RobustActionProblem(
        blocks=problem.blocks,
        lower_response_matrix=destroyed.lower_response_matrix,
        upper_response_matrix=destroyed.upper_response_matrix,
        target_importance=problem.target_importance,
        coordinate_caps=problem.coordinate_caps,
        linear_costs=problem.linear_costs,
        total_budget=problem.total_budget,
        principal_support=problem.principal_support,
    )
    return solve_robust_action(destroyed_problem).selected_action


def score_coarse_block_mean_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("coarse-block-mean"),
        TransferMethod.COARSE_BLOCK_MEAN,
        solve_coarse_block_mean_action,
    )


def score_coarse_block_min_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("coarse-block-min"),
        TransferMethod.COARSE_BLOCK_MIN,
        solve_coarse_block_min_action,
    )


def score_orbit_mean_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("orbit-mean"),
        TransferMethod.ORBIT_MEAN,
        solve_orbit_mean_action,
    )


def score_coupling_destroyed_fedorbit_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("coupling-destroyed-fedorbit"),
        TransferMethod.COUPLING_DESTROYED_FEDORBIT,
        solve_coupling_destroyed_action,
    )


def score_local_sir_cell_adapter(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    del source, source_materialized
    return score_local_sir_cell(store, layout, target, target_materialized, seed, device)


def _confirm_assimilate_score_capturing_verdict(
    verdicts: MutableCell[ConfirmationVerdict],
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    source: DatasetId,
    target: DatasetId,
    overwrite_policy: OverwritePolicy,
) -> Callable[
    [
        torch.nn.Module,
        torch.optim.AdamW,
        BaseCheckpoint,
        SplitTensors,
        SplitTensors,
        SplitTensors,
        CurriculumMultipliers,
        CurriculumAction,
        RandomSeed,
        ContrastCoordinates,
        AssimilationCoordinates,
        ClassCount,
    ],
    ScoreArtifact,
]:
    def settle_and_score(
        model: torch.nn.Module,
        optimizer: torch.optim.AdamW,
        checkpoint: BaseCheckpoint,
        train: SplitTensors,
        confirm: SplitTensors,
        test: SplitTensors,
        multipliers: CurriculumMultipliers,
        action: CurriculumAction,
        seed: RandomSeed,
        contrast_coordinates: ContrastCoordinates,
        assimilation_coordinates: AssimilationCoordinates,
        n_classes: ClassCount,
    ) -> ScoreArtifact:
        del action
        access = AccessLogger()
        access.record(ClientRole.TARGET, ResourceKind.ANONYMOUS_SOURCE_PACKET)
        access.record(ClientRole.TARGET, ResourceKind.TRAIN)
        access.record(ClientRole.TARGET, ResourceKind.META)
        access.record(ClientRole.TARGET, ResourceKind.CONFIRM)
        pre_confirm = capture_pre_confirm_pair(model, optimizer)
        optimizer_step_ledger = TargetOptimizerStepLedger(
            TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
            assimilation_coordinates.directed_pair,
            seed,
        )
        verdict = run_proposal_confirmation(
            ConfirmationRequest(
                model,
                pre_confirm.baseline,
                pre_confirm.curriculum,
                train.features,
                train.targets,
                confirm.features,
                confirm.targets,
                checkpoint.train_class_weights,
                multipliers,
                checkpoint.selected_hyperparameters,
                seed,
                contrast_coordinates,
            ),
            optimizer_step_ledger,
        )
        verdicts.value = verdict
        lifecycle = PreTestLifecycle()
        lifecycle.complete_phase(PreTestPhase.SOURCE_SELECTION_FINALIZED)
        lifecycle.complete_phase(PreTestPhase.ACTION_FINALIZED)
        lifecycle.complete_phase(PreTestPhase.CONFIRMATION_DECISION_FINALIZED)
        if verdict.accepted:
            apply_accepted_assimilation(
                model,
                optimizer,
                pre_confirm.curriculum,
                train.features,
                train.targets,
                checkpoint.train_class_weights,
                multipliers,
                seed,
                assimilation_coordinates,
                optimizer_step_ledger,
            )
        else:
            settle_rejected_proposal(model, optimizer, pre_confirm.baseline)
        lifecycle.complete_phase(PreTestPhase.ASSIMILATION_SETTLED)
        lifecycle.complete_phase(PreTestPhase.PRE_TEST_ARTIFACTS_COMMITTED)
        lifecycle.open_test()
        lifecycle.assert_opened()
        access.record(ClientRole.TARGET, ResourceKind.TEST, transfer_finalized=True)
        validate_dynamic_access_log_scan(
            access.trace(),
            method_resource_manifest(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER),
        )
        reserved = active_config().scientific.target_optimizer_budget.reserved
        step_input_ids = (
            assimilation_coordinates.clean_pretransfer_checkpoint_artifact_id,
            assimilation_coordinates.source_packet_artifact_id,
        )
        for metric_name, category, reserved_steps in (
            (
                MetricId.TARGET_CONFIRMATION_OPTIMIZER_STEPS,
                TargetOptimizerBudgetCategory.CONFIRMATION_CANDIDATES,
                reserved.confirmation_candidates,
            ),
            (
                MetricId.LIVE_ASSIMILATION_OPTIMIZER_STEPS,
                TargetOptimizerBudgetCategory.LIVE_ASSIMILATION,
                reserved.live_assimilation,
            ),
        ):
            consumed_steps = reserved_steps - optimizer_step_ledger.remaining(category)
            persist_primary_transfer_metric(
                store,
                layout,
                experiment,
                assimilation_coordinates.directed_pair,
                source,
                target,
                TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
                seed,
                metric_name,
                float(consumed_steps),
                MetricUnit.COUNT,
                MetricDirection.DESCRIPTIVE,
                step_input_ids,
                overwrite_policy,
                assimilation_coordinates.condition,
            )
        return score_model(
            ScoringRequest(model, test.features, test.targets, LocalClassCount(n_classes))
        )

    return settle_and_score


def score_fedorbit_with_confirmation_verdict_cell(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    experiment: ExperimentName,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
    verdicts: MutableCell[ConfirmationVerdict],
    overwrite_policy: OverwritePolicy,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    return score_robust_action_cell(
        store,
        layout,
        source,
        target,
        source_materialized,
        target_materialized,
        seed,
        device,
        FilesystemSlug("fedorbit-exact-sparse-solver"),
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        solve_fedorbit_exact_sparse_action,
        _confirm_assimilate_score_capturing_verdict(
            verdicts, store, layout, experiment, source, target, overwrite_policy
        ),
    )


def score_local_only_cell_adapter(
    store: ArtifactStore,
    layout: WorkspaceLayout,
    source: DatasetId,
    target: DatasetId,
    source_materialized: MaterializedClient,
    target_materialized: MaterializedClient,
    seed: RandomSeed,
    device: torch.device,
) -> tuple[ScoreArtifact, ClassCount, tuple[ArtifactIdentifier, ...]] | None:
    del source, source_materialized
    scored = score_local_only_cell(store, layout, target, target_materialized, seed, device)
    if scored is None:
        return None
    score, n_classes, artifact_id = scored
    return score, n_classes, (artifact_id,)


def solve_fedorbit_exact_sparse_action_at_support(
    support_limit: SupportCount,
) -> Callable[..., CurriculumAction | None]:
    def solve(
        problem: RobustActionProblem,
        seed: RandomSeed,
        certified_value_sink: list[Score] | None = None,
    ) -> CurriculumAction | None:
        del seed
        solution = solve_robust_action(problem, support_limit)
        if certified_value_sink is not None:
            certified_value_sink.append(solution.certified_robust_value)
        return solution.selected_action

    return solve


def solve_dense_ccp_fallback_action(
    problem: RobustActionProblem, seed: RandomSeed
) -> CurriculumAction | None:
    outcome = solve_dense_ccp(
        problem, seed, SemanticCoordinates(f"sparsity-and-dense-fallback:{seed}")
    )
    return outcome.selected_action


def solve_matched_resource_rectangular_action(
    problem: RobustActionProblem,
    seed: RandomSeed,
    certified_value_sink: MutableCell[Score] | None = None,
) -> CurriculumAction | None:
    del seed
    hull = build_rectangular_hull(
        problem.blocks, problem.lower_response_matrix, problem.upper_response_matrix
    )
    solution = optimize_against_fixed_matrix(problem, hull.lower_bounds)
    if certified_value_sink is not None:
        certified_value_sink.value = solution.objective_value
    return solution.selected_action


WeakSignalPerturbation = Callable[
    [PaddedBlockStructure, ResponseMatrix, ResponseMatrix], tuple[ResponseMatrix, ResponseMatrix]
]
