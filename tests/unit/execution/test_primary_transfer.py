from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

import fedorbit.experiments.scoring as scoring
from fedorbit.analysis.metrics import (
    ClassEntropySet,
    Probability,
    TrueClassProbabilities,
    balanced_accuracy,
    macro_f1,
)
from fedorbit.analysis.records import (
    MetricDirection,
)
from fedorbit.datasets.common import AdapterSchema, FieldRole
from fedorbit.datasets.materialization import MaterializedClient, TransferConceptGroup
from fedorbit.experiments.scoring import (
    assemble_cross_client_response_matrix,
    assemble_self_response_matrix,
    assemble_target_response_matrix,
    assess_pair_seed_structure,
    class_metric_sets,
    cross_client_padded_blocks,
    curriculum_multipliers_from_action,
    eligible_groups_by_coarse,
    pair_seed_structure,
    persist_ineligible_transfer_cell,
    persist_primary_transfer_metric,
    self_padded_blocks,
    solve_coarse_block_mean_action,
    solve_coarse_block_min_action,
    solve_coupling_destroyed_action,
    solve_orbit_mean_action,
    strict_pair_resource_validity,
    target_node_risks,
)
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import WorkspaceLayout, build_layout
from fedorbit.learning.scoring import (
    CrossEntropy,
    LocalClassIndex,
    ScoreArtifact,
    ScoreRow,
    ScoreRowIndex,
)
from fedorbit.optimization.objective import ActionSpaceError, CurriculumAction, RobustActionProblem
from fedorbit.response.packet import SourcePacket, build_source_packet
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate
from fedorbit.types import (
    AnonymousNodeDisplayId,
    ArtifactIdentifier,
    ClassCount,
    ClassIndex,
    ClientRole,
    CoarseGroup,
    DatasetId,
    DirectedPairName,
    ExperimentName,
    ExposedCoarseGroupId,
    FeatureName,
    MetricId,
    MetricUnit,
    OracleTransferConcept,
    OverwritePolicy,
    PairSeedIneligibilityReason,
    RandomSeed,
    Rfc3339UtcTimestamp,
    Sha256Digest,
    StrictResourceValidity,
    TabularColumnName,
    TransferMethod,
)


def _score_row(row_index: int, target: int, predicted: int, probability: float) -> ScoreRow:
    other = 1.0 - probability
    probabilities = (
        TrueClassProbabilities((Probability(probability), Probability(other)))
        if target == 0
        else TrueClassProbabilities((Probability(other), Probability(probability)))
    )
    return ScoreRow(
        row_index=ScoreRowIndex(row_index),
        target=LocalClassIndex(target),
        predicted_class=LocalClassIndex(predicted),
        probabilities=probabilities,
        cross_entropy=CrossEntropy(-math.log(max(probability, 1e-6))),
    )


def test_class_metric_sets_match_hand_computed_values() -> None:
    rows = (
        _score_row(0, target=0, predicted=0, probability=0.9),
        _score_row(1, target=0, predicted=1, probability=0.4),
        _score_row(2, target=1, predicted=1, probability=0.8),
        _score_row(3, target=1, predicted=1, probability=0.7),
    )
    score = ScoreArtifact(
        rows=rows,
        class_conditional_cross_entropy=ClassEntropySet((CrossEntropy(0.1), CrossEntropy(0.2))),
        macro_cross_entropy=CrossEntropy(0.15),
    )
    class_count: ClassCount = 2
    f1_set, recall_set = class_metric_sets(score, class_count)
    assert balanced_accuracy(recall_set).value == (0.5 + 1.0) / 2
    macro = macro_f1(f1_set).value
    assert 0.0 < macro < 1.0


def test_persist_primary_transfer_metric_round_trips_and_dedupes(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    manifest = persist_primary_transfer_metric(
        store,
        layout,
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        DirectedPairName("ton_iot_linux_process_host -> ton_iot_windows10_host"),
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_WINDOWS10_HOST,
        TransferMethod.LOCAL_ONLY,
        3319,
        MetricId.MACRO_CROSS_ENTROPY,
        0.42,
        MetricUnit("nats"),
        MetricDirection.LOWER_IS_BETTER,
        (ArtifactIdentifier("checkpoint-stub"),),
        OverwritePolicy.REPLACE,
    )
    assert manifest is not None
    resolved = store.resolve(manifest.artifact_id)
    assert resolved.state.value == "Completed"
    payload_path = Path(resolved.payload_paths[0])
    assert payload_path.is_file()
    reused = persist_primary_transfer_metric(
        store,
        layout,
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        DirectedPairName("ton_iot_linux_process_host -> ton_iot_windows10_host"),
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_WINDOWS10_HOST,
        TransferMethod.LOCAL_ONLY,
        3319,
        MetricId.MACRO_CROSS_ENTROPY,
        0.42,
        MetricUnit("nats"),
        MetricDirection.LOWER_IS_BETTER,
        (ArtifactIdentifier("checkpoint-stub"),),
        OverwritePolicy.REUSE,
    )
    assert reused is not None
    assert reused.artifact_id == manifest.artifact_id


def test_ineligible_transfer_cell_persists_typed_pair_seed_reason(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    reason = PairSeedIneligibilityReason.STRICT_RESOURCE_VIOLATION

    manifest = persist_ineligible_transfer_cell(
        store,
        layout,
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        DirectedPairName("ton_iot_linux_process_host -> ton_iot_windows10_host"),
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_WINDOWS10_HOST,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        3319,
        OverwritePolicy.REPLACE,
        pair_seed_ineligibility_reason=reason,
    )

    assert manifest is not None
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    assert payload["metric_record"]["invalid_reason"] == reason.value


def test_persist_primary_transfer_metric_keeps_distinct_metric_identities(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    first = persist_primary_transfer_metric(
        store,
        layout,
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        DirectedPairName("ton_iot_linux_process_host -> ton_iot_windows10_host"),
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_WINDOWS10_HOST,
        TransferMethod.LOCAL_ONLY,
        3319,
        MetricId.MACRO_CROSS_ENTROPY,
        0.42,
        MetricUnit("nats"),
        MetricDirection.LOWER_IS_BETTER,
        (ArtifactIdentifier("checkpoint-stub"),),
        OverwritePolicy.REUSE,
    )
    second = persist_primary_transfer_metric(
        store,
        layout,
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        DirectedPairName("ton_iot_linux_process_host -> ton_iot_windows10_host"),
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_WINDOWS10_HOST,
        TransferMethod.LOCAL_ONLY,
        3319,
        MetricId.MACRO_F1,
        0.7,
        MetricUnit("fraction"),
        MetricDirection.HIGHER_IS_BETTER,
        (ArtifactIdentifier("checkpoint-stub"),),
        OverwritePolicy.REUSE,
    )
    assert first is not None
    assert second is not None
    assert first.artifact_id != second.artifact_id
    assert len(store.all_manifests()) == 2


def _synthetic_packet(node_count: int, base_value: float) -> SourcePacket:
    entries = tuple(
        FinalResponseEntry(row, col, base_value, 0.01, base_value, base_value, True)
        for row in range(node_count)
        for col in range(node_count)
    )
    estimate = FinalResponseEstimate(
        entries=entries,
        critical_value=4.0,
        useful_intervention_columns=node_count,
        median_band_width_ratio=1.0,
        stability_rule_passed=True,
    )
    return build_source_packet(
        estimate,
        anonymous_fine_node_ids=tuple(
            AnonymousNodeDisplayId(f"node-{index}") for index in range(node_count)
        ),
        exposed_coarse_group_id=ExposedCoarseGroupId("Disruption"),
        per_node_train_support=tuple(10 for _ in range(node_count)),
        per_node_meta_support=tuple(5 for _ in range(node_count)),
        per_node_effective_replicate_count=tuple(1 for _ in range(node_count)),
        source_checkpoint_sha256=Sha256Digest("a" * 64),
        response_configuration_sha256=Sha256Digest("b" * 64),
        creation_timestamp=Rfc3339UtcTimestamp("2026-08-22T00:00:00Z"),
    )


def test_self_padded_blocks_and_response_matrix_assembly() -> None:
    eligible_groups_by_coarse = {
        CoarseGroup.DISRUPTION: (
            TransferConceptGroup(OracleTransferConcept.DDOS, (ClassIndex(0),), 10, 5, True),
            TransferConceptGroup(OracleTransferConcept.RANSOMWARE, (ClassIndex(1),), 10, 5, True),
        ),
        CoarseGroup.EXPLOITATION: (
            TransferConceptGroup(OracleTransferConcept.BACKDOOR, (ClassIndex(2),), 10, 5, True),
        ),
    }
    blocks = self_padded_blocks(eligible_groups_by_coarse)
    assert blocks.total_padded_nodes == 3
    packets = {
        CoarseGroup.DISRUPTION: _synthetic_packet(2, 0.5),
        CoarseGroup.EXPLOITATION: _synthetic_packet(1, 0.9),
    }
    matrix = assemble_self_response_matrix(blocks, packets)
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 0.5 and matrix[0, 1] == 0.5 and matrix[1, 0] == 0.5
    assert matrix[2, 2] == 0.9
    assert matrix[0, 2] == 0.0 and matrix[2, 0] == 0.0 and matrix[1, 2] == 0.0


def test_target_node_risks_averages_member_classes_and_flags_null_nodes() -> None:
    eligible_groups_by_coarse = {
        CoarseGroup.DISRUPTION: (
            TransferConceptGroup(
                OracleTransferConcept.DDOS, (ClassIndex(0), ClassIndex(1)), 10, 5, True
            ),
        ),
    }
    blocks = self_padded_blocks(eligible_groups_by_coarse)
    risks = target_node_risks(blocks, eligible_groups_by_coarse, (0.2, 0.4, float("nan")))
    assert len(risks) == 1
    assert risks[0].is_actionable
    assert risks[0].meta_class_risk == (0.2 + 0.4) / 2


def test_curriculum_multipliers_apply_uniformly_within_a_concept_group() -> None:
    eligible_groups_by_coarse = {
        CoarseGroup.DISRUPTION: (
            TransferConceptGroup(
                OracleTransferConcept.DDOS, (ClassIndex(0), ClassIndex(1)), 10, 5, True
            ),
        ),
    }
    blocks = self_padded_blocks(eligible_groups_by_coarse)
    problem = RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=np.zeros((1, 1)),
        upper_response_matrix=np.zeros((1, 1)),
        target_importance=np.array([1.0]),
        coordinate_caps=np.array([1.0]),
        linear_costs=np.array([0.0]),
        total_budget=1.0,
        principal_support=1,
    )
    action = CurriculumAction(problem=problem, coordinates=np.array([0.3]))
    class_count: ClassCount = 3
    multipliers = curriculum_multipliers_from_action(
        action, blocks, eligible_groups_by_coarse, class_count
    )
    assert torch.allclose(multipliers.values, torch.tensor([1.3, 1.3, 1.0], dtype=torch.float64))


def _two_node_problem(lower: np.ndarray, upper: np.ndarray) -> RobustActionProblem:
    eligible_groups_by_coarse = {
        CoarseGroup.DISRUPTION: (
            TransferConceptGroup(OracleTransferConcept.DDOS, (ClassIndex(0),), 10, 5, True),
            TransferConceptGroup(OracleTransferConcept.RANSOMWARE, (ClassIndex(1),), 10, 5, True),
        ),
    }
    blocks = self_padded_blocks(eligible_groups_by_coarse)
    return RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=lower,
        upper_response_matrix=upper,
        target_importance=np.array([0.5, 0.5]),
        coordinate_caps=np.array([1.0, 1.0]),
        linear_costs=np.array([0.0, 0.0]),
        total_budget=2.0,
        principal_support=2,
    )


def test_ablation_solve_actions_return_valid_curriculum_actions() -> None:
    lower = np.array([[0.1, 0.3], [0.2, 0.4]])
    upper = np.array([[0.5, 0.7], [0.6, 0.8]])
    problem = _two_node_problem(lower, upper)
    seed: RandomSeed = 3319
    for solve in (
        solve_coarse_block_mean_action,
        solve_coarse_block_min_action,
        solve_orbit_mean_action,
        solve_coupling_destroyed_action,
    ):
        action = solve(problem, seed)
        assert action is not None
        assert action.coordinates.shape == (2,)
        assert np.all(action.coordinates >= 0.0)
        assert np.all(action.coordinates <= problem.coordinate_caps)


def test_curriculum_action_rejects_total_budget_violation() -> None:
    problem = replace(_two_node_problem(np.zeros((2, 2)), np.zeros((2, 2))), total_budget=0.5)
    with pytest.raises(ActionSpaceError, match="total budget"):
        CurriculumAction(problem=problem, coordinates=np.array([0.3, 0.3]))


def test_curriculum_action_accepts_exact_boundaries_and_rejects_null_cap() -> None:
    problem = replace(
        _two_node_problem(np.zeros((2, 2)), np.zeros((2, 2))),
        coordinate_caps=np.array([0.25, 0.0]),
        total_budget=0.25,
    )

    action = CurriculumAction(problem=problem, coordinates=np.array([0.25, 0.0]))
    assert action.realized_support_size == 1
    with pytest.raises(ActionSpaceError, match="coordinate cap"):
        CurriculumAction(problem=problem, coordinates=np.array([0.0, 0.01]))


def test_eligible_groups_apply_role_specific_support_predicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = TransferConceptGroup(
        OracleTransferConcept.DDOS,
        (ClassIndex(0),),
        10,
        5,
        True,
        target_eligible=False,
    )
    materialized = cast(MaterializedClient, SimpleNamespace())

    def fixed_groups(
        _dataset: DatasetId, _client: MaterializedClient
    ) -> tuple[TransferConceptGroup, ...]:
        return (group,)

    monkeypatch.setattr(scoring, "transfer_concept_groups", fixed_groups)

    source_groups = eligible_groups_by_coarse(
        DatasetId.TON_IOT_NETWORK, materialized, ClientRole.SOURCE
    )
    target_groups = eligible_groups_by_coarse(
        DatasetId.TON_IOT_NETWORK, materialized, ClientRole.TARGET
    )

    assert tuple(source_groups) == (CoarseGroup.DISRUPTION,)
    assert not target_groups


def test_pair_seed_structure_requires_configured_target_concepts_and_real_response_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def group(concept: OracleTransferConcept) -> TransferConceptGroup:
        return TransferConceptGroup(concept, (ClassIndex(0),), 10, 5, True, target_eligible=True)

    source = {
        CoarseGroup.DISRUPTION: (
            group(OracleTransferConcept.DDOS),
            group(OracleTransferConcept.RANSOMWARE),
        ),
        CoarseGroup.EXPLOITATION: (
            group(OracleTransferConcept.BACKDOOR),
            group(OracleTransferConcept.INJECTION),
        ),
        CoarseGroup.ACCESS_AND_DISCOVERY: (group(OracleTransferConcept.PASSWORD_ATTACK),),
    }
    target = {
        CoarseGroup.DISRUPTION: (
            group(OracleTransferConcept.DDOS),
            group(OracleTransferConcept.RANSOMWARE),
        ),
        CoarseGroup.EXPLOITATION: (
            group(OracleTransferConcept.BACKDOOR),
            group(OracleTransferConcept.INJECTION),
        ),
    }
    available = frozenset(target)
    strict_resources = StrictResourceValidity(True)

    assert pair_seed_structure(source, target, available, strict_resources).is_eligible
    insufficient_target = {
        CoarseGroup.DISRUPTION: target[CoarseGroup.DISRUPTION],
        CoarseGroup.EXPLOITATION: (group(OracleTransferConcept.BACKDOOR),),
    }
    assert (
        pair_seed_structure(
            source, insufficient_target, available, strict_resources
        ).ineligibility_reason
        is PairSeedIneligibilityReason.INSUFFICIENT_ACTIONABLE_TARGET_CONCEPTS
    )

    minimums = SimpleNamespace(
        minimum_actionable_target_concepts=3,
        minimum_nontrivial_block_size=2,
    )
    monkeypatch.setattr(
        scoring,
        "active_config",
        lambda: SimpleNamespace(scientific=SimpleNamespace(transfer_support=minimums)),
    )
    target_without_nontrivial_block = {
        CoarseGroup.DISRUPTION: (group(OracleTransferConcept.DDOS),),
        CoarseGroup.EXPLOITATION: (group(OracleTransferConcept.BACKDOOR),),
        CoarseGroup.ACCESS_AND_DISCOVERY: (group(OracleTransferConcept.PASSWORD_ATTACK),),
    }
    target_block_structure = pair_seed_structure(
        source, target_without_nontrivial_block, frozenset(CoarseGroup), strict_resources
    )
    assert (
        target_block_structure.ineligibility_reason
        is PairSeedIneligibilityReason.NO_NONTRIVIAL_TARGET_BLOCK
    )
    source_without_nontrivial_response = {
        CoarseGroup.DISRUPTION: (group(OracleTransferConcept.DDOS),),
        CoarseGroup.EXPLOITATION: (group(OracleTransferConcept.BACKDOOR),),
        CoarseGroup.ACCESS_AND_DISCOVERY: (group(OracleTransferConcept.PASSWORD_ATTACK),),
    }
    source_response_structure = pair_seed_structure(
        source_without_nontrivial_response, target, available, strict_resources
    )
    assert (
        source_response_structure.ineligibility_reason
        is PairSeedIneligibilityReason.NO_NONTRIVIAL_SOURCE_RESPONSE_BLOCK
    )
    assert (
        pair_seed_structure(
            source, target, available, StrictResourceValidity(False)
        ).ineligibility_reason
        is PairSeedIneligibilityReason.STRICT_RESOURCE_VIOLATION
    )


def test_strict_pair_resource_validity_rejects_shared_feature_namespace() -> None:
    shared_feature = FeatureName("shared")
    source = cast(
        MaterializedClient,
        SimpleNamespace(
            feature_names=(shared_feature,),
            schema=AdapterSchema(DatasetId.TON_IOT_WINDOWS10_HOST, ()),
        ),
    )
    target = cast(
        MaterializedClient,
        SimpleNamespace(
            feature_names=(shared_feature,),
            schema=AdapterSchema(
                DatasetId.TON_IOT_LINUX_PROCESS_HOST,
                (),
                roles={
                    TabularColumnName("entity"): FieldRole.FORBIDDEN_IDENTITY,
                },
            ),
        ),
    )

    assert not strict_pair_resource_validity(source, target)


def test_assess_pair_seed_structure_records_missing_source_response_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    groups = {
        CoarseGroup.DISRUPTION: (
            TransferConceptGroup(OracleTransferConcept.DDOS, (ClassIndex(0),), 10, 5, True),
        ),
    }
    source = cast(MaterializedClient, SimpleNamespace())
    target = cast(MaterializedClient, SimpleNamespace())

    def common_groups(
        _source_dataset: DatasetId,
        _target_dataset: DatasetId,
        _source_materialized: MaterializedClient,
        _target_materialized: MaterializedClient,
    ) -> tuple[dict[CoarseGroup, tuple[TransferConceptGroup, ...]], ...]:
        return groups, groups

    def strict_resources(
        _source_materialized: MaterializedClient,
        _target_materialized: MaterializedClient,
    ) -> StrictResourceValidity:
        return StrictResourceValidity(True)

    def no_source_packet(
        _layout: WorkspaceLayout,
        _source_dataset: DatasetId,
        _seed: RandomSeed,
        _coarse: CoarseGroup,
    ) -> None:
        return None

    monkeypatch.setattr(scoring, "common_eligible_groups", common_groups)
    monkeypatch.setattr(scoring, "strict_pair_resource_validity", strict_resources)
    monkeypatch.setattr(scoring, "load_dataset_source_packet", no_source_packet)

    result = assess_pair_seed_structure(
        build_layout(root=tmp_path),
        DatasetId.TON_IOT_WINDOWS10_HOST,
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        source,
        target,
        3319,
    )

    assert (
        result.ineligibility_reason is PairSeedIneligibilityReason.NO_SHARED_ELIGIBLE_COARSE_GROUP
    )


def test_cross_client_padded_blocks_places_source_submatrix_and_pads_the_rest() -> None:
    source_eligible = {
        CoarseGroup.DISRUPTION: (
            TransferConceptGroup(OracleTransferConcept.DDOS, (ClassIndex(0),), 10, 5, True),
        ),
    }
    target_eligible = {
        CoarseGroup.DISRUPTION: (
            TransferConceptGroup(OracleTransferConcept.DDOS, (ClassIndex(0),), 10, 5, True),
            TransferConceptGroup(OracleTransferConcept.RANSOMWARE, (ClassIndex(1),), 10, 5, True),
        ),
    }
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    assert blocks.source_real_counts == (1,)
    assert blocks.target_real_counts == (2,)
    assert blocks.total_padded_nodes == 2
    packet = _synthetic_packet(1, 0.7)
    matrix = assemble_cross_client_response_matrix(
        blocks, {CoarseGroup.DISRUPTION: packet}, SourcePacket.lower_matrix
    )
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 0.7
    assert matrix[0, 1] == 0.0 and matrix[1, 0] == 0.0 and matrix[1, 1] == 0.0


def test_assemble_target_response_matrix_places_target_submatrix_and_pads_the_rest() -> None:
    source_eligible = {
        CoarseGroup.DISRUPTION: (
            TransferConceptGroup(OracleTransferConcept.DDOS, (ClassIndex(0),), 10, 5, True),
            TransferConceptGroup(OracleTransferConcept.RANSOMWARE, (ClassIndex(1),), 10, 5, True),
        ),
    }
    target_eligible = {
        CoarseGroup.DISRUPTION: (
            TransferConceptGroup(OracleTransferConcept.DDOS, (ClassIndex(0),), 10, 5, True),
        ),
    }
    blocks = cross_client_padded_blocks(source_eligible, target_eligible)
    assert blocks.source_real_counts == (2,)
    assert blocks.target_real_counts == (1,)
    packet = _synthetic_packet(1, 0.4)
    matrix = assemble_target_response_matrix(
        blocks, {CoarseGroup.DISRUPTION: packet}, SourcePacket.lower_matrix
    )
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 0.4
    assert matrix[0, 1] == 0.0 and matrix[1, 0] == 0.0 and matrix[1, 1] == 0.0
