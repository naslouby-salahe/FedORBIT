from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

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
from fedorbit.datasets.materialization import TransferConceptGroup
from fedorbit.infrastructure.execution import (
    ArtifactStore,
    assemble_cross_client_response_matrix,
    assemble_self_response_matrix,
    assemble_target_response_matrix,
    class_metric_sets,
    cross_client_padded_blocks,
    curriculum_multipliers_from_action,
    persist_primary_transfer_metric,
    self_padded_blocks,
    solve_coarse_block_mean_action,
    solve_coarse_block_min_action,
    solve_coupling_destroyed_action,
    solve_orbit_mean_action,
    target_node_risks,
)
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.learning.scoring import (
    CrossEntropy,
    LocalClassIndex,
    ScoreArtifact,
    ScoreRow,
    ScoreRowIndex,
)
from fedorbit.optimization.objective import CurriculumAction, RobustActionProblem
from fedorbit.response.packet import SourcePacket, build_source_packet
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate
from fedorbit.types import (
    AnonymousNodeDisplayId,
    ArtifactIdentifier,
    ClassCount,
    ClassIndex,
    CoarseGroup,
    DatasetId,
    DirectedPairName,
    ExperimentName,
    ExposedCoarseGroupId,
    MetricId,
    MetricUnit,
    OracleTransferConcept,
    OverwritePolicy,
    RandomSeed,
    Rfc3339UtcTimestamp,
    Sha256Digest,
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
