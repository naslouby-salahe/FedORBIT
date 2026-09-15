from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import pytest

from fedorbit.datasets.materialization import TransferConceptGroup
from fedorbit.experiments.scoring import registered_exact_correspondence
from fedorbit.infrastructure.runtime import RandomSeed
from fedorbit.interface import (
    AnonymityCoordinate,
    AnonymityCoordinateEntry,
    anonymous_node_order,
)
from fedorbit.optimization.certificates import build_rectangular_hull
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    BlockNodeCounts,
    CorrespondenceError,
    PaddedBlockStructure,
    active_image_assignment_count,
    build_padded_block_structure,
    correspondence_block_id,
    enumerate_block_permutations,
    falling_factorial,
)
from fedorbit.optimization.diagnostics import fixed_action_rectangularization_gap
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    evaluate_objective,
    h_orb,
    h_rect,
)
from fedorbit.response.packet import (
    PacketField,
    SourcePacket,
    anonymized_response_estimate,
    build_source_packet,
)
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate
from fedorbit.types import (
    ClientRole,
    CoarseGroup,
    CorrespondenceBlockId,
    ExposedCoarseGroupId,
    OracleTransferConcept,
    Rfc3339UtcTimestamp,
    Sha256Digest,
)


def _two_by_three_blocks() -> PaddedBlockStructure:
    return build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
        {CoarseGroup.DISRUPTION: 3, CoarseGroup.EXPLOITATION: 1},
    )


def test_total_padded_node_count() -> None:
    assert _two_by_three_blocks().total_padded_nodes == 5


def test_block_of_node_resolves_group_membership() -> None:
    blocks = _two_by_three_blocks()
    assert [blocks.block_of_node(node) for node in range(5)] == [0, 0, 0, 1, 1]
    with pytest.raises(CorrespondenceError):
        blocks.block_of_node(5)


def test_orbit_size_is_product_of_block_factorials() -> None:
    blocks = _two_by_three_blocks()
    assert blocks.orbit_size == math.factorial(3) * math.factorial(2) == 12


def test_enumerated_permutations_are_exactly_the_full_orbit() -> None:
    blocks = _two_by_three_blocks()
    correspondences = tuple(enumerate_block_permutations(blocks))
    assert len(correspondences) == blocks.orbit_size
    unique_images = {correspondence.images for correspondence in correspondences}
    assert len(unique_images) == blocks.orbit_size


def test_every_enumerated_correspondence_is_block_preserving_bijection() -> None:
    blocks = _two_by_three_blocks()
    for correspondence in enumerate_block_permutations(blocks):
        assert sorted(correspondence.images) == list(range(blocks.total_padded_nodes))
        for target_index, image in enumerate(correspondence.images):
            assert blocks.block_of_node(image) == blocks.block_of_node(target_index)


def test_correspondence_rejects_cross_block_mapping() -> None:
    blocks = _two_by_three_blocks()
    with pytest.raises(CorrespondenceError):
        BlockCorrespondence(blocks, (0, 1, 4, 3, 2))


def test_correspondence_rejects_duplicate_images() -> None:
    blocks = _two_by_three_blocks()
    with pytest.raises(CorrespondenceError):
        BlockCorrespondence(blocks, (0, 0, 2, 3, 4))


def test_correspondence_rejects_wrong_length() -> None:
    blocks = _two_by_three_blocks()
    with pytest.raises(CorrespondenceError):
        BlockCorrespondence(blocks, (0, 1, 2, 3))


def test_correspondence_block_id_preserves_coarse_group_value() -> None:
    assert correspondence_block_id(CoarseGroup.DISRUPTION) == CorrespondenceBlockId(
        CoarseGroup.DISRUPTION.value
    )


def test_fine_concept_singleton_blocks_have_trivial_orbit() -> None:
    block_ids = (
        CorrespondenceBlockId(OracleTransferConcept.DDOS.value),
        CorrespondenceBlockId(OracleTransferConcept.SCANNING.value),
    )
    blocks = build_padded_block_structure(
        block_ids,
        {block_ids[0]: 1, block_ids[1]: 1},
        {block_ids[0]: 1, block_ids[1]: 1},
    )
    assert blocks.total_padded_nodes == 2
    assert blocks.orbit_size == 1
    assert next(iter(enumerate_block_permutations(blocks))).images == (0, 1)


def test_permute_rejects_shape_mismatch() -> None:
    blocks = _two_by_three_blocks()
    correspondence = BlockCorrespondence.identity(blocks)
    with pytest.raises(CorrespondenceError):
        correspondence.permute_response_matrix(np.zeros((4, 4)))


def test_falling_factorial_matches_enumeration_count() -> None:
    assert falling_factorial(3, 2) == 6
    assert falling_factorial(4, 1) == 4
    assert falling_factorial(2, 2) == 2
    assert falling_factorial(5, 0) == 1
    with pytest.raises(CorrespondenceError):
        falling_factorial(2, 3)


def test_active_image_assignment_count_formula() -> None:
    blocks = _two_by_three_blocks()
    counts = BlockNodeCounts(blocks=blocks, per_block=(2, 1))
    expected = falling_factorial(3, 2) * falling_factorial(2, 1)
    assert expected == 6 * 2
    assert active_image_assignment_count(blocks, counts) == expected


def test_map_conditioned_objective_matches_direct_permuted_matrix_calculation() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 2},
        {CoarseGroup.DISRUPTION: 2},
    )
    lower_response = np.asarray(((1.0, 2.0), (3.0, 5.0)), dtype=np.float64)
    target_importance = np.asarray((0.25, 0.75), dtype=np.float64)
    action_coordinates = np.asarray((0.1, 0.2), dtype=np.float64)
    linear_costs = np.asarray((0.01, 0.02), dtype=np.float64)
    problem = RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=lower_response,
        upper_response_matrix=lower_response,
        target_importance=target_importance,
        coordinate_caps=np.asarray((0.25, 0.25), dtype=np.float64),
        linear_costs=linear_costs,
        total_budget=0.5,
        principal_support=2,
    )
    action = CurriculumAction(problem, action_coordinates)
    correspondence = BlockCorrespondence(blocks, (1, 0))

    direct_permuted = lower_response[np.ix_((1, 0), (1, 0))]
    expected = float(target_importance @ direct_permuted @ action_coordinates)
    expected -= float(linear_costs @ action_coordinates)

    assert evaluate_objective(action, correspondence) == pytest.approx(expected, abs=1e-12)


def _two_equal_blocks() -> PaddedBlockStructure:
    return build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
    )


def _rectangular_hull_fixture() -> tuple[RobustActionProblem, np.ndarray]:
    blocks = _two_equal_blocks()
    lower_response = np.asarray(
        (
            (10.0, 20.0, 30.0, 40.0),
            (21.0, 11.0, 31.0, 41.0),
            (50.0, 60.0, 70.0, 80.0),
            (51.0, 61.0, 81.0, 71.0),
        ),
        dtype=np.float64,
    )
    problem = RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=lower_response,
        upper_response_matrix=lower_response + 100.0,
        target_importance=np.asarray((0.1, 0.2, 0.3, 0.4), dtype=np.float64),
        coordinate_caps=np.full(blocks.total_padded_nodes, 0.5, dtype=np.float64),
        linear_costs=np.asarray((0.01, 0.02, 0.03, 0.04), dtype=np.float64),
        total_budget=1.0,
        principal_support=4,
    )
    return problem, lower_response


def test_rectangular_hull_matches_manual_block_extrema() -> None:
    problem, lower_response = _rectangular_hull_fixture()

    hull = build_rectangular_hull(problem.blocks, lower_response, problem.upper_response_matrix)

    expected_lower = np.asarray(
        (
            (10.0, 20.0, 30.0, 30.0),
            (20.0, 10.0, 30.0, 30.0),
            (50.0, 50.0, 70.0, 80.0),
            (50.0, 50.0, 80.0, 70.0),
        ),
        dtype=np.float64,
    )
    expected_upper = np.asarray(
        (
            (111.0, 121.0, 141.0, 141.0),
            (121.0, 111.0, 141.0, 141.0),
            (161.0, 161.0, 171.0, 181.0),
            (161.0, 161.0, 181.0, 171.0),
        ),
        dtype=np.float64,
    )

    np.testing.assert_array_equal(hull.lower_bounds, expected_lower)
    np.testing.assert_array_equal(hull.upper_bounds, expected_upper)


def test_fixed_action_rectangularization_gap_matches_direct_orbit_minimum() -> None:
    problem, lower_response = _rectangular_hull_fixture()
    action = CurriculumAction(problem, np.asarray((0.1, 0.2, 0.3, 0.4), dtype=np.float64))
    orbit = tuple(enumerate_block_permutations(problem.blocks))
    hull = build_rectangular_hull(problem.blocks, lower_response, problem.upper_response_matrix)

    direct_orbit_values = tuple(
        float(
            problem.target_importance
            @ lower_response[np.ix_(correspondence.images, correspondence.images)]
            @ action.coordinates
        )
        for correspondence in orbit
    )
    expected_orbit = min(direct_orbit_values)
    expected_rectangular = float(problem.target_importance @ hull.lower_bounds @ action.coordinates)
    expected_gap = expected_orbit - expected_rectangular

    assert h_orb(action, orbit) == pytest.approx(expected_orbit, abs=1e-12)
    assert h_rect(action, hull.lower_bounds) == pytest.approx(expected_rectangular, abs=1e-12)
    assert fixed_action_rectangularization_gap(action, orbit, hull.lower_bounds) == pytest.approx(
        expected_gap, abs=1e-12
    )
    assert expected_gap >= 0.0


def _block_keyed(
    entries: tuple[tuple[CoarseGroup, tuple[TransferConceptGroup, ...]], ...],
) -> Mapping[CorrespondenceBlockId, tuple[TransferConceptGroup, ...]]:
    return OrderedDict((correspondence_block_id(group), groups) for group, groups in entries)


def _exact_map_blocks() -> PaddedBlockStructure:
    return build_padded_block_structure(
        (CoarseGroup.DISRUPTION,), {CoarseGroup.DISRUPTION: 2}, {CoarseGroup.DISRUPTION: 2}
    )


def _exact_map_group(concept: OracleTransferConcept) -> TransferConceptGroup:
    return TransferConceptGroup(concept, (), 100, 50, True, 40, 40, True)


def _exact_map_packet(seed: RandomSeed) -> SourcePacket:
    entries = tuple(
        FinalResponseEntry(
            outcome, intervention, 0.0, 0.0, float(outcome * 2 + intervention), 0.0, True
        )
        for outcome in range(2)
        for intervention in range(2)
    )
    estimate = FinalResponseEstimate(entries, 1.0, 2, 0.0, True)
    order = anonymous_node_order(
        2,
        seed,
        ClientRole.SOURCE,
        CoarseGroup.DISRUPTION,
        AnonymityCoordinate(
            (
                AnonymityCoordinateEntry(
                    PacketField.SOURCE_CHECKPOINT_SHA256, Sha256Digest("a" * 64)
                ),
                AnonymityCoordinateEntry(
                    PacketField.RESPONSE_CONFIGURATION_SHA256, Sha256Digest("b" * 64)
                ),
            )
        ),
    )
    anonymized = anonymized_response_estimate(estimate, order)
    return build_source_packet(
        anonymized,
        anonymous_fine_node_ids=order.display_ids,
        exposed_coarse_group_id=ExposedCoarseGroupId("Disruption"),
        per_node_train_support=(1, 1),
        per_node_meta_support=(1, 1),
        per_node_effective_replicate_count=(1, 1),
        source_checkpoint_sha256=Sha256Digest("a" * 64),
        response_configuration_sha256=Sha256Digest("b" * 64),
        creation_timestamp=Rfc3339UtcTimestamp("2026-08-22T00:00:00Z"),
    )


def test_registered_exact_correspondence_maps_each_concept_to_its_anonymous_source_position() -> (
    None
):
    seed = 0
    blocks = _exact_map_blocks()
    groups = (
        _exact_map_group(OracleTransferConcept.DDOS),
        _exact_map_group(OracleTransferConcept.RANSOMWARE),
    )
    source_groups = _block_keyed(((CoarseGroup.DISRUPTION, groups),))
    target_groups = _block_keyed(((CoarseGroup.DISRUPTION, groups),))
    packet = _exact_map_packet(seed)
    assert packet.registered_node_order(seed).permutation != (0, 1)
    positions = packet.registered_node_order(seed).anonymous_position_of_semantic_index()
    correspondence = registered_exact_correspondence(
        blocks,
        source_groups,
        target_groups,
        {CoarseGroup.DISRUPTION: packet},
        seed,
    )
    assert correspondence.images == positions


def test_registered_exact_correspondence_is_a_block_preserving_bijection_with_unequal_blocks() -> (
    None
):
    seed = 31
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 1},
        {CoarseGroup.DISRUPTION: 1, CoarseGroup.EXPLOITATION: 2},
    )
    source_groups = _block_keyed(
        (
            (
                CoarseGroup.DISRUPTION,
                (
                    _exact_map_group(OracleTransferConcept.DDOS),
                    _exact_map_group(OracleTransferConcept.RANSOMWARE),
                ),
            ),
            (CoarseGroup.EXPLOITATION, (_exact_map_group(OracleTransferConcept.BACKDOOR),)),
        )
    )
    target_groups = _block_keyed(
        (
            (CoarseGroup.DISRUPTION, (_exact_map_group(OracleTransferConcept.RANSOMWARE),)),
            (
                CoarseGroup.EXPLOITATION,
                (
                    _exact_map_group(OracleTransferConcept.BACKDOOR),
                    _exact_map_group(OracleTransferConcept.INJECTION),
                ),
            ),
        )
    )
    packets: dict[CoarseGroup, SourcePacket] = {
        CoarseGroup.DISRUPTION: _exact_map_packet(seed),
        CoarseGroup.EXPLOITATION: _exact_map_packet(seed),
    }
    correspondence = registered_exact_correspondence(
        blocks, source_groups, target_groups, packets, seed
    )
    assert sorted(correspondence.images) == list(range(blocks.total_padded_nodes))
    for node, image in enumerate(correspondence.images):
        assert blocks.block_of_node(node) == blocks.block_of_node(image)
