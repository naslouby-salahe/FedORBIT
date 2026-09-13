from __future__ import annotations

import math

import numpy as np
import pytest

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
from fedorbit.types import CoarseGroup, CorrespondenceBlockId, OracleTransferConcept


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
