from __future__ import annotations

import math

import numpy as np
import pytest

from fedorbit.optimization.correspondence import (
    ActiveImageMap,
    BlockCorrespondence,
    BlockNodeCounts,
    CorrespondenceError,
    PaddedBlockStructure,
    active_image_assignment_count,
    build_padded_block_structure,
    compare_correspondences_lexicographically,
    enumerate_active_image_maps,
    enumerate_block_permutations,
    falling_factorial,
    support_per_block,
)
from fedorbit.types import CoarseGroup


def _two_by_three_blocks() -> PaddedBlockStructure:
    return build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
        {CoarseGroup.DISRUPTION: 3, CoarseGroup.EXPLOITATION: 1},
    )


def test_padded_block_size_is_endpoint_maximum() -> None:
    blocks = _two_by_three_blocks()
    assert blocks.padded_sizes.per_block == (3, 2)


def test_null_padding_counts_follow_deficit() -> None:
    blocks = _two_by_three_blocks()
    assert blocks.source_null_counts().per_block == (1, 0)
    assert blocks.target_null_counts().per_block == (0, 1)


def test_total_padded_node_count() -> None:
    assert _two_by_three_blocks().total_padded_nodes == 5


def test_block_index_ranges_partition_padded_space() -> None:
    blocks = _two_by_three_blocks()
    covered: set[int] = set()
    boundaries: list[tuple[int, int]] = []
    for block_index in range(len(blocks.padded_sizes.per_block)):
        index_range = blocks.block_index_range(block_index)
        assert not covered.intersection(index_range)
        covered.update(index_range)
        boundaries.append((index_range.start, index_range.stop))
    assert covered == set(range(blocks.total_padded_nodes))
    assert boundaries == [(0, 3), (3, 5)]


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


def test_identity_is_lexicographically_smallest() -> None:
    blocks = _two_by_three_blocks()
    identity = BlockCorrespondence.identity(blocks)
    smallest = BlockCorrespondence.lexicographically_smallest(blocks)
    assert smallest.images == identity.images
    for correspondence in enumerate_block_permutations(blocks):
        assert compare_correspondences_lexicographically(smallest, correspondence) <= 0


def test_compare_correspondences_orders_numerically() -> None:
    blocks = _two_by_three_blocks()
    first = BlockCorrespondence(blocks, (0, 1, 2, 3, 4))
    second = BlockCorrespondence(blocks, (0, 2, 1, 3, 4))
    assert compare_correspondences_lexicographically(first, second) == -1
    assert compare_correspondences_lexicographically(second, first) == 1
    assert compare_correspondences_lexicographically(first, first) == 0


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


def test_permutation_matrix_is_orthogonal_and_leaves_identity_invariant() -> None:
    blocks = _two_by_three_blocks()
    for correspondence in enumerate_block_permutations(blocks):
        matrix = correspondence.permutation_matrix()
        assert np.allclose(matrix.T @ matrix, np.eye(5))
        assert np.allclose(correspondence.permute_response_matrix(np.eye(5)), np.eye(5))


def test_permutation_matches_conjugation_definition() -> None:
    blocks = _two_by_three_blocks()
    rng = np.random.default_rng(7)
    matrix = rng.uniform(-0.5, 0.5, size=(5, 5))
    for correspondence in enumerate_block_permutations(blocks):
        images = np.asarray(correspondence.images)
        permutation = correspondence.permutation_matrix()
        expected = permutation.T @ matrix @ permutation
        actual = correspondence.permute_response_matrix(matrix)
        assert np.allclose(actual, expected)
        assert images.size == 5


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


def test_support_per_block_counts_active_nodes_per_group() -> None:
    blocks = _two_by_three_blocks()
    counts = support_per_block(blocks, (0, 3))
    assert counts.per_block == (1, 1)
    counts_all_first = support_per_block(blocks, (0, 1))
    assert counts_all_first.per_block == (2, 0)


def test_active_image_assignment_count_formula() -> None:
    blocks = _two_by_three_blocks()
    counts = BlockNodeCounts(blocks=blocks, per_block=(2, 1))
    expected = falling_factorial(3, 2) * falling_factorial(2, 1)
    assert expected == 6 * 2
    assert active_image_assignment_count(blocks, counts) == expected


def test_enumerated_active_image_maps_count_matches_n_s() -> None:
    blocks = _two_by_three_blocks()
    active_nodes = (0, 3)
    counts = support_per_block(blocks, active_nodes)
    expected = active_image_assignment_count(blocks, counts)
    maps = tuple(enumerate_active_image_maps(blocks, active_nodes))
    assert len(maps) == expected


def test_active_image_maps_are_injective_and_block_compatible() -> None:
    blocks = _two_by_three_blocks()
    active_nodes = (0, 1, 3)
    for mapping in enumerate_active_image_maps(blocks, active_nodes):
        pairs = mapping.fixed_pairs()
        images = [image for _, image in pairs]
        assert len(set(images)) == len(images)
        for target, image in pairs:
            assert blocks.block_of_node(target) == blocks.block_of_node(image)
        for target in active_nodes:
            assert mapping.image_of(target) is not None
    assert enumerate_active_image_maps(blocks, ()).__next__() == ActiveImageMap(blocks, ())
