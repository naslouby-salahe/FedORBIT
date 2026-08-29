from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fedorbit.domain.enums import CoarseGroup


class CorrespondenceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BlockNodeCounts:
    blocks: PaddedBlockStructure
    per_block: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.per_block) != len(self.blocks.coarse_groups):
            raise CorrespondenceError("block-count length mismatch")
        if any(count < 0 for count in self.per_block):
            raise CorrespondenceError("block counts must be nonnegative")

    def for_block(self, block_index: int) -> int:
        return self.per_block[block_index]

    def total(self) -> int:
        return sum(self.per_block)


@dataclass(frozen=True, slots=True)
class PaddedBlockStructure:
    coarse_groups: tuple[CoarseGroup, ...]
    source_real_counts: tuple[int, ...]
    target_real_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.coarse_groups) == 0:
            raise CorrespondenceError("padded block structure requires at least one coarse group")
        if len(self.source_real_counts) != len(self.coarse_groups):
            raise CorrespondenceError("source real count length mismatch")
        if len(self.target_real_counts) != len(self.coarse_groups):
            raise CorrespondenceError("target real count length mismatch")
        if any(count < 1 for count in self.source_real_counts):
            raise CorrespondenceError("coarse group without real source nodes")
        if any(count < 1 for count in self.target_real_counts):
            raise CorrespondenceError("coarse group without real target nodes")

    @property
    def padded_size_tuple(self) -> tuple[int, ...]:
        return tuple(
            max(source_count, target_count)
            for source_count, target_count in zip(
                self.source_real_counts, self.target_real_counts, strict=True
            )
        )

    @property
    def padded_sizes(self) -> BlockNodeCounts:
        return BlockNodeCounts(blocks=self, per_block=self.padded_size_tuple)

    @property
    def total_padded_nodes(self) -> int:
        return sum(self.padded_size_tuple)

    @property
    def orbit_size(self) -> int:
        return math.prod(math.factorial(size) for size in self.padded_size_tuple)

    def block_index_range(self, block_index: int) -> range:
        offset = sum(self.padded_size_tuple[:block_index])
        return range(offset, offset + self.padded_size_tuple[block_index])

    def block_of_node(self, node_index: int) -> int:
        total = self.total_padded_nodes
        if node_index < 0 or node_index >= total:
            raise CorrespondenceError(
                f"node index {node_index} outside padded space of size {total}"
            )
        cumulative = 0
        for block_index, size in enumerate(self.padded_size_tuple):
            if node_index < cumulative + size:
                return block_index
            cumulative += size
        raise CorrespondenceError(f"node index {node_index} outside padded space")

    def source_null_counts(self) -> BlockNodeCounts:
        return BlockNodeCounts(
            blocks=self,
            per_block=tuple(
                size - real
                for size, real in zip(self.padded_size_tuple, self.source_real_counts, strict=True)
            ),
        )

    def target_null_counts(self) -> BlockNodeCounts:
        return BlockNodeCounts(
            blocks=self,
            per_block=tuple(
                size - real
                for size, real in zip(self.padded_size_tuple, self.target_real_counts, strict=True)
            ),
        )


def build_padded_block_structure(
    coarse_groups: Sequence[CoarseGroup],
    source_real_counts: Mapping[CoarseGroup, int],
    target_real_counts: Mapping[CoarseGroup, int],
) -> PaddedBlockStructure:
    ordered_groups = tuple(coarse_groups)
    missing = [
        group
        for group in ordered_groups
        if group not in source_real_counts or group not in target_real_counts
    ]
    if missing:
        raise CorrespondenceError(f"missing real node counts for coarse groups: {missing}")
    return PaddedBlockStructure(
        coarse_groups=ordered_groups,
        source_real_counts=tuple(source_real_counts[group] for group in ordered_groups),
        target_real_counts=tuple(target_real_counts[group] for group in ordered_groups),
    )


@dataclass(frozen=True, slots=True)
class BlockCorrespondence:
    blocks: PaddedBlockStructure
    images: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = self.blocks.total_padded_nodes
        if len(self.images) != expected:
            raise CorrespondenceError(
                f"correspondence length {len(self.images)} does not match padded size {expected}"
            )
        seen: set[int] = set()
        for target_index, image in enumerate(self.images):
            if image < 0 or image >= expected:
                raise CorrespondenceError(f"image {image} outside padded space of size {expected}")
            if self.blocks.block_of_node(image) != self.blocks.block_of_node(target_index):
                raise CorrespondenceError(
                    f"correspondence maps target node {target_index} across coarse-group boundary"
                )
            if image in seen:
                raise CorrespondenceError(
                    f"correspondence assigns source node {image} more than once"
                )
            seen.add(image)

    @classmethod
    def identity(cls, blocks: PaddedBlockStructure) -> BlockCorrespondence:
        return cls(blocks=blocks, images=tuple(range(blocks.total_padded_nodes)))

    @classmethod
    def lexicographically_smallest(cls, blocks: PaddedBlockStructure) -> BlockCorrespondence:
        return cls.identity(blocks)

    def ordering_key(self) -> tuple[int, ...]:
        return self.images

    def permutation_matrix(self) -> NDArray[np.float64]:
        total = self.blocks.total_padded_nodes
        matrix = np.zeros((total, total), dtype=np.float64)
        for target_index, image in enumerate(self.images):
            matrix[image, target_index] = 1.0
        return matrix

    def permute_response_matrix(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        expected = self.blocks.total_padded_nodes
        if matrix.shape != (expected, expected):
            raise CorrespondenceError(
                f"response matrix shape {matrix.shape} does not match padded size "
                f"({expected}, {expected})"
            )
        row_permutation = np.asarray(self.images, dtype=np.intp)
        return matrix[np.ix_(row_permutation, row_permutation)]


def enumerate_block_permutations(blocks: PaddedBlockStructure) -> Iterator[BlockCorrespondence]:
    per_block = [
        list(blocks.block_index_range(block_index))
        for block_index in range(len(blocks.padded_size_tuple))
    ]
    for combined in itertools.product(*(itertools.permutations(indices) for indices in per_block)):
        images = tuple(index for block_images in combined for index in block_images)
        yield BlockCorrespondence(blocks=blocks, images=images)


def compare_correspondences_lexicographically(
    left: BlockCorrespondence, right: BlockCorrespondence
) -> int:
    for left_value, right_value in zip(left.images, right.images, strict=True):
        if left_value != right_value:
            return -1 if left_value < right_value else 1
    return 0


@dataclass(frozen=True, slots=True)
class ActiveImageMap:
    blocks: PaddedBlockStructure
    assignments: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        targets_seen: set[int] = set()
        images_seen: set[int] = set()
        for target, image in self.assignments:
            if target in targets_seen:
                raise CorrespondenceError(f"active-image map assigns target {target} twice")
            targets_seen.add(target)
            if image in images_seen:
                raise CorrespondenceError(f"active-image map reuses source image {image}")
            images_seen.add(image)
            if self.blocks.block_of_node(target) != self.blocks.block_of_node(image):
                raise CorrespondenceError(
                    f"active-image assignment ({target}, {image}) crosses coarse-group boundary"
                )

    def image_of(self, target_node: int) -> int | None:
        for target, image in self.assignments:
            if target == target_node:
                return image
        return None

    def fixed_pairs(self) -> tuple[tuple[int, int], ...]:
        return self.assignments


def active_support_of_action(alpha: NDArray[np.float64]) -> tuple[int, ...]:
    return tuple(int(node) for node in np.flatnonzero(alpha > 0.0))


def support_per_block(blocks: PaddedBlockStructure, active_nodes: Sequence[int]) -> BlockNodeCounts:
    counts = [0] * len(blocks.padded_size_tuple)
    for node in active_nodes:
        if node < 0 or node >= blocks.total_padded_nodes:
            raise CorrespondenceError(f"active node {node} outside padded space")
        counts[blocks.block_of_node(node)] += 1
    return BlockNodeCounts(blocks=blocks, per_block=tuple(counts))


def falling_factorial(n: int, r: int) -> int:
    if r < 0 or r > n:
        raise CorrespondenceError(f"invalid falling factorial arguments n={n} r={r}")
    result = 1
    for offset in range(r):
        result *= n - offset
    return result


def active_image_assignment_count(
    blocks: PaddedBlockStructure, support_counts: BlockNodeCounts
) -> int:
    count = 1
    for block_index, support_size in enumerate(support_counts.per_block):
        count *= falling_factorial(blocks.padded_size_tuple[block_index], support_size)
    return count


@dataclass(frozen=True, slots=True)
class BlockActiveImageChoice:
    block_index: int
    candidate_assignments: tuple[tuple[tuple[int, int], ...], ...]


def enumerate_active_image_maps(
    blocks: PaddedBlockStructure,
    active_support_nodes: Sequence[int],
) -> Iterator[ActiveImageMap]:
    by_block: defaultdict[int, list[int]] = defaultdict(list)
    for node in sorted(active_support_nodes):
        by_block.setdefault(blocks.block_of_node(node), []).append(node)
    choices: list[BlockActiveImageChoice] = []
    for block_index in range(len(blocks.padded_size_tuple)):
        active_targets = by_block.get(block_index, [])
        block_sources = list(blocks.block_index_range(block_index))
        if len(block_sources) < len(active_targets):
            raise CorrespondenceError(
                "active-image enumeration requires at least as many source nodes as active targets"
            )
        if active_targets:
            candidates = tuple(
                tuple(zip(active_targets, assignment, strict=True))
                for assignment in itertools.permutations(block_sources, len(active_targets))
            )
        else:
            candidates = ((),)
        choices.append(
            BlockActiveImageChoice(block_index=block_index, candidate_assignments=candidates)
        )
    for combination in itertools.product(*(choice.candidate_assignments for choice in choices)):
        assignments: list[tuple[int, int]] = []
        for pairs in combination:
            assignments.extend(pairs)
        yield ActiveImageMap(blocks=blocks, assignments=tuple(assignments))
