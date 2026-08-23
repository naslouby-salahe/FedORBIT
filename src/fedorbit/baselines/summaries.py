from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fedorbit.orbit.correspondence import PaddedBlockStructure
from fedorbit.orbit.diagnostics import analytic_orbit_mean
from fedorbit.orbit.rectangular import build_rectangular_hull


@dataclass(frozen=True, slots=True)
class CoarseBlockSummary:
    matrix: NDArray[np.float64]


def _real_source_indices(blocks: PaddedBlockStructure, block_index: int) -> range:
    return range(
        blocks.block_index_range(block_index).start,
        blocks.block_index_range(block_index).start + blocks.source_real_counts[block_index],
    )


def _target_block_range(blocks: PaddedBlockStructure, block_index: int) -> range:
    return blocks.block_index_range(block_index)


def coarse_block_mean_matrix(
    blocks: PaddedBlockStructure,
    response_matrix: NDArray[np.float64],
) -> CoarseBlockSummary:
    size = blocks.total_padded_nodes
    summary = np.zeros((size, size), dtype=np.float64)
    block_count = len(blocks.padded_size_tuple)
    for group_source in range(block_count):
        source_rows = _real_source_indices(blocks, group_source)
        for group_target in range(block_count):
            target_columns = _real_source_indices(blocks, group_target)
            block_entries = response_matrix[np.ix_(source_rows, target_columns)]
            if block_entries.size == 0:
                continue
            block_mean = float(np.mean(block_entries))
            for row in _target_block_range(blocks, group_source):
                for column in _target_block_range(blocks, group_target):
                    summary[row, column] = block_mean
    return CoarseBlockSummary(matrix=summary)


def coarse_block_min_matrix(
    blocks: PaddedBlockStructure,
    response_matrix: NDArray[np.float64],
) -> CoarseBlockSummary:
    size = blocks.total_padded_nodes
    summary = np.zeros((size, size), dtype=np.float64)
    block_count = len(blocks.padded_size_tuple)
    for group_source in range(block_count):
        source_rows = _real_source_indices(blocks, group_source)
        for group_target in range(block_count):
            target_columns = _real_source_indices(blocks, group_target)
            block_entries = response_matrix[np.ix_(source_rows, target_columns)]
            if block_entries.size == 0:
                continue
            block_minimum = float(np.min(block_entries))
            for row in _target_block_range(blocks, group_source):
                for column in _target_block_range(blocks, group_target):
                    summary[row, column] = block_minimum
    return CoarseBlockSummary(matrix=summary)


def orbit_mean_matrix(
    blocks: PaddedBlockStructure,
    response_matrix: NDArray[np.float64],
) -> CoarseBlockSummary:
    return CoarseBlockSummary(matrix=analytic_orbit_mean(blocks, response_matrix))


def matched_resource_rectangular_lower_bounds(
    blocks: PaddedBlockStructure,
    lower_response_matrix: NDArray[np.float64],
    upper_response_matrix: NDArray[np.float64],
) -> CoarseBlockSummary:
    hull = build_rectangular_hull(blocks, lower_response_matrix, upper_response_matrix)
    return CoarseBlockSummary(matrix=hull.lower_bounds)
