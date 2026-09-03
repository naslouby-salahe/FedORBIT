from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
    enumerate_block_permutations,
)
from fedorbit.optimization.objective import (
    ActionSpaceError,
    CurriculumAction,
    RobustActionProblem,
    h_orb,
    h_rect,
    robust_post_map_value,
    robust_pre_map_value,
)


def fixed_action_rectangularization_gap(
    alpha: CurriculumAction,
    orbit: Sequence[BlockCorrespondence],
    lower_hull: NDArray[np.float64],
) -> float:
    gap = h_orb(alpha, orbit) - h_rect(alpha, lower_hull)
    if gap < 0.0:
        raise ActionSpaceError(
            f"fixed-action rectangularization gap must be nonnegative, got {gap}"
        )
    return gap


def _same_group_block_means(
    block_entries: NDArray[np.float64],
) -> tuple[float, float]:
    diagonal_mean = float(np.mean(np.diag(block_entries)))
    off_diagonal_mask = ~np.eye(block_entries.shape[0], dtype=bool)
    off_diagonal_values = block_entries[off_diagonal_mask]
    off_diagonal_mean = float(np.mean(off_diagonal_values)) if off_diagonal_values.size else 0.0
    return diagonal_mean, off_diagonal_mean


def _fill_target_block(
    mean: NDArray[np.float64],
    targets_rows: range,
    targets_columns: range,
    fill_diagonal: float,
    fill_off_diagonal: float,
) -> None:
    for target_k in targets_rows:
        for target_j in targets_columns:
            value = fill_diagonal if target_k == target_j else fill_off_diagonal
            mean[target_k, target_j] = value


def analytic_orbit_mean(
    blocks: PaddedBlockStructure,
    response_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    size = blocks.total_padded_nodes
    if response_matrix.shape != (size, size):
        raise ActionSpaceError("response matrix shape mismatch for analytic orbit mean")
    mean = np.zeros((size, size), dtype=np.float64)
    block_count = len(blocks.padded_size_tuple)
    for group_source in range(block_count):
        rows = blocks.block_index_range(group_source)
        for group_target in range(block_count):
            columns = blocks.block_index_range(group_target)
            block_entries = response_matrix[np.ix_(rows, columns)]
            if group_source == group_target:
                diagonal_mean, off_diagonal_mean = _same_group_block_means(block_entries)
                _fill_target_block(mean, rows, columns, diagonal_mean, off_diagonal_mean)
            else:
                block_mean = float(np.mean(block_entries))
                _fill_target_block(mean, rows, columns, block_mean, block_mean)
    return mean


@dataclass(frozen=True, slots=True)
class OrbitRadius:
    radius: float


def orbit_radius_2_norm(
    blocks: PaddedBlockStructure,
    response_matrix: NDArray[np.float64],
) -> OrbitRadius:
    mean = analytic_orbit_mean(blocks, response_matrix)
    radius = 0.0
    for correspondence in enumerate_block_permutations(blocks):
        permuted = correspondence.permute_response_matrix(response_matrix)
        spectral = float(np.linalg.norm(permuted - mean, ord=2))
        radius = max(radius, spectral)
    return OrbitRadius(radius=radius)


@dataclass(frozen=True, slots=True)
class MapValueDiagnostics:
    pre_map_value: float
    post_map_value: float
    exact_map_action_value: float
    action_radius: float
    importance_norm: float
    orbit_radius_bound: float
    bound: float

    def violates_bound(self, tolerance: float) -> bool:
        return self.exact_map_action_value > self.bound + tolerance


def map_value_diagnostics(
    action_candidates: Sequence[CurriculumAction],
    problem: RobustActionProblem,
    orbit: Sequence[BlockCorrespondence],
    tolerance: float,
) -> MapValueDiagnostics:
    pre_map = robust_pre_map_value(action_candidates, orbit)
    post_map = robust_post_map_value(action_candidates, orbit)
    delta_map = post_map - pre_map
    action_radius = max(
        (float(np.linalg.norm(candidate.coordinates, ord=2)) for candidate in action_candidates),
        default=0.0,
    )
    importance_norm = float(np.linalg.norm(problem.target_importance, ord=2))
    radius = orbit_radius_2_norm(problem.blocks, problem.lower_response_matrix).radius
    bound = 2.0 * radius * importance_norm * action_radius
    diagnostics = MapValueDiagnostics(
        pre_map_value=pre_map,
        post_map_value=post_map,
        exact_map_action_value=delta_map,
        action_radius=action_radius,
        importance_norm=importance_norm,
        orbit_radius_bound=radius,
        bound=bound,
    )
    if diagnostics.violates_bound(tolerance):
        raise ActionSpaceError(
            f"exact-map action value {delta_map} exceeds orbit-radius bound {bound}"
        )
    return diagnostics


@dataclass(frozen=True, slots=True)
class CouplingUpperBoundDiagnostic:
    value: float


def coupling_upper_bound_diagnostic(
    action_candidates: Sequence[CurriculumAction],
    problem: RobustActionProblem,
    hull_lower_bounds: NDArray[np.float64],
    hull_upper_bounds: NDArray[np.float64],
) -> CouplingUpperBoundDiagnostic:
    expected = (problem.size, problem.size)
    if hull_lower_bounds.shape != expected or hull_upper_bounds.shape != expected:
        raise ActionSpaceError("hull bounds shape mismatch for coupling upper-bound diagnostic")
    width = hull_upper_bounds - hull_lower_bounds
    best = -math.inf
    for candidate in action_candidates:
        value = float(problem.target_importance @ width @ candidate.coordinates)
        best = max(best, value)
    if math.isinf(best):
        raise ActionSpaceError(
            "coupling upper-bound diagnostic requires at least one candidate action"
        )
    return CouplingUpperBoundDiagnostic(value=best)


def orbit_is_nontrivial(orbit_size: int) -> bool:
    return orbit_size > 1
