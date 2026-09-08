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
)
from fedorbit.types import Coefficient, Score, Tolerance


def verify_correspondence_certificate(
    correspondence: BlockCorrespondence,
    reported_objective: Coefficient,
    action: CurriculumAction,
    objective_tolerance: Tolerance,
) -> bool:
    from fedorbit.optimization.objective import evaluate_objective

    recomputed = evaluate_objective(action, correspondence)
    return abs(recomputed - reported_objective) <= objective_tolerance


def verify_exactness_certificate(
    solver_value: Coefficient,
    exhaustive_truth_value: Coefficient,
    exact_tolerance: Tolerance,
) -> bool:
    return abs(solver_value - exhaustive_truth_value) <= exact_tolerance


@dataclass(frozen=True, slots=True)
class RectangularHull:
    blocks: PaddedBlockStructure
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]

    def __post_init__(self) -> None:
        size = self.blocks.total_padded_nodes
        expected = (size, size)
        if self.lower_bounds.shape != expected or self.upper_bounds.shape != expected:
            raise ActionSpaceError(
                f"rectangular hull bounds shape must be {expected}, got "
                f"{self.lower_bounds.shape} and {self.upper_bounds.shape}"
            )


def build_rectangular_hull(
    blocks: PaddedBlockStructure,
    lower_response_matrix: NDArray[np.float64],
    upper_response_matrix: NDArray[np.float64],
) -> RectangularHull:
    size = blocks.total_padded_nodes
    lower = np.full((size, size), math.inf, dtype=np.float64)
    upper = np.full((size, size), -math.inf, dtype=np.float64)
    for correspondence in enumerate_block_permutations(blocks):
        permuted_lower = correspondence.permute_response_matrix(lower_response_matrix)
        permuted_upper = correspondence.permute_response_matrix(upper_response_matrix)
        lower = np.minimum(lower, permuted_lower)
        upper = np.maximum(upper, permuted_upper)
    return RectangularHull(blocks=blocks, lower_bounds=lower, upper_bounds=upper)


def h_rect_from_hull(alpha: CurriculumAction, hull: RectangularHull) -> Score:
    return Score(float(alpha.problem.target_importance @ hull.lower_bounds @ alpha.coordinates))


def orbit_value_over_candidates(
    action_candidates: Sequence[CurriculumAction],
    problem: RobustActionProblem,
    orbit: Sequence[BlockCorrespondence],
) -> Score:
    best = -math.inf
    for candidate in action_candidates:
        minimum_response = math.inf
        for correspondence in orbit:
            permuted = correspondence.permute_response_matrix(problem.lower_response_matrix)
            response_value = float(problem.target_importance @ permuted @ candidate.coordinates)
            minimum_response = min(minimum_response, response_value)
        objective = minimum_response - float(problem.linear_costs @ candidate.coordinates)
        best = max(best, objective)
    if math.isinf(best):
        raise ActionSpaceError("orbit value requires at least one candidate action")
    return Score(best)


def rectangular_value_over_candidates(
    action_candidates: Sequence[CurriculumAction],
    problem: RobustActionProblem,
    hull: RectangularHull,
) -> Score:
    best = -math.inf
    for candidate in action_candidates:
        objective = h_rect_from_hull(candidate, hull) - float(
            problem.linear_costs @ candidate.coordinates
        )
        best = max(best, objective)
    if math.isinf(best):
        raise ActionSpaceError("rectangular value requires at least one candidate action")
    return Score(best)


def robust_coupling_gap(
    action_candidates: Sequence[CurriculumAction],
    problem: RobustActionProblem,
    orbit: Sequence[BlockCorrespondence],
    hull: RectangularHull,
) -> Score:
    return Score(
        orbit_value_over_candidates(action_candidates, problem, orbit)
        - rectangular_value_over_candidates(action_candidates, problem, hull)
    )
