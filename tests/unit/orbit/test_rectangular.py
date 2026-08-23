from __future__ import annotations

import numpy as np
import pytest

from fedorbit.domain.enums import CoarseGroup
from fedorbit.orbit.correspondence import (
    BlockCorrespondence,
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.orbit.objective import (
    ActionSpaceError,
    CurriculumAction,
    RobustActionProblem,
)
from fedorbit.orbit.rectangular import (
    build_rectangular_hull,
    h_rect_from_hull,
    orbit_value_over_candidates,
    rectangular_value_over_candidates,
    robust_coupling_gap,
)


def _two_block_problem() -> RobustActionProblem:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
    )
    return RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=np.array(
            [
                [0.10, 0.90, 0.00, 0.00],
                [0.70, 0.30, 0.00, 0.00],
                [0.00, 0.00, 0.20, -0.10],
                [0.00, 0.00, -0.05, 0.40],
            ]
        ),
        upper_response_matrix=np.zeros((4, 4)),
        target_importance=np.array([1.0, 0.5, 0.25, 0.125]),
        coordinate_caps=np.array([0.25, 0.25, 0.25, 0.25]),
        linear_costs=np.zeros(4),
        total_budget=0.50,
        principal_support=2,
    )


def test_hull_matches_entrywise_brute_force_extrema() -> None:
    problem = _two_block_problem()
    upper_response = problem.lower_response_matrix + 0.07
    hull = build_rectangular_hull(problem.blocks, problem.lower_response_matrix, upper_response)
    size = problem.size
    lower = np.full((size, size), np.inf)
    upper = np.full((size, size), -np.inf)
    for correspondence in enumerate_block_permutations(problem.blocks):
        permuted_lower = correspondence.permute_response_matrix(problem.lower_response_matrix)
        permuted_upper = correspondence.permute_response_matrix(upper_response)
        lower = np.minimum(lower, permuted_lower)
        upper = np.maximum(upper, permuted_upper)
    assert np.allclose(hull.lower_bounds, lower)
    assert np.allclose(hull.upper_bounds, upper)


def test_upper_hull_bounds_dominate_permuted_upper_entries() -> None:
    problem = _two_block_problem()
    upper_response = problem.lower_response_matrix + 0.07
    hull = build_rectangular_hull(problem.blocks, problem.lower_response_matrix, upper_response)
    for correspondence in enumerate_block_permutations(problem.blocks):
        permuted_upper = correspondence.permute_response_matrix(upper_response)
        assert np.all(hull.upper_bounds >= permuted_upper - 1e-15)
        permuted_lower = correspondence.permute_response_matrix(problem.lower_response_matrix)
        assert np.all(hull.lower_bounds <= permuted_lower + 1e-15)


def test_hull_bounds_are_ordered() -> None:
    problem = _two_block_problem()
    rng_values = np.abs(problem.lower_response_matrix) + 0.01
    hull = build_rectangular_hull(problem.blocks, problem.lower_response_matrix, rng_values)
    assert np.all(hull.lower_bounds <= hull.upper_bounds + 1e-15)


def test_orbit_and_rectangular_values_over_candidates() -> None:
    problem = _two_block_problem()
    orbit = list(enumerate_block_permutations(problem.blocks))
    alpha = CurriculumAction(problem, np.array([0.25, 0.25, 0.0, 0.0]))
    candidates = (alpha,)
    exact = orbit_value_over_candidates(candidates, problem, orbit)
    manual_min_response = min(
        float(
            problem.target_importance
            @ correspondence.permute_response_matrix(problem.lower_response_matrix)
            @ alpha.coordinates
        )
        for correspondence in orbit
    )
    assert exact == pytest.approx(manual_min_response)
    hull = build_rectangular_hull(
        problem.blocks, problem.lower_response_matrix, problem.upper_response_matrix
    )
    rect = rectangular_value_over_candidates(candidates, problem, hull)
    assert rect == pytest.approx(h_rect_from_hull(alpha, hull))
    with pytest.raises(ActionSpaceError):
        orbit_value_over_candidates((), problem, orbit)
    with pytest.raises(ActionSpaceError):
        rectangular_value_over_candidates((), problem, hull)


def test_coupling_gap_matches_hand_computed_single_block_fixture() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 2},
        {CoarseGroup.DISRUPTION: 2},
    )
    problem = RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=np.array([[0.1, 0.9], [0.7, 0.3]]),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.array([1.0, 0.0]),
        coordinate_caps=np.array([0.5, 0.5]),
        linear_costs=np.zeros(2),
        total_budget=1.0,
        principal_support=2,
    )
    alpha = CurriculumAction(problem, np.array([0.4, 0.1]))
    identity = BlockCorrespondence.identity(blocks)
    swap = BlockCorrespondence(blocks, (1, 0))
    orbit = [identity, swap]
    hull = build_rectangular_hull(
        blocks, problem.lower_response_matrix, problem.upper_response_matrix
    )
    gap = robust_coupling_gap((alpha,), problem, orbit, hull)
    h_orb_value = min(
        float(
            problem.target_importance
            @ c.permute_response_matrix(problem.lower_response_matrix)
            @ alpha.coordinates
        )
        for c in orbit
    )
    assert h_orb_value == pytest.approx(0.13)
    assert h_rect_from_hull(alpha, hull) == pytest.approx(0.11)
    assert gap == pytest.approx(0.02)
