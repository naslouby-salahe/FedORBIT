from __future__ import annotations

import numpy as np
import pytest

from fedorbit.optimization.certificates import build_rectangular_hull
from fedorbit.optimization.correspondence import (
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.optimization.diagnostics import (
    MapValueDiagnostics,
    analytic_orbit_mean,
    coupling_upper_bound_diagnostic,
    fixed_action_rectangularization_gap,
    map_value_diagnostics,
    orbit_radius_2_norm,
)
from fedorbit.optimization.objective import (
    ActionSpaceError,
    CurriculumAction,
    RobustActionProblem,
    zero_action,
)
from fedorbit.types import CoarseGroup


def _problem_with(
    lower: np.ndarray, importance: np.ndarray, caps: np.ndarray, costs: np.ndarray
) -> RobustActionProblem:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: lower.shape[0]},
        {CoarseGroup.DISRUPTION: lower.shape[0]},
    )
    return RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=lower,
        upper_response_matrix=np.zeros_like(lower),
        target_importance=importance,
        coordinate_caps=caps,
        linear_costs=costs,
        total_budget=float(caps.sum()),
        principal_support=2,
    )


def test_fixed_action_gap_nonnegative_and_matches_hand_value() -> None:
    problem = _problem_with(
        np.array([[0.1, 0.9], [0.7, 0.3]]),
        np.array([1.0, 0.0]),
        np.array([0.5, 0.5]),
        np.zeros(2),
    )
    alpha = CurriculumAction(problem, np.array([0.4, 0.1]))
    orbit = list(enumerate_block_permutations(problem.blocks))
    hull = build_rectangular_hull(
        problem.blocks, problem.lower_response_matrix, problem.upper_response_matrix
    )
    gap = fixed_action_rectangularization_gap(alpha, orbit, hull.lower_bounds)
    assert gap == pytest.approx(0.02)
    with pytest.raises(ActionSpaceError):
        fixed_action_rectangularization_gap(alpha, orbit, hull.lower_bounds + 1.0)


def test_fixed_action_gap_never_negative_on_deterministic_instances() -> None:
    rng = np.random.default_rng(11)
    for _ in range(25):
        size = int(rng.integers(2, 4))
        problem = _problem_with(
            rng.uniform(-1.0, 1.0, size=(size, size)),
            np.abs(rng.uniform(0.0, 1.0, size=size)) + 0.01,
            np.full(size, 1.0 / size),
            np.zeros(size),
        )
        alpha = CurriculumAction(problem, rng.uniform(0.0, 1.0 / size, size=size))
        orbit = list(enumerate_block_permutations(problem.blocks))
        hull = build_rectangular_hull(
            problem.blocks, problem.lower_response_matrix, problem.upper_response_matrix
        )
        gap = fixed_action_rectangularization_gap(alpha, orbit, hull.lower_bounds)
        assert gap >= 0.0


def test_analytic_orbit_mean_equals_enumerated_mean() -> None:
    rng = np.random.default_rng(23)
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
    )
    matrix = rng.uniform(-1.0, 1.0, size=(4, 4))
    mean = analytic_orbit_mean(blocks, matrix)
    enumerated = np.mean(
        [
            correspondence.permute_response_matrix(matrix)
            for correspondence in enumerate_block_permutations(blocks)
        ],
        axis=0,
    )
    assert np.allclose(mean, enumerated)


def test_analytic_mean_constant_blocks_are_fixed_points() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
    )
    constant = np.array(
        [
            [0.10, 0.10, 0.30, 0.30],
            [0.10, 0.10, 0.30, 0.30],
            [0.30, 0.30, 0.20, 0.20],
            [0.30, 0.30, 0.20, 0.20],
        ]
    )
    mean = analytic_orbit_mean(blocks, constant)
    for correspondence in enumerate_block_permutations(blocks):
        assert np.allclose(correspondence.permute_response_matrix(constant), mean)


def test_orbit_radius_is_max_spectral_deviation() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,), {CoarseGroup.DISRUPTION: 3}, {CoarseGroup.DISRUPTION: 3}
    )
    matrix = np.array([[0.2, -0.1, 0.05], [0.15, 0.3, -0.2], [-0.05, 0.1, 0.25]])
    radius = orbit_radius_2_norm(blocks, matrix).radius
    mean = analytic_orbit_mean(blocks, matrix)
    brute = max(
        float(np.linalg.norm(correspondence.permute_response_matrix(matrix) - mean, ord=2))
        for correspondence in enumerate_block_permutations(blocks)
    )
    assert radius == pytest.approx(brute)
    assert radius >= 0.0


def test_map_value_diagnostics_respect_orbit_radius_bound() -> None:
    rng = np.random.default_rng(31)
    tolerance = 1e-9
    for _ in range(8):
        size = int(rng.integers(2, 4))
        problem = _problem_with(
            rng.uniform(-0.5, 0.5, size=(size, size)),
            np.abs(rng.uniform(0.0, 1.0, size=size)) + 0.01,
            np.full(size, 1.0 / size),
            np.zeros(size),
        )
        candidates = (
            *(CurriculumAction(problem, rng.uniform(0.0, 1.0 / size, size=size)) for _ in range(3)),
            zero_action(problem),
        )
        orbit = list(enumerate_block_permutations(problem.blocks))
        diagnostics = map_value_diagnostics(candidates, problem, orbit, tolerance)
        assert diagnostics.exact_map_action_value >= -tolerance
        bound = (
            2.0
            * diagnostics.orbit_radius_bound
            * diagnostics.importance_norm
            * diagnostics.action_radius
        )
        assert diagnostics.bound == pytest.approx(bound)
        assert not diagnostics.violates_bound(tolerance)


def test_map_value_diagnostics_flag_bound_violation() -> None:
    problem = _problem_with(
        np.array([[7.0, 4.0], [3.0, 2.0]]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.zeros(2),
    )
    candidates = (CurriculumAction(problem, np.array([1.0, 0.0])),)
    orbit = list(enumerate_block_permutations(problem.blocks))
    diagnostics = map_value_diagnostics(candidates, problem, orbit, 1e-9)
    tampered = MapValueDiagnostics(
        pre_map_value=diagnostics.pre_map_value,
        post_map_value=diagnostics.post_map_value,
        exact_map_action_value=diagnostics.bound + 1.0,
        action_radius=diagnostics.action_radius,
        importance_norm=diagnostics.importance_norm,
        orbit_radius_bound=diagnostics.orbit_radius_bound,
        bound=diagnostics.bound,
    )
    assert tampered.violates_bound(1e-9)


def test_coupling_upper_bound_diagnostic_dominates_realized_gap() -> None:
    problem = _problem_with(
        np.array([[0.1, 0.9], [0.7, 0.3]]),
        np.array([1.0, 0.0]),
        np.array([0.5, 0.5]),
        np.zeros(2),
    )
    alpha = CurriculumAction(problem, np.array([0.4, 0.1]))
    orbit = list(enumerate_block_permutations(problem.blocks))
    upper_response = problem.lower_response_matrix + 0.1
    hull = build_rectangular_hull(problem.blocks, problem.lower_response_matrix, upper_response)
    diagnostic = coupling_upper_bound_diagnostic(
        (alpha,), problem, hull.lower_bounds, hull.upper_bounds
    )
    gap = fixed_action_rectangularization_gap(alpha, orbit, hull.lower_bounds)
    assert diagnostic.value >= gap
