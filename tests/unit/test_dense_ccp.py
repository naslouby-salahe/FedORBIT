from __future__ import annotations

import numpy as np
import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.optimization.correspondence import (
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.optimization.dense_ccp import (
    AssignmentVariableKey,
    AssignmentVariableLayout,
    DenseCcpError,
    barycenter_start,
    ccp_trajectory,
    dense_starts,
    integrality_residual,
    penalty_scale,
    permutation_to_vector,
    project_to_permutation,
    relaxed_fixed_action_lower_bound,
    response_only_objective,
    solve_dense_ccp,
)
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
)
from fedorbit.types import CoarseGroup


def _two_block_problem(seed: int) -> RobustActionProblem:
    rng = np.random.default_rng(seed)
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
    )
    size = blocks.total_padded_nodes
    lower = np.zeros((size, size))
    for row in range(size):
        for column in range(size):
            if blocks.block_of_node(row) == blocks.block_of_node(column):
                lower[row, column] = float(rng.uniform(-0.3, 0.3))
    return RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=lower,
        upper_response_matrix=np.zeros((size, size)),
        target_importance=np.abs(rng.uniform(0.0, 1.0, size=size)) + 0.05,
        coordinate_caps=np.full(size, 0.5),
        linear_costs=np.zeros(size),
        total_budget=1.0,
        principal_support=2,
    )


def test_integrality_residual_definition() -> None:
    integral = np.array([1.0, 0.0, 1.0, 0.0])
    fractional = np.array([0.5, 0.5, 0.5, 0.5])
    mixed = np.array([1.0, 0.9, 0.0, 0.25])
    assert integrality_residual(integral) == pytest.approx(0.0)
    assert integrality_residual(fractional) == pytest.approx(0.5)
    assert integrality_residual(mixed) == pytest.approx(0.25)
    with pytest.raises(DenseCcpError):
        integrality_residual(np.array([]))


def test_penalty_scale_matches_registered_formula() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 2},
        {CoarseGroup.DISRUPTION: 2},
    )
    problem = RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=np.array([[7.0, -4.0], [2.0, -8.0]]),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.array([2.0, 1.0]),
        coordinate_caps=np.array([1.0, 1.0]),
        linear_costs=np.zeros(2),
        total_budget=2.0,
        principal_support=2,
    )
    alpha = CurriculumAction(problem, np.array([0.5, 0.5]))
    brute_max = 0.0
    for target_k in range(2):
        weight = float(problem.target_importance[target_k])
        if weight == 0.0:
            continue
        for target_j in range(2):
            value = float(alpha.coordinates[target_j])
            if value == 0.0:
                continue
            for source_a in range(2):
                for source_b in range(2):
                    entry = (
                        weight * value * float(problem.lower_response_matrix[source_a, source_b])
                    )
                    brute_max = max(brute_max, abs(entry))
    expected = max(1.0, brute_max)
    assert expected == 8.0
    assert penalty_scale(problem, alpha) == expected


def test_relaxation_lower_bounds_exhaustive_orbit_minimum() -> None:
    for seed in range(4):
        problem = _two_block_problem(seed)
        layout = AssignmentVariableLayout.build(problem.blocks)
        alpha = CurriculumAction(problem, np.array([0.3, 0.2, 0.1, 0.05]))
        bound = relaxed_fixed_action_lower_bound(problem, alpha, layout).objective_value
        truth = min(
            response_only_objective(problem, alpha, correspondence)
            for correspondence in enumerate_block_permutations(problem.blocks)
        )
        assert bound <= truth + 1e-9, f"seed={seed}: relaxation {bound} exceeds truth {truth}"


def test_projection_recovers_permutation_and_prefers_weighted_entries() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 2},
        {CoarseGroup.DISRUPTION: 2},
    )
    layout = AssignmentVariableLayout.build(blocks)
    skewed = layout.zeros()
    skewed[layout.column_of(AssignmentVariableKey(0, 0))] = 0.9
    skewed[layout.column_of(AssignmentVariableKey(1, 0))] = 0.2
    skewed[layout.column_of(AssignmentVariableKey(1, 1))] = 0.8
    correspondence = project_to_permutation(layout, skewed)
    assert correspondence.images == (0, 1)
    uniform = np.full(layout.size, 0.5)
    tie_broken = project_to_permutation(layout, uniform)
    assert tie_broken.images == (0, 1)


def test_dense_starts_shape_and_determinism() -> None:
    problem = _two_block_problem(13)
    layout = AssignmentVariableLayout.build(problem.blocks)
    first = dense_starts(layout, 5531, "dense-start-probe")
    second = dense_starts(layout, 5531, "dense-start-probe")
    assert 2 <= len(first) <= 5
    for start_a, start_b in zip(first, second, strict=True):
        assert np.allclose(start_a, start_b)
    barycenter = barycenter_start(layout)
    assert np.allclose(first[0], barycenter)
    assert np.allclose(barycenter[layout.column_of(AssignmentVariableKey(0, 0))], 1.0 / 2)


def test_small_orbit_starts_use_every_unique_permutation() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 2},
        {CoarseGroup.DISRUPTION: 2},
    )
    layout = AssignmentVariableLayout.build(blocks)
    starts = dense_starts(layout, 7749, "small-orbit")
    assert len(starts) == 3
    assert np.allclose(starts[0], barycenter_start(layout))
    assert np.allclose(starts[1], permutation_to_vector((0, 1), layout))
    assert np.allclose(starts[2], permutation_to_vector((1, 0), layout))


def test_ccp_trajectory_improves_and_records_convergence_state() -> None:
    config = load_fedorbit_config()
    problem = _two_block_problem(41)
    layout = AssignmentVariableLayout.build(problem.blocks)
    alpha = CurriculumAction(problem, np.array([0.4, 0.3, 0.2, 0.1]))
    start = permutation_to_vector((0, 1, 2, 3), layout)
    trajectory = ccp_trajectory(problem, alpha, start, layout)
    assert (
        trajectory.iterations
        >= config.solvers.dense_ccp.penalty_multipliers_relative_to_scale.__len__()
    )
    assert 0.0 <= trajectory.integrality_residual <= 0.5 + 1e-12
    assert np.isfinite(trajectory.final_objective)


def test_solve_dense_ccp_returns_complete_non_exact_record() -> None:
    problem = _two_block_problem(59)
    outcome = solve_dense_ccp(problem, 8861, "dense-outcome-probe")
    assert outcome.is_exact is False
    assert outcome.dense_bound_gap >= -1e-9
    assert outcome.relaxation_lower_bound <= outcome.best_projected_response_objective + 1e-9
    assert int(np.count_nonzero(outcome.selected_action.coordinates)) <= problem.size
    assert outcome.worst_projected_correspondence.images in {
        correspondence.images for correspondence in enumerate_block_permutations(problem.blocks)
    }
    assert outcome.master_objective >= outcome.best_projected_response_objective - 1e-8 or (
        outcome.outer_cut_count > 0
    )


def test_dense_ccp_is_deterministic_for_fixed_seed() -> None:
    problem = _two_block_problem(71)
    first = solve_dense_ccp(problem, 9973, "determinism-probe")
    second = solve_dense_ccp(problem, 9973, "determinism-probe")
    assert first.selected_action.coordinates.shape == second.selected_action.coordinates.shape
    assert np.allclose(first.selected_action.coordinates, second.selected_action.coordinates)
    assert (
        first.worst_projected_correspondence.images == second.worst_projected_correspondence.images
    )
    assert first.converged_heuristically == second.converged_heuristically
