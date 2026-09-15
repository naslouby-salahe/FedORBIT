from __future__ import annotations

import numpy as np

from fedorbit.config.loading import active_config
from fedorbit.experiments.synthetic import (
    UnresolvedMapWorld,
    UnresolvedMapWorldKind,
    UnresolvedMapWorldRequest,
    generate_unresolved_map_world,
)
from fedorbit.methods.baselines import optimize_against_fixed_matrix
from fedorbit.optimization.correspondence import (
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.optimization.diagnostics import map_value_diagnostics
from fedorbit.optimization.exact_sparse import solve_robust_action
from fedorbit.optimization.objective import (
    RobustActionProblem,
    build_robust_action_problem,
    exact_map_action_value,
    robust_pre_map_value,
    rounded_action_vector,
    zero_action,
)
from fedorbit.types import CoarseGroup


def test_common_action_world_is_block_constant_and_weight_normalized() -> None:
    world = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.COMMON_ACTION, 101)
    )
    response = world.lower_response_matrix
    assert world.block_pattern == (2, 2)
    assert np.all(response[:2, :2] == response[0, 0])
    assert np.all(response[:2, 2:] == response[0, 2])
    assert np.all(response[2:, :2] == response[2, 0])
    assert np.all(response[2:, 2:] == response[2, 2])
    assert np.isclose(world.target_importance.sum(), 1.0)


def test_unresolved_map_worlds_are_seed_deterministic() -> None:
    request = UnresolvedMapWorldRequest(UnresolvedMapWorldKind.MAP_DEPENDENT, 202)
    first = generate_unresolved_map_world(request)
    second = generate_unresolved_map_world(request)
    assert np.array_equal(first.lower_response_matrix, second.lower_response_matrix)
    assert np.array_equal(first.target_importance, second.target_importance)


def _problem_for(world: UnresolvedMapWorld) -> RobustActionProblem:
    pattern = world.block_pattern
    groups = tuple(CoarseGroup)[: len(pattern)]
    counts = dict(zip(groups, pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    return build_robust_action_problem(
        blocks,
        world.lower_response_matrix,
        world.lower_response_matrix,
        world.target_importance,
        tuple(range(sum(pattern))),
    )


def test_common_action_world_has_positive_robust_value() -> None:
    world = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.COMMON_ACTION, 101)
    )
    problem = _problem_for(world)
    assert solve_robust_action(problem).certified_robust_value > 0.0


def test_exact_sparse_solver_enforces_registered_support_and_budget() -> None:
    world = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.COMMON_ACTION, 101)
    )
    problem = _problem_for(world)
    solution = solve_robust_action(problem)

    assert solution.selected_action.realized_support_size <= problem.principal_support
    assert float(solution.selected_action.coordinates.sum()) <= problem.total_budget
    assert all(
        len(candidate.support_nodes) <= problem.principal_support
        and candidate.certified_action.realized_support_size <= problem.principal_support
        and float(candidate.certified_action.coordinates.sum()) <= problem.total_budget
        for candidate in solution.support_solutions
    )


def test_robust_compromise_world_has_disjoint_map_winners_and_positive_pre_map_value() -> None:
    world = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.ROBUST_COMPROMISE, 101)
    )
    problem = _problem_for(world)
    orbit = list(enumerate_block_permutations(problem.blocks))
    assert len(orbit) > 1

    winners = {
        rounded_action_vector(
            optimize_against_fixed_matrix(
                problem, correspondence.permute_response_matrix(problem.lower_response_matrix)
            ).selected_action,
            1e-12,
        )
        for correspondence in orbit
    }
    assert len(winners) > 1

    per_map = tuple(
        optimize_against_fixed_matrix(
            problem, correspondence.permute_response_matrix(problem.lower_response_matrix)
        ).selected_action
        for correspondence in orbit
    )
    candidates = (solve_robust_action(problem).selected_action, *per_map, zero_action(problem))
    assert robust_pre_map_value(candidates, orbit) > 0.005


def test_map_dependent_world_meets_the_registered_delta_map_floor() -> None:
    world = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.MAP_DEPENDENT, 101)
    )
    problem = _problem_for(world)
    orbit = list(enumerate_block_permutations(problem.blocks))
    per_map = tuple(
        optimize_against_fixed_matrix(
            problem, correspondence.permute_response_matrix(problem.lower_response_matrix)
        ).selected_action
        for correspondence in orbit
    )
    candidates = (solve_robust_action(problem).selected_action, *per_map, zero_action(problem))
    assert exact_map_action_value(candidates, orbit) >= 0.01


def test_map_value_diagnostics_match_direct_joint_orbit_calculation() -> None:
    world = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.MAP_DEPENDENT, 101)
    )
    problem = _problem_for(world)
    orbit = tuple(enumerate_block_permutations(problem.blocks))
    map_actions = tuple(
        optimize_against_fixed_matrix(
            problem, correspondence.permute_response_matrix(problem.lower_response_matrix)
        ).selected_action
        for correspondence in orbit
    )
    candidates = (solve_robust_action(problem).selected_action, *map_actions, zero_action(problem))
    direct_values = np.asarray(
        [
            [
                float(
                    problem.target_importance
                    @ correspondence.permute_response_matrix(problem.lower_response_matrix)
                    @ action.coordinates
                )
                - float(problem.linear_costs @ action.coordinates)
                for correspondence in orbit
            ]
            for action in candidates
        ],
        dtype=np.float64,
    )
    expected_pre = float(np.max(np.min(direct_values, axis=1)))
    expected_post = float(np.min(np.max(direct_values, axis=0)))
    diagnostics = map_value_diagnostics(
        candidates,
        problem,
        orbit,
        active_config().solvers.exact_sparse.exact_validation_absolute_tolerance,
    )

    assert all(action.realized_support_size <= problem.principal_support for action in candidates)
    assert diagnostics.pre_map_value == expected_pre
    assert diagnostics.post_map_value == expected_post
    assert diagnostics.exact_map_action_value == expected_post - expected_pre
    assert 0.0 <= diagnostics.exact_map_action_value <= diagnostics.bound


def test_map_value_diagnostics_has_zero_gap_for_common_action_world() -> None:
    world = generate_unresolved_map_world(
        UnresolvedMapWorldRequest(UnresolvedMapWorldKind.COMMON_ACTION, 101)
    )
    problem = _problem_for(world)
    orbit = tuple(enumerate_block_permutations(problem.blocks))
    candidates = (
        solve_robust_action(problem).selected_action,
        *(
            optimize_against_fixed_matrix(
                problem, correspondence.permute_response_matrix(problem.lower_response_matrix)
            ).selected_action
            for correspondence in orbit
        ),
        zero_action(problem),
    )
    diagnostics = map_value_diagnostics(
        candidates,
        problem,
        orbit,
        active_config().solvers.exact_sparse.exact_validation_absolute_tolerance,
    )

    assert diagnostics.exact_map_action_value == 0.0
    assert diagnostics.bound >= 0.0
