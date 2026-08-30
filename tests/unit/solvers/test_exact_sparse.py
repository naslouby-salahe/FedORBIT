from __future__ import annotations

import numpy as np
import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.domain.enums import CoarseGroup
from fedorbit.orbit.correspondence import (
    BlockCorrespondence,
    BlockNodeCounts,
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.orbit.objective import (
    CurriculumAction,
    RobustActionProblem,
    enumerate_support_coordinate_sets,
    evaluate_objective,
)
from fedorbit.solvers.certificates import SeparatorWorkCertificate, verify_exactness_certificate
from fedorbit.solvers.exact_sparse import (
    RobustActionSolution,
    SparseMasterNonConvergenceError,
    fixed_action_worst_correspondence,
    solve_robust_action,
)


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
                lower[row, column] = float(rng.uniform(-0.2, 0.2))
    return RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=lower,
        upper_response_matrix=np.zeros((size, size)),
        target_importance=np.abs(rng.uniform(0.0, 1.0, size=size)) + 0.05,
        coordinate_caps=np.full(size, 0.25),
        linear_costs=np.zeros(size),
        total_budget=1.0,
        principal_support=2,
    )


def _exhaustive_worst(
    problem: RobustActionProblem, alpha: CurriculumAction
) -> tuple[float, BlockCorrespondence]:
    best_value = np.inf
    best = None
    for correspondence in enumerate_block_permutations(problem.blocks):
        value = evaluate_objective(alpha, correspondence)
        if value < best_value:
            best_value = value
            best = correspondence
    assert best is not None
    return float(best_value), best


def test_separator_matches_exhaustive_orbit_truth() -> None:
    config = load_fedorbit_config()
    settings = config.solvers.exact_sparse
    for seed in range(6):
        problem = _two_block_problem(seed)
        for support in enumerate_support_coordinate_sets(problem, support_limit=2):
            coordinates = np.zeros(problem.size)
            for node in support.nodes:
                coordinates[node] = float(
                    np.random.default_rng(seed * 31 + node).uniform(0.05, 0.25)
                )
            alpha = CurriculumAction(
                problem, coordinates / (2.0 if coordinates.sum() > 1.0 else 1.0)
            )
            outcome = fixed_action_worst_correspondence(
                problem, alpha, settings.lap_objective_tie_tolerance, settings.action_tie_tolerance
            )
            truth_value, truth_correspondence = _exhaustive_worst(problem, alpha)
            assert verify_exactness_certificate(
                outcome.separator_objective,
                truth_value,
                settings.exact_validation_absolute_tolerance,
            ), f"seed={seed} support={support.nodes}"
            assert outcome.worst_correspondence.images == truth_correspondence.images


def test_work_counters_match_registered_formulas() -> None:
    config = load_fedorbit_config()
    settings = config.solvers.exact_sparse
    problem = _two_block_problem(17)
    alpha = CurriculumAction(problem, np.array([0.25, 0.0, 0.25, 0.0]))
    outcome = fixed_action_worst_correspondence(
        problem, alpha, settings.lap_objective_tie_tolerance, settings.action_tie_tolerance
    )
    certificate = SeparatorWorkCertificate(
        active_image_candidates=outcome.active_image_candidates,
        lap_calls=outcome.lap_calls,
    )
    counts = BlockNodeCounts(blocks=problem.blocks, per_block=(1, 1))
    assert certificate.verify_against(problem.blocks, counts.per_block)
    assert outcome.active_image_candidates == 4
    assert outcome.lap_calls == 4 * 2


def test_size_one_completion_counts_as_lap_call() -> None:
    config = load_fedorbit_config()
    settings = config.solvers.exact_sparse
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 3},
        {CoarseGroup.DISRUPTION: 3},
    )
    problem = RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=np.array([[0.1, -0.2, 0.05], [0.3, 0.15, -0.1], [0.0, 0.2, 0.12]]),
        upper_response_matrix=np.zeros((3, 3)),
        target_importance=np.array([1.0, 0.5, 0.25]),
        coordinate_caps=np.full(3, 0.25),
        linear_costs=np.zeros(3),
        total_budget=0.5,
        principal_support=2,
    )
    alpha = CurriculumAction(problem, np.array([0.25, 0.0, 0.0]))
    outcome = fixed_action_worst_correspondence(
        problem, alpha, settings.lap_objective_tie_tolerance, settings.action_tie_tolerance
    )
    assert outcome.active_image_candidates == 3
    assert outcome.lap_calls == 3 * 1


def test_robust_master_certifies_known_hand_solution() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 2},
        {CoarseGroup.DISRUPTION: 2},
    )
    problem = RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=np.array([[7.0, 4.0], [3.0, 2.0]]),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.array([1.0, 1.0]),
        coordinate_caps=np.array([1.0, 1.0]),
        linear_costs=np.zeros(2),
        total_budget=2.0,
        principal_support=2,
    )
    solution = solve_robust_action(problem)
    assert isinstance(solution, RobustActionSolution)
    assert solution.certified_robust_value >= solution.zero_action_value
    orbit_values = [
        evaluate_objective(solution.selected_action, correspondence)
        for correspondence in enumerate_block_permutations(blocks)
    ]
    worst_case = min(orbit_values)
    assert solution.certified_robust_value == pytest.approx(worst_case, abs=1e-8)


def test_cut_cap_exhaustion_raises_non_convergence() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
    )
    rng = np.random.default_rng(99)
    size = blocks.total_padded_nodes
    lower = rng.uniform(-0.2, 0.2, size=(size, size))
    for row in range(size):
        for column in range(size):
            if blocks.block_of_node(row) != blocks.block_of_node(column):
                lower[row, column] = 0.0
    problem = RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=lower,
        upper_response_matrix=np.zeros((size, size)),
        target_importance=np.ones(size),
        coordinate_caps=np.full(size, 0.25),
        linear_costs=np.zeros(size),
        total_budget=1.0,
        principal_support=2,
    )
    with pytest.raises(SparseMasterNonConvergenceError):
        solve_robust_action(problem, maximum_cuts=0)


def test_zero_action_is_explicit_candidate() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 2},
        {CoarseGroup.DISRUPTION: 2},
    )
    negative_problem = RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=np.array([[-0.5, -0.1], [-0.2, -0.4]]),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.array([1.0, 1.0]),
        coordinate_caps=np.array([0.25, 0.25]),
        linear_costs=np.full(2, 0.01),
        total_budget=0.5,
        principal_support=2,
    )
    solution = solve_robust_action(negative_problem)
    assert int(np.count_nonzero(solution.selected_action.coordinates)) == 0
    assert solution.certified_robust_value == pytest.approx(0.0, abs=1e-12)
