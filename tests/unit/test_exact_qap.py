from __future__ import annotations

import numpy as np
import pytest

from fedorbit.config.loading import configured, load_fedorbit_config
from fedorbit.optimization.correspondence import (
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.optimization.exact_qap import (
    QapUncertifiedError,
    fixed_action_worst_correspondence_qap,
    point_correspondence_commitment,
    solve_robust_action_qap,
)
from fedorbit.optimization.exact_sparse import solve_robust_action
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    evaluate_objective,
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


def _exhaustive_worst_value(problem: RobustActionProblem, alpha: CurriculumAction) -> float:
    return min(evaluate_objective(alpha, c) for c in enumerate_block_permutations(problem.blocks))


def test_qap_separator_matches_exhaustive_orbit_truth() -> None:
    for seed in range(3):
        problem = _two_block_problem(seed)
        alpha = CurriculumAction(problem, np.array([0.2, 0.1, 0.15, 0.05]))
        result = fixed_action_worst_correspondence_qap(problem, alpha)
        certificate = result.require_certified()
        correspondence = certificate.correspondence
        value = certificate.objective_value
        truth = _exhaustive_worst_value(problem, alpha)
        assert value == pytest.approx(truth, abs=1e-9)
        assert evaluate_objective(alpha, correspondence) == pytest.approx(truth, abs=1e-9)


def test_uncertified_result_refuses_to_release_action() -> None:
    config = load_fedorbit_config()
    problem = _two_block_problem(21)
    alpha = CurriculumAction(problem, np.array([0.2, 0.0, 0.0, 0.0]))
    tiny_limit = config.solvers.generic_exact_qap.model_copy(
        update={"wall_time_seconds_per_solve": 0.001}
    )
    limited_config = config.model_copy(
        update={"solvers": config.solvers.model_copy(update={"generic_exact_qap": tiny_limit})}
    )
    with configured(limited_config):
        result = fixed_action_worst_correspondence_qap(problem, alpha)
    if not result.certified:
        with pytest.raises(QapUncertifiedError):
            result.require_certified()
        assert result.correspondence is None
        assert result.objective_value is None


def test_point_correspondence_recovers_structural_match() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 2},
        {CoarseGroup.DISRUPTION: 2},
    )
    source_matrix = np.array([[0.9, -0.4], [0.5, -0.1]])
    target_matrix = np.array([[-0.1, 0.5], [-0.4, 0.9]])
    result = point_correspondence_commitment(source_matrix, target_matrix, blocks)
    certificate = result.require_certified()
    distance, images = certificate.objective_value, certificate.correspondence.images
    del distance
    brute: dict[tuple[int, ...], float] = {}
    for correspondence in enumerate_block_permutations(blocks):
        permuted = correspondence.permute_response_matrix(source_matrix)
        squared = float(np.sum((permuted - target_matrix) ** 2))
        brute[correspondence.images] = squared
    best_distance = min(brute.values())
    assert brute[images] == pytest.approx(best_distance, abs=1e-12)
    assert result.objective_value == pytest.approx(best_distance, abs=1e-9)


def test_point_correspondence_tie_prefers_lexicographically_smallest() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 2},
        {CoarseGroup.DISRUPTION: 2},
    )
    symmetric = np.array([[0.5, 0.2], [0.2, 0.8]])
    result = point_correspondence_commitment(symmetric, symmetric, blocks)
    correspondence = result.require_certified().correspondence
    assert correspondence.images == (0, 1)


def test_qap_robust_action_agrees_with_exact_sparse_solver() -> None:
    problem = _two_block_problem(33)
    sparse_solution = solve_robust_action(problem, support_limit=1)
    qap_outcome = solve_robust_action_qap(problem, support_limit=1)
    assert qap_outcome.is_exact
    assert qap_outcome.certified_solution is not None
    qap_solution = qap_outcome.certified_solution
    assert qap_solution.certified_robust_value == pytest.approx(
        sparse_solution.certified_robust_value, abs=1e-8
    )
    worst_sparse = min(
        evaluate_objective(sparse_solution.selected_action, correspondence)
        for correspondence in enumerate_block_permutations(problem.blocks)
    )
    worst_qap = min(
        evaluate_objective(qap_solution.certified_action, correspondence)
        for correspondence in enumerate_block_permutations(problem.blocks)
    )
    assert worst_qap == pytest.approx(worst_sparse, abs=1e-8)


def test_zero_action_candidate_present_in_qap_method() -> None:
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
    outcome = solve_robust_action_qap(negative_problem)
    assert outcome.certified_solution is not None
    selected = outcome.certified_solution.certified_action
    assert int(np.count_nonzero(selected.coordinates)) == 0
