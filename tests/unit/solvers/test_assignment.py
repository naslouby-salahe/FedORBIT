from __future__ import annotations

import numpy as np
import pytest

from fedorbit.solvers.assignment import (
    AssignmentError,
    solve_minimum_cost_assignment,
)


def test_optimal_value_matches_enumeration_on_small_matrix() -> None:
    costs = np.array([[4.0, 1.0], [2.0, 3.0]])
    result = solve_minimum_cost_assignment(costs, 0.0)
    assert result.column_for_row == (1, 0)
    assert result.objective_value == pytest.approx(3.0)
    assert sorted(result.column_for_row) == [0, 1]


def test_lexicographically_smallest_among_ties() -> None:
    costs = np.array([[0.0, 0.0], [0.0, 0.0]])
    result = solve_minimum_cost_assignment(costs, 1e-12)
    assert result.column_for_row == (0, 1)
    transposed = np.array([[1.0, 1.0], [1.0, 1.0]])
    tied = solve_minimum_cost_assignment(transposed, 1e-12)
    assert tied.column_for_row == (0, 1)
    assert tied.objective_value == pytest.approx(2.0)


def test_near_tie_within_tolerance_prefers_smaller_columns() -> None:
    costs = np.array([[1.0, 1.0 + 5e-13], [1.0 + 5e-13, 1.0]])
    strict = solve_minimum_cost_assignment(costs, 0.0)
    tolerant = solve_minimum_cost_assignment(costs, 1e-9)
    assert strict.column_for_row in {(0, 1), (1, 0)}
    assert tolerant.column_for_row == (0, 1)


def test_three_by_three_permutation_optimum() -> None:
    rng = np.random.default_rng(5)
    costs = rng.uniform(0.0, 10.0, size=(3, 3))
    result = solve_minimum_cost_assignment(costs, 1e-12)
    brute = min(
        sum(float(costs[row, column]) for row, column in enumerate(permutation))
        for permutation in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    )
    assert result.objective_value == pytest.approx(brute)
    assert len(set(result.column_for_row)) == 3


def test_rejects_nonsquare_nan_and_negative_tolerance() -> None:
    with pytest.raises(AssignmentError):
        solve_minimum_cost_assignment(np.zeros((2, 3)), 0.0)
    with pytest.raises(AssignmentError):
        matrix = np.zeros((2, 2))
        matrix[0, 0] = np.nan
        solve_minimum_cost_assignment(matrix, 0.0)
    with pytest.raises(AssignmentError):
        solve_minimum_cost_assignment(np.zeros((2, 2)), -1.0)
    with pytest.raises(AssignmentError):
        solve_minimum_cost_assignment(np.zeros((0, 0)), 0.0)
