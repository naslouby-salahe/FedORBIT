from __future__ import annotations

import numpy as np

from fedorbit.optimization.assignment import solve_minimum_cost_assignment


def test_solver_pipeline_returns_lexicographic_optimum_under_tie() -> None:
    costs = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    result = solve_minimum_cost_assignment(costs, 1e-12)
    assert result.column_for_row == (0, 1)
    assert result.objective_value == 0.0
