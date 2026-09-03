from __future__ import annotations

import numpy as np

from fedorbit.optimization.assignment import solve_minimum_cost_assignment


def test_assignment_math_uses_lexicographic_tie_resolution() -> None:
    result = solve_minimum_cost_assignment(np.asarray(((0.0, 0.0), (0.0, 0.0))), 1e-12)
    assert result.column_for_row == (0, 1)
    assert result.objective_value == 0.0
