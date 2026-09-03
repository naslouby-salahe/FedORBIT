from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


class AssignmentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BlockwiseAssignmentResult:
    column_for_row: tuple[int, ...]
    objective_value: float


def _completion_cost(
    costs: NDArray[np.float64],
    fixed_rows: tuple[int, ...],
    fixed_columns: tuple[int, ...],
) -> float:
    free_rows = [row for row in range(costs.shape[0]) if row not in fixed_rows]
    free_columns = [column for column in range(costs.shape[1]) if column not in fixed_columns]
    if not free_rows:
        return 0.0
    reduced = costs[np.ix_(free_rows, free_columns)]
    row_indices, column_indices = linear_sum_assignment(reduced)
    selected_rows = np.asarray(row_indices, dtype=np.intp)
    selected_columns = np.asarray(column_indices, dtype=np.intp)
    return float(reduced[selected_rows, selected_columns].sum())


def solve_minimum_cost_assignment(
    costs: NDArray[np.float64],
    tie_tolerance: float,
) -> BlockwiseAssignmentResult:
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise AssignmentError(f"assignment requires a square cost matrix, got shape {costs.shape}")
    if costs.shape[0] == 0:
        raise AssignmentError("assignment requires at least one row")
    if tie_tolerance < 0.0:
        raise AssignmentError("tie tolerance must be nonnegative")
    if bool(np.any(np.isnan(costs))):
        raise AssignmentError("assignment cost matrix contains NaN entries")
    optimum = _completion_cost(costs, (), ())
    if not np.isfinite(optimum):
        raise AssignmentError("assignment optimum is not finite")
    fixed_rows: list[int] = []
    fixed_columns: list[int] = []
    assigned_cost = 0.0
    for row in range(costs.shape[0]):
        for column in sorted(set(range(costs.shape[1])) - set(fixed_columns)):
            candidate_fixed_rows = (*fixed_rows, row)
            candidate_fixed_columns = (*fixed_columns, column)
            completion = _completion_cost(costs, candidate_fixed_rows, candidate_fixed_columns)
            partial = assigned_cost + float(costs[row, column])
            if partial + completion <= optimum + tie_tolerance:
                fixed_rows = list(candidate_fixed_rows)
                fixed_columns = list(candidate_fixed_columns)
                assigned_cost = partial
                break
        else:
            raise AssignmentError(f"no feasible completion found for assignment row {row}")
    return BlockwiseAssignmentResult(
        column_for_row=tuple(fixed_columns),
        objective_value=assigned_cost,
    )
