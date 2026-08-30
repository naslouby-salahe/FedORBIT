from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fedorbit.config.context import active_config
from fedorbit.config.models import ExactSparseSolverConfig
from fedorbit.orbit.objective import (
    CurriculumAction,
    RobustActionProblem,
    SupportCoordinateSet,
    actions_tied_within_tolerance,
    enumerate_support_coordinate_sets,
    rounded_action_vector,
    zero_action,
)


class FixedMatrixOptimizationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FixedMatrixActionSolution:
    selected_action: CurriculumAction
    objective_value: float


def linear_objective_row(
    problem: RobustActionProblem, matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.asarray(problem.target_importance @ matrix) - problem.linear_costs


def _solve_support_lp(
    problem: RobustActionProblem,
    support: SupportCoordinateSet,
    objective_row: NDArray[np.float64],
    settings: ExactSparseSolverConfig,
) -> CurriculumAction:
    import highspy

    columns = 1 + support.size
    infinity = highspy.kHighsInf
    col_cost = [0.0] * columns
    for offset, node in enumerate(support.nodes):
        col_cost[1 + offset] = -float(objective_row[node])
    budget_coefficients = [0.0] * columns
    for offset in range(support.size):
        budget_coefficients[1 + offset] = 1.0
    lp = highspy.HighsLp()
    lp.num_col_ = columns
    lp.num_row_ = 1
    lp.col_cost_ = col_cost
    lp.col_lower_ = [0.0] * columns
    lp.col_upper_ = [infinity] + [float(problem.coordinate_caps[node]) for node in support.nodes]
    lp.row_lower_ = [-infinity]
    lp.row_upper_ = [float(problem.total_budget)]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = [0, columns]
    lp.a_matrix_.index_ = list(range(columns))
    lp.a_matrix_.value_ = budget_coefficients
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("solver", "simplex")
    highs.setOptionValue("presolve", "on")
    highs.setOptionValue("threads", settings.lp_threads_per_solve)
    highs.setOptionValue("random_seed", settings.deterministic_random_seed)
    highs.setOptionValue("primal_feasibility_tolerance", settings.lp_primal_feasibility_tolerance)
    highs.passModel(lp)
    highs.run()
    if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise FixedMatrixOptimizationError(f"fixed-matrix LP status {highs.getModelStatus()}")
    solution = highs.getSolution()
    values = np.asarray([float(value) for value in solution.col_value], dtype=np.float64)
    embedded = np.zeros(problem.size, dtype=np.float64)
    for offset, node in enumerate(support.nodes):
        embedded[node] = values[1 + offset]
    return CurriculumAction(problem, embedded)


def optimize_against_fixed_matrix(
    problem: RobustActionProblem,
    matrix: NDArray[np.float64],
    support_limit: int | None = None,
) -> FixedMatrixActionSolution:
    expected_shape = (problem.size, problem.size)
    if matrix.shape != expected_shape:
        raise FixedMatrixOptimizationError(
            f"fixed matrix shape {matrix.shape} does not match {expected_shape}"
        )
    settings = active_config().solvers.exact_sparse
    objective_row = linear_objective_row(problem, matrix)
    supports = enumerate_support_coordinate_sets(problem, support_limit)

    def objective_of(action: CurriculumAction) -> float:
        return float(objective_row @ action.coordinates)

    zero_candidate = zero_action(problem)
    candidates: list[tuple[float, CurriculumAction]] = [
        (objective_of(zero_candidate), zero_candidate)
    ]
    candidates.extend(
        (objective_of(action), action)
        for action in (
            _solve_support_lp(problem, support, objective_row, settings) for support in supports
        )
    )
    best_value = max(value for value, _ in candidates)
    rounding_precision = settings.action_tie_comparison_rounding_precision
    tied = [
        (value, action)
        for value, action in candidates
        if actions_tied_within_tolerance(value, best_value, settings.action_tie_tolerance)
    ]
    winner_value, winning_action = min(
        tied,
        key=lambda entry: (
            -entry[0],
            entry[1].realized_support_size,
            entry[1].active_support_nodes,
            rounded_action_vector(entry[1], rounding_precision),
        ),
    )
    return FixedMatrixActionSolution(selected_action=winning_action, objective_value=winner_value)
