from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import highspy
import numpy as np
from numpy.typing import NDArray

from fedorbit.config.models import ExactSparseSolverConfig, FedorbitConfig
from fedorbit.orbit.correspondence import (
    ActiveImageMap,
    BlockCorrespondence,
    PaddedBlockStructure,
    enumerate_active_image_maps,
)
from fedorbit.orbit.objective import (
    CurriculumAction,
    RobustActionProblem,
    SupportCoordinateSet,
    actions_tied_within_tolerance,
    enumerate_support_coordinate_sets,
    evaluate_objective,
    rounded_action_vector,
    zero_action,
)
from fedorbit.solvers.assignment import solve_minimum_cost_assignment


class SparseMasterNonConvergenceError(RuntimeError):
    pass


class SolverExecutionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SeparatorOutcome:
    worst_correspondence: BlockCorrespondence
    separator_objective: float
    active_image_candidates: int
    lap_calls: int


@dataclass(frozen=True, slots=True)
class SupportMasterSolution:
    support_nodes: tuple[int, ...]
    certified_action: CurriculumAction
    certified_robust_value: float
    worst_correspondence: BlockCorrespondence
    iterations: int
    cut_count: int


@dataclass(frozen=True, slots=True)
class RobustActionSolution:
    selected_action: CurriculumAction
    certified_robust_value: float
    support_solutions: tuple[SupportMasterSolution, ...]
    zero_action_value: float


def fixed_action_worst_correspondence(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    lap_tie_tolerance: float,
    action_tie_tolerance: float,
) -> SeparatorOutcome:
    blocks = problem.blocks
    active_nodes = alpha.active_support_nodes
    if not active_nodes:
        raise SolverExecutionError("separator requires a nonzero action")
    support_counts = _support_block_counts(blocks, active_nodes)
    candidates = _active_image_candidates(blocks.padded_size_tuple, support_counts)
    lap_calls_per_map = sum(
        1
        for block_index, size in enumerate(blocks.padded_size_tuple)
        if size - support_counts[block_index] > 0
    )
    best_value = math.inf
    best_images: tuple[int, ...] | None = None
    for mapping in enumerate_active_image_maps(blocks, active_nodes):
        value, images = _evaluate_active_image_map(problem, alpha, mapping, lap_tie_tolerance)
        if value < best_value - action_tie_tolerance:
            best_value = value
            best_images = images
        elif abs(value - best_value) <= action_tie_tolerance:
            if best_images is None or images < best_images:
                best_images = images
                best_value = min(best_value, value)
    if best_images is None or any(image < 0 for image in best_images):
        raise SolverExecutionError("separator produced no admissible correspondence")
    correspondence = BlockCorrespondence(blocks=blocks, images=best_images)
    return SeparatorOutcome(
        worst_correspondence=correspondence,
        separator_objective=evaluate_objective(alpha, correspondence),
        active_image_candidates=candidates,
        lap_calls=candidates * lap_calls_per_map,
    )


def _active_image_candidates(padded_sizes: tuple[int, ...], support_counts: tuple[int, ...]) -> int:
    candidates = 1
    for size, support_size in zip(padded_sizes, support_counts, strict=True):
        falling_factorial = 1
        for offset in range(support_size):
            falling_factorial *= size - offset
        candidates *= falling_factorial
    return candidates


def _support_block_counts(
    blocks: PaddedBlockStructure, active_nodes: tuple[int, ...]
) -> tuple[int, ...]:
    counts = [0] * len(blocks.padded_size_tuple)
    for node in active_nodes:
        counts[blocks.block_of_node(node)] += 1
    return tuple(counts)


def _evaluate_active_image_map(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    mapping: ActiveImageMap,
    lap_tie_tolerance: float,
) -> tuple[float, tuple[int, ...]]:
    blocks = problem.blocks
    coordinates = alpha.coordinates
    image_by_target: dict[int, int] = dict(mapping.fixed_pairs())
    used_sources = set(image_by_target.values())
    active_targets = sorted(image_by_target.keys())
    total_cost = 0.0
    for target in active_targets:
        weight = float(problem.target_importance[target])
        for j in active_targets:
            source_image = image_by_target[target]
            column_image = image_by_target[j]
            total_cost += (
                weight
                * float(coordinates[j])
                * float(problem.lower_response_matrix[source_image, column_image])
            )
    full_assignment = dict(image_by_target)
    for block_index in range(len(blocks.padded_size_tuple)):
        remaining_targets = [
            node for node in blocks.block_index_range(block_index) if node not in image_by_target
        ]
        unused_sources = [
            node for node in blocks.block_index_range(block_index) if node not in used_sources
        ]
        if not remaining_targets:
            continue
        if len(remaining_targets) != len(unused_sources):
            raise SolverExecutionError("block completion sizes differ")
        cost_matrix = np.zeros((len(remaining_targets), len(unused_sources)), dtype=np.float64)
        for row_index, target_node in enumerate(remaining_targets):
            weight = float(problem.target_importance[target_node])
            for column_index, source_node in enumerate(unused_sources):
                accumulated = 0.0
                for j in active_targets:
                    column_image = image_by_target[j]
                    accumulated += float(coordinates[j]) * float(
                        problem.lower_response_matrix[source_node, column_image]
                    )
                cost_matrix[row_index, column_index] = weight * accumulated
        assignment = solve_minimum_cost_assignment(cost_matrix, lap_tie_tolerance)
        total_cost += assignment.objective_value
        for local_row, local_column in enumerate(assignment.column_for_row):
            full_assignment[remaining_targets[local_row]] = unused_sources[local_column]
    images = tuple(full_assignment[node] for node in range(blocks.total_padded_nodes))
    return total_cost, images


def scenario_cut_row(
    problem: RobustActionProblem, correspondence: BlockCorrespondence
) -> NDArray[np.float64]:
    permuted = correspondence.permute_response_matrix(problem.lower_response_matrix)
    return np.asarray(problem.target_importance @ permuted) - problem.linear_costs


def solve_support_master(
    problem: RobustActionProblem,
    support: SupportCoordinateSet,
    config: FedorbitConfig,
    maximum_cuts: int | None = None,
) -> SupportMasterSolution:
    settings = config.solvers.exact_sparse
    cut_cap = maximum_cuts if maximum_cuts is not None else settings.maximum_cuts_per_support
    initial = BlockCorrespondence.lexicographically_smallest(problem.blocks)
    scenario_rows: list[NDArray[np.float64]] = [scenario_cut_row(problem, initial)]
    scenarios: list[BlockCorrespondence] = [initial]
    iterations = 0
    while True:
        z_value, alpha_values = run_support_master_lp(problem, support, scenario_rows, settings)
        iterations += 1
        alpha = CurriculumAction(problem, alpha_values)
        if not alpha.active_support_nodes:
            zero_objective = evaluate_objective(alpha, initial)
            if 0.0 - zero_objective <= settings.separator_cut_stopping_tolerance:
                return SupportMasterSolution(
                    support_nodes=support.nodes,
                    certified_action=alpha,
                    certified_robust_value=zero_objective,
                    worst_correspondence=initial,
                    iterations=iterations,
                    cut_count=len(scenarios),
                )
        outcome = fixed_action_worst_correspondence(
            problem,
            alpha,
            settings.lap_objective_tie_tolerance,
            settings.action_tie_tolerance,
        )
        gap = z_value - outcome.separator_objective
        if gap <= settings.separator_cut_stopping_tolerance:
            return SupportMasterSolution(
                support_nodes=support.nodes,
                certified_action=alpha,
                certified_robust_value=outcome.separator_objective,
                worst_correspondence=outcome.worst_correspondence,
                iterations=iterations,
                cut_count=len(scenarios),
            )
        if len(scenarios) >= cut_cap:
            raise SparseMasterNonConvergenceError(
                f"support {support.nodes} reached the configured cut cap of {cut_cap}"
            )
        if any(existing == outcome.worst_correspondence for existing in scenarios):
            raise SparseMasterNonConvergenceError(
                "robust master repeated a scenario without converging"
            )
        scenarios.append(outcome.worst_correspondence)
        scenario_rows.append(scenario_cut_row(problem, outcome.worst_correspondence))


def run_support_master_lp(
    problem: RobustActionProblem,
    support: SupportCoordinateSet,
    scenario_rows: Sequence[NDArray[np.float64]],
    settings: ExactSparseSolverConfig,
) -> tuple[float, NDArray[np.float64]]:
    columns = 1 + support.size
    infinity = highspy.kHighsInf
    row_lower: list[float] = []
    row_upper: list[float] = []
    matrix_start = [0]
    matrix_index: list[int] = []
    matrix_value: list[float] = []

    def add_row(coefficients: list[float], upper: float) -> None:
        matrix_index.extend(range(columns))
        matrix_value.extend(coefficients)
        matrix_start.append(len(matrix_index))
        row_lower.append(-infinity)
        row_upper.append(upper)

    budget_coefficients = [0.0] * columns
    for offset in range(support.size):
        budget_coefficients[1 + offset] = 1.0
    add_row(budget_coefficients, float(problem.total_budget))
    for scenario in scenario_rows:
        cut_coefficients = [1.0] + [0.0] * support.size
        for offset, node in enumerate(support.nodes):
            cut_coefficients[1 + offset] = -float(scenario[node])
        add_row(cut_coefficients, 0.0)

    lp = highspy.HighsLp()
    lp.num_col_ = columns
    lp.num_row_ = len(row_lower)
    lp.col_cost_ = [-1.0] + [0.0] * support.size
    lp.col_lower_ = [-infinity] + [0.0] * support.size
    lp.col_upper_ = [infinity] + [float(problem.coordinate_caps[node]) for node in support.nodes]
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = matrix_start
    lp.a_matrix_.index_ = matrix_index
    lp.a_matrix_.value_ = matrix_value

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("solver", "simplex")
    highs.setOptionValue("presolve", "on")
    highs.setOptionValue("threads", settings.lp_threads_per_solve)
    highs.setOptionValue("random_seed", settings.deterministic_random_seed)
    highs.setOptionValue("primal_feasibility_tolerance", settings.lp_primal_feasibility_tolerance)
    highs.setOptionValue("dual_feasibility_tolerance", settings.lp_dual_feasibility_tolerance)
    highs.passModel(lp)
    highs.run()
    model_status = highs.getModelStatus()
    if model_status != highspy.HighsModelStatus.kOptimal:
        raise SolverExecutionError(f"robust master LP status {model_status}")
    solution = highs.getSolution()
    objective = float(highs.getInfo().objective_function_value)
    column_values = [float(value) for value in solution.col_value]
    embedded = np.zeros(problem.size, dtype=np.float64)
    for offset, node in enumerate(support.nodes):
        embedded[node] = column_values[1 + offset]
    return -objective, embedded


def solve_robust_action(
    problem: RobustActionProblem,
    config: FedorbitConfig,
    support_limit: int | None = None,
    maximum_cuts: int | None = None,
) -> RobustActionSolution:
    settings = config.solvers.exact_sparse
    supports = enumerate_support_coordinate_sets(problem, support_limit)
    concurrency = max(1, min(settings.maximum_concurrent_supports, len(supports)))
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(solve_support_master, problem, support, config, maximum_cuts)
                for support in supports
            ]
            solutions = tuple(future.result() for future in futures)
    else:
        solutions = tuple(
            solve_support_master(problem, support, config, maximum_cuts) for support in supports
        )
    identity = BlockCorrespondence.lexicographically_smallest(problem.blocks)
    zero_candidate = zero_action(problem)
    zero_value = evaluate_objective(zero_candidate, identity)
    candidates: list[tuple[float, CurriculumAction]] = [(zero_value, zero_candidate)]
    candidates.extend(
        (solution.certified_robust_value, solution.certified_action) for solution in solutions
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
    return RobustActionSolution(
        selected_action=winning_action,
        certified_robust_value=winner_value,
        support_solutions=solutions,
        zero_action_value=zero_value,
    )
