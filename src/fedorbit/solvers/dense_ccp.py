from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import highspy
import numpy as np
from numpy.typing import NDArray

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import RngNamespace, TerminalState
from fedorbit.orbit.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
    enumerate_block_permutations,
)
from fedorbit.orbit.objective import (
    CurriculumAction,
    RobustActionProblem,
    SupportCoordinateSet,
    actions_tied_within_tolerance,
    evaluate_objective,
    zero_action,
)
from fedorbit.runtime.seeds import derive_seed32
from fedorbit.solvers.assignment import solve_minimum_cost_assignment
from fedorbit.solvers.exact_sparse import (
    run_support_master_lp,
    scenario_cut_row,
)


class DenseCcpError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AssignmentVariableKey:
    source_node: int
    target_node: int


@dataclass(frozen=True, slots=True)
class AssignmentVariableLayout:
    blocks: PaddedBlockStructure
    columns: tuple[AssignmentVariableKey, ...]
    column_index: Mapping[AssignmentVariableKey, int]

    @classmethod
    def build(cls, blocks: PaddedBlockStructure) -> AssignmentVariableLayout:
        columns: list[AssignmentVariableKey] = []
        for block_index in range(len(blocks.padded_size_tuple)):
            sources = blocks.block_index_range(block_index)
            targets = blocks.block_index_range(block_index)
            for source in sources:
                for target in targets:
                    columns.append(AssignmentVariableKey(int(source), int(target)))
        index_map = {key: index for index, key in enumerate(columns)}
        return cls(blocks=blocks, columns=tuple(columns), column_index=index_map)

    @property
    def size(self) -> int:
        return len(self.columns)

    def column_of(self, key: AssignmentVariableKey) -> int:
        return self.column_index[key]

    def zeros(self) -> NDArray[np.float64]:
        return np.zeros(self.size, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LiftedRelaxationSolution:
    objective_value: float
    assignment_values: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CcpTrajectoryOutcome:
    final_assignment: NDArray[np.float64]
    final_objective: float
    integrality_residual: float
    converged_final_level: bool
    iterations: int


@dataclass(frozen=True, slots=True)
class ProjectedCandidate:
    correspondence: BlockCorrespondence
    response_objective: float
    integrality_residual: float


@dataclass(frozen=True, slots=True)
class DenseCcpOutcome:
    selected_action: CurriculumAction
    master_objective: float
    best_projected_response_objective: float
    relaxation_lower_bound: float
    dense_bound_gap: float
    integrality_residual: float
    outer_cut_count: int
    converged_heuristically: bool
    terminal_state: TerminalState | None
    worst_projected_correspondence: BlockCorrespondence

    @property
    def is_exact(self) -> bool:
        return False


def assignment_variable_keys(
    blocks: PaddedBlockStructure,
) -> AssignmentVariableLayout:
    return AssignmentVariableLayout.build(blocks)


def _nonzero_product_coefficients(
    problem: RobustActionProblem, alpha: CurriculumAction
) -> dict[tuple[int, int, int, int], float]:
    coefficients: dict[tuple[int, int, int, int], float] = {}
    blocks = problem.blocks
    lower = problem.lower_response_matrix
    importance = problem.target_importance
    coordinates = alpha.coordinates
    for target_k in range(blocks.total_padded_nodes):
        weight = float(importance[target_k])
        if weight == 0.0:
            continue
        block_of_k = blocks.block_of_node(target_k)
        for target_j in range(blocks.total_padded_nodes):
            if blocks.block_of_node(target_j) != block_of_k:
                continue
            action_value = float(coordinates[target_j])
            if action_value == 0.0:
                continue
            for source_a in blocks.block_index_range(block_of_k):
                for source_b in blocks.block_index_range(block_of_k):
                    coefficient = weight * action_value * float(lower[source_a, source_b])
                    if coefficient != 0.0:
                        coefficients[(source_a, source_b, target_k, target_j)] = coefficient
    return coefficients


def penalty_scale(problem: RobustActionProblem, alpha: CurriculumAction) -> float:
    largest = max(
        (abs(value) for value in _nonzero_product_coefficients(problem, alpha).values()),
        default=0.0,
    )
    return max(1.0, largest)


def integrality_residual(assignment_values: NDArray[np.float64]) -> float:
    if assignment_values.size == 0:
        raise DenseCcpError("empty assignment vector has no integrality residual")
    distances = np.minimum(assignment_values, 1.0 - assignment_values)
    return float(np.max(distances))


def solve_lifted_lp(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    config: FedorbitConfig,
    layout: AssignmentVariableLayout,
    penalty_coefficient: float,
    linearization_point: NDArray[np.float64],
) -> LiftedRelaxationSolution:
    blocks = problem.blocks
    product_map = _nonzero_product_coefficients(problem, alpha)
    product_keys = sorted(product_map)
    product_column = {key: layout.size + offset for offset, key in enumerate(product_keys)}
    total_columns = layout.size + len(product_keys)

    infinity = highspy.kHighsInf
    rows_lower: list[float] = []
    rows_upper: list[float] = []
    starts = [0]
    indices: list[int] = []
    values: list[float] = []

    def add_row(entries: dict[int, float], lower: float, upper: float) -> None:
        for column in sorted(entries):
            indices.append(column)
            values.append(entries[column])
        starts.append(len(indices))
        rows_lower.append(lower)
        rows_upper.append(upper)

    for block_index in range(len(blocks.padded_size_tuple)):
        targets = list(blocks.block_index_range(block_index))
        sources = list(blocks.block_index_range(block_index))
        for target in targets:
            add_row(
                {
                    layout.column_of(AssignmentVariableKey(source, target)): 1.0
                    for source in sources
                },
                1.0,
                1.0,
            )
        for source in sources:
            add_row(
                {
                    layout.column_of(AssignmentVariableKey(source, target)): 1.0
                    for target in targets
                },
                1.0,
                1.0,
            )
    for product_key in product_keys:
        source_a, source_b, target_k, target_j = product_key
        y_column = product_column[product_key]
        upper_p_ak: dict[int, float] = {
            y_column: 1.0,
            layout.column_of(AssignmentVariableKey(source_a, target_k)): -1.0,
        }
        add_row(upper_p_ak, -infinity, 0.0)
        upper_p_bj: dict[int, float] = {
            y_column: 1.0,
            layout.column_of(AssignmentVariableKey(source_b, target_j)): -1.0,
        }
        add_row(upper_p_bj, -infinity, 0.0)
        lower_combined: dict[int, float] = {}
        for node_pair in (
            AssignmentVariableKey(source_a, target_k),
            AssignmentVariableKey(source_b, target_j),
        ):
            column = layout.column_of(node_pair)
            lower_combined[column] = lower_combined.get(column, 0.0) + 1.0
        lower_combined[y_column] = -1.0
        add_row(lower_combined, -infinity, 1.0)

    col_cost = [0.0] * total_columns
    for key in product_keys:
        col_cost[product_column[key]] = product_map[key]
    if penalty_coefficient > 0.0:
        for position, _key in enumerate(layout.columns):
            point_value = float(linearization_point[position])
            col_cost[position] += penalty_coefficient * (1.0 - 2.0 * point_value)

    lp = highspy.HighsLp()
    lp.num_col_ = total_columns
    lp.num_row_ = len(rows_lower)
    lp.col_cost_ = col_cost
    lp.col_lower_ = [0.0] * layout.size + [-infinity] * len(product_keys)
    lp.col_upper_ = [1.0] * layout.size + [infinity] * len(product_keys)
    lp.row_lower_ = rows_lower
    lp.row_upper_ = rows_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = values

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("solver", "simplex")
    highs.setOptionValue("presolve", "on")
    highs.setOptionValue("threads", config.solvers.dense_ccp.lp_threads)
    highs.passModel(lp)
    highs.run()
    status = highs.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise DenseCcpError(f"dense lifted LP status {status}")
    solution = highs.getSolution()
    raw_values = np.asarray([float(value) for value in solution.col_value], dtype=np.float64)
    assignment_values = raw_values[: layout.size].copy()
    unpenalized = 0.0
    for key in product_keys:
        unpenalized += product_map[key] * float(raw_values[product_column[key]])
    return LiftedRelaxationSolution(
        objective_value=unpenalized,
        assignment_values=assignment_values,
    )


def relaxed_fixed_action_lower_bound(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    config: FedorbitConfig,
    layout: AssignmentVariableLayout,
) -> LiftedRelaxationSolution:
    return solve_lifted_lp(problem, alpha, config, layout, 0.0, layout.zeros())


def unpenalized_fixed_action_objective(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    layout: AssignmentVariableLayout,
    assignment_values: NDArray[np.float64],
) -> float:
    total = 0.0
    for (source_a, source_b, target_k, target_j), coefficient in _nonzero_product_coefficients(
        problem, alpha
    ).items():
        p_ak = float(assignment_values[layout.column_of(AssignmentVariableKey(source_a, target_k))])
        p_bj = float(assignment_values[layout.column_of(AssignmentVariableKey(source_b, target_j))])
        total += coefficient * p_ak * p_bj
    return total


def ccp_trajectory(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    start_assignment: NDArray[np.float64],
    config: FedorbitConfig,
    layout: AssignmentVariableLayout,
) -> CcpTrajectoryOutcome:
    settings = config.solvers.dense_ccp
    scale = penalty_scale(problem, alpha)
    current = start_assignment.copy()
    previous_objective: float | None = None
    residual = integrality_residual(current)
    iterations = 0
    converged_final_level = False
    for multiplier in settings.penalty_multipliers_relative_to_scale:
        converged_final_level = False
        for _ in range(settings.maximum_iterations_per_penalty_level):
            step = solve_lifted_lp(problem, alpha, config, layout, multiplier * scale, current)
            current = step.assignment_values
            objective = step.objective_value
            residual = integrality_residual(current)
            iterations += 1
            if previous_objective is not None:
                relative_change = abs(objective - previous_objective) / max(
                    1.0, abs(previous_objective)
                )
                if (
                    relative_change <= settings.relative_objective_convergence_tolerance
                    and residual <= settings.assignment_integrality_residual
                ):
                    converged_final_level = True
                    break
            previous_objective = objective
    final_objective = unpenalized_fixed_action_objective(problem, alpha, layout, current)
    return CcpTrajectoryOutcome(
        final_assignment=current,
        final_objective=final_objective,
        integrality_residual=residual,
        converged_final_level=converged_final_level,
        iterations=iterations,
    )


def permutation_to_vector(
    images: tuple[int, ...],
    layout: AssignmentVariableLayout,
) -> NDArray[np.float64]:
    vector = layout.zeros()
    for target, image in enumerate(images):
        vector[layout.column_of(AssignmentVariableKey(image, target))] = 1.0
    return vector


def barycenter_start(layout: AssignmentVariableLayout) -> NDArray[np.float64]:
    vector = layout.zeros()
    for block_index, size in enumerate(layout.blocks.padded_size_tuple):
        uniform = 1.0 / size
        for source in layout.blocks.block_index_range(block_index):
            for target in layout.blocks.block_index_range(block_index):
                vector[layout.column_of(AssignmentVariableKey(int(source), int(target)))] = uniform
    return vector


def dense_starts(
    layout: AssignmentVariableLayout,
    seed: int,
    coordinates: str,
) -> tuple[NDArray[np.float64], ...]:
    unique_permutations: dict[tuple[int, ...], NDArray[np.float64]] = {}
    for correspondence in enumerate_block_permutations(layout.blocks):
        unique_permutations.setdefault(
            correspondence.images,
            permutation_to_vector(correspondence.images, layout),
        )
    ordered_permutations = sorted(unique_permutations)
    starts: list[NDArray[np.float64]] = [barycenter_start(layout)]
    if len(ordered_permutations) < 4:
        for images in ordered_permutations:
            starts.append(unique_permutations[images].copy())
        return tuple(starts[:5])
    starts.append(unique_permutations[ordered_permutations[0]].copy())
    rng_seed = derive_seed32(seed, RngNamespace.DENSE_START, coordinates)
    rng = np.random.default_rng(rng_seed)
    seen_orders: list[tuple[int, ...]] = [ordered_permutations[0]]
    attempts = 0
    while len(starts) < 5 and attempts < 24:
        attempts += 1
        order: list[int] = []
        for block_index in range(len(layout.blocks.padded_size_tuple)):
            sources = [int(node) for node in layout.blocks.block_index_range(block_index)]
            shuffled = [int(node) for node in rng.permutation(sources)]
            order.extend(shuffled)
        order_tuple = tuple(order)
        if order_tuple in seen_orders:
            continue
        seen_orders.append(order_tuple)
        starts.append(permutation_to_vector(order_tuple, layout))
    return tuple(starts[:5])


def project_to_permutation(
    config: FedorbitConfig,
    layout: AssignmentVariableLayout,
    assignment_values: NDArray[np.float64],
) -> BlockCorrespondence:
    blocks = layout.blocks
    images: list[int] = [-1] * blocks.total_padded_nodes
    lap_tie_tolerance = config.solvers.exact_sparse.lap_objective_tie_tolerance
    for block_index in range(len(blocks.padded_size_tuple)):
        targets = list(blocks.block_index_range(block_index))
        sources = list(blocks.block_index_range(block_index))
        cost_matrix = np.zeros((len(targets), len(sources)), dtype=np.float64)
        for row_index, target in enumerate(targets):
            for column_index, source in enumerate(sources):
                value = float(
                    assignment_values[layout.column_of(AssignmentVariableKey(source, target))]
                )
                cost_matrix[row_index, column_index] = -value
        assignment = solve_minimum_cost_assignment(cost_matrix, lap_tie_tolerance)
        for row_index, column_index in enumerate(assignment.column_for_row):
            images[targets[row_index]] = sources[column_index]
    return BlockCorrespondence(blocks=blocks, images=tuple(images))


def response_only_objective(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    correspondence: BlockCorrespondence,
) -> float:
    permuted = correspondence.permute_response_matrix(problem.lower_response_matrix)
    return float(problem.target_importance @ permuted @ alpha.coordinates)


def solve_dense_ccp(
    problem: RobustActionProblem,
    config: FedorbitConfig,
    seed: int,
    contrast_coordinates: str,
) -> DenseCcpOutcome:
    started = time.monotonic()
    settings = config.solvers.dense_ccp
    exact_settings = config.solvers.exact_sparse
    deadline = started + settings.wall_time_seconds
    layout = AssignmentVariableLayout.build(problem.blocks)
    actionable = tuple(problem.actionable_nodes())
    full_support = SupportCoordinateSet(problem=problem, nodes=actionable)
    initial = BlockCorrespondence.lexicographically_smallest(problem.blocks)
    scenario_rows = [scenario_cut_row(problem, initial)]
    scenarios: list[BlockCorrespondence] = [initial]
    selected_action = zero_action(problem)
    master_objective = 0.0
    best_candidate: ProjectedCandidate | None = None
    lower_bound = 0.0
    outer_cut_count = 0
    converged_heuristically = False
    terminal_state: TerminalState | None = None
    while True:
        z_value, alpha_values = run_support_master_lp(
            problem, full_support, scenario_rows, exact_settings
        )
        alpha = CurriculumAction(problem, alpha_values)
        master_objective = z_value
        selected_action = alpha
        relaxation = relaxed_fixed_action_lower_bound(problem, alpha, config, layout)
        lower_bound = relaxation.objective_value
        candidates: list[ProjectedCandidate] = []
        for start in dense_starts(layout, seed, contrast_coordinates):
            if time.monotonic() > deadline:
                terminal_state = TerminalState.TIME_LIMIT
                break
            trajectory = ccp_trajectory(problem, alpha, start, config, layout)
            correspondence = project_to_permutation(config, layout, trajectory.final_assignment)
            candidates.append(
                ProjectedCandidate(
                    correspondence=correspondence,
                    response_objective=response_only_objective(problem, alpha, correspondence),
                    integrality_residual=trajectory.integrality_residual,
                )
            )
        if terminal_state is not None or not candidates:
            break
        best_response = min(candidate.response_objective for candidate in candidates)
        tied = [
            candidate
            for candidate in candidates
            if actions_tied_within_tolerance(
                candidate.response_objective,
                best_response,
                exact_settings.action_tie_tolerance,
            )
        ]
        best_candidate = min(tied, key=lambda entry: entry.correspondence.ordering_key())
        full_objective = evaluate_objective(alpha, best_candidate.correspondence)
        violation = master_objective - full_objective
        is_new = best_candidate.correspondence not in scenarios
        if is_new and violation > exact_settings.separator_cut_stopping_tolerance:
            scenarios.append(best_candidate.correspondence)
            scenario_rows.append(scenario_cut_row(problem, best_candidate.correspondence))
            outer_cut_count += 1
            if outer_cut_count >= settings.outer_action_cuts:
                break
            if time.monotonic() > deadline:
                terminal_state = TerminalState.TIME_LIMIT
                break
        else:
            converged_heuristically = True
            break
    if best_candidate is None:
        raise DenseCcpError("dense CCP terminated without a projected correspondence")
    projected_response = response_only_objective(
        problem, selected_action, best_candidate.correspondence
    )
    return DenseCcpOutcome(
        selected_action=selected_action,
        master_objective=master_objective,
        best_projected_response_objective=projected_response,
        relaxation_lower_bound=lower_bound,
        dense_bound_gap=projected_response - lower_bound,
        integrality_residual=best_candidate.integrality_residual,
        outer_cut_count=outer_cut_count,
        converged_heuristically=converged_heuristically,
        terminal_state=terminal_state,
        worst_projected_correspondence=best_candidate.correspondence,
    )
