from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass

import highspy
import numpy as np
from numpy.typing import NDArray

from fedorbit.config.models import (
    DenseCcpSolverConfig,
    ExactSparseSolverConfig,
    FedorbitConfig,
)
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


def _collect_block_products_for_target_pair(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    target_k: int,
    target_j: int,
    block_index: int,
    coefficients: MutableMapping[tuple[int, int, int, int], float],
) -> None:
    blocks = problem.blocks
    weight = float(problem.target_importance[target_k])
    action_value = float(alpha.coordinates[target_j])
    lower = problem.lower_response_matrix
    sources = list(blocks.block_index_range(block_index))
    for source_a in sources:
        for source_b in sources:
            coefficient = weight * action_value * float(lower[source_a, source_b])
            if coefficient:
                coefficients[(source_a, source_b, target_k, target_j)] = coefficient


def _nonzero_product_coefficients(
    problem: RobustActionProblem, alpha: CurriculumAction
) -> Mapping[tuple[int, int, int, int], float]:
    coefficients: OrderedDict[tuple[int, int, int, int], float] = OrderedDict()
    blocks = problem.blocks
    for target_k in range(blocks.total_padded_nodes):
        block_of_k = blocks.block_of_node(target_k)
        for target_j in range(blocks.total_padded_nodes):
            if blocks.block_of_node(target_j) != block_of_k:
                continue
            _collect_block_products_for_target_pair(
                problem, alpha, target_k, target_j, block_of_k, coefficients
            )
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


@dataclass(frozen=True, slots=True)
class _LiftedConstraintMatrix:
    row_lower: tuple[float, ...]
    row_upper: tuple[float, ...]
    start: tuple[int, ...]
    index: tuple[int, ...]
    value: tuple[float, ...]


def _append_lifted_assignment_rows(
    blocks: PaddedBlockStructure,
    layout: AssignmentVariableLayout,
    add_row: Callable[[Mapping[int, float], float, float], None],
) -> None:
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


def _build_lifted_constraint_matrix(
    blocks: PaddedBlockStructure,
    layout: AssignmentVariableLayout,
    product_keys: list[tuple[int, int, int, int]],
    product_column: Mapping[tuple[int, int, int, int], int],
) -> _LiftedConstraintMatrix:
    infinity = highspy.kHighsInf
    row_lower: list[float] = []
    row_upper: list[float] = []
    start: list[int] = [0]
    index: list[int] = []
    value: list[float] = []

    def add_row(entries: Mapping[int, float], lower: float, upper: float) -> None:
        for column in sorted(entries):
            index.append(column)
            value.append(entries[column])
        start.append(len(index))
        row_lower.append(lower)
        row_upper.append(upper)

    _append_lifted_assignment_rows(blocks, layout, add_row)
    for product_key in product_keys:
        source_a, source_b, target_k, target_j = product_key
        y_column = product_column[product_key]
        add_row(
            OrderedDict(
                (
                    (y_column, 1.0),
                    (layout.column_of(AssignmentVariableKey(source_a, target_k)), -1.0),
                )
            ),
            -infinity,
            0.0,
        )
        add_row(
            OrderedDict(
                (
                    (y_column, 1.0),
                    (layout.column_of(AssignmentVariableKey(source_b, target_j)), -1.0),
                )
            ),
            -infinity,
            0.0,
        )
        lower_combined: OrderedDict[int, float] = OrderedDict()
        for node_pair in (
            AssignmentVariableKey(source_a, target_k),
            AssignmentVariableKey(source_b, target_j),
        ):
            column = layout.column_of(node_pair)
            lower_combined[column] = lower_combined.get(column, 0.0) + 1.0
        lower_combined[y_column] = -1.0
        add_row(lower_combined, -infinity, 1.0)
    return _LiftedConstraintMatrix(
        row_lower=tuple(row_lower),
        row_upper=tuple(row_upper),
        start=tuple(start),
        index=tuple(index),
        value=tuple(value),
    )


def _lifted_objective_vector(
    layout: AssignmentVariableLayout,
    product_map: Mapping[tuple[int, int, int, int], float],
    product_keys: list[tuple[int, int, int, int]],
    penalty_coefficient: float,
    linearization_point: NDArray[np.float64],
) -> list[float]:
    product_column = {key: layout.size + offset for offset, key in enumerate(product_keys)}
    col_cost = [0.0] * (layout.size + len(product_keys))
    for key in product_keys:
        col_cost[product_column[key]] = product_map[key]
    if penalty_coefficient > 0.0:
        for position in range(layout.size):
            point_value = float(linearization_point[position])
            col_cost[position] += penalty_coefficient * (1.0 - 2.0 * point_value)
    return col_cost


def solve_lifted_lp(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    config: FedorbitConfig,
    layout: AssignmentVariableLayout,
    penalty_coefficient: float,
    linearization_point: NDArray[np.float64],
) -> LiftedRelaxationSolution:
    product_map = _nonzero_product_coefficients(problem, alpha)
    product_keys = sorted(product_map)
    product_column = {key: layout.size + offset for offset, key in enumerate(product_keys)}
    matrix = _build_lifted_constraint_matrix(problem.blocks, layout, product_keys, product_column)
    col_cost = _lifted_objective_vector(
        layout,
        product_map,
        product_keys,
        penalty_coefficient,
        linearization_point,
    )

    infinity = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_ = layout.size + len(product_keys)
    lp.num_row_ = len(matrix.row_lower)
    lp.col_cost_ = col_cost
    lp.col_lower_ = [0.0] * layout.size + [-infinity] * len(product_keys)
    lp.col_upper_ = [1.0] * layout.size + [infinity] * len(product_keys)
    lp.row_lower_ = list(matrix.row_lower)
    lp.row_upper_ = list(matrix.row_upper)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = list(matrix.start)
    lp.a_matrix_.index_ = list(matrix.index)
    lp.a_matrix_.value_ = list(matrix.value)

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


def _run_penalty_level(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    config: FedorbitConfig,
    layout: AssignmentVariableLayout,
    penalty: float,
    start_assignment: NDArray[np.float64],
    start_residual: float,
    start_iterations: int,
) -> tuple[NDArray[np.float64], float, int, bool]:
    settings = config.solvers.dense_ccp
    current = start_assignment.copy()
    previous_objective: float | None = None
    residual = start_residual
    iterations = start_iterations
    for _ in range(settings.maximum_iterations_per_penalty_level):
        step = solve_lifted_lp(problem, alpha, config, layout, penalty, current)
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
                return current, residual, iterations, True
        previous_objective = objective
    return current, residual, iterations, False


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
    residual = integrality_residual(current)
    iterations = 0
    converged_final_level = False
    for multiplier in settings.penalty_multipliers_relative_to_scale:
        current, residual, iterations, level_converged = _run_penalty_level(
            problem,
            alpha,
            config,
            layout,
            multiplier * scale,
            current,
            residual,
            iterations,
        )
        converged_final_level = level_converged
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
    unique_permutations: OrderedDict[tuple[int, ...], NDArray[np.float64]] = OrderedDict()
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


def _evaluate_projected_candidates(
    problem: RobustActionProblem,
    config: FedorbitConfig,
    layout: AssignmentVariableLayout,
    alpha: CurriculumAction,
    seed: int,
    contrast_coordinates: str,
    deadline: float,
) -> tuple[list[ProjectedCandidate], bool]:
    candidates: list[ProjectedCandidate] = []
    for start in dense_starts(layout, seed, contrast_coordinates):
        if time.monotonic() > deadline:
            return candidates, True
        trajectory = ccp_trajectory(problem, alpha, start, config, layout)
        correspondence = project_to_permutation(config, layout, trajectory.final_assignment)
        candidates.append(
            ProjectedCandidate(
                correspondence=correspondence,
                response_objective=response_only_objective(problem, alpha, correspondence),
                integrality_residual=trajectory.integrality_residual,
            )
        )
    return candidates, False


def _select_best_projected_candidate(
    candidates: list[ProjectedCandidate],
    exact_settings: ExactSparseSolverConfig,
) -> ProjectedCandidate:
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
    return min(tied, key=lambda entry: entry.correspondence.ordering_key())


@dataclass(frozen=True, slots=True)
class OuterIterationOutcome:
    cut_added: bool
    converged: bool
    terminal_state: TerminalState | None


def _classify_outer_iteration(
    problem: RobustActionProblem,
    dense_settings: DenseCcpSolverConfig,
    exact_settings: ExactSparseSolverConfig,
    scenarios: list[BlockCorrespondence],
    scenario_rows: list[NDArray[np.float64]],
    alpha: CurriculumAction,
    correspondence: BlockCorrespondence,
    master_objective: float,
    deadline: float,
) -> OuterIterationOutcome:
    full_objective = evaluate_objective(alpha, correspondence)
    violation = master_objective - full_objective
    is_new = correspondence not in scenarios
    violated = violation > exact_settings.separator_cut_stopping_tolerance
    if not (is_new and violated):
        return OuterIterationOutcome(cut_added=False, converged=True, terminal_state=None)
    scenarios.append(correspondence)
    scenario_rows.append(scenario_cut_row(problem, correspondence))
    cap_reached = len(scenarios) - 1 >= dense_settings.outer_action_cuts
    if cap_reached:
        return OuterIterationOutcome(cut_added=True, converged=False, terminal_state=None)
    if time.monotonic() > deadline:
        return OuterIterationOutcome(
            cut_added=True, converged=False, terminal_state=TerminalState.TIME_LIMIT
        )
    return OuterIterationOutcome(cut_added=True, converged=False, terminal_state=None)


@dataclass(frozen=True, slots=True)
class DenseOuterLoopResult:
    selected_action: CurriculumAction
    master_objective: float
    best_candidate: ProjectedCandidate | None
    lower_bound: float
    outer_cut_count: int
    converged_heuristically: bool
    terminal_state: TerminalState | None


def _outer_iteration_timed_out(timed_out: bool) -> bool:
    return timed_out


def _action_relaxation_bound(
    problem: RobustActionProblem,
    config: FedorbitConfig,
    layout: AssignmentVariableLayout,
    alpha: CurriculumAction,
) -> float:
    return relaxed_fixed_action_lower_bound(problem, alpha, config, layout).objective_value


@dataclass(frozen=True, slots=True)
class _LoopAccounting:
    outer_cut_count: int
    converged_heuristically: bool
    should_stop: bool


def _apply_outer_outcome(
    outcome: OuterIterationOutcome,
    cuts_so_far: int,
    previously_converged: bool,
) -> _LoopAccounting:
    cut_count = cuts_so_far + (1 if outcome.cut_added else 0)
    converged = previously_converged or outcome.converged
    should_stop = not outcome.cut_added or outcome.converged or outcome.terminal_state is not None
    return _LoopAccounting(cut_count, converged, should_stop)


def _run_dense_outer_loop(
    problem: RobustActionProblem,
    config: FedorbitConfig,
    seed: int,
    contrast_coordinates: str,
) -> DenseOuterLoopResult:
    settings = config.solvers.dense_ccp
    exact_settings = config.solvers.exact_sparse
    deadline = time.monotonic() + settings.wall_time_seconds
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
        selected_action = CurriculumAction(problem, alpha_values)
        master_objective = z_value
        lower_bound = _action_relaxation_bound(problem, config, layout, selected_action)
        candidates, timed_out = _evaluate_projected_candidates(
            problem, config, layout, selected_action, seed, contrast_coordinates, deadline
        )
        if _outer_iteration_timed_out(timed_out):
            terminal_state = TerminalState.TIME_LIMIT
        if terminal_state is not None or not candidates:
            break
        best_candidate = _select_best_projected_candidate(candidates, exact_settings)
        outcome = _classify_outer_iteration(
            problem,
            settings,
            exact_settings,
            scenarios,
            scenario_rows,
            selected_action,
            best_candidate.correspondence,
            master_objective,
            deadline,
        )
        loop = _apply_outer_outcome(outcome, outer_cut_count, converged_heuristically)
        outer_cut_count, converged_heuristically = (
            loop.outer_cut_count,
            loop.converged_heuristically,
        )
        terminal_state = outcome.terminal_state or terminal_state
        if loop.should_stop:
            break
    return DenseOuterLoopResult(
        selected_action=selected_action,
        master_objective=master_objective,
        best_candidate=best_candidate,
        lower_bound=lower_bound,
        outer_cut_count=outer_cut_count,
        converged_heuristically=converged_heuristically,
        terminal_state=terminal_state,
    )


def solve_dense_ccp(
    problem: RobustActionProblem,
    config: FedorbitConfig,
    seed: int,
    contrast_coordinates: str,
) -> DenseCcpOutcome:
    result = _run_dense_outer_loop(problem, config, seed, contrast_coordinates)
    best_candidate = result.best_candidate
    if best_candidate is None:
        raise DenseCcpError("dense CCP terminated without a projected correspondence")
    projected_response = response_only_objective(
        problem, result.selected_action, best_candidate.correspondence
    )
    return DenseCcpOutcome(
        selected_action=result.selected_action,
        master_objective=result.master_objective,
        best_projected_response_objective=projected_response,
        relaxation_lower_bound=result.lower_bound,
        dense_bound_gap=projected_response - result.lower_bound,
        integrality_residual=best_candidate.integrality_residual,
        outer_cut_count=result.outer_cut_count,
        converged_heuristically=result.converged_heuristically,
        terminal_state=result.terminal_state,
        worst_projected_correspondence=best_candidate.correspondence,
    )
