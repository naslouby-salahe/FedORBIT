from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray
from pyscipopt import Expr, Model, quicksum

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import TerminalState
from fedorbit.orbit.correspondence import BlockCorrespondence, PaddedBlockStructure
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
from fedorbit.solvers.exact_sparse import (
    SolverExecutionError,
    SupportMasterSolution,
    run_support_master_lp,
    scenario_cut_row,
)


class QapUncertifiedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class QapSeparatorResult:
    correspondence: BlockCorrespondence | None
    objective_value: float | None
    certified: bool
    terminal_state: TerminalState | None

    def require_certified(self) -> tuple[BlockCorrespondence, float]:
        if not self.certified or self.correspondence is None or self.objective_value is None:
            raise QapUncertifiedError(
                f"QAP separator result carries no exact certificate: {self.terminal_state}"
            )
        return self.correspondence, self.objective_value


@dataclass(frozen=True, slots=True)
class QapRobustOutcome:
    certified_solution: SupportMasterSolution | None
    terminal_state: TerminalState | None

    @property
    def is_exact(self) -> bool:
        return self.certified_solution is not None


def _terminal_state_for(status: str) -> TerminalState | None:
    if status == "timelimit":
        return TerminalState.TIME_LIMIT
    if status in {"memlimit", "nodelimit", "gaplimit"}:
        return TerminalState.RESOURCE_LIMIT
    return None


def _configure_model(model: Model, config: FedorbitConfig, deadline: float | None) -> None:
    settings = config.solvers.generic_exact_qap
    model.hideOutput()
    remaining = settings.wall_time_seconds_per_solve
    if deadline is not None:
        remaining = min(remaining, max(0.0, deadline - time.monotonic()))
    model.setRealParam("limits/time", float(remaining))
    model.setIntParam("parallel/maxnthreads", int(settings.threads))
    model.setIntParam("randomization/randomseedshift", int(settings.random_seed))
    model.setIntParam("randomization/permutationseed", int(settings.random_seed))
    model.setRealParam("numerics/feastol", float(settings.feasibility_tolerance))
    model.setRealParam("limits/gap", float(settings.relative_mip_gap))


def _build_assignment_structure(
    model: Model,
    blocks: PaddedBlockStructure,
    prefix: str,
) -> dict[tuple[int, int], Expr]:
    assignment_variables: dict[tuple[int, int], Expr] = {}
    for block_index in range(len(blocks.padded_size_tuple)):
        targets = list(blocks.block_index_range(block_index))
        sources = list(blocks.block_index_range(block_index))
        for target in targets:
            variables = [
                model.addVar(vtype="B", name=f"{prefix}_p_{source}_{target}") for source in sources
            ]
            for source, variable in zip(sources, variables, strict=True):
                assignment_variables[(source, target)] = variable
            model.addCons(quicksum(variables) == 1, name=f"tgt_{prefix}_{target}")
        for source in sources:
            model.addCons(
                quicksum(assignment_variables[(source, target)] for target in targets) == 1,
                name=f"src_{prefix}_{source}",
            )
    return assignment_variables


def _add_mccormick_products(
    model: Model,
    assignment_variables: dict[tuple[int, int], Expr],
    coefficients: dict[tuple[int, int, int, int], float],
    prefix: str,
) -> list[Expr]:
    product_variables: dict[tuple[int, int, int, int], Expr] = {}
    objective_terms: list[Expr] = []
    for key, coefficient in sorted(coefficients.items()):
        source_a, source_b, target_k, target_j = key
        existing = product_variables.get(key)
        if isinstance(existing, Expr):
            objective_terms.append(coefficient * existing)
            continue
        p_ak = assignment_variables[(source_a, target_k)]
        p_bj = assignment_variables[(source_b, target_j)]
        y: Expr = model.addVar(vtype="C", lb=0.0, ub=1.0, name=f"y_{prefix}_{key}")
        model.addCons(y <= p_ak, name=f"mc1_{prefix}_{key}")
        model.addCons(y <= p_bj, name=f"mc2_{prefix}_{key}")
        model.addCons(y >= p_ak + p_bj - 1.0, name=f"mc3_{prefix}_{key}")
        product_variables[key] = y
        objective_terms.append(coefficient * y)
    return objective_terms


def fixed_action_worst_correspondence_qap(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    config: FedorbitConfig,
) -> QapSeparatorResult:
    active_nodes = alpha.active_support_nodes
    if not active_nodes:
        raise SolverExecutionError("QAP separator requires a nonzero action")
    blocks = problem.blocks
    lower = problem.lower_response_matrix
    importance = problem.target_importance
    coordinates = alpha.coordinates
    coefficients: dict[tuple[int, int, int, int], float] = {}
    for target_k in range(blocks.total_padded_nodes):
        weight = float(importance[target_k])
        if weight == 0.0:
            continue
        for target_j in active_nodes:
            action_value = float(coordinates[target_j])
            if action_value == 0.0:
                continue
            for source_a in blocks.block_index_range(blocks.block_of_node(target_k)):
                for source_b in blocks.block_index_range(blocks.block_of_node(target_j)):
                    coefficient = weight * action_value * float(lower[source_a, source_b])
                    if coefficient != 0.0:
                        coefficients[(source_a, source_b, target_k, target_j)] = coefficient
    deadline = time.monotonic() + config.solvers.generic_exact_qap.wall_time_seconds_per_solve
    model = Model("qap_fixed_action")
    _configure_model(model, config, deadline)
    assignment_variables = _build_assignment_structure(model, blocks, "fa")
    objective_terms = _add_mccormick_products(model, assignment_variables, coefficients, "fa")
    model.setObjective(quicksum(objective_terms) if objective_terms else 0.0, "minimize")
    model.optimize()
    status = model.getStatus()
    limit_state = _terminal_state_for(status)
    if status != "optimal":
        return QapSeparatorResult(
            correspondence=None,
            objective_value=None,
            certified=False,
            terminal_state=limit_state or TerminalState.FAILED_SCIENTIFIC_ALGORITHMIC,
        )
    gap = float(model.getGap())
    if not math.isfinite(gap) or gap > config.solvers.generic_exact_qap.relative_mip_gap:
        return QapSeparatorResult(
            correspondence=None,
            objective_value=None,
            certified=False,
            terminal_state=TerminalState.TIME_LIMIT,
        )
    images = _extract_images(model, assignment_variables, blocks)
    correspondence = BlockCorrespondence(blocks=blocks, images=images)
    return QapSeparatorResult(
        correspondence=correspondence,
        objective_value=float(model.getObjVal()),
        certified=True,
        terminal_state=None,
    )


def point_correspondence_commitment(
    source_response_matrix: NDArray[np.float64],
    target_response_matrix: NDArray[np.float64],
    blocks: PaddedBlockStructure,
    config: FedorbitConfig,
) -> QapSeparatorResult:
    size = blocks.total_padded_nodes
    if source_response_matrix.shape != (size, size) or target_response_matrix.shape != (size, size):
        raise SolverExecutionError("point-correspondence matrices must match padded size")
    coefficients: dict[tuple[int, int, int, int], float] = {}
    for source_a, source_b, target_k, target_j in product(range(size), repeat=4):
        if blocks.block_of_node(source_a) != blocks.block_of_node(target_k):
            continue
        if blocks.block_of_node(source_b) != blocks.block_of_node(target_j):
            continue
        coefficient = -float(
            source_response_matrix[source_a, source_b] * target_response_matrix[target_k, target_j]
        )
        if coefficient != 0.0:
            coefficients[(source_a, source_b, target_k, target_j)] = coefficient
    deadline = time.monotonic() + config.solvers.generic_exact_qap.wall_time_seconds_per_solve
    model = Model("qap_point_correspondence")
    _configure_model(model, config, deadline)
    assignment_variables = _build_assignment_structure(model, blocks, "pc")
    objective_terms = _add_mccormick_products(model, assignment_variables, coefficients, "pc")
    model.setObjective(quicksum(objective_terms) if objective_terms else 0.0, "minimize")
    model.optimize()
    status = model.getStatus()
    limit_state = _terminal_state_for(status)
    if status != "optimal":
        return QapSeparatorResult(
            correspondence=None,
            objective_value=None,
            certified=False,
            terminal_state=limit_state or TerminalState.FAILED_SCIENTIFIC_ALGORITHMIC,
        )
    tie_tolerance = config.scientific.baselines.point_correspondence_commitment.qap_tie_tolerance
    best_objective = float(model.getObjVal())
    refined = _refine_lexicographic_correspondence(
        model,
        assignment_variables,
        objective_terms,
        blocks,
        best_objective,
        tie_tolerance,
        config,
        deadline,
    )
    if refined is None:
        return QapSeparatorResult(
            correspondence=None,
            objective_value=None,
            certified=False,
            terminal_state=TerminalState.TIME_LIMIT,
        )
    images, inner_product = refined
    squared_distance = (
        float(np.sum(np.square(source_response_matrix)))
        + float(np.sum(np.square(target_response_matrix)))
        - 2.0 * inner_product
    )
    return QapSeparatorResult(
        correspondence=BlockCorrespondence(blocks=blocks, images=tuple(images)),
        objective_value=squared_distance,
        certified=True,
        terminal_state=None,
    )


def _refine_lexicographic_correspondence(
    model: Model,
    assignment_variables: dict[tuple[int, int], Expr],
    objective_terms: list[Expr],
    blocks: PaddedBlockStructure,
    best_objective: float,
    tie_tolerance: float,
    config: FedorbitConfig,
    deadline: float,
) -> tuple[tuple[int, ...], float] | None:
    model.freeTransform()
    model.addCons(
        quicksum(objective_terms) <= best_objective + tie_tolerance,
        name="tie_bound",
        removable=True,
    )
    chosen_images: list[int] = []
    for target in range(blocks.total_padded_nodes):
        block_sources = list(blocks.block_index_range(blocks.block_of_node(target)))
        fixed = False
        for source in sorted(block_sources):
            model.freeTransform()
            constraint = model.addCons(
                assignment_variables[(source, target)] == 1,
                name=f"lex_fix_{source}_{target}",
                removable=True,
            )
            _configure_model(model, config, deadline)
            model.optimize()
            status = model.getStatus()
            if status == "optimal":
                fixed = True
                break
            model.freeTransform()
            model.delCons(constraint)
            if _terminal_state_for(status) is not None:
                return None
        if not fixed:
            raise SolverExecutionError(
                f"lexicographic refinement found no feasible image for target {target}"
            )
        selected = [
            source
            for source in block_sources
            if round(float(model.getVal(assignment_variables[(source, target)]))) == 1
        ]
        if len(selected) != 1:
            raise SolverExecutionError("lexicographic refinement lost the bijection")
        chosen_images.append(selected[0])
    inner_product = -float(model.getObjVal())
    return tuple(chosen_images), inner_product


def solve_support_master_qap(
    problem: RobustActionProblem,
    support: SupportCoordinateSet,
    config: FedorbitConfig,
) -> SupportMasterSolution | TerminalState:
    settings = config.solvers.exact_sparse
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
            return SupportMasterSolution(
                support_nodes=support.nodes,
                certified_action=alpha,
                certified_robust_value=zero_objective,
                worst_correspondence=initial,
                iterations=iterations,
                cut_count=len(scenarios),
            )
        separator = fixed_action_worst_correspondence_qap(problem, alpha, config)
        if not separator.certified:
            assert separator.terminal_state is not None
            return separator.terminal_state
        worst_correspondence, worst_value = separator.require_certified()
        gap = z_value - worst_value
        if gap <= settings.separator_cut_stopping_tolerance:
            return SupportMasterSolution(
                support_nodes=support.nodes,
                certified_action=alpha,
                certified_robust_value=worst_value,
                worst_correspondence=worst_correspondence,
                iterations=iterations,
                cut_count=len(scenarios),
            )
        if any(existing == worst_correspondence for existing in scenarios):
            raise SolverExecutionError("QAP robust master repeated a scenario without progress")
        scenarios.append(worst_correspondence)
        scenario_rows.append(scenario_cut_row(problem, worst_correspondence))


def solve_robust_action_qap(
    problem: RobustActionProblem,
    config: FedorbitConfig,
    support_limit: int | None = None,
) -> QapRobustOutcome:
    settings = config.solvers.exact_sparse
    supports = enumerate_support_coordinate_sets(problem, support_limit)
    identity = BlockCorrespondence.lexicographically_smallest(problem.blocks)
    zero_candidate = zero_action(problem)
    candidates: list[tuple[float, CurriculumAction]] = [
        (evaluate_objective(zero_candidate, identity), zero_candidate)
    ]
    solutions: list[SupportMasterSolution] = []
    for support in supports:
        outcome = solve_support_master_qap(problem, support, config)
        if isinstance(outcome, TerminalState):
            return QapRobustOutcome(certified_solution=None, terminal_state=outcome)
        solutions.append(outcome)
        candidates.append((outcome.certified_robust_value, outcome.certified_action))
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
    solution = SupportMasterSolution(
        support_nodes=winning_action.active_support_nodes,
        certified_action=winning_action,
        certified_robust_value=winner_value,
        worst_correspondence=identity,
        iterations=sum(solution.iterations for solution in solutions),
        cut_count=sum(solution.cut_count for solution in solutions),
    )
    return QapRobustOutcome(certified_solution=solution, terminal_state=None)


def _extract_images(
    model: Model,
    assignment_variables: dict[tuple[int, int], Expr],
    blocks: PaddedBlockStructure,
) -> tuple[int, ...]:
    images: list[int] = []
    for target in range(blocks.total_padded_nodes):
        selected = [
            source
            for source in blocks.block_index_range(blocks.block_of_node(target))
            if round(float(model.getVal(assignment_variables[(source, target)]))) == 1
        ]
        if len(selected) != 1:
            raise SolverExecutionError("QAP assignment variables do not form a bijection")
        images.append(selected[0])
    return tuple(images)
