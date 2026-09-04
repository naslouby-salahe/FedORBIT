from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import RandomSeed, SeedDerivationRequest, derive_seed32
from fedorbit.optimization.certificates import build_rectangular_hull
from fedorbit.optimization.correspondence import BlockCorrespondence, PaddedBlockStructure
from fedorbit.optimization.diagnostics import analytic_orbit_mean
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    SupportCoordinateSet,
    actions_tied_within_tolerance,
    enumerate_support_coordinate_sets,
    rounded_action_vector,
    zero_action,
)
from fedorbit.types import Budget, RngNamespace, Score, StepCount, SupportCount, TransferMethod


class CouplingDestructionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CouplingDestroyedMatrices:
    lower_response_matrix: NDArray[np.float64]
    upper_response_matrix: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CommittedMapAction:
    correspondence: BlockCorrespondence
    selected_action: CurriculumAction


def _block_pair_permutation(
    entry_count: int,
    seed: RandomSeed,
    coordinates: str,
    block_pair_index: int,
) -> NDArray[np.intp]:
    rng_seed = derive_seed32(
        SeedDerivationRequest(
            seed,
            RngNamespace.COUPLING_DESTRUCTION,
            OrderedDict[str, str | int](coordinates=coordinates, block_pair=block_pair_index),
        )
    )
    generator = np.random.default_rng(rng_seed)
    return generator.permutation(entry_count)


def coupling_destroyed_matrices(
    blocks: PaddedBlockStructure,
    lower_response_matrix: NDArray[np.float64],
    upper_response_matrix: NDArray[np.float64],
    seed: RandomSeed,
    contrast_coordinates: str,
) -> CouplingDestroyedMatrices:
    size = blocks.total_padded_nodes
    if lower_response_matrix.shape != (size, size):
        raise CouplingDestructionError("lower matrix shape mismatch")
    if upper_response_matrix.shape != (size, size):
        raise CouplingDestructionError("upper matrix shape mismatch")
    destroyed_lower = np.array(lower_response_matrix, dtype=np.float64, copy=True)
    destroyed_upper = np.array(upper_response_matrix, dtype=np.float64, copy=True)
    block_pair_index = 0
    for group_source in range(len(blocks.padded_size_tuple)):
        rows = blocks.block_index_range(group_source)
        for group_target in range(len(blocks.padded_size_tuple)):
            columns = blocks.block_index_range(group_target)
            lower_block = lower_response_matrix[np.ix_(rows, columns)]
            upper_block = upper_response_matrix[np.ix_(rows, columns)]
            flat_entries = lower_block.reshape(-1)
            permutation = _block_pair_permutation(
                flat_entries.shape[0],
                seed,
                contrast_coordinates,
                block_pair_index,
            )
            permuted_lower = flat_entries[permutation].reshape(lower_block.shape)
            permuted_upper = upper_block.reshape(-1)[permutation].reshape(upper_block.shape)
            destroyed_lower[np.ix_(rows, columns)] = permuted_lower
            destroyed_upper[np.ix_(rows, columns)] = permuted_upper
            block_pair_index += 1
    return CouplingDestroyedMatrices(destroyed_lower, destroyed_upper)


def committed_map_action(
    problem: RobustActionProblem,
    source_matrix: NDArray[np.float64],
    target_matrix: NDArray[np.float64],
) -> CommittedMapAction:
    from fedorbit.methods.baselines import optimize_against_fixed_matrix
    from fedorbit.optimization.exact_qap import point_correspondence_commitment

    result = point_correspondence_commitment(source_matrix, target_matrix, problem.blocks)
    correspondence = result.require_certified().correspondence
    committed = correspondence.permute_response_matrix(problem.lower_response_matrix)
    solution = optimize_against_fixed_matrix(problem, committed)
    return CommittedMapAction(correspondence, solution.selected_action)


class FairnessViolationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ComparatorResources:
    source_packet_id: str
    target_checkpoint_artifact_id: str
    target_importance_vector_sha256: str
    action_budget_cap: Budget
    support_cap: SupportCount
    seed: RandomSeed
    confirmation_opportunity: bool
    live_assimilation_step_allowance: StepCount
    test_access_granted: bool
    extra_target_labels: bool
    additional_tuning_seeds: tuple[int, ...]
    local_base_checkpoint_favorable: bool

    def validate_contract(self) -> None:
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.test_access_granted:
            raise FairnessViolationError(
                "comparator resources must never include pre-decision TEST access"
            )
        if self.extra_target_labels:
            raise FairnessViolationError("comparator resources must never add target labels")
        if self.additional_tuning_seeds:
            raise FairnessViolationError("comparator resources must never add tuning seeds")
        if self.local_base_checkpoint_favorable:
            raise FairnessViolationError(
                "comparator resources must never grant a more favorable base checkpoint"
            )


def assert_identical_resources(
    method_name: TransferMethod,
    reference: ComparatorResources,
    candidate: ComparatorResources,
) -> None:
    for field in fields(ComparatorResources):
        if getattr(reference, field.name) != getattr(candidate, field.name):
            raise FairnessViolationError(
                f"method {method_name} received different {field.name} from the principal bundle"
            )


REGISTERED_METHOD_NAMES = frozenset(method.value for method in TransferMethod)


def assert_registered_method_name(name: str) -> None:
    if name not in REGISTERED_METHOD_NAMES:
        raise FairnessViolationError(f"unregistered comparator name: {name}")


class FixedMatrixOptimizationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FixedMatrixActionSolution:
    selected_action: CurriculumAction
    objective_value: Score


def linear_objective_row(
    problem: RobustActionProblem, matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.asarray(problem.target_importance @ matrix) - problem.linear_costs


def _solve_support_lp(
    problem: RobustActionProblem,
    support: SupportCoordinateSet,
    objective_row: NDArray[np.float64],
) -> CurriculumAction:
    settings = active_config().solvers.exact_sparse
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
        for action in (_solve_support_lp(problem, support, objective_row) for support in supports)
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


@dataclass(frozen=True, slots=True)
class CoarseBlockSummary:
    matrix: NDArray[np.float64]


def _real_source_indices(blocks: PaddedBlockStructure, block_index: int) -> range:
    return range(
        blocks.block_index_range(block_index).start,
        blocks.block_index_range(block_index).start + blocks.source_real_counts[block_index],
    )


def _target_block_range(blocks: PaddedBlockStructure, block_index: int) -> range:
    return blocks.block_index_range(block_index)


def coarse_block_mean_matrix(
    blocks: PaddedBlockStructure,
    response_matrix: NDArray[np.float64],
) -> CoarseBlockSummary:
    size = blocks.total_padded_nodes
    summary = np.zeros((size, size), dtype=np.float64)
    block_count = len(blocks.padded_size_tuple)
    for group_source in range(block_count):
        source_rows = _real_source_indices(blocks, group_source)
        for group_target in range(block_count):
            target_columns = _real_source_indices(blocks, group_target)
            block_entries = response_matrix[np.ix_(source_rows, target_columns)]
            if block_entries.size == 0:
                continue
            block_mean = float(np.mean(block_entries))
            for row in _target_block_range(blocks, group_source):
                for column in _target_block_range(blocks, group_target):
                    summary[row, column] = block_mean
    return CoarseBlockSummary(matrix=summary)


def coarse_block_min_matrix(
    blocks: PaddedBlockStructure,
    response_matrix: NDArray[np.float64],
) -> CoarseBlockSummary:
    size = blocks.total_padded_nodes
    summary = np.zeros((size, size), dtype=np.float64)
    block_count = len(blocks.padded_size_tuple)
    for group_source in range(block_count):
        source_rows = _real_source_indices(blocks, group_source)
        for group_target in range(block_count):
            target_columns = _real_source_indices(blocks, group_target)
            block_entries = response_matrix[np.ix_(source_rows, target_columns)]
            if block_entries.size == 0:
                continue
            block_minimum = float(np.min(block_entries))
            for row in _target_block_range(blocks, group_source):
                for column in _target_block_range(blocks, group_target):
                    summary[row, column] = block_minimum
    return CoarseBlockSummary(matrix=summary)


def orbit_mean_matrix(
    blocks: PaddedBlockStructure,
    response_matrix: NDArray[np.float64],
) -> CoarseBlockSummary:
    return CoarseBlockSummary(matrix=analytic_orbit_mean(blocks, response_matrix))


def matched_resource_rectangular_lower_bounds(
    blocks: PaddedBlockStructure,
    lower_response_matrix: NDArray[np.float64],
    upper_response_matrix: NDArray[np.float64],
) -> CoarseBlockSummary:
    hull = build_rectangular_hull(blocks, lower_response_matrix, upper_response_matrix)
    return CoarseBlockSummary(matrix=hull.lower_bounds)
