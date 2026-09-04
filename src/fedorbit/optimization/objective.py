from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fedorbit.config.loading import active_config
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    BlockNodeCounts,
    PaddedBlockStructure,
    support_per_block,
)
from fedorbit.types import Budget, Coefficient, Score, SupportCount, Tolerance


class ActionSpaceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RobustActionProblem:
    blocks: PaddedBlockStructure
    lower_response_matrix: NDArray[np.float64]
    upper_response_matrix: NDArray[np.float64]
    target_importance: NDArray[np.float64]
    coordinate_caps: NDArray[np.float64]
    linear_costs: NDArray[np.float64]
    total_budget: Budget
    principal_support: SupportCount

    def __post_init__(self) -> None:
        size = self.blocks.total_padded_nodes
        expected = (size, size)
        for name, matrix in (
            ("lower_response_matrix", self.lower_response_matrix),
            ("upper_response_matrix", self.upper_response_matrix),
        ):
            if matrix.shape != expected:
                raise ActionSpaceError(
                    f"{name} shape {matrix.shape} does not match padded size {expected}"
                )
        for name, vector in (
            ("target_importance", self.target_importance),
            ("coordinate_caps", self.coordinate_caps),
            ("linear_costs", self.linear_costs),
        ):
            if vector.shape != (size,):
                raise ActionSpaceError(
                    f"{name} length {vector.shape[0]} does not match padded size {size}"
                )
        if np.any(self.target_importance < 0.0):
            raise ActionSpaceError("target importance must be coordinatewise nonnegative")
        if np.any(self.coordinate_caps < 0.0):
            raise ActionSpaceError("coordinate caps must be coordinatewise nonnegative")
        if np.any(self.linear_costs < 0.0):
            raise ActionSpaceError("linear costs must be coordinatewise nonnegative")
        if np.any(np.isnan(self.lower_response_matrix)) or np.any(
            np.isnan(self.upper_response_matrix)
        ):
            raise ActionSpaceError("response matrices must not contain NaN entries")
        if self.total_budget < 0.0:
            raise ActionSpaceError("total curriculum budget must be nonnegative")
        if self.principal_support < 1:
            raise ActionSpaceError("principal support must be at least one")

    @property
    def size(self) -> int:
        return self.blocks.total_padded_nodes

    def actionable_nodes(self) -> tuple[int, ...]:
        return tuple(int(node) for node in np.flatnonzero(self.coordinate_caps > 0.0))


@dataclass(frozen=True, slots=True)
class CurriculumAction:
    problem: RobustActionProblem
    coordinates: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.coordinates.shape != (self.problem.size,):
            raise ActionSpaceError(
                f"action length {self.coordinates.shape[0]} does not match padded size "
                f"{self.problem.size}"
            )
        if np.any(self.coordinates < 0.0):
            raise ActionSpaceError("curriculum action coordinates must be nonnegative")
        if np.any(self.coordinates > self.problem.coordinate_caps + 0.0):
            raise ActionSpaceError("curriculum action violates a coordinate cap")

    @property
    def realized_support_size(self) -> int:
        return int(np.count_nonzero(self.coordinates))

    @property
    def active_support_nodes(self) -> tuple[int, ...]:
        return tuple(int(node) for node in np.flatnonzero(self.coordinates > 0.0))

    def is_within_budget(self) -> bool:
        return bool(float(np.sum(self.coordinates)) <= self.problem.total_budget + 0.0)

    def is_support_limited(self, support_limit: SupportCount) -> bool:
        return self.realized_support_size <= support_limit


def build_robust_action_problem(
    blocks: PaddedBlockStructure,
    lower_response_matrix: NDArray[np.float64],
    upper_response_matrix: NDArray[np.float64],
    target_importance: NDArray[np.float64],
    actionable_nodes: Sequence[int],
) -> RobustActionProblem:
    action_config = active_config().scientific.action
    size = blocks.total_padded_nodes
    actionable = frozenset(actionable_nodes)
    invalid = sorted(node for node in actionable if node < 0 or node >= size)
    if invalid:
        raise ActionSpaceError(f"actionable nodes outside padded space: {invalid}")
    caps = np.zeros(size, dtype=np.float64)
    costs = np.zeros(size, dtype=np.float64)
    for node in sorted(actionable):
        caps[node] = action_config.coordinate_cap
        costs[node] = action_config.linear_cost_per_actionable_node
    return RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=lower_response_matrix.copy(),
        upper_response_matrix=upper_response_matrix.copy(),
        target_importance=target_importance.astype(np.float64).copy(),
        coordinate_caps=caps,
        linear_costs=costs,
        total_budget=action_config.total_curriculum_budget,
        principal_support=action_config.principal_sparse_support,
    )


def zero_action(problem: RobustActionProblem) -> CurriculumAction:
    return CurriculumAction(problem=problem, coordinates=np.zeros(problem.size, dtype=np.float64))


def curriculum_action_from_entries(
    problem: RobustActionProblem, nonzero_entries: Sequence[tuple[int, float]]
) -> CurriculumAction:
    vector = np.zeros(problem.size, dtype=np.float64)
    for node, value in nonzero_entries:
        if node < 0 or node >= problem.size:
            raise ActionSpaceError(f"action node {node} outside padded space")
        if value < 0.0:
            raise ActionSpaceError("action coordinates must be nonnegative")
        vector[node] = value
    return CurriculumAction(problem=problem, coordinates=vector)


def evaluate_objective(alpha: CurriculumAction, correspondence: BlockCorrespondence) -> float:
    permuted = correspondence.permute_response_matrix(alpha.problem.lower_response_matrix)
    response_term = float(alpha.problem.target_importance @ permuted @ alpha.coordinates)
    cost_term = float(alpha.problem.linear_costs @ alpha.coordinates)
    return response_term - cost_term


def zero_action_objective() -> float:
    return 0.0


def orbit_response_minimum(
    alpha: CurriculumAction,
    orbit: Sequence[BlockCorrespondence],
) -> float:
    minimum = math.inf
    for correspondence in orbit:
        permuted = correspondence.permute_response_matrix(alpha.problem.lower_response_matrix)
        response_value = float(alpha.problem.target_importance @ permuted @ alpha.coordinates)
        minimum = min(minimum, response_value)
    if math.isinf(minimum):
        raise ActionSpaceError("orbit is empty")
    return minimum


def h_orb(alpha: CurriculumAction, orbit: Sequence[BlockCorrespondence]) -> float:
    return orbit_response_minimum(alpha, orbit)


def h_rect(alpha: CurriculumAction, lower_hull: NDArray[np.float64]) -> float:
    expected = (alpha.problem.size, alpha.problem.size)
    if lower_hull.shape != expected:
        raise ActionSpaceError(
            f"rectangular hull shape {lower_hull.shape} does not match {expected}"
        )
    return float(alpha.problem.target_importance @ lower_hull @ alpha.coordinates)


def map_conditioned_optimum(
    correspondence: BlockCorrespondence,
    action_candidates: Sequence[CurriculumAction],
) -> float:
    values = [evaluate_objective(candidate, correspondence) for candidate in action_candidates]
    if not values:
        raise ActionSpaceError("map-conditioned optimum requires at least one candidate action")
    return max(values)


def robust_pre_map_value(
    action_candidates: Sequence[CurriculumAction],
    orbit: Sequence[BlockCorrespondence],
) -> float:
    best = -math.inf
    for candidate in action_candidates:
        conditioned_min = min(
            evaluate_objective(candidate, correspondence) for correspondence in orbit
        )
        best = max(best, conditioned_min)
    if math.isinf(best):
        raise ActionSpaceError("robust pre-map value requires at least one candidate action")
    return best


def robust_post_map_value(
    action_candidates: Sequence[CurriculumAction],
    orbit: Sequence[BlockCorrespondence],
) -> float:
    per_map_values = [
        map_conditioned_optimum(correspondence, action_candidates) for correspondence in orbit
    ]
    return min(per_map_values)


def exact_map_action_value(
    action_candidates: Sequence[CurriculumAction],
    orbit: Sequence[BlockCorrespondence],
) -> float:
    return robust_post_map_value(action_candidates, orbit) - robust_pre_map_value(
        action_candidates, orbit
    )


@dataclass(frozen=True, slots=True)
class SupportCoordinateSet:
    problem: RobustActionProblem
    nodes: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(set(self.nodes)) != len(self.nodes):
            raise ActionSpaceError("support coordinate set contains duplicate nodes")
        actionable = set(self.problem.actionable_nodes())
        outside = sorted(node for node in self.nodes if node not in actionable)
        if outside:
            raise ActionSpaceError(f"support coordinates without action eligibility: {outside}")

    @classmethod
    def empty(cls, problem: RobustActionProblem) -> SupportCoordinateSet:
        return cls(problem=problem, nodes=())

    @property
    def size(self) -> int:
        return len(self.nodes)

    def block_support_counts(self) -> BlockNodeCounts:
        return support_per_block(self.problem.blocks, self.nodes)


def enumerate_support_coordinate_sets(
    problem: RobustActionProblem, support_limit: int | None = None
) -> tuple[SupportCoordinateSet, ...]:
    limit = support_limit if support_limit is not None else problem.principal_support
    actionable = list(problem.actionable_nodes())
    sets: list[SupportCoordinateSet] = []
    for size in range(1, limit + 1):
        if size > len(actionable):
            break
        for combination in itertools.combinations(actionable, size):
            sets.append(SupportCoordinateSet(problem=problem, nodes=combination))
    return tuple(sets)


def rounded_action_vector(
    alpha: CurriculumAction, rounding_precision: Tolerance
) -> tuple[float, ...]:
    if rounding_precision <= 0.0:
        raise ActionSpaceError("action tie comparison rounding precision must be positive")
    decimals = max(0, round(-math.log10(rounding_precision)))
    return tuple(float(value) for value in np.round(alpha.coordinates, decimals))


def actions_tied_within_tolerance(
    left: Coefficient, right: Coefficient, tolerance: Tolerance
) -> bool:
    if tolerance < 0.0:
        raise ActionSpaceError("tie tolerance must be nonnegative")
    return abs(left - right) <= tolerance


@dataclass(frozen=True, slots=True, order=True)
class CertifiedActionOrderingKey:
    negated_certified_value: Score
    realized_support_size: SupportCount
    target_node_sequence: tuple[int, ...]
    rounded_coordinates: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CertifiedActionCandidate:
    action: CurriculumAction
    certified_robust_value: Score
    target_node_sequence: tuple[int, ...]

    def deterministic_ordering_key(
        self, rounding_precision: Tolerance
    ) -> CertifiedActionOrderingKey:
        return CertifiedActionOrderingKey(
            negated_certified_value=-self.certified_robust_value,
            realized_support_size=self.action.realized_support_size,
            target_node_sequence=self.target_node_sequence,
            rounded_coordinates=rounded_action_vector(self.action, rounding_precision),
        )


def select_deterministic_candidate(
    candidates: Sequence[CertifiedActionCandidate],
    tie_tolerance: Tolerance,
    rounding_precision: Tolerance,
) -> CertifiedActionCandidate | None:
    if not candidates:
        return None
    best_value = max(candidate.certified_robust_value for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if actions_tied_within_tolerance(
            candidate.certified_robust_value, best_value, tie_tolerance
        )
    ]
    return min(tied, key=lambda candidate: candidate.deterministic_ordering_key(rounding_precision))
