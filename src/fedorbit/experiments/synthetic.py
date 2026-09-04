from __future__ import annotations

import hashlib
import itertools
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np

from fedorbit.config.loading import active_config
from fedorbit.infrastructure.runtime import RandomSeed, SeedDerivationRequest, derive_seed32
from fedorbit.methods.baselines import optimize_against_fixed_matrix
from fedorbit.optimization.certificates import RectangularHull, build_rectangular_hull
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.optimization.diagnostics import fixed_action_rectangularization_gap
from fedorbit.optimization.exact_sparse import solve_robust_action
from fedorbit.optimization.objective import (
    CurriculumAction,
    RobustActionProblem,
    build_robust_action_problem,
    exact_map_action_value,
    robust_pre_map_value,
    rounded_action_vector,
    zero_action,
)
from fedorbit.types import (
    CoarseGroup,
    ConceptCount,
    CouplingCompatibility,
    RngNamespace,
    ScalabilityBlockPattern,
    StableJsonPayload,
    SupportCount,
    stable_json,
)


@dataclass(frozen=True, slots=True)
class SyntheticRandomRequest:
    seed: RandomSeed
    coordinates: StableJsonPayload


@dataclass(frozen=True, slots=True)
class SyntheticRandomStream:
    generator: np.random.Generator


def create_float64_random_stream(request: SyntheticRandomRequest) -> SyntheticRandomStream:
    instance_seed = derive_seed32(
        SeedDerivationRequest(request.seed, RngNamespace.SYNTHETIC_INSTANCE, request.coordinates)
    )
    return SyntheticRandomStream(np.random.Generator(np.random.PCG64(instance_seed)))


class SyntheticGenerationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExactSeparatorInstanceRequest:
    block_pattern: tuple[int, ...]
    seed: RandomSeed
    active_support_size: SupportCount = 1

    def __post_init__(self) -> None:
        if not self.block_pattern or any(size < 1 for size in self.block_pattern):
            raise SyntheticGenerationError("synthetic block pattern must contain positive blocks")
        if self.active_support_size > sum(self.block_pattern):
            raise SyntheticGenerationError("active support size exceeds the padded node count")


@dataclass(frozen=True, slots=True)
class ExactSeparatorInstance:
    block_pattern: tuple[int, ...]
    lower_response_matrix: np.ndarray
    upper_response_matrix: np.ndarray
    target_importance: np.ndarray
    active_action: np.ndarray
    generation_seed: RandomSeed


def generate_exact_separator_instance(
    request: ExactSeparatorInstanceRequest,
) -> ExactSeparatorInstance:
    generator_config = active_config().generators.exact_separator_theorem
    coordinates = cast(
        StableJsonPayload,
        OrderedDict(generator="exact_separator_theorem", block_pattern=list(request.block_pattern)),
    )
    random = create_float64_random_stream(
        SyntheticRandomRequest(request.seed, coordinates)
    ).generator
    size = sum(request.block_pattern)
    response_lower, response_upper = generator_config.response_uniform
    lower = random.uniform(response_lower, response_upper, size=(size, size)).astype(np.float64)
    increment_lower, increment_upper = generator_config.serialization_upper_band_increment_uniform
    upper = lower + random.uniform(increment_lower, increment_upper, size=(size, size))
    gamma = generator_config.target_importance_gamma
    importance = random.gamma(gamma.shape, gamma.scale, size=size).astype(np.float64)
    active_action = _draw_support_restricted_action(random, size, request.active_support_size)
    return ExactSeparatorInstance(
        block_pattern=request.block_pattern,
        lower_response_matrix=lower,
        upper_response_matrix=upper,
        target_importance=importance,
        active_action=active_action,
        generation_seed=request.seed,
    )


def _draw_support_restricted_action(
    random: np.random.Generator,
    size: int,
    active_support_size: SupportCount,
) -> np.ndarray:
    combinations = tuple(itertools.combinations(range(size), active_support_size))
    support = combinations[int(random.integers(0, len(combinations)))]
    action_lower, action_upper = (
        active_config().generators.exact_separator_theorem.active_action_uniform
    )
    raw = random.uniform(action_lower, action_upper, size=active_support_size).astype(np.float64)
    budget = active_config().scientific.action.total_curriculum_budget
    total = float(raw.sum())
    scaled = raw if total <= budget else raw * (budget / total)
    active_action = np.zeros(size, dtype=np.float64)
    active_action[list(support)] = scaled
    return active_action


class MechanismGenerationError(ValueError):
    pass


class UnresolvedMapWorldKind(StrEnum):
    COMMON_ACTION = "common_action"
    ROBUST_COMPROMISE = "robust_compromise"
    MAP_DEPENDENT = "map_dependent"


@dataclass(frozen=True, slots=True)
class UnresolvedMapWorldRequest:
    world_kind: UnresolvedMapWorldKind
    seed: RandomSeed


@dataclass(frozen=True, slots=True)
class UnresolvedMapWorld:
    block_pattern: tuple[int, int]
    lower_response_matrix: np.ndarray
    target_importance: np.ndarray
    generation_seed: RandomSeed
    world_kind: UnresolvedMapWorldKind


def generate_unresolved_map_world(request: UnresolvedMapWorldRequest) -> UnresolvedMapWorld:
    max_attempts = _max_attempts_for(request.world_kind)
    for attempt in range(max_attempts):
        coordinates = cast(
            StableJsonPayload,
            OrderedDict(
                generator="unresolved_map_world",
                world_kind=request.world_kind.value,
                attempt=attempt,
            ),
        )
        random = create_float64_random_stream(
            SyntheticRandomRequest(request.seed, coordinates)
        ).generator
        pattern, response = _draw_world_response(request.world_kind, random)
        importance = _gamma_normalized_importance(random, sum(pattern))
        if _world_is_accepted(request.world_kind, pattern, response, importance):
            return UnresolvedMapWorld(
                block_pattern=pattern,
                lower_response_matrix=response,
                target_importance=importance,
                generation_seed=request.seed,
                world_kind=request.world_kind,
            )
    raise MechanismGenerationError(
        f"unresolved-map generator exhausted attempts for {request.world_kind.value}"
    )


def _max_attempts_for(world_kind: UnresolvedMapWorldKind) -> int:
    if world_kind == UnresolvedMapWorldKind.COMMON_ACTION:
        return active_config().generators.common_action_unresolved_map.maximum_attempts
    if world_kind == UnresolvedMapWorldKind.ROBUST_COMPROMISE:
        return (
            active_config().generators.robust_compromise_unresolved_map.maximum_attempts_per_fixture
        )
    return active_config().generators.map_dependent.maximum_attempts


def _draw_world_response(
    world_kind: UnresolvedMapWorldKind,
    random: np.random.Generator,
) -> tuple[tuple[int, int], np.ndarray]:
    if world_kind == UnresolvedMapWorldKind.COMMON_ACTION:
        pattern = active_config().generators.common_action_unresolved_map.block_pattern
        return (pattern[0], pattern[1]), _common_action_response(random, (pattern[0], pattern[1]))
    if world_kind == UnresolvedMapWorldKind.ROBUST_COMPROMISE:
        configuration = active_config().generators.robust_compromise_unresolved_map
        pattern = configuration.block_pattern
        response = _independent_response(
            random, (pattern[0], pattern[1]), configuration.response_uniform
        )
        return (pattern[0], pattern[1]), response
    configuration = active_config().generators.map_dependent
    pattern = configuration.block_pattern
    response = _independent_response(
        random, (pattern[0], pattern[1]), configuration.response_uniform
    )
    return (pattern[0], pattern[1]), response


def _unresolved_map_problem(
    pattern: tuple[int, int],
    response: np.ndarray,
    importance: np.ndarray,
) -> RobustActionProblem:
    groups = tuple(CoarseGroup)[: len(pattern)]
    counts = OrderedDict((group, size) for group, size in zip(groups, pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    return build_robust_action_problem(
        blocks, response, response, importance, tuple(range(sum(pattern)))
    )


def _map_conditioned_winner(
    problem: RobustActionProblem,
    correspondence: BlockCorrespondence,
) -> CurriculumAction:
    matrix = correspondence.permute_response_matrix(problem.lower_response_matrix)
    return optimize_against_fixed_matrix(problem, matrix).selected_action


def _map_conditioned_winners_disjoint(
    problem: RobustActionProblem,
    orbit: Sequence[BlockCorrespondence],
) -> bool:
    rounding = active_config().solvers.exact_sparse.action_tie_comparison_rounding_precision
    winners = {
        rounded_action_vector(_map_conditioned_winner(problem, correspondence), rounding)
        for correspondence in orbit
    }
    return len(winners) > 1


def _world_is_accepted(
    world_kind: UnresolvedMapWorldKind,
    pattern: tuple[int, int],
    response: np.ndarray,
    importance: np.ndarray,
) -> bool:
    problem = _unresolved_map_problem(pattern, response, importance)
    orbit = list(enumerate_block_permutations(problem.blocks))
    if world_kind == UnresolvedMapWorldKind.COMMON_ACTION:
        return solve_robust_action(problem).certified_robust_value > 0.0
    per_map_winners = tuple(
        _map_conditioned_winner(problem, correspondence) for correspondence in orbit
    )
    candidates = (
        solve_robust_action(problem).selected_action,
        *per_map_winners,
        zero_action(problem),
    )
    if world_kind == UnresolvedMapWorldKind.ROBUST_COMPROMISE:
        configuration = active_config().generators.robust_compromise_unresolved_map
        threshold = configuration.robust_pre_map_value_strictly_greater_than
        return (
            len(orbit) > 1
            and _map_conditioned_winners_disjoint(problem, orbit)
            and robust_pre_map_value(candidates, orbit) > threshold
        )
    threshold = active_config().generators.map_dependent.map_value_minimum
    return exact_map_action_value(candidates, orbit) >= threshold


def _common_action_response(
    random: np.random.Generator,
    pattern: tuple[int, int],
) -> np.ndarray:
    lower, upper = (
        active_config().generators.common_action_unresolved_map.block_pair_response_uniform
    )
    block_values = random.uniform(lower, upper, size=(2, 2)).astype(np.float64)
    first, second = pattern
    response = np.empty((first + second, first + second), dtype=np.float64)
    response[:first, :first] = block_values[0, 0]
    response[:first, first:] = block_values[0, 1]
    response[first:, :first] = block_values[1, 0]
    response[first:, first:] = block_values[1, 1]
    return response


def _independent_response(
    random: np.random.Generator,
    pattern: tuple[int, int],
    bounds: tuple[float, float],
) -> np.ndarray:
    size = sum(pattern)
    return random.uniform(bounds[0], bounds[1], size=(size, size)).astype(np.float64)


def _gamma_normalized_importance(random: np.random.Generator, size: int) -> np.ndarray:
    gamma = active_config().generators.exact_separator_theorem.target_importance_gamma
    weights = random.gamma(gamma.shape, gamma.scale, size=size).astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise MechanismGenerationError("gamma target-importance draw has nonpositive total")
    return weights / total


class ScalabilityGenerationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ScalabilityInstanceRequest:
    node_count: ConceptCount
    block_pattern: ScalabilityBlockPattern
    support_size: SupportCount
    seed: RandomSeed

    def __post_init__(self) -> None:
        if self.node_count < 2:
            raise ScalabilityGenerationError("scalability node count must be at least two")
        if self.support_size < 1:
            raise ScalabilityGenerationError("scalability support size must be positive")


@dataclass(frozen=True, slots=True)
class ScalabilityInstance:
    block_pattern: tuple[int, int]
    lower_response_matrix: np.ndarray
    target_importance: np.ndarray
    fixed_action: np.ndarray
    generation_seed: RandomSeed


def generate_scalability_instance(request: ScalabilityInstanceRequest) -> ScalabilityInstance:
    block_pattern = _block_pattern(request)
    largest_block_size = max(block_pattern)
    if request.support_size > largest_block_size:
        raise ScalabilityGenerationError("support size exceeds the largest coarse block")
    lower, upper = active_config().generators.scalability.response_uniform
    coordinates = cast(
        StableJsonPayload,
        OrderedDict(
            generator="scalability",
            node_count=request.node_count,
            block_pattern=request.block_pattern.value,
            support_size=request.support_size,
        ),
    )
    random = create_float64_random_stream(
        SyntheticRandomRequest(request.seed, coordinates)
    ).generator
    response = random.uniform(lower, upper, size=(request.node_count, request.node_count)).astype(
        np.float64
    )
    target_importance = np.full(request.node_count, 1.0 / request.node_count, dtype=np.float64)
    fixed_action = np.zeros(request.node_count, dtype=np.float64)
    action = active_config().scientific.action
    action_value = min(action.coordinate_cap, action.total_curriculum_budget / request.support_size)
    fixed_action[: request.support_size] = action_value
    return ScalabilityInstance(
        block_pattern=block_pattern,
        lower_response_matrix=response,
        target_importance=target_importance,
        fixed_action=fixed_action,
        generation_seed=request.seed,
    )


def _block_pattern(request: ScalabilityInstanceRequest) -> tuple[int, int]:
    if request.block_pattern == ScalabilityBlockPattern.BALANCED:
        lower = request.node_count // 2
        return (lower, request.node_count - lower)
    return (request.node_count - 1, 1)


class CouplingGenerationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CouplingInstanceRequest:
    compatibility: CouplingCompatibility
    response_heterogeneity: float
    directed_asymmetry: float
    response_sparsity: float
    block_pattern: tuple[int, ...]
    support_size: SupportCount
    seed: RandomSeed
    instance_index: int


@dataclass(frozen=True, slots=True)
class CouplingInstance:
    lower_response_matrix: np.ndarray
    target_importance: np.ndarray
    active_action: np.ndarray
    block_pattern: tuple[int, ...]
    classification: CouplingCompatibility
    generation_seed: RandomSeed


def _coupling_coordinates(
    request: CouplingInstanceRequest,
    attempt: int | None,
) -> OrderedDict[str, str | int | float | list[int]]:
    fields: OrderedDict[str, str | int | float | list[int]] = OrderedDict(
        generator="coupling_structure",
        compatibility=request.compatibility.value,
        response_heterogeneity=request.response_heterogeneity,
        directed_asymmetry=request.directed_asymmetry,
        response_sparsity=request.response_sparsity,
        block_pattern=list(request.block_pattern),
        support=request.support_size,
        instance_index=request.instance_index,
    )
    if attempt is not None:
        fields["attempt"] = attempt
    return fields


def _deterministic_sparsity_mask(
    size: int,
    sparsity: float,
    seed: RandomSeed,
    coordinates_text: str,
) -> frozenset[tuple[int, int]]:
    total = size * size
    keep = max(1, round(sparsity * total))
    ranked: list[tuple[int, int, int]] = []
    for a in range(size):
        for b in range(size):
            payload = f"FedORBIT|synthetic-sparsity|{seed}|{coordinates_text}|{a}|{b}"
            digest_int = int.from_bytes(
                hashlib.sha256(payload.encode("utf-8")).digest(), byteorder="big"
            )
            ranked.append((digest_int, a, b))
    ranked.sort()
    return frozenset((a, b) for _, a, b in ranked[:keep])


def _apply_heterogeneity(
    matrix: np.ndarray,
    blocks: PaddedBlockStructure,
    heterogeneity: float,
) -> np.ndarray:
    result = matrix.copy()
    for g in range(len(blocks.coarse_groups)):
        g_range = list(blocks.block_index_range(g))
        for h in range(len(blocks.coarse_groups)):
            h_range = list(blocks.block_index_range(h))
            submatrix = matrix[np.ix_(g_range, h_range)]
            mean = float(submatrix.mean())
            result[np.ix_(g_range, h_range)] = mean + heterogeneity * (submatrix - mean)
    return result


def _apply_deterministic_sparsity(
    matrix: np.ndarray,
    keep_mask: frozenset[tuple[int, int]],
) -> np.ndarray:
    result = np.zeros_like(matrix)
    for a, b in keep_mask:
        result[a, b] = matrix[a, b]
    return result


_CorrespondenceImageMinimizerSet = frozenset[tuple[int, ...]]


def _active_term_minimizer_sets(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    orbit: Sequence[BlockCorrespondence],
) -> tuple[_CorrespondenceImageMinimizerSet, ...]:
    minimizer_sets: list[_CorrespondenceImageMinimizerSet] = []
    for j in alpha.active_support_nodes:
        column_values: list[tuple[float, tuple[int, ...]]] = []
        for correspondence in orbit:
            permuted = correspondence.permute_response_matrix(problem.lower_response_matrix)
            column_value = float(problem.target_importance @ permuted[:, j])
            column_values.append((column_value, correspondence.images))
        best = min(value for value, _ in column_values)
        tie_tolerance = active_config().solvers.exact_sparse.lap_objective_tie_tolerance
        minimizer_sets.append(
            frozenset(
                images for value, images in column_values if abs(value - best) <= tie_tolerance
            )
        )
    return tuple(minimizer_sets)


def _classify_coupling(
    problem: RobustActionProblem,
    alpha: CurriculumAction,
    orbit: Sequence[BlockCorrespondence],
    hull: RectangularHull,
    incompatible_gap_threshold: float,
) -> CouplingCompatibility | None:
    minimizer_sets = _active_term_minimizer_sets(problem, alpha, orbit)
    if not minimizer_sets:
        return None
    intersection = minimizer_sets[0]
    for candidate_set in minimizer_sets[1:]:
        intersection = intersection & candidate_set
    if intersection:
        return CouplingCompatibility.JOINTLY_REALIZABLE
    gap = fixed_action_rectangularization_gap(alpha, orbit, hull.lower_bounds)
    if gap > incompatible_gap_threshold:
        return CouplingCompatibility.INCOMPATIBLE
    return None


def generate_coupling_instance(request: CouplingInstanceRequest) -> CouplingInstance:
    configuration = active_config().generators.coupling_structure
    groups = tuple(CoarseGroup)[: len(request.block_pattern)]
    counts = OrderedDict(
        (group, size) for group, size in zip(groups, request.block_pattern, strict=True)
    )
    blocks = build_padded_block_structure(groups, counts, counts)
    orbit = tuple(enumerate_block_permutations(blocks))
    size = blocks.total_padded_nodes
    sparsity_coordinates_text = stable_json(
        cast(StableJsonPayload, _coupling_coordinates(request, None))
    )
    keep_mask = _deterministic_sparsity_mask(
        size, request.response_sparsity, request.seed, sparsity_coordinates_text
    )
    threshold = configuration.incompatible_fixed_action_gap_strictly_greater_than
    for attempt in range(configuration.maximum_attempts_per_instance):
        coordinates = cast(StableJsonPayload, _coupling_coordinates(request, attempt))
        random = create_float64_random_stream(
            SyntheticRandomRequest(request.seed, coordinates)
        ).generator
        lower, upper_bound = configuration.unconstrained_response_uniform
        r = random.uniform(lower, upper_bound, size=(size, size)).astype(np.float64)
        symmetric = (r + r.T) / 2.0
        q = (1.0 - request.directed_asymmetry) * symmetric + request.directed_asymmetry * r
        q = _apply_heterogeneity(q, blocks, request.response_heterogeneity)
        q = _apply_deterministic_sparsity(q, keep_mask)
        importance = _gamma_normalized_importance(random, size)
        active_action = _draw_support_restricted_action(random, size, request.support_size)
        problem = build_robust_action_problem(blocks, q, q, importance, tuple(range(size)))
        alpha = CurriculumAction(problem=problem, coordinates=active_action)
        hull = build_rectangular_hull(blocks, q, q)
        classification = _classify_coupling(problem, alpha, orbit, hull, threshold)
        if classification is not None and classification == request.compatibility:
            return CouplingInstance(
                lower_response_matrix=q,
                target_importance=importance,
                active_action=active_action,
                block_pattern=request.block_pattern,
                classification=classification,
                generation_seed=request.seed,
            )
    raise CouplingGenerationError(
        f"coupling generator exhausted attempts for {request.compatibility.value}"
    )
