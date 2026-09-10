from __future__ import annotations

import json
from collections import OrderedDict

from fedorbit.config.loading import active_config
from fedorbit.experiments.synthetic import (
    ExactSeparatorInstanceRequest,
    generate_exact_separator_instance,
)
from fedorbit.experiments.validation import theorem_exhaustive_validation_instance
from fedorbit.optimization.correspondence import (
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.types import CoarseGroup, stable_json


def test_theorem_exhaustive_validation_instance_matches_registered_generator() -> None:
    pattern = (2,)
    support = 1
    seed = 1103
    instance_index = 0
    groups = tuple(CoarseGroup)[: len(pattern)]
    counts = OrderedDict(zip(groups, pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    orbit = tuple(enumerate_block_permutations(blocks))
    solver_config = active_config().solvers.exact_sparse

    cell = json.loads(
        stable_json(
            theorem_exhaustive_validation_instance(
                pattern,
                support,
                seed,
                instance_index,
                blocks,
                orbit,
                solver_config.lap_objective_tie_tolerance,
                solver_config.action_tie_tolerance,
                solver_config.exact_validation_absolute_tolerance,
            )
        )
    )

    assert cell["block_pattern"] == list(pattern)
    assert cell["support"] == support
    assert cell["seed"] == seed
    assert cell["instance_index"] == instance_index
    assert isinstance(cell["absolute_objective_error"], float)
    assert cell["exact_minima"] in (True, False)
    assert cell["valid_certificate"] in (True, False)


def test_theorem_exhaustive_validation_instance_is_deterministic_per_key() -> None:
    pattern = (2,)
    support = 1
    seed = 2207
    groups = tuple(CoarseGroup)[: len(pattern)]
    counts = OrderedDict(zip(groups, pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    orbit = tuple(enumerate_block_permutations(blocks))
    solver_config = active_config().solvers.exact_sparse

    first = theorem_exhaustive_validation_instance(
        pattern,
        support,
        seed,
        0,
        blocks,
        orbit,
        solver_config.lap_objective_tie_tolerance,
        solver_config.action_tie_tolerance,
        solver_config.exact_validation_absolute_tolerance,
    )
    second = theorem_exhaustive_validation_instance(
        pattern,
        support,
        seed,
        0,
        blocks,
        orbit,
        solver_config.lap_objective_tie_tolerance,
        solver_config.action_tie_tolerance,
        solver_config.exact_validation_absolute_tolerance,
    )
    third = json.loads(
        stable_json(
            theorem_exhaustive_validation_instance(
                pattern,
                support,
                seed,
                1,
                blocks,
                orbit,
                solver_config.lap_objective_tie_tolerance,
                solver_config.action_tie_tolerance,
                solver_config.exact_validation_absolute_tolerance,
            )
        )
    )

    assert first == second
    generated_first = generate_exact_separator_instance(
        ExactSeparatorInstanceRequest(pattern, seed, support, 0)
    )
    generated_third = generate_exact_separator_instance(
        ExactSeparatorInstanceRequest(pattern, seed, support, 1)
    )
    assert not (
        generated_first.lower_response_matrix == generated_third.lower_response_matrix
    ).all()
    assert third["instance_index"] == 1
