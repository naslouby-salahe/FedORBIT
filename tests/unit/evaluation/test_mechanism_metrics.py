from __future__ import annotations

import numpy as np
import pytest

from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.optimization.diagnostics import (
    analytic_orbit_mean,
)
from fedorbit.optimization.objective import (
    RobustActionProblem,
)
from fedorbit.types import CoarseGroup


def _problem(seed: int) -> RobustActionProblem:
    rng = np.random.default_rng(seed)
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
    )
    size = blocks.total_padded_nodes
    lower = np.zeros((size, size))
    for row in range(size):
        for column in range(size):
            if blocks.block_of_node(row) == blocks.block_of_node(column):
                lower[row, column] = float(rng.uniform(-0.3, 0.3))
    return RobustActionProblem(
        blocks=blocks,
        lower_response_matrix=lower,
        upper_response_matrix=np.zeros((size, size)),
        target_importance=np.abs(rng.uniform(0.0, 1.0, size=size)) + 0.05,
        coordinate_caps=np.full(size, 0.5),
        linear_costs=np.zeros(size),
        total_budget=1.0,
        principal_support=2,
    )


def test_orbit_radius_computation_matches_enumerated_maximum() -> None:
    problem = _problem(5)
    matrix = problem.lower_response_matrix
    mean = analytic_orbit_mean(problem.blocks, matrix)
    brute = max(
        float(np.linalg.norm(correspondence.permute_response_matrix(matrix) - mean, ord=2))
        for correspondence in enumerate_block_permutations(problem.blocks)
    )
    from fedorbit.optimization.diagnostics import orbit_radius_2_norm

    assert orbit_radius_2_norm(problem.blocks, matrix).radius == pytest.approx(brute)


def test_identity_correspondence_is_in_every_orbit() -> None:
    problem = _problem(9)
    identity = BlockCorrespondence.identity(problem.blocks)
    images = {
        correspondence.images for correspondence in enumerate_block_permutations(problem.blocks)
    }
    assert identity.images in images
