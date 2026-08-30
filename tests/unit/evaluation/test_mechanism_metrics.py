from __future__ import annotations

import math

import numpy as np
import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.domain.enums import CoarseGroup
from fedorbit.evaluation.spearman import SpearmanError, descriptive_spearman
from fedorbit.orbit.correspondence import (
    BlockCorrespondence,
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.orbit.diagnostics import (
    analytic_orbit_mean,
    map_value_diagnostics,
)
from fedorbit.orbit.objective import (
    CurriculumAction,
    RobustActionProblem,
)


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


def test_map_value_diagnostics_respect_bound() -> None:
    config = load_fedorbit_config()
    tolerance = config.solvers.exact_sparse.exact_validation_absolute_tolerance
    for seed in range(4):
        problem = _problem(seed)
        orbit = list(enumerate_block_permutations(problem.blocks))
        candidates = tuple(
            CurriculumAction(problem, np.array([0.4, 0.1, 0.25, 0.25])) for _ in range(2)
        )
        diagnostics = map_value_diagnostics(candidates, problem, orbit, tolerance)
        assert diagnostics.exact_map_action_value >= -tolerance
        bound = (
            2.0
            * diagnostics.orbit_radius_bound
            * diagnostics.importance_norm
            * diagnostics.action_radius
        )
        assert diagnostics.bound == pytest.approx(bound)


def test_orbit_radius_computation_matches_enumerated_maximum() -> None:
    problem = _problem(5)
    matrix = problem.lower_response_matrix
    mean = analytic_orbit_mean(problem.blocks, matrix)
    brute = max(
        float(np.linalg.norm(correspondence.permute_response_matrix(matrix) - mean, ord=2))
        for correspondence in enumerate_block_permutations(problem.blocks)
    )
    from fedorbit.orbit.diagnostics import orbit_radius_2_norm

    assert orbit_radius_2_norm(problem.blocks, matrix).radius == pytest.approx(brute)


def test_identity_correspondence_is_in_every_orbit() -> None:
    problem = _problem(9)
    identity = BlockCorrespondence.identity(problem.blocks)
    images = {
        correspondence.images for correspondence in enumerate_block_permutations(problem.blocks)
    }
    assert identity.images in images


def test_spearman_reports_rho_n_pair_and_gates_min_points() -> None:
    config = load_fedorbit_config()
    minimum = config.scientific.statistics.spearman_minimum_valid_points
    predicted = tuple(float(value) for value in range(minimum))
    realized = tuple(float(value) * 2 for value in range(minimum))
    report = descriptive_spearman(predicted, realized, "edge -> windows")
    assert report is not None
    assert report.rho == pytest.approx(1.0)
    assert report.point_count == minimum
    assert report.pair == "edge -> windows"

    short = descriptive_spearman(
        predicted[: minimum - 1], realized[: minimum - 1], "edge -> windows"
    )
    assert short is None


def test_spearman_perfect_negative_and_ties() -> None:
    config = load_fedorbit_config()
    count = config.scientific.statistics.spearman_minimum_valid_points
    predicted = tuple(float(value) for value in range(count))
    descending = tuple(float(count - 1 - value) for value in range(count))
    negative = descriptive_spearman(predicted, descending, "pair")
    assert negative is not None
    assert negative.rho == pytest.approx(-1.0)
    constant = (1.0,) * count
    tied = descriptive_spearman(predicted, constant, "pair")
    assert tied is not None
    assert math.isfinite(tied.rho)


def test_spearman_rejects_length_mismatch() -> None:
    with pytest.raises(SpearmanError):
        descriptive_spearman((1.0, 2.0), (1.0,), "pair")
