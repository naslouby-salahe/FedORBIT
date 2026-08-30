from __future__ import annotations

import numpy as np
import pytest

from fedorbit.baselines import (
    FixedMatrixActionSolution,
    coarse_block_mean_matrix,
    coarse_block_min_matrix,
    committed_map_action,
    coupling_destroyed_matrices,
    matched_resource_rectangular_lower_bounds,
    optimize_against_fixed_matrix,
    orbit_mean_matrix,
)
from fedorbit.baselines.fairness import (
    ComparatorResources,
    FairnessViolationError,
    assert_identical_resources,
    assert_registered_method_name,
)
from fedorbit.domain.enums import CoarseGroup, DatasetId, TransferMethod
from fedorbit.orbit.correspondence import (
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.orbit.objective import CurriculumAction, RobustActionProblem
from fedorbit.solvers.exact_sparse import solve_robust_action


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
        upper_response_matrix=lower + 0.05,
        target_importance=np.abs(rng.uniform(0.0, 1.0, size=size)) + 0.05,
        coordinate_caps=np.full(size, 0.5),
        linear_costs=np.zeros(size),
        total_budget=1.0,
        principal_support=2,
    )


def test_fixed_matrix_optimizer_matches_exhaustive_support_truth() -> None:
    for seed in range(3):
        problem = _problem(seed)
        matrix = problem.lower_response_matrix
        solution = optimize_against_fixed_matrix(problem, matrix)
        brute_best = -np.inf
        for support_size in (1, 2):
            from itertools import combinations

            for support in combinations(range(problem.size), support_size):
                grid = np.linspace(0.0, 0.5, 9)
                grids = [grid] * support_size
                for combination in zip(*grids, strict=True):
                    alpha = np.zeros(problem.size)
                    for node, value in zip(support, combination, strict=True):
                        alpha[node] = value
                    if alpha.sum() > problem.total_budget:
                        continue
                    value = float(
                        (problem.target_importance @ matrix @ alpha) - problem.linear_costs @ alpha
                    )
                    brute_best = max(brute_best, value)
        assert isinstance(solution, FixedMatrixActionSolution)
        assert solution.objective_value <= brute_best + 1e-9


def test_local_only_is_identity_and_local_sir_uses_target_response() -> None:
    problem = _problem(11)
    local_only = optimize_against_fixed_matrix(problem, np.zeros((problem.size, problem.size)))
    assert int(np.count_nonzero(local_only.selected_action.coordinates)) == 0
    sir = optimize_against_fixed_matrix(problem, problem.lower_response_matrix)
    brute_sir = -np.inf
    grid = np.linspace(0.0, 0.5, 21)
    for first in grid:
        for second in grid:
            if first + second > problem.total_budget:
                continue
            alpha = np.array([first, second, 0.0, 0.0])
            brute_sir = max(
                brute_sir,
                float(problem.target_importance @ problem.lower_response_matrix @ alpha),
            )
    assert sir.objective_value >= brute_sir - 1e-6


def test_coarse_block_summaries_lift_to_fine_space() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 2},
    )
    lower = np.array(
        [
            [0.10, 0.90, 0.00, 0.00],
            [0.70, 0.30, 0.00, 0.00],
            [0.00, 0.00, 0.20, -0.10],
            [0.00, 0.00, -0.05, 0.40],
        ]
    )
    mean_summary = coarse_block_mean_matrix(blocks, lower).matrix
    min_summary = coarse_block_min_matrix(blocks, lower).matrix
    assert mean_summary[0, 1] == pytest.approx(0.5)
    assert mean_summary[2, 3] == pytest.approx(0.1125)
    assert mean_summary[0, 2] == 0.0
    assert min_summary[0, 1] == pytest.approx(0.1)
    assert min_summary[2, 3] == pytest.approx(-0.1)


def test_orbit_mean_matches_enumerated_average() -> None:
    problem = _problem(17)
    enumerated = np.mean(
        [
            correspondence.permute_response_matrix(problem.lower_response_matrix)
            for correspondence in enumerate_block_permutations(problem.blocks)
        ],
        axis=0,
    )
    analytic = orbit_mean_matrix(problem.blocks, problem.lower_response_matrix).matrix
    assert np.allclose(analytic, enumerated)


def test_matched_resource_rectangular_lower_bounds_dominate_every_permutation() -> None:
    problem = _problem(23)
    bounds = matched_resource_rectangular_lower_bounds(
        problem.blocks, problem.lower_response_matrix, problem.upper_response_matrix
    ).matrix
    for correspondence in enumerate_block_permutations(problem.blocks):
        permuted = correspondence.permute_response_matrix(problem.lower_response_matrix)
        assert np.all(bounds <= permuted + 1e-12)


def test_coupling_destruction_preserves_multisets_pairing_and_dimensions() -> None:
    problem = _problem(29)
    seed = 8861
    destroyed = coupling_destroyed_matrices(
        problem.blocks,
        problem.lower_response_matrix,
        problem.upper_response_matrix,
        seed,
        "destruction-probe",
    )
    assert destroyed.lower_response_matrix.shape == problem.lower_response_matrix.shape
    assert destroyed.upper_response_matrix.shape == problem.upper_response_matrix.shape
    for block_index in range(len(problem.blocks.padded_size_tuple)):
        rows = problem.blocks.block_index_range(block_index)
        for other in range(len(problem.blocks.padded_size_tuple)):
            columns = problem.blocks.block_index_range(other)
            original_lower = problem.lower_response_matrix[np.ix_(rows, columns)].reshape(-1)
            original_upper = problem.upper_response_matrix[np.ix_(rows, columns)].reshape(-1)
            new_lower = destroyed.lower_response_matrix[np.ix_(rows, columns)].reshape(-1)
            new_upper = destroyed.upper_response_matrix[np.ix_(rows, columns)].reshape(-1)
            assert sorted(original_lower.tolist()) == sorted(new_lower.tolist())
            assert sorted(original_upper.tolist()) == sorted(new_upper.tolist())
            original_pairs = set(zip(original_lower.tolist(), original_upper.tolist(), strict=True))
            new_pairs = set(zip(new_lower.tolist(), new_upper.tolist(), strict=True))
            assert original_pairs == new_pairs
    replay = coupling_destroyed_matrices(
        problem.blocks,
        problem.lower_response_matrix,
        problem.upper_response_matrix,
        seed,
        "destruction-probe",
    )
    assert np.allclose(destroyed.lower_response_matrix, replay.lower_response_matrix)
    assert np.allclose(destroyed.upper_response_matrix, replay.upper_response_matrix)


def test_committed_map_optimizes_under_chosen_correspondence() -> None:
    problem = _problem(31)
    target_matrix = problem.lower_response_matrix.copy()
    committed_action = committed_map_action(problem, problem.lower_response_matrix, target_matrix)
    committed = committed_action.correspondence.permute_response_matrix(
        problem.lower_response_matrix
    )
    expected_objective = float(
        problem.target_importance @ committed @ committed_action.selected_action.coordinates
        - problem.linear_costs @ committed_action.selected_action.coordinates
    )
    del expected_objective
    assert isinstance(committed_action.selected_action, CurriculumAction)


def test_exact_map_oracle_uses_given_correspondence() -> None:
    from fedorbit.oracle import OracleCorrespondence, exact_map_action

    problem = _problem(37)
    identity = next(iter(enumerate_block_permutations(problem.blocks)))
    outcome = exact_map_action(
        problem,
        OracleCorrespondence(
            source_client=DatasetId.EDGE_IIOTSET_NETWORK,
            target_client=DatasetId.TON_IOT_WINDOWS10_HOST,
            correspondence=identity,
        ),
    )
    committed = identity.permute_response_matrix(problem.lower_response_matrix)
    expected = float(
        problem.target_importance @ committed @ outcome.selected_action.coordinates
        - problem.linear_costs @ outcome.selected_action.coordinates
    )
    assert outcome.objective_value == pytest.approx(expected)


def test_without_confirmation_matches_principal_ranking_then_assimilates() -> None:
    problem = _problem(41)
    principal = solve_robust_action(problem, support_limit=1)
    no_confirm_matrix = principal.selected_action.active_support_nodes
    assert len(no_confirm_matrix) <= 1


def test_method_names_are_stable() -> None:
    for name in (
        TransferMethod.LOCAL_ONLY,
        TransferMethod.LOCAL_SIR,
        TransferMethod.COARSE_BLOCK_MEAN,
        TransferMethod.COARSE_BLOCK_MIN,
        TransferMethod.ORBIT_MEAN,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
        TransferMethod.GENERIC_EXACT_QAP,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.FEDORBIT_DENSE_CCP_FALLBACK,
        TransferMethod.EXACT_MAP_ORACLE,
        TransferMethod.FEDORBIT_WITHOUT_CONFIRMATION,
        TransferMethod.COUPLING_DESTROYED_FEDORBIT,
    ):
        assert_registered_method_name(name.value)
    with pytest.raises(FairnessViolationError):
        assert_registered_method_name("baseline-A")


def _resources_bundle() -> ComparatorResources:
    return ComparatorResources(
        source_packet_id="packet-1",
        target_checkpoint_artifact_id="checkpoint-1",
        target_importance_vector_sha256="a" * 64,
        action_budget_cap=0.5,
        support_cap=2,
        seed=1103,
        confirmation_opportunity=True,
        live_assimilation_step_allowance=500,
        test_access_granted=False,
        extra_target_labels=False,
        additional_tuning_seeds=(),
        local_base_checkpoint_favorable=False,
    )


def test_identical_resources_required_across_methods() -> None:
    from dataclasses import replace

    bundle = _resources_bundle()
    assert_identical_resources("Local-Only", bundle, bundle)
    tampered = replace(bundle, support_cap=3)
    with pytest.raises(FairnessViolationError):
        assert_identical_resources("Local-SIR", bundle, tampered)


def test_fairness_violations_rejected_at_construction() -> None:
    from dataclasses import replace

    base = _resources_bundle()
    with pytest.raises(FairnessViolationError):
        replace(base, test_access_granted=True).validate_contract()
    with pytest.raises(FairnessViolationError):
        replace(base, extra_target_labels=True).validate_contract()
    with pytest.raises(FairnessViolationError):
        replace(base, additional_tuning_seeds=(7,)).validate_contract()
    with pytest.raises(FairnessViolationError):
        replace(base, local_base_checkpoint_favorable=True).validate_contract()
