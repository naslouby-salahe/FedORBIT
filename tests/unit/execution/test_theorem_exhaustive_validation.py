from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

import fedorbit.experiments.solvers as experiment_solvers
import fedorbit.optimization.exact_qap as exact_qap
from fedorbit.config.loading import active_config
from fedorbit.experiments.synthetic import (
    ExactSeparatorInstanceRequest,
    generate_exact_separator_instance,
)
from fedorbit.experiments.validation import theorem_exhaustive_validation_instance
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import WorkspaceLayout
from fedorbit.optimization.certificates import build_rectangular_hull
from fedorbit.optimization.correspondence import (
    build_padded_block_structure,
    enumerate_active_image_maps,
    enumerate_block_permutations,
)
from fedorbit.optimization.dense_ccp import (
    AssignmentVariableLayout,
    barycenter_start,
    ccp_trajectory,
    project_to_permutation,
    solve_dense_ccp,
)
from fedorbit.optimization.exact_qap import (
    QapUncertifiedError,
    fixed_action_worst_correspondence_qap,
)
from fedorbit.optimization.exact_sparse import (
    SparseMasterNonConvergenceError,
    fixed_action_worst_correspondence,
    solve_robust_action,
    solve_support_master,
)
from fedorbit.optimization.objective import (
    CurriculumAction,
    SupportCoordinateSet,
    build_robust_action_problem,
    enumerate_support_coordinate_sets,
    evaluate_objective,
)
from fedorbit.types import (
    CoarseGroup,
    DenseCcpCertificationStatus,
    EvaluationConditionName,
    ExperimentName,
    MetricId,
    MetricUnit,
    OverwritePolicy,
    SemanticCoordinates,
    TerminalState,
    TransferMethod,
    stable_json,
)


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


def test_fixed_action_separator_matches_exhaustive_joint_orbit_and_counters() -> None:
    pattern = (2, 3)
    support = 2
    seed = 1103
    groups = tuple(CoarseGroup)[: len(pattern)]
    counts = OrderedDict(zip(groups, pattern, strict=True))
    blocks = build_padded_block_structure(groups, counts, counts)
    instance = generate_exact_separator_instance(
        ExactSeparatorInstanceRequest(pattern, seed, support, 0)
    )
    problem = build_robust_action_problem(
        blocks,
        instance.lower_response_matrix,
        instance.upper_response_matrix,
        instance.target_importance / instance.target_importance.sum(),
        tuple(range(sum(pattern))),
    )
    action = CurriculumAction(problem, instance.active_action)
    settings = active_config().solvers.exact_sparse
    outcome = fixed_action_worst_correspondence(
        problem,
        action,
        settings.lap_objective_tie_tolerance,
        settings.action_tie_tolerance,
    )
    qap = fixed_action_worst_correspondence_qap(problem, action)
    orbit = tuple(enumerate_block_permutations(blocks))
    exhaustive_truth = min(evaluate_objective(action, correspondence) for correspondence in orbit)
    expected_active_maps = len(
        tuple(enumerate_active_image_maps(blocks, action.active_support_nodes))
    )
    support_counts = tuple(
        sum(blocks.block_of_node(node) == block_index for node in action.active_support_nodes)
        for block_index in range(len(blocks.padded_size_tuple))
    )
    expected_lap_calls = expected_active_maps * sum(
        size - support_count > 0
        for size, support_count in zip(blocks.padded_size_tuple, support_counts, strict=True)
    )

    assert outcome.separator_objective == exhaustive_truth
    assert outcome.worst_correspondence in orbit
    assert outcome.active_image_candidates == expected_active_maps
    assert outcome.lap_calls == expected_lap_calls
    assert qap.certified
    assert qap.require_certified().objective_value == pytest.approx(
        exhaustive_truth, abs=settings.exact_validation_absolute_tolerance
    )
    assert qap.require_certified().correspondence in orbit


def test_principal_sparse_robust_action_matches_exhaustive_joint_orbit_not_marginals() -> None:
    group = CoarseGroup.DISRUPTION
    blocks = build_padded_block_structure((group,), {group: 3}, {group: 3})
    response = np.asarray(((9.0, 1.0, 2.0), (3.0, 8.0, 4.0), (5.0, 6.0, 7.0)))
    initial_problem = build_robust_action_problem(
        blocks,
        response,
        response,
        np.asarray((0.6, 0.3, 0.1), dtype=np.float64),
        (0, 1, 2),
    )
    problem = replace(
        initial_problem,
        coordinate_caps=np.full(3, 0.2, dtype=np.float64),
        linear_costs=np.zeros(3, dtype=np.float64),
        total_budget=0.2,
        principal_support=1,
    )
    orbit = tuple(enumerate_block_permutations(blocks))
    actions = (
        CurriculumAction(problem, np.zeros(3, dtype=np.float64)),
        *(
            CurriculumAction(problem, np.eye(3, dtype=np.float64)[node] * problem.total_budget)
            for node in range(problem.size)
        ),
    )
    expected_action, expected_value = max(
        (
            (action, min(evaluate_objective(action, correspondence) for correspondence in orbit))
            for action in actions
        ),
        key=lambda entry: entry[1],
    )
    solution = solve_robust_action(problem, support_limit=1)
    hull = build_rectangular_hull(blocks, response, response)
    rectangular_value = float(
        problem.target_importance @ hull.lower_bounds @ solution.selected_action.coordinates
    )

    np.testing.assert_array_equal(solution.selected_action.coordinates, expected_action.coordinates)
    assert solution.certified_robust_value == pytest.approx(expected_value, abs=1e-12)
    assert expected_value > rectangular_value


def test_coupling_metric_artifact_records_the_principal_sparse_action_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = CoarseGroup.DISRUPTION
    blocks = build_padded_block_structure((group,), {group: 2}, {group: 2})
    problem = build_robust_action_problem(
        blocks,
        np.asarray(((0.8, 0.1), (0.2, 0.7)), dtype=np.float64),
        np.asarray(((0.8, 0.1), (0.2, 0.7)), dtype=np.float64),
        np.asarray((0.5, 0.5), dtype=np.float64),
        (0, 1),
    )
    problem = replace(problem, principal_support=1)
    orbit = tuple(enumerate_block_permutations(blocks))
    alpha = CurriculumAction(problem, np.asarray((0.2, 0.0), dtype=np.float64))
    hull = build_rectangular_hull(
        blocks, problem.lower_response_matrix, problem.upper_response_matrix
    )
    persisted: list[tuple[MetricId, float, MetricUnit]] = []

    def record_metric(*arguments: object) -> None:
        metric_name, metric_value, metric_unit = arguments[7:10]
        assert isinstance(metric_name, MetricId)
        assert isinstance(metric_value, float)
        assert isinstance(metric_unit, MetricUnit)
        persisted.append((metric_name, metric_value, metric_unit))

    monkeypatch.setattr(experiment_solvers, "persist_synthetic_benchmark_metric", record_metric)
    experiment_solvers.persist_coupling_mechanism_metrics(
        cast(ArtifactStore, None),
        cast(WorkspaceLayout, None),
        ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION,
        EvaluationConditionName("bounded-action-set"),
        1,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        1103,
        problem,
        orbit,
        alpha,
        hull,
        OverwritePolicy.REUSE,
    )

    assert (MetricId.COUPLING_ACTION_SET_SUPPORT, 1.0, MetricUnit.COUNT) in persisted


def test_exact_separator_and_qap_choose_lexicographic_correspondence_under_ties() -> None:
    group = CoarseGroup.DISRUPTION
    blocks = build_padded_block_structure((group,), {group: 2}, {group: 2})
    problem = build_robust_action_problem(
        blocks,
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.asarray((0.5, 0.5), dtype=np.float64),
        (0, 1),
    )
    action = CurriculumAction(problem, np.asarray((0.25, 0.0), dtype=np.float64))
    settings = active_config().solvers.exact_sparse
    outcomes = tuple(
        fixed_action_worst_correspondence(
            problem,
            action,
            settings.lap_objective_tie_tolerance,
            settings.action_tie_tolerance,
        )
        for _ in range(3)
    )
    qap = fixed_action_worst_correspondence_qap(problem, action)

    assert {outcome.worst_correspondence.images for outcome in outcomes} == {(0, 1)}
    assert qap.certified
    assert qap.require_certified().correspondence.images == (0, 1)


def test_support_master_certifies_after_a_joint_scenario_cut_and_reports_cut_cap() -> None:
    group = CoarseGroup.DISRUPTION
    blocks = build_padded_block_structure((group,), {group: 2}, {group: 2})
    problem = build_robust_action_problem(
        blocks,
        np.asarray(((10.0, 0.0), (0.0, 0.0)), dtype=np.float64),
        np.asarray(((10.0, 0.0), (0.0, 0.0)), dtype=np.float64),
        np.asarray((1.0, 0.0), dtype=np.float64),
        (0, 1),
    )
    support = SupportCoordinateSet(problem, (0,))

    with pytest.raises(SparseMasterNonConvergenceError):
        solve_support_master(problem, support, maximum_cuts=1)

    solution = solve_support_master(problem, support, maximum_cuts=2)

    assert solution.iterations == 2
    assert solution.cut_count == 2
    assert solution.certified_robust_value == 0.0
    assert solution.worst_correspondence.images == (0, 1)


def test_sparse_support_enumeration_and_final_ties_are_complete_and_deterministic() -> None:
    group = CoarseGroup.DISRUPTION
    blocks = build_padded_block_structure((group,), {group: 3}, {group: 3})
    problem = build_robust_action_problem(
        blocks,
        np.ones((3, 3), dtype=np.float64),
        np.ones((3, 3), dtype=np.float64),
        np.full(3, 1.0 / 3.0, dtype=np.float64),
        (0, 1, 2),
    )
    problem = replace(problem, total_budget=0.25)
    supports = enumerate_support_coordinate_sets(problem, support_limit=2)
    solution = solve_robust_action(problem, support_limit=2)

    assert tuple(support.nodes for support in supports) == (
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
    )
    assert len(solution.support_solutions) == len(supports)
    assert solution.selected_action.active_support_nodes == (0,)
    assert solution.selected_action.realized_support_size == 1


def test_qap_timeout_status_cannot_be_promoted_to_an_exact_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimedOutModel:
        def __init__(self, name: str) -> None:
            del name

        def setObjective(self, objective: object, sense: str) -> None:
            del objective, sense

        def optimize(self) -> None:
            return None

        def getStatus(self) -> str:
            return "timelimit"

    def ignore_configuration(*arguments: object) -> None:
        del arguments

    def empty_assignment(*arguments: object) -> dict[object, object]:
        del arguments
        return {}

    def empty_products(*arguments: object) -> list[object]:
        del arguments
        return []

    group = CoarseGroup.DISRUPTION
    blocks = build_padded_block_structure((group,), {group: 2}, {group: 2})
    problem = build_robust_action_problem(
        blocks,
        np.eye(2, dtype=np.float64),
        np.eye(2, dtype=np.float64),
        np.asarray((0.5, 0.5), dtype=np.float64),
        (0, 1),
    )
    action = CurriculumAction(problem, np.asarray((0.25, 0.0), dtype=np.float64))
    monkeypatch.setattr(exact_qap, "Model", TimedOutModel)
    monkeypatch.setattr(exact_qap, "_configure_model", ignore_configuration)
    monkeypatch.setattr(exact_qap, "_build_assignment_structure", empty_assignment)
    monkeypatch.setattr(exact_qap, "_add_mccormick_products", empty_products)

    result = fixed_action_worst_correspondence_qap(problem, action)

    assert not result.certified
    assert result.correspondence is None
    assert result.objective_value is None
    assert result.terminal_state == TerminalState.TIME_LIMIT
    with pytest.raises(QapUncertifiedError):
        result.require_certified()


def test_dense_ccp_reports_heuristic_only_status_and_valid_lap_projection() -> None:
    group = CoarseGroup.DISRUPTION
    blocks = build_padded_block_structure((group,), {group: 2}, {group: 2})
    problem = build_robust_action_problem(
        blocks,
        np.asarray(((0.7, 0.1), (0.2, 0.8)), dtype=np.float64),
        np.asarray(((0.7, 0.1), (0.2, 0.8)), dtype=np.float64),
        np.asarray((0.5, 0.5), dtype=np.float64),
        (0, 1),
    )
    action = CurriculumAction(problem, np.asarray((0.2, 0.1), dtype=np.float64))
    layout = AssignmentVariableLayout.build(blocks)
    trajectory = ccp_trajectory(problem, action, barycenter_start(layout), layout)
    projection = project_to_permutation(layout, trajectory.final_assignment)
    outcome = solve_dense_ccp(problem, 1103, SemanticCoordinates("dense-ccp-audit"))

    assert trajectory.iterations > 0
    assert 0.0 <= trajectory.integrality_residual <= 0.5
    assert projection in tuple(enumerate_block_permutations(blocks))
    assert outcome.worst_projected_correspondence in tuple(enumerate_block_permutations(blocks))
    tolerance = active_config().solvers.exact_sparse.exact_validation_absolute_tolerance
    assert outcome.dense_bound_gap >= -tolerance
    assert outcome.certification_status is DenseCcpCertificationStatus.HEURISTIC_ONLY
