from __future__ import annotations

import numpy as np
import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.domain.enums import CoarseGroup
from fedorbit.orbit.correspondence import (
    BlockCorrespondence,
    PaddedBlockStructure,
    build_padded_block_structure,
    enumerate_block_permutations,
)
from fedorbit.orbit.objective import (
    ActionSpaceError,
    CertifiedActionCandidate,
    CurriculumAction,
    RobustActionProblem,
    SupportCoordinateSet,
    actions_tied_within_tolerance,
    build_robust_action_problem,
    curriculum_action_from_entries,
    enumerate_support_coordinate_sets,
    evaluate_objective,
    exact_map_action_value,
    h_orb,
    h_rect,
    map_conditioned_optimum,
    robust_post_map_value,
    robust_pre_map_value,
    rounded_action_vector,
    select_deterministic_candidate,
    zero_action,
    zero_action_objective,
)


def _single_block() -> PaddedBlockStructure:
    return build_padded_block_structure(
        (CoarseGroup.DISRUPTION,),
        {CoarseGroup.DISRUPTION: 2},
        {CoarseGroup.DISRUPTION: 2},
    )


def test_action_configuration_values_come_from_authoritative_yaml() -> None:
    config = load_fedorbit_config()
    blocks = _single_block()
    problem = build_robust_action_problem(
        config,
        blocks,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.array([1.0, 0.0]),
        actionable_nodes=(0, 1),
    )
    action = config.scientific.action
    assert problem.total_budget == action.total_curriculum_budget
    assert problem.principal_support == action.principal_sparse_support
    assert np.all(problem.coordinate_caps == action.coordinate_cap)
    assert np.all(problem.linear_costs == action.linear_cost_per_actionable_node)


def test_non_actionable_nodes_have_zero_cap_and_zero_cost() -> None:
    config = load_fedorbit_config()
    blocks = _single_block()
    problem = build_robust_action_problem(
        config,
        blocks,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.array([1.0, 1.0]),
        actionable_nodes=(0,),
    )
    assert problem.coordinate_caps[1] == 0.0
    assert problem.linear_costs[1] == 0.0
    assert problem.coordinate_caps[0] > 0.0
    assert problem.actionable_nodes() == (0,)


def test_rejects_negative_target_importance() -> None:
    bad_importance = np.array([1.0, -0.1])
    with pytest.raises(ActionSpaceError):
        RobustActionProblem(
            blocks=_single_block(),
            lower_response_matrix=np.zeros((2, 2)),
            upper_response_matrix=np.zeros((2, 2)),
            target_importance=bad_importance,
            coordinate_caps=np.ones(2),
            linear_costs=np.zeros(2),
            total_budget=1.0,
            principal_support=2,
        )


def test_rejects_nan_response_matrices() -> None:
    nan_matrix = np.full((2, 2), np.nan)
    with pytest.raises(ActionSpaceError):
        RobustActionProblem(
            blocks=_single_block(),
            lower_response_matrix=nan_matrix,
            upper_response_matrix=np.zeros((2, 2)),
            target_importance=np.ones(2),
            coordinate_caps=np.ones(2),
            linear_costs=np.zeros(2),
            total_budget=1.0,
            principal_support=2,
        )


def test_curriculum_action_enforces_caps_and_budget() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.zeros((2, 2)),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.ones(2),
        coordinate_caps=np.array([0.25, 0.25]),
        linear_costs=np.zeros(2),
        total_budget=0.4,
        principal_support=2,
    )
    within = CurriculumAction(problem, np.array([0.25, 0.15]))
    assert within.is_within_budget()
    assert within.realized_support_size == 2
    over_budget = CurriculumAction(problem, np.array([0.25, 0.25]))
    assert not over_budget.is_within_budget()


def test_curriculum_action_rejects_cap_violation() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.zeros((2, 2)),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.ones(2),
        coordinate_caps=np.array([0.25, 0.25]),
        linear_costs=np.zeros(2),
        total_budget=0.4,
        principal_support=2,
    )
    with pytest.raises(ActionSpaceError):
        CurriculumAction(problem, np.array([0.3, 0.05]))


def test_curriculum_action_rejects_negative_coordinates() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.zeros((2, 2)),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.ones(2),
        coordinate_caps=np.array([0.25, 0.25]),
        linear_costs=np.zeros(2),
        total_budget=0.4,
        principal_support=2,
    )
    with pytest.raises(ActionSpaceError):
        CurriculumAction(problem, np.array([-0.1, 0.2]))


def test_zero_action_has_exactly_zero_objective() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.array([[0.5, -0.2], [0.1, 0.3]]),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.array([1.0, 1.0]),
        coordinate_caps=np.ones(2),
        linear_costs=np.array([0.01, 0.01]),
        total_budget=1.0,
        principal_support=2,
    )
    orbit = list(enumerate_block_permutations(problem.blocks))
    zero = zero_action(problem)
    assert zero_action_objective() == 0.0
    assert evaluate_objective(zero, orbit[0]) - zero_action_objective() < 1e-300
    assert h_orb(zero, orbit) == 0.0


def test_objective_matches_hand_computed_value() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.array([[0.1, 0.2], [0.3, 0.4]]),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.array([1.0, 0.0]),
        coordinate_caps=np.ones(2),
        linear_costs=np.array([0.01, 0.0]),
        total_budget=1.0,
        principal_support=2,
    )
    alpha = curriculum_action_from_entries(problem, ((0, 0.5),))
    identity = BlockCorrespondence.identity(problem.blocks)
    swap = BlockCorrespondence(problem.blocks, (1, 0))
    assert evaluate_objective(alpha, identity) == pytest.approx(0.1 * 0.5 - 0.005)
    assert evaluate_objective(alpha, swap) == pytest.approx(0.4 * 0.5 - 0.005)
    assert h_orb(alpha, [identity, swap]) == pytest.approx(0.05)
    hull_lower = np.array([[0.1, 0.2], [0.2, 0.1]])
    assert h_rect(alpha, hull_lower) == pytest.approx(0.05)


def test_h_rect_requires_matching_hull_shape() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.zeros((2, 2)),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.ones(2),
        coordinate_caps=np.ones(2),
        linear_costs=np.zeros(2),
        total_budget=1.0,
        principal_support=2,
    )
    alpha = zero_action(problem)
    with pytest.raises(ActionSpaceError):
        h_rect(alpha, np.zeros((3, 3)))


def test_map_value_hand_fixture_with_trade_off() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.array([[7.0, 4.0], [3.0, 2.0]]),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.array([1.0, 1.0]),
        coordinate_caps=np.array([1.0, 1.0]),
        linear_costs=np.zeros(2),
        total_budget=2.0,
        principal_support=2,
    )
    alpha_first = CurriculumAction(problem, np.array([1.0, 0.0]))
    alpha_second = CurriculumAction(problem, np.array([0.0, 1.0]))
    candidates = (zero_action(problem), alpha_first, alpha_second)
    identity = BlockCorrespondence.identity(problem.blocks)
    swap = BlockCorrespondence(problem.blocks, (1, 0))
    assert evaluate_objective(alpha_first, identity) == pytest.approx(10.0)
    assert evaluate_objective(alpha_first, swap) == pytest.approx(6.0)
    assert evaluate_objective(alpha_second, identity) == pytest.approx(6.0)
    assert evaluate_objective(alpha_second, swap) == pytest.approx(10.0)
    assert map_conditioned_optimum(identity, candidates) == pytest.approx(10.0)
    assert map_conditioned_optimum(swap, candidates) == pytest.approx(10.0)
    pre_map = robust_pre_map_value(candidates, [identity, swap])
    post_map = robust_post_map_value(candidates, [identity, swap])
    assert pre_map == pytest.approx(6.0)
    assert post_map == pytest.approx(10.0)
    assert exact_map_action_value(candidates, [identity, swap]) == pytest.approx(4.0)


def test_support_coordinate_sets_enumerate_all_actionable_subsets() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.zeros((2, 2)),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.ones(2),
        coordinate_caps=np.array([0.25, 0.25]),
        linear_costs=np.zeros(2),
        total_budget=1.0,
        principal_support=2,
    )
    sets = enumerate_support_coordinate_sets(problem)
    sizes = sorted({support.size for support in sets})
    assert sizes == [1, 2]
    assert len([support for support in sets if support.size == 1]) == 2
    assert len([support for support in sets if support.size == 2]) == 1
    for support in sets:
        counts = support.block_support_counts()
        assert sum(counts.per_block) == support.size


def test_support_coordinate_set_rejects_ineligible_nodes() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.zeros((2, 2)),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.ones(2),
        coordinate_caps=np.array([0.25, 0.0]),
        linear_costs=np.zeros(2),
        total_budget=1.0,
        principal_support=2,
    )
    with pytest.raises(ActionSpaceError):
        SupportCoordinateSet(problem=problem, nodes=(0, 1))


def test_negative_tolerance_rejected() -> None:
    with pytest.raises(ActionSpaceError):
        actions_tied_within_tolerance(1.0, 1.0, -1.0)


def test_actions_tied_within_tolerance_and_rounding() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.zeros((2, 2)),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.ones(2),
        coordinate_caps=np.ones(2),
        linear_costs=np.zeros(2),
        total_budget=1.0,
        principal_support=2,
    )
    assert actions_tied_within_tolerance(1.0, 1.0 + 1e-12, 1e-10)
    assert not actions_tied_within_tolerance(1.0, 1.0 + 1e-8, 1e-10)
    alpha = CurriculumAction(problem, np.array([0.123456789, 0.0]))
    rounded = rounded_action_vector(alpha, 1e-6)
    assert rounded == (0.123457, 0.0)


def test_deterministic_candidate_selection_prefers_higher_then_lexicographic() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.zeros((2, 2)),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.ones(2),
        coordinate_caps=np.ones(2),
        linear_costs=np.zeros(2),
        total_budget=1.0,
        principal_support=2,
    )
    first = CertifiedActionCandidate(
        action=CurriculumAction(problem, np.array([0.4, 0.1])),
        certified_robust_value=1.0 + 1e-13,
        target_node_sequence=(0, 1),
    )
    second = CertifiedActionCandidate(
        action=CurriculumAction(problem, np.array([0.1, 0.4])),
        certified_robust_value=1.0,
        target_node_sequence=(0, 1),
    )
    selected = select_deterministic_candidate((second, first), 1e-10, 1e-12)
    assert selected is not None
    assert selected.deterministic_ordering_key(1e-12) <= second.deterministic_ordering_key(1e-12)
    worse = CertifiedActionCandidate(
        action=zero_action(problem),
        certified_robust_value=0.5,
        target_node_sequence=(0,),
    )
    best = select_deterministic_candidate((worse, first, second), 1e-10, 1e-12)
    assert best is not None
    assert best.certified_robust_value >= 1.0
    assert select_deterministic_candidate((), 1e-10, 1e-12) is None


def test_build_rejects_out_of_range_actionable_nodes() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.zeros((2, 2)),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.ones(2),
        coordinate_caps=np.ones(2),
        linear_costs=np.zeros(2),
        total_budget=1.0,
        principal_support=2,
    )
    with pytest.raises(ActionSpaceError):
        SupportCoordinateSet(problem=problem, nodes=(5,))


def test_math_inf_guard_raises_on_empty_orbit() -> None:
    problem = RobustActionProblem(
        blocks=_single_block(),
        lower_response_matrix=np.zeros((2, 2)),
        upper_response_matrix=np.zeros((2, 2)),
        target_importance=np.ones(2),
        coordinate_caps=np.ones(2),
        linear_costs=np.zeros(2),
        total_budget=1.0,
        principal_support=2,
    )
    empty_orbit: list[BlockCorrespondence] = []
    alpha = zero_action(problem)
    with pytest.raises(ActionSpaceError):
        h_orb(alpha, empty_orbit)
