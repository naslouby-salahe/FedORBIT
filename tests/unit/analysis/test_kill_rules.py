from __future__ import annotations

from fedorbit.analysis.kill_rules import (
    confirmation_has_no_safety_value,
    coupling_destruction_retains_gain,
    exactness_failure,
    generic_qap_dominates,
    local_sir_is_sufficient,
    point_matching_is_sufficient,
    rectangularization_is_sufficient,
    source_response_is_too_unstable,
    sparse_support_is_operationally_irrelevant,
    strict_interface_removes_gain,
    theory_classification_failure,
    unresolved_map_regime_lacks_practical_motivation,
)


def test_exactness_failure_triggers_on_either_condition() -> None:
    assert exactness_failure(True, False)
    assert exactness_failure(False, True)
    assert not exactness_failure(False, False)


def test_rectangularization_is_sufficient_requires_both_conditions() -> None:
    assert rectangularization_is_sufficient(0.95, False)
    assert not rectangularization_is_sufficient(0.95, True)
    assert not rectangularization_is_sufficient(0.5, False)


def test_theory_classification_failure_triggers_on_either_condition() -> None:
    assert theory_classification_failure(True, False)
    assert theory_classification_failure(False, True)
    assert not theory_classification_failure(False, False)


def test_generic_qap_dominates_requires_all_thresholds() -> None:
    assert generic_qap_dominates(True, 1.0, 1.2, 1.0, False)
    assert not generic_qap_dominates(False, 1.0, 1.2, 1.0, False)
    assert not generic_qap_dominates(True, 1.01, 1.2, 1.0, False)
    assert not generic_qap_dominates(True, 1.0, 1.21, 1.0, False)
    assert not generic_qap_dominates(True, 1.0, 1.2, 1.01, False)
    assert not generic_qap_dominates(True, 1.0, 1.2, 1.0, True)


def test_sparse_support_is_operationally_irrelevant_requires_every_configured_support() -> None:
    assert sparse_support_is_operationally_irrelevant(0.02, 0.75, {1: True, 2: True, 3: True})
    assert not sparse_support_is_operationally_irrelevant(0.02, 0.75, {1: True, 2: False, 3: True})
    assert not sparse_support_is_operationally_irrelevant(0.01, 0.75, {1: True, 2: True, 3: True})
    assert not sparse_support_is_operationally_irrelevant(0.02, 0.5, {1: True, 2: True, 3: True})


def test_local_sir_is_sufficient_requires_all_three_conditions() -> None:
    assert local_sir_is_sufficient(True, 4, True)
    assert not local_sir_is_sufficient(False, 4, True)
    assert not local_sir_is_sufficient(True, 3, True)
    assert not local_sir_is_sufficient(True, 4, False)


def test_point_matching_is_sufficient_requires_every_pair() -> None:
    assert point_matching_is_sufficient((0.0, 0.02), (0.01, 0.05))
    assert not point_matching_is_sufficient((0.0, 0.03), (0.01, 0.05))
    assert not point_matching_is_sufficient((0.0, 0.02), (0.01, 0.005))
    assert not point_matching_is_sufficient((), ())


def test_coupling_destruction_retains_gain_is_a_direct_pass_through() -> None:
    assert coupling_destruction_retains_gain(True)
    assert not coupling_destruction_retains_gain(False)


def test_strict_interface_removes_gain_requires_majority_of_qualifying_pairs() -> None:
    assert strict_interface_removes_gain(
        (0.0, -0.01, -0.02, 0.0),
        (0.005, 0.005, 0.005, 0.005),
        (True, True, True, True),
    )
    assert not strict_interface_removes_gain(
        (0.0, -0.01, -0.02, 0.0),
        (0.005, 0.005, 0.005, 0.005),
        (True, False, False, True),
    )
    assert not strict_interface_removes_gain(
        (0.01, 0.02, -0.02, 0.0),
        (0.005, 0.005, 0.005, 0.005),
        (True, True, True, True),
    )
    assert not strict_interface_removes_gain(
        (0.0, -0.01, -0.02, 0.0),
        (0.02, 0.02, 0.005, 0.005),
        (True, True, True, True),
    )


def test_confirmation_has_no_safety_value_triggers_on_either_failure_mode() -> None:
    assert confirmation_has_no_safety_value(False, False, False)
    assert confirmation_has_no_safety_value(True, True, True)
    assert not confirmation_has_no_safety_value(True, False, False)
    assert not confirmation_has_no_safety_value(False, True, False)


def test_source_response_is_too_unstable_triggers_on_either_condition() -> None:
    assert source_response_is_too_unstable(0.51, False)
    assert source_response_is_too_unstable(0.0, True)
    assert not source_response_is_too_unstable(0.5, False)


def test_unresolved_map_regime_lacks_practical_motivation_is_a_direct_pass_through() -> None:
    assert unresolved_map_regime_lacks_practical_motivation(True)
    assert not unresolved_map_regime_lacks_practical_motivation(False)
