from __future__ import annotations

from collections.abc import Mapping

from fedorbit.config.loading import active_config
from fedorbit.types import Fraction, RelativeGain

EXACT_SPARSE_ROADMAP_COMPLEXITY_COUNT = 5
GENERIC_QAP_ROADMAP_COMPLEXITY_COUNT = 5


def exactness_failure(
    any_reproducible_exact_sparse_error_outside_exactness_bound: bool,
    any_invalid_correspondence_certificate: bool,
) -> bool:
    return (
        any_reproducible_exact_sparse_error_outside_exactness_bound
        or any_invalid_correspondence_certificate
    )


def rectangularization_is_sufficient(
    valid_real_packet_fraction_below_coupling_materiality: float,
    any_primary_pair_satisfies_coupling_mechanism_criteria: bool,
) -> bool:
    rule = active_config().scientific.simplification_rules.rectangularization_is_sufficient
    return (
        valid_real_packet_fraction_below_coupling_materiality
        >= rule.valid_real_packet_fraction_below_coupling_materiality_minimum
        and not any_primary_pair_satisfies_coupling_mechanism_criteria
    )


def theory_classification_failure(
    designed_classification_disagrees_with_exhaustive_truth: bool,
    map_bound_disagrees_with_exhaustive_truth: bool,
) -> bool:
    return (
        designed_classification_disagrees_with_exhaustive_truth
        or map_bound_disagrees_with_exhaustive_truth
    )


def generic_qap_dominates(
    qap_exact_on_every_required_intended_exact_case: bool,
    median_runtime_ratio_to_exact_sparse: float,
    p95_runtime_ratio_to_exact_sparse: float,
    peak_memory_ratio_to_exact_sparse: float,
    generic_qap_has_more_timeouts_than_exact_sparse: bool,
) -> bool:
    rule = active_config().scientific.simplification_rules.generic_qap_dominates
    return (
        qap_exact_on_every_required_intended_exact_case
        and median_runtime_ratio_to_exact_sparse
        <= rule.median_runtime_ratio_to_exact_sparse_maximum
        and p95_runtime_ratio_to_exact_sparse <= rule.p95_runtime_ratio_to_exact_sparse_maximum
        and peak_memory_ratio_to_exact_sparse <= rule.peak_memory_ratio_to_exact_sparse_maximum
        and not generic_qap_has_more_timeouts_than_exact_sparse
        and GENERIC_QAP_ROADMAP_COMPLEXITY_COUNT <= EXACT_SPARSE_ROADMAP_COMPLEXITY_COUNT
    )


def sparse_support_is_operationally_irrelevant(
    dense_minus_support_3_test_relative_macro_ce_gain: RelativeGain,
    fraction_of_valid_primary_units_with_that_gain: Fraction,
    useful_transfer_materiality_failure_by_support: Mapping[int, bool],
) -> bool:
    rule = (
        active_config().scientific.simplification_rules.sparse_support_is_operationally_irrelevant
    )
    every_configured_support_fails = all(
        useful_transfer_materiality_failure_by_support.get(int(support), False)
        for support in rule.sparse_supports_that_must_fail_useful_materiality
    )
    return (
        dense_minus_support_3_test_relative_macro_ce_gain
        >= rule.dense_gain_advantage_over_support_3_minimum
        and fraction_of_valid_primary_units_with_that_gain
        >= rule.valid_primary_unit_fraction_minimum
        and every_configured_support_fails
    )


def local_sir_is_sufficient(
    external_source_value_evidence_rule_failed: bool,
    primary_pairs_local_sir_equivalent_or_superior: int,
    no_remaining_valid_primary_pair_shows_material_advantage_over_local_sir: bool,
) -> bool:
    criteria = active_config().scientific.evaluation_criteria.external_source_value_vs_local_sir
    threshold = criteria.successful_primary_pairs_required
    return (
        external_source_value_evidence_rule_failed
        and primary_pairs_local_sir_equivalent_or_superior >= threshold
        and no_remaining_valid_primary_pair_shows_material_advantage_over_local_sir
    )


def point_matching_is_sufficient(
    per_eligible_primary_pair_harmful_rate_worsening_vs_fedorbit: tuple[float, ...],
    per_eligible_primary_pair_mean_gain_advantage_over_fedorbit: tuple[float, ...],
) -> bool:
    if len(per_eligible_primary_pair_harmful_rate_worsening_vs_fedorbit) == 0:
        return False
    if len(per_eligible_primary_pair_harmful_rate_worsening_vs_fedorbit) != len(
        per_eligible_primary_pair_mean_gain_advantage_over_fedorbit
    ):
        raise ValueError("primary pair measurement tuples must have matching length")
    rule = active_config().scientific.simplification_rules.point_matching_is_sufficient
    return all(
        harmful_rate_worsening <= rule.harmful_rate_worsening_maximum
        and mean_gain_advantage >= rule.utility_advantage_over_fedorbit_minimum
        for harmful_rate_worsening, mean_gain_advantage in zip(
            per_eligible_primary_pair_harmful_rate_worsening_vs_fedorbit,
            per_eligible_primary_pair_mean_gain_advantage_over_fedorbit,
            strict=True,
        )
    )


def coupling_destruction_retains_gain(mechanism_retention_condition_present: bool) -> bool:
    return mechanism_retention_condition_present


def strict_interface_removes_gain(
    per_primary_pair_fedorbit_mean_gain: tuple[float, ...],
    per_primary_pair_fedorbit_bca_upper_bound: tuple[float, ...],
    per_primary_pair_exact_map_oracle_relaxed_diagnostic_succeeds: tuple[bool, ...],
) -> bool:
    if not (
        len(per_primary_pair_fedorbit_mean_gain)
        == len(per_primary_pair_fedorbit_bca_upper_bound)
        == len(per_primary_pair_exact_map_oracle_relaxed_diagnostic_succeeds)
    ):
        raise ValueError("primary pair measurement tuples must have matching length")
    rule = active_config().scientific.simplification_rules.strict_interface_removes_gain
    qualifying_pairs = sum(
        1
        for mean_gain, bca_upper_bound, oracle_succeeds in zip(
            per_primary_pair_fedorbit_mean_gain,
            per_primary_pair_fedorbit_bca_upper_bound,
            per_primary_pair_exact_map_oracle_relaxed_diagnostic_succeeds,
            strict=True,
        )
        if mean_gain <= 0.0 and bca_upper_bound < rule.bca_upper_bound_maximum and oracle_succeeds
    )
    return qualifying_pairs >= rule.primary_pair_majority_required


def confirmation_has_no_safety_value(
    meets_absolute_harm_reduction_requirement: bool,
    meets_relative_harm_reduction_requirement: bool,
    confirmation_gap_exceeds_ceiling: bool,
) -> bool:
    return (
        not meets_absolute_harm_reduction_requirement
        and not meets_relative_harm_reduction_requirement
    ) or confirmation_gap_exceeds_ceiling


def source_response_is_too_unstable(
    principal_source_packet_failure_fraction: Fraction,
    any_required_primary_source_domain_has_no_eligible_pilot_setting: bool,
) -> bool:
    rule = active_config().scientific.simplification_rules.source_response_is_too_unstable
    return (
        principal_source_packet_failure_fraction
        > rule.principal_source_packet_failure_fraction_strictly_greater_than
        or any_required_primary_source_domain_has_no_eligible_pilot_setting
    )


def unresolved_map_regime_lacks_practical_motivation(
    applicability_audit_shows_exact_fine_map_trivial_under_intended_resources: bool,
) -> bool:
    return applicability_audit_shows_exact_fine_map_trivial_under_intended_resources
