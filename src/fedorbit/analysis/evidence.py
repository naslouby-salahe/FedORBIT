from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

from fedorbit.analysis.kill_rules import (
    confirmation_has_no_safety_value,
    coupling_destruction_retains_gain,
    exactness_failure,
    local_sir_is_sufficient,
    sparse_support_is_operationally_irrelevant,
)
from fedorbit.config.loading import active_config
from fedorbit.types import EvidenceProposition, EvidenceStatus, Fraction, RelativeGain


@dataclass(frozen=True, slots=True)
class ExactSparseSeparatorExactnessInputs:
    required_evidence_is_complete: bool
    any_reproducible_exact_sparse_error_outside_exactness_bound: bool
    any_invalid_correspondence_certificate: bool


def classify_exact_sparse_separator_exactness(
    inputs: ExactSparseSeparatorExactnessInputs,
) -> EvidenceStatus:
    if not inputs.required_evidence_is_complete:
        return EvidenceStatus.NOT_TESTED
    if exactness_failure(
        inputs.any_reproducible_exact_sparse_error_outside_exactness_bound,
        inputs.any_invalid_correspondence_certificate,
    ):
        return EvidenceStatus.NOT_SUPPORTED
    return EvidenceStatus.SUPPORTED


@dataclass(frozen=True, slots=True)
class JointCorrespondenceAvoidsRectangularPessimismInputs:
    required_evidence_is_complete: bool
    designed_synthetic_theorem_mechanism_passes: bool
    real_packet_materiality_criterion_reached: bool
    mechanism_retention_condition_present: bool


def classify_joint_correspondence_avoids_rectangular_pessimism(
    inputs: JointCorrespondenceAvoidsRectangularPessimismInputs,
) -> EvidenceStatus:
    if not inputs.required_evidence_is_complete:
        return EvidenceStatus.NOT_TESTED
    if (
        coupling_destruction_retains_gain(inputs.mechanism_retention_condition_present)
        or not inputs.designed_synthetic_theorem_mechanism_passes
    ):
        return EvidenceStatus.NOT_SUPPORTED
    if inputs.real_packet_materiality_criterion_reached:
        return EvidenceStatus.SUPPORTED
    return EvidenceStatus.MECHANISM_ONLY


@dataclass(frozen=True, slots=True)
class ActionCertificationWithoutFineMapIdentificationInputs:
    required_evidence_is_complete: bool
    common_action_world_construction_succeeds: bool
    robust_compromise_world_construction_succeeds: bool
    exact_map_recovery_is_always_required: bool
    orbit_radius_bound_is_valid: bool


def classify_action_certification_without_fine_map_identification(
    inputs: ActionCertificationWithoutFineMapIdentificationInputs,
) -> EvidenceStatus:
    if not inputs.required_evidence_is_complete:
        return EvidenceStatus.NOT_TESTED
    if inputs.exact_map_recovery_is_always_required or not inputs.orbit_radius_bound_is_valid:
        return EvidenceStatus.NOT_SUPPORTED
    if (
        inputs.common_action_world_construction_succeeds
        and inputs.robust_compromise_world_construction_succeeds
    ):
        return EvidenceStatus.SUPPORTED
    if (
        inputs.common_action_world_construction_succeeds
        or inputs.robust_compromise_world_construction_succeeds
    ):
        return EvidenceStatus.PARTIALLY_SUPPORTED
    return EvidenceStatus.NULL_RESULT


@dataclass(frozen=True, slots=True)
class StrictCrossTelemetryTransferUtilityInputs:
    required_evidence_is_complete: bool
    primary_pairs_satisfying_full_positive_criteria: int
    any_primary_pair_materially_harmful: bool
    strict_resource_validation_passes: bool
    one_pair_removed_pre_outcome_for_eligibility_reasons: bool
    remaining_analyzable_primary_pairs_after_exclusion: int
    remaining_pairs_individually_satisfy_positive_criteria: bool
    equal_pair_mean_meets_materiality: bool


def classify_strict_cross_telemetry_transfer_utility(
    inputs: StrictCrossTelemetryTransferUtilityInputs,
) -> EvidenceStatus:
    if not inputs.required_evidence_is_complete:
        return EvidenceStatus.NOT_TESTED
    if inputs.any_primary_pair_materially_harmful or not inputs.strict_resource_validation_passes:
        return EvidenceStatus.NOT_SUPPORTED
    if (
        inputs.one_pair_removed_pre_outcome_for_eligibility_reasons
        and inputs.remaining_analyzable_primary_pairs_after_exclusion == 3
        and inputs.remaining_pairs_individually_satisfy_positive_criteria
        and inputs.equal_pair_mean_meets_materiality
    ):
        return EvidenceStatus.CONDITIONAL
    required = (
        active_config().scientific.evaluation_criteria.strict_cross_telemetry_utility
    ).successful_primary_pairs_required
    if inputs.primary_pairs_satisfying_full_positive_criteria >= required:
        return EvidenceStatus.SUPPORTED
    if inputs.primary_pairs_satisfying_full_positive_criteria >= 1:
        return EvidenceStatus.PARTIALLY_SUPPORTED
    return EvidenceStatus.NULL_RESULT


@dataclass(frozen=True, slots=True)
class ValueOfExternalProceduralEvidenceInputs:
    required_evidence_is_complete: bool
    primary_pairs_satisfying_full_positive_criteria: int
    any_primary_pair_materially_harmful: bool
    strict_resource_validation_passes: bool
    one_pair_removed_pre_outcome_for_eligibility_reasons: bool
    remaining_analyzable_primary_pairs_after_exclusion: int
    remaining_pairs_individually_satisfy_positive_criteria: bool
    equal_pair_mean_meets_materiality: bool
    external_source_value_evidence_rule_failed: bool
    primary_pairs_local_sir_equivalent_or_superior: int
    no_remaining_valid_primary_pair_shows_material_advantage_over_local_sir: bool


def classify_value_of_external_procedural_evidence(
    inputs: ValueOfExternalProceduralEvidenceInputs,
) -> EvidenceStatus:
    if not inputs.required_evidence_is_complete:
        return EvidenceStatus.NOT_TESTED
    if local_sir_is_sufficient(
        inputs.external_source_value_evidence_rule_failed,
        inputs.primary_pairs_local_sir_equivalent_or_superior,
        inputs.no_remaining_valid_primary_pair_shows_material_advantage_over_local_sir,
    ):
        return EvidenceStatus.NOT_SUPPORTED
    return classify_strict_cross_telemetry_transfer_utility(
        StrictCrossTelemetryTransferUtilityInputs(
            required_evidence_is_complete=inputs.required_evidence_is_complete,
            primary_pairs_satisfying_full_positive_criteria=(
                inputs.primary_pairs_satisfying_full_positive_criteria
            ),
            any_primary_pair_materially_harmful=inputs.any_primary_pair_materially_harmful,
            strict_resource_validation_passes=inputs.strict_resource_validation_passes,
            one_pair_removed_pre_outcome_for_eligibility_reasons=(
                inputs.one_pair_removed_pre_outcome_for_eligibility_reasons
            ),
            remaining_analyzable_primary_pairs_after_exclusion=(
                inputs.remaining_analyzable_primary_pairs_after_exclusion
            ),
            remaining_pairs_individually_satisfy_positive_criteria=(
                inputs.remaining_pairs_individually_satisfy_positive_criteria
            ),
            equal_pair_mean_meets_materiality=inputs.equal_pair_mean_meets_materiality,
        )
    )


@dataclass(frozen=True, slots=True)
class OperationalRelevanceOfSparseSupportInputs:
    required_evidence_is_complete: bool
    dense_minus_support_3_test_relative_macro_ce_gain: RelativeGain
    fraction_of_valid_primary_units_with_that_gain: Fraction
    useful_transfer_materiality_failure_by_support: Mapping[int, bool]
    full_sparse_operational_rule_passes: bool


def classify_operational_relevance_of_sparse_support(
    inputs: OperationalRelevanceOfSparseSupportInputs,
) -> EvidenceStatus:
    if not inputs.required_evidence_is_complete:
        return EvidenceStatus.NOT_TESTED
    if sparse_support_is_operationally_irrelevant(
        inputs.dense_minus_support_3_test_relative_macro_ce_gain,
        inputs.fraction_of_valid_primary_units_with_that_gain,
        inputs.useful_transfer_materiality_failure_by_support,
    ):
        return EvidenceStatus.NOT_SUPPORTED
    if inputs.full_sparse_operational_rule_passes:
        return EvidenceStatus.SUPPORTED
    every_support_fails_useful_materiality = all(
        inputs.useful_transfer_materiality_failure_by_support.values()
    )
    if every_support_fails_useful_materiality:
        return EvidenceStatus.NULL_RESULT
    return EvidenceStatus.PARTIALLY_SUPPORTED


@dataclass(frozen=True, slots=True)
class TargetConfirmationSafetyInputs:
    required_evidence_is_complete: bool
    meets_absolute_harm_reduction_requirement: bool
    meets_relative_harm_reduction_requirement: bool
    confirmation_gap_exceeds_ceiling: bool
    primary_pairs_meeting_either_harm_reduction_criterion: int


def classify_target_confirmation_safety(
    inputs: TargetConfirmationSafetyInputs,
) -> EvidenceStatus:
    if not inputs.required_evidence_is_complete:
        return EvidenceStatus.NOT_TESTED
    if confirmation_has_no_safety_value(
        inputs.meets_absolute_harm_reduction_requirement,
        inputs.meets_relative_harm_reduction_requirement,
        inputs.confirmation_gap_exceeds_ceiling,
    ):
        return EvidenceStatus.NOT_SUPPORTED
    required = active_config().scientific.evaluation_criteria.confirmation_safety
    if (
        inputs.primary_pairs_meeting_either_harm_reduction_criterion
        >= required.qualifying_primary_pairs_required
    ):
        return EvidenceStatus.SUPPORTED
    if inputs.primary_pairs_meeting_either_harm_reduction_criterion >= 1:
        return EvidenceStatus.PARTIALLY_SUPPORTED
    return EvidenceStatus.NULL_RESULT


@dataclass(frozen=True, slots=True)
class SparseSolverWorkStructureAgreementInputs:
    required_evidence_is_complete: bool
    every_exact_counter_formula_matches: bool
    exactness_is_retained: bool
    every_eligible_runtime_trend_stratum_is_positive_with_sufficient_evidence: bool
    approximation_was_required: bool


def classify_sparse_solver_work_structure_agreement(
    inputs: SparseSolverWorkStructureAgreementInputs,
) -> EvidenceStatus:
    if not inputs.required_evidence_is_complete:
        return EvidenceStatus.NOT_TESTED
    if inputs.approximation_was_required or not inputs.every_exact_counter_formula_matches:
        return EvidenceStatus.NOT_SUPPORTED
    if (
        inputs.exactness_is_retained
        and inputs.every_eligible_runtime_trend_stratum_is_positive_with_sufficient_evidence
    ):
        return EvidenceStatus.SUPPORTED
    if inputs.exactness_is_retained:
        return EvidenceStatus.PARTIALLY_SUPPORTED
    return EvidenceStatus.NOT_SUPPORTED


@dataclass(frozen=True, slots=True)
class EvidenceClassificationInputs:
    exact_sparse_separator_exactness: ExactSparseSeparatorExactnessInputs
    joint_correspondence_avoids_rectangular_pessimism: (
        JointCorrespondenceAvoidsRectangularPessimismInputs
    )
    action_certification_without_fine_map_identification: (
        ActionCertificationWithoutFineMapIdentificationInputs
    )
    strict_cross_telemetry_transfer_utility: StrictCrossTelemetryTransferUtilityInputs
    value_of_external_procedural_evidence: ValueOfExternalProceduralEvidenceInputs
    operational_relevance_of_sparse_support: OperationalRelevanceOfSparseSupportInputs
    target_confirmation_safety: TargetConfirmationSafetyInputs
    sparse_solver_work_structure_agreement: SparseSolverWorkStructureAgreementInputs


def classify_all_propositions(
    inputs: EvidenceClassificationInputs,
) -> OrderedDict[EvidenceProposition, EvidenceStatus]:
    return OrderedDict(
        (
            (
                EvidenceProposition.EXACT_SPARSE_SEPARATOR_EXACTNESS,
                classify_exact_sparse_separator_exactness(inputs.exact_sparse_separator_exactness),
            ),
            (
                EvidenceProposition.JOINT_CORRESPONDENCE_AVOIDS_RECTANGULAR_PESSIMISM,
                classify_joint_correspondence_avoids_rectangular_pessimism(
                    inputs.joint_correspondence_avoids_rectangular_pessimism
                ),
            ),
            (
                EvidenceProposition.ACTION_CERTIFICATION_WITHOUT_FINE_MAP_IDENTIFICATION,
                classify_action_certification_without_fine_map_identification(
                    inputs.action_certification_without_fine_map_identification
                ),
            ),
            (
                EvidenceProposition.STRICT_CROSS_TELEMETRY_TRANSFER_UTILITY,
                classify_strict_cross_telemetry_transfer_utility(
                    inputs.strict_cross_telemetry_transfer_utility
                ),
            ),
            (
                EvidenceProposition.VALUE_OF_EXTERNAL_PROCEDURAL_EVIDENCE,
                classify_value_of_external_procedural_evidence(
                    inputs.value_of_external_procedural_evidence
                ),
            ),
            (
                EvidenceProposition.OPERATIONAL_RELEVANCE_OF_SPARSE_SUPPORT,
                classify_operational_relevance_of_sparse_support(
                    inputs.operational_relevance_of_sparse_support
                ),
            ),
            (
                EvidenceProposition.TARGET_CONFIRMATION_SAFETY,
                classify_target_confirmation_safety(inputs.target_confirmation_safety),
            ),
            (
                EvidenceProposition.SPARSE_SOLVER_WORK_STRUCTURE_AGREEMENT,
                classify_sparse_solver_work_structure_agreement(
                    inputs.sparse_solver_work_structure_agreement
                ),
            ),
        )
    )
