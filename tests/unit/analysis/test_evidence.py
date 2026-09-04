from __future__ import annotations

from fedorbit.analysis.evidence import (
    ActionCertificationWithoutFineMapIdentificationInputs,
    EvidenceClassificationInputs,
    ExactSparseSeparatorExactnessInputs,
    JointCorrespondenceAvoidsRectangularPessimismInputs,
    OperationalRelevanceOfSparseSupportInputs,
    SparseSolverWorkStructureAgreementInputs,
    StrictCrossTelemetryTransferUtilityInputs,
    TargetConfirmationSafetyInputs,
    ValueOfExternalProceduralEvidenceInputs,
    classify_action_certification_without_fine_map_identification,
    classify_all_propositions,
    classify_exact_sparse_separator_exactness,
    classify_joint_correspondence_avoids_rectangular_pessimism,
    classify_operational_relevance_of_sparse_support,
    classify_sparse_solver_work_structure_agreement,
    classify_strict_cross_telemetry_transfer_utility,
    classify_target_confirmation_safety,
    classify_value_of_external_procedural_evidence,
)
from fedorbit.types import EvidenceProposition, EvidenceStatus


def test_exact_sparse_separator_exactness_states() -> None:
    assert (
        classify_exact_sparse_separator_exactness(
            ExactSparseSeparatorExactnessInputs(False, False, False)
        )
        == EvidenceStatus.NOT_TESTED
    )
    assert (
        classify_exact_sparse_separator_exactness(
            ExactSparseSeparatorExactnessInputs(True, True, False)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_exact_sparse_separator_exactness(
            ExactSparseSeparatorExactnessInputs(True, False, False)
        )
        == EvidenceStatus.SUPPORTED
    )


def test_joint_correspondence_avoids_rectangular_pessimism_states() -> None:
    assert (
        classify_joint_correspondence_avoids_rectangular_pessimism(
            JointCorrespondenceAvoidsRectangularPessimismInputs(False, True, True, False)
        )
        == EvidenceStatus.NOT_TESTED
    )
    assert (
        classify_joint_correspondence_avoids_rectangular_pessimism(
            JointCorrespondenceAvoidsRectangularPessimismInputs(True, False, True, False)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_joint_correspondence_avoids_rectangular_pessimism(
            JointCorrespondenceAvoidsRectangularPessimismInputs(True, True, False, True)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_joint_correspondence_avoids_rectangular_pessimism(
            JointCorrespondenceAvoidsRectangularPessimismInputs(True, True, True, False)
        )
        == EvidenceStatus.SUPPORTED
    )
    assert (
        classify_joint_correspondence_avoids_rectangular_pessimism(
            JointCorrespondenceAvoidsRectangularPessimismInputs(True, True, False, False)
        )
        == EvidenceStatus.MECHANISM_ONLY
    )


def test_action_certification_without_fine_map_identification_states() -> None:
    assert (
        classify_action_certification_without_fine_map_identification(
            ActionCertificationWithoutFineMapIdentificationInputs(False, True, True, False, True)
        )
        == EvidenceStatus.NOT_TESTED
    )
    assert (
        classify_action_certification_without_fine_map_identification(
            ActionCertificationWithoutFineMapIdentificationInputs(True, True, True, True, True)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_action_certification_without_fine_map_identification(
            ActionCertificationWithoutFineMapIdentificationInputs(True, True, True, False, False)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_action_certification_without_fine_map_identification(
            ActionCertificationWithoutFineMapIdentificationInputs(True, True, True, False, True)
        )
        == EvidenceStatus.SUPPORTED
    )
    assert (
        classify_action_certification_without_fine_map_identification(
            ActionCertificationWithoutFineMapIdentificationInputs(True, True, False, False, True)
        )
        == EvidenceStatus.PARTIALLY_SUPPORTED
    )
    assert (
        classify_action_certification_without_fine_map_identification(
            ActionCertificationWithoutFineMapIdentificationInputs(True, False, False, False, True)
        )
        == EvidenceStatus.NULL_RESULT
    )


def test_strict_cross_telemetry_transfer_utility_states() -> None:
    assert (
        classify_strict_cross_telemetry_transfer_utility(
            StrictCrossTelemetryTransferUtilityInputs(False, 0, False, True, False, 0, False, False)
        )
        == EvidenceStatus.NOT_TESTED
    )
    assert (
        classify_strict_cross_telemetry_transfer_utility(
            StrictCrossTelemetryTransferUtilityInputs(True, 4, True, True, False, 0, False, False)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_strict_cross_telemetry_transfer_utility(
            StrictCrossTelemetryTransferUtilityInputs(True, 2, False, False, False, 0, False, False)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_strict_cross_telemetry_transfer_utility(
            StrictCrossTelemetryTransferUtilityInputs(True, 2, False, True, True, 3, True, True)
        )
        == EvidenceStatus.CONDITIONAL
    )
    assert (
        classify_strict_cross_telemetry_transfer_utility(
            StrictCrossTelemetryTransferUtilityInputs(True, 4, False, True, False, 0, False, False)
        )
        == EvidenceStatus.SUPPORTED
    )
    assert (
        classify_strict_cross_telemetry_transfer_utility(
            StrictCrossTelemetryTransferUtilityInputs(True, 2, False, True, False, 0, False, False)
        )
        == EvidenceStatus.PARTIALLY_SUPPORTED
    )
    assert (
        classify_strict_cross_telemetry_transfer_utility(
            StrictCrossTelemetryTransferUtilityInputs(True, 0, False, True, False, 0, False, False)
        )
        == EvidenceStatus.NULL_RESULT
    )


def test_value_of_external_procedural_evidence_states() -> None:
    not_supported_by_local_sir = ValueOfExternalProceduralEvidenceInputs(
        required_evidence_is_complete=True,
        primary_pairs_satisfying_full_positive_criteria=4,
        any_primary_pair_materially_harmful=False,
        strict_resource_validation_passes=True,
        one_pair_removed_pre_outcome_for_eligibility_reasons=False,
        remaining_analyzable_primary_pairs_after_exclusion=0,
        remaining_pairs_individually_satisfy_positive_criteria=False,
        equal_pair_mean_meets_materiality=False,
        external_source_value_evidence_rule_failed=True,
        primary_pairs_local_sir_equivalent_or_superior=4,
        no_remaining_valid_primary_pair_shows_material_advantage_over_local_sir=True,
    )
    assert (
        classify_value_of_external_procedural_evidence(not_supported_by_local_sir)
        == EvidenceStatus.NOT_SUPPORTED
    )
    supported = ValueOfExternalProceduralEvidenceInputs(
        required_evidence_is_complete=True,
        primary_pairs_satisfying_full_positive_criteria=4,
        any_primary_pair_materially_harmful=False,
        strict_resource_validation_passes=True,
        one_pair_removed_pre_outcome_for_eligibility_reasons=False,
        remaining_analyzable_primary_pairs_after_exclusion=0,
        remaining_pairs_individually_satisfy_positive_criteria=False,
        equal_pair_mean_meets_materiality=False,
        external_source_value_evidence_rule_failed=False,
        primary_pairs_local_sir_equivalent_or_superior=0,
        no_remaining_valid_primary_pair_shows_material_advantage_over_local_sir=False,
    )
    assert classify_value_of_external_procedural_evidence(supported) == EvidenceStatus.SUPPORTED


def test_operational_relevance_of_sparse_support_states() -> None:
    assert (
        classify_operational_relevance_of_sparse_support(
            OperationalRelevanceOfSparseSupportInputs(
                False, 0.0, 0.0, {1: False, 2: False, 3: False}, False
            )
        )
        == EvidenceStatus.NOT_TESTED
    )
    assert (
        classify_operational_relevance_of_sparse_support(
            OperationalRelevanceOfSparseSupportInputs(
                True, 0.02, 0.75, {1: True, 2: True, 3: True}, False
            )
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_operational_relevance_of_sparse_support(
            OperationalRelevanceOfSparseSupportInputs(
                True, 0.0, 0.0, {1: False, 2: False, 3: False}, True
            )
        )
        == EvidenceStatus.SUPPORTED
    )
    assert (
        classify_operational_relevance_of_sparse_support(
            OperationalRelevanceOfSparseSupportInputs(
                True, 0.0, 0.0, {1: True, 2: True, 3: True}, False
            )
        )
        == EvidenceStatus.NULL_RESULT
    )
    assert (
        classify_operational_relevance_of_sparse_support(
            OperationalRelevanceOfSparseSupportInputs(
                True, 0.0, 0.0, {1: False, 2: True, 3: True}, False
            )
        )
        == EvidenceStatus.PARTIALLY_SUPPORTED
    )


def test_target_confirmation_safety_states() -> None:
    assert (
        classify_target_confirmation_safety(
            TargetConfirmationSafetyInputs(False, False, False, False, 0)
        )
        == EvidenceStatus.NOT_TESTED
    )
    assert (
        classify_target_confirmation_safety(
            TargetConfirmationSafetyInputs(True, False, False, False, 0)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_target_confirmation_safety(
            TargetConfirmationSafetyInputs(True, False, False, True, 4)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_target_confirmation_safety(
            TargetConfirmationSafetyInputs(True, True, False, False, 4)
        )
        == EvidenceStatus.SUPPORTED
    )
    assert (
        classify_target_confirmation_safety(
            TargetConfirmationSafetyInputs(True, True, False, False, 1)
        )
        == EvidenceStatus.PARTIALLY_SUPPORTED
    )


def test_sparse_solver_work_structure_agreement_states() -> None:
    assert (
        classify_sparse_solver_work_structure_agreement(
            SparseSolverWorkStructureAgreementInputs(False, True, True, True, False)
        )
        == EvidenceStatus.NOT_TESTED
    )
    assert (
        classify_sparse_solver_work_structure_agreement(
            SparseSolverWorkStructureAgreementInputs(True, True, True, True, True)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_sparse_solver_work_structure_agreement(
            SparseSolverWorkStructureAgreementInputs(True, False, True, True, False)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )
    assert (
        classify_sparse_solver_work_structure_agreement(
            SparseSolverWorkStructureAgreementInputs(True, True, True, True, False)
        )
        == EvidenceStatus.SUPPORTED
    )
    assert (
        classify_sparse_solver_work_structure_agreement(
            SparseSolverWorkStructureAgreementInputs(True, True, True, False, False)
        )
        == EvidenceStatus.PARTIALLY_SUPPORTED
    )
    assert (
        classify_sparse_solver_work_structure_agreement(
            SparseSolverWorkStructureAgreementInputs(True, True, False, False, False)
        )
        == EvidenceStatus.NOT_SUPPORTED
    )


def test_classify_all_propositions_returns_every_registered_proposition_not_tested() -> None:
    statuses = classify_all_propositions(_all_not_tested_inputs())
    assert set(statuses.keys()) == set(EvidenceProposition)
    assert set(statuses.values()) == {EvidenceStatus.NOT_TESTED}


def _all_not_tested_inputs() -> EvidenceClassificationInputs:
    return EvidenceClassificationInputs(
        exact_sparse_separator_exactness=ExactSparseSeparatorExactnessInputs(False, False, False),
        joint_correspondence_avoids_rectangular_pessimism=(
            JointCorrespondenceAvoidsRectangularPessimismInputs(False, False, False, False)
        ),
        action_certification_without_fine_map_identification=(
            ActionCertificationWithoutFineMapIdentificationInputs(False, False, False, False, False)
        ),
        strict_cross_telemetry_transfer_utility=StrictCrossTelemetryTransferUtilityInputs(
            False, 0, False, False, False, 0, False, False
        ),
        value_of_external_procedural_evidence=ValueOfExternalProceduralEvidenceInputs(
            required_evidence_is_complete=False,
            primary_pairs_satisfying_full_positive_criteria=0,
            any_primary_pair_materially_harmful=False,
            strict_resource_validation_passes=False,
            one_pair_removed_pre_outcome_for_eligibility_reasons=False,
            remaining_analyzable_primary_pairs_after_exclusion=0,
            remaining_pairs_individually_satisfy_positive_criteria=False,
            equal_pair_mean_meets_materiality=False,
            external_source_value_evidence_rule_failed=False,
            primary_pairs_local_sir_equivalent_or_superior=0,
            no_remaining_valid_primary_pair_shows_material_advantage_over_local_sir=False,
        ),
        operational_relevance_of_sparse_support=OperationalRelevanceOfSparseSupportInputs(
            False, 0.0, 0.0, {1: False, 2: False, 3: False}, False
        ),
        target_confirmation_safety=TargetConfirmationSafetyInputs(False, False, False, False, 0),
        sparse_solver_work_structure_agreement=SparseSolverWorkStructureAgreementInputs(
            False, False, False, False, False
        ),
    )
