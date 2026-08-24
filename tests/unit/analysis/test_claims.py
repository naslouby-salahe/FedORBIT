from __future__ import annotations

import pytest

from fedorbit.analysis.claims import (
    ConfirmationPairEvidence,
    CouplingMechanismEvidence,
    MechanismRetentionEvidence,
    SparsePairGainEvidence,
    SparseUnitEvidence,
    evaluate_confirmation_safety,
    evaluate_coupling_mechanism,
    evaluate_external_source_value,
    evaluate_sparse_operational_relevance,
    evaluate_transfer_style_criteria,
)
from fedorbit.analysis.comparisons import PairContrastEvidence, PairContrastEvidenceSet
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig


@pytest.fixture
def config() -> FedorbitConfig:
    return load_fedorbit_config()


def _evidence(
    pair: str,
    mean_gain: float,
    holm_p: float = 0.01,
    bca_lower: float = 0.005,
    valid_seed_count: int = 10,
    strict_valid: bool = True,
) -> PairContrastEvidence:
    return PairContrastEvidence(
        pair,
        mean_gain,
        holm_p,
        bca_lower,
        strict_valid,
        valid_seed_count,
    )


def _set(*entries: PairContrastEvidence) -> PairContrastEvidenceSet:
    return PairContrastEvidenceSet(entries)


def test_full_scope_supported_when_three_pairs_pass(config: FedorbitConfig) -> None:
    decision = evaluate_transfer_style_criteria(
        config,
        _set(
            _evidence("p1", 0.05),
            _evidence("p2", 0.04),
            _evidence("p3", 0.06),
            _evidence("p4", -0.001),
        ),
        removed_before_outcome_inspection=False,
    )
    assert decision.supported
    assert not decision.conditional
    assert set(decision.successful_pairs) == {"p1", "p2", "p3"}


def test_harmful_pair_makes_claim_not_supported(config: FedorbitConfig) -> None:
    decision = evaluate_transfer_style_criteria(
        config,
        _set(
            _evidence("p1", 0.05),
            _evidence("p2", 0.04),
            _evidence("p3", -0.02),
            _evidence("p4", 0.03),
        ),
        removed_before_outcome_inspection=False,
    )
    assert decision.not_supported
    assert "p3" in decision.harmful_pairs


def test_reduced_scope_is_conditional_only_for_pre_outcome_removal(config: FedorbitConfig) -> None:
    decision = evaluate_transfer_style_criteria(
        config,
        _set(
            _evidence("p1", 0.05, valid_seed_count=0),
            _evidence("p2", 0.04),
            _evidence("p3", 0.06),
            _evidence("p4", 0.03),
        ),
        removed_before_outcome_inspection=True,
    )
    assert decision.conditional
    assert not decision.supported


def test_pair_below_minimum_seeds_cannot_be_positive(config: FedorbitConfig) -> None:
    decision = evaluate_transfer_style_criteria(
        config,
        _set(
            _evidence("p1", 0.05, valid_seed_count=7),
            _evidence("p2", 0.04),
            _evidence("p3", 0.06),
            _evidence("p4", 0.03),
        ),
        removed_before_outcome_inspection=False,
    )
    assert decision.not_supported


def test_two_positive_pairs_is_partial_support(config: FedorbitConfig) -> None:
    decision = evaluate_transfer_style_criteria(
        config,
        _set(
            _evidence("p1", 0.05),
            _evidence("p2", 0.04),
            _evidence("p3", 0.001),
            _evidence("p4", 0.002),
        ),
        removed_before_outcome_inspection=False,
    )
    assert decision.partially_supported
    assert len(decision.successful_pairs) == 2


def test_no_material_benefit_without_harm_is_null_result(config: FedorbitConfig) -> None:
    decision = evaluate_transfer_style_criteria(
        config,
        _set(
            _evidence("p1", 0.001),
            _evidence("p2", 0.0005),
            _evidence("p3", 0.002),
            _evidence("p4", -0.001),
        ),
        removed_before_outcome_inspection=False,
    )
    assert decision.null_result
    assert not decision.not_supported


def test_strict_resource_failure_blocks_claim_support(config: FedorbitConfig) -> None:
    decision = evaluate_transfer_style_criteria(
        config,
        _set(
            _evidence("p1", 0.05),
            _evidence("p2", 0.04),
            _evidence("p3", 0.06, strict_valid=False),
            _evidence("p4", 0.03),
        ),
        removed_before_outcome_inspection=False,
    )
    assert decision.not_supported
    assert not decision.supported


def test_external_source_value_uses_its_own_registered_evaluator(config: FedorbitConfig) -> None:
    decision = evaluate_external_source_value(
        config,
        _set(
            _evidence("p1", 0.05),
            _evidence("p2", 0.04),
            _evidence("p3", 0.06),
            _evidence("p4", -0.001),
        ),
        removed_before_outcome_inspection=False,
    )
    assert decision.supported


def test_coupling_claim_requires_no_mechanism_retention(config: FedorbitConfig) -> None:
    pair_gaps = _set(
        _evidence("p1", 0.02, bca_lower=0.0),
        _evidence("p2", 0.02, bca_lower=0.0),
        _evidence("p3", 0.001, bca_lower=0.0),
        _evidence("p4", 0.001, bca_lower=0.0),
    )
    clean = CouplingMechanismEvidence(
        theorem_classification_accuracy=1.0,
        real_packet_gaps=(0.02, 0.01, 0.006, 0.0),
        pair_gap_evidence=pair_gaps,
        retention_evidence=(
            MechanismRetentionEvidence("p1", 0.04, 0.01, 0.2, 10),
            MechanismRetentionEvidence("p2", 0.04, 0.01, 0.2, 10),
        ),
    )
    assert evaluate_coupling_mechanism(config, clean).supported
    retained = clean.__class__(
        clean.theorem_classification_accuracy,
        clean.real_packet_gaps,
        clean.pair_gap_evidence,
        (
            MechanismRetentionEvidence("p1", 0.04, 0.04, 0.01, 10),
            MechanismRetentionEvidence("p2", 0.04, 0.04, 0.01, 10),
        ),
    )
    decision = evaluate_coupling_mechanism(config, retained)
    assert not decision.supported
    assert set(decision.mechanism_retention_pairs) == {"p1", "p2"}


def test_sparse_operational_relevance_requires_dense_closeness_useful_gain_and_exactness(
    config: FedorbitConfig,
) -> None:
    units = tuple(
        SparseUnitEvidence(f"p{index}", index, 3, 0.03, 0.035, True) for index in range(1, 5)
    )
    pair_gains = (
        SparsePairGainEvidence("p1", 2, 0.03, 10),
        SparsePairGainEvidence("p2", 2, 0.02, 10),
    )
    decision = evaluate_sparse_operational_relevance(config, units, pair_gains, True)
    assert decision.supported
    assert decision.useful_supports == (2,)
    failed = evaluate_sparse_operational_relevance(config, units, pair_gains, False)
    assert not failed.supported


def test_confirmation_safety_applies_pair_and_equal_pair_criteria(config: FedorbitConfig) -> None:
    evidence = tuple(
        ConfirmationPairEvidence(f"p{index}", 0.10, 0.05, 0.05, 10) for index in range(1, 5)
    )
    decision = evaluate_confirmation_safety(config, evidence)
    assert decision.supported
    assert len(decision.qualifying_pairs) == 4
    failed = (*evidence[:3], ConfirmationPairEvidence("p4", 0.10, 0.15, 0.05, 10))
    assert not evaluate_confirmation_safety(config, failed).supported
