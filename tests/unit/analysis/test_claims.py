from __future__ import annotations

import pytest

from fedorbit.analysis.claims import evaluate_transfer_style_criteria
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
