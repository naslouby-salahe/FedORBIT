from __future__ import annotations

import pytest

from fedorbit.analysis.comparisons import (
    PairContrastEvidence,
    PairContrastEvidenceSet,
    evaluate_transfer_style_criteria,
)
from fedorbit.analysis.families import build_family_states, registered_family_inputs
from fedorbit.analysis.statistics import NamedPValue, PValueSet
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import MultiplicityFamily


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
        directed_pair=pair,
        mean_gain=mean_gain,
        holm_p=holm_p,
        bca_lower=bca_lower,
        strict_resource_valid=strict_valid,
        valid_seed_count=valid_seed_count,
    )


def _set(*entries: PairContrastEvidence) -> PairContrastEvidenceSet:
    return PairContrastEvidenceSet(entries)


def test_full_scope_supported_when_three_pairs_pass(config: FedorbitConfig) -> None:
    evidence = _set(
        _evidence("p1", mean_gain=0.05),
        _evidence("p2", mean_gain=0.04),
        _evidence("p3", mean_gain=0.06),
        _evidence("p4", mean_gain=-0.001),
    )
    decision = evaluate_transfer_style_criteria(
        config, evidence, removed_before_outcome_inspection=False
    )
    assert decision.supported
    assert not decision.conditional
    assert set(decision.successful_pairs) == {"p1", "p2", "p3"}
    assert decision.not_supported is False


def test_harmful_pair_makes_not_supported(config: FedorbitConfig) -> None:
    evidence = _set(
        _evidence("p1", mean_gain=0.05),
        _evidence("p2", mean_gain=0.04),
        _evidence("p3", mean_gain=-0.02),
        _evidence("p4", mean_gain=0.03),
    )
    decision = evaluate_transfer_style_criteria(
        config, evidence, removed_before_outcome_inspection=False
    )
    assert decision.not_supported
    assert "p3" in decision.harmful_pairs


def test_reduced_scope_conditional_with_one_pre_outcome_removal(
    config: FedorbitConfig,
) -> None:
    evidence = _set(
        _evidence("p1", mean_gain=0.05, valid_seed_count=0),
        _evidence("p2", mean_gain=0.04),
        _evidence("p3", mean_gain=0.06),
        _evidence("p4", mean_gain=0.03),
    )
    decision = evaluate_transfer_style_criteria(
        config, evidence, removed_before_outcome_inspection=True
    )
    assert decision.conditional
    assert not decision.supported
    assert len(decision.successful_pairs) == 3


def test_two_positive_pairs_is_partially_supported(config: FedorbitConfig) -> None:
    evidence = _set(
        _evidence("p1", mean_gain=0.05),
        _evidence("p2", mean_gain=0.04),
        _evidence("p3", mean_gain=0.001),
        _evidence("p4", mean_gain=0.002),
    )
    decision = evaluate_transfer_style_criteria(
        config, evidence, removed_before_outcome_inspection=False
    )
    assert decision.partially_supported
    assert len(decision.successful_pairs) == 2


def test_no_benefit_no_harm_is_null_result(config: FedorbitConfig) -> None:
    evidence = _set(
        _evidence("p1", mean_gain=0.001),
        _evidence("p2", mean_gain=0.0005),
        _evidence("p3", mean_gain=0.002),
        _evidence("p4", mean_gain=-0.001),
    )
    decision = evaluate_transfer_style_criteria(
        config, evidence, removed_before_outcome_inspection=False
    )
    assert decision.null_result or decision.partially_supported
    assert not decision.not_supported
    assert not decision.supported


def test_strict_resource_failure_blocks_support(config: FedorbitConfig) -> None:
    evidence = _set(
        _evidence("p1", mean_gain=0.05),
        _evidence("p2", mean_gain=0.04),
        _evidence("p3", mean_gain=0.06, strict_valid=False),
        _evidence("p4", mean_gain=0.03),
    )
    decision = evaluate_transfer_style_criteria(
        config, evidence, removed_before_outcome_inspection=False
    )
    assert not decision.supported
    assert "p3" not in decision.successful_pairs


def test_family_registry_counts_match_roadmap() -> None:
    registry = registered_family_inputs()
    assert len(registry.contrasts_for(MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY)) == 4
    assert len(registry.contrasts_for(MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR)) == 8
    assert len(registry.contrasts_for(MultiplicityFamily.COUPLING_MECHANISM)) == 4
    assert len(registry.contrasts_for(MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY)) == 8
    assert len(registry.contrasts_for(MultiplicityFamily.MECHANISM_ABLATIONS)) == 8
    assert len(registry.contrasts_for(MultiplicityFamily.SPARSITY_SENSITIVITY)) == 12
    assert len(registry.contrasts_for(MultiplicityFamily.CONFIRMATION_SAFETY)) == 4
    assert sum(len(entry.contrasts) for entry in registry.entries) == 48


def test_missing_family_inputs_recorded_without_p_values(config: FedorbitConfig) -> None:
    registry = registered_family_inputs()
    name = registry.contrasts_for(MultiplicityFamily.COUPLING_MECHANISM)[0].name
    states = build_family_states(config, PValueSet((NamedPValue(name, 0.02),)))
    unavailable = [
        state for group in states.entries for state in group.states if not state.available
    ]
    assert unavailable
    assert all(state.raw_p_value is None for state in unavailable)
    assert all(state.unavailable_reason for state in unavailable)


def test_available_family_inputs_carry_raw_p_values(config: FedorbitConfig) -> None:
    registry = registered_family_inputs()
    name = registry.contrasts_for(MultiplicityFamily.COUPLING_MECHANISM)[0].name
    states = build_family_states(config, PValueSet((NamedPValue(name, 0.02),)))
    coupling_states = states.states_for(MultiplicityFamily.COUPLING_MECHANISM)
    available = [state for state in coupling_states if state.available]
    assert len(available) == 4
    assert all(state.raw_p_value == pytest.approx(0.02) for state in available)
