from __future__ import annotations

import pytest

from fedorbit.analysis.comparisons import (
    PairContrastEvidence,
    PairContrastEvidenceSet,
    build_family_states,
    registered_family_inputs,
)
from fedorbit.analysis.statistics import NamedPValue, PValueSet
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import MultiplicityFamily


@pytest.fixture
def config() -> FedorbitConfig:
    return load_fedorbit_config()


def test_pair_evidence_set_rejects_duplicate_directed_pairs() -> None:
    evidence = PairContrastEvidence("pair", 0.1, 0.01, 0.01, True, 10)
    with pytest.raises(ValueError):
        PairContrastEvidenceSet((evidence, evidence))


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
