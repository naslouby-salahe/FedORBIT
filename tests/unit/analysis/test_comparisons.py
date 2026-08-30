from __future__ import annotations

import pytest

from fedorbit.analysis.comparisons import (
    ContrastPValue,
    ContrastPValueSet,
    PairContrastEvidence,
    PairContrastEvidenceSet,
    PairedObservation,
    PairingError,
    PairingLineage,
    build_family_states,
    registered_family_inputs,
    validate_paired_observations,
)
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import MultiplicityFamily, Split, TransferMethod


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


def _pvalue(
    family: MultiplicityFamily,
    index: int,
    p_value: float,
    valid_seed_count: int = 10,
) -> ContrastPValue:
    contrast = registered_family_inputs().contrasts_for(family)[index]
    return ContrastPValue(
        family,
        contrast.name,
        contrast.directed_pair,
        p_value,
        valid_seed_count,
    )


def test_pair_specific_p_value_does_not_populate_other_pairs() -> None:
    states = build_family_states(
        ContrastPValueSet((_pvalue(MultiplicityFamily.COUPLING_MECHANISM, 0, 0.02),)),
    )
    coupling_states = states.states_for(MultiplicityFamily.COUPLING_MECHANISM)
    available = [state for state in coupling_states if state.available]
    assert len(available) == 1
    assert available[0].raw_p_value == pytest.approx(0.02)
    assert available[0].holm_p_value == pytest.approx(0.02)
    assert available[0].family_size == 1


def test_holm_is_applied_within_family_only() -> None:
    values = ContrastPValueSet(
        (
            _pvalue(MultiplicityFamily.COUPLING_MECHANISM, 0, 0.01),
            _pvalue(MultiplicityFamily.COUPLING_MECHANISM, 1, 0.03),
            _pvalue(MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY, 0, 0.02),
        )
    )
    states = build_family_states(values)
    coupling = [
        state
        for state in states.states_for(MultiplicityFamily.COUPLING_MECHANISM)
        if state.available
    ]
    primary = [
        state
        for state in states.states_for(MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY)
        if state.available
    ]
    assert [state.holm_p_value for state in coupling] == pytest.approx([0.02, 0.03])
    assert primary[0].holm_p_value == pytest.approx(0.02)
    assert primary[0].family_size == 1


def test_inputs_with_insufficient_paired_seeds_are_not_in_holm_family() -> None:
    states = build_family_states(
        ContrastPValueSet(
            (
                _pvalue(MultiplicityFamily.COUPLING_MECHANISM, 0, 0.01),
                _pvalue(MultiplicityFamily.COUPLING_MECHANISM, 1, 0.001, valid_seed_count=7),
            )
        ),
    )
    coupling = states.states_for(MultiplicityFamily.COUPLING_MECHANISM)
    assert coupling[0].family_size == 1
    assert coupling[0].holm_p_value == pytest.approx(0.01)
    assert coupling[1].available is False
    assert coupling[1].unavailable_reason == "insufficient valid paired seeds"


def _lineage(seed: int = 1103, source_packet: str | None = "packet-1") -> PairingLineage:
    return PairingLineage(
        raw_dataset_lineage_sha256="a" * 64,
        directed_pair="Edge→Windows",
        seed=seed,
        split=Split.TEST,
        target_pre_transfer_checkpoint_artifact_id="checkpoint-1",
        target_importance_artifact_id="importance-1",
        source_packet_artifact_id=source_packet,
        action_budget=0.5,
        support_budget=2,
        confirmation_budget=200,
        environment_lineage_sha256="b" * 64,
    )


def test_pairing_engine_accepts_only_identical_registered_lineage() -> None:
    method = (
        PairedObservation(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, 0.2, _lineage(1103)),
        PairedObservation(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, 0.3, _lineage(2089)),
    )
    reference = (
        PairedObservation(TransferMethod.LOCAL_ONLY, 0.1, _lineage(2089)),
        PairedObservation(TransferMethod.LOCAL_ONLY, 0.15, _lineage(1103)),
    )
    paired = validate_paired_observations(method, reference)
    assert paired.directed_pair == "Edge→Windows"
    assert paired.seeds == (1103, 2089)
    assert paired.method_values == pytest.approx((0.2, 0.3))
    assert paired.reference_values == pytest.approx((0.15, 0.1))


def test_pairing_engine_rejects_source_packet_mismatch() -> None:
    method = (PairedObservation(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, 0.2, _lineage()),)
    reference = (
        PairedObservation(TransferMethod.LOCAL_ONLY, 0.1, _lineage(source_packet="packet-2")),
    )
    with pytest.raises(PairingError, match="lineage mismatch"):
        validate_paired_observations(method, reference)
