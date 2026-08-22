from __future__ import annotations

import numpy as np
import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.response.bootstrap import BootstrapError, max_t_critical_value
from fedorbit.response.final import (
    FinalResponseEntry,
    FinalResponseEstimate,
    build_source_packet,
)
from fedorbit.strict_interface.packet import SourcePacket


def test_max_t_bootstrap_is_deterministic() -> None:
    config = load_fedorbit_config()
    entries = ((1.0, 1.5, 2.0), (0.5, 1.0, 1.5))
    first = max_t_critical_value(config, entries, 7)
    second = max_t_critical_value(config, entries, 7)
    assert first == second
    assert first > 0


def test_max_t_bootstrap_rejects_empty_entries() -> None:
    config = load_fedorbit_config()
    with pytest.raises(BootstrapError):
        max_t_critical_value(config, (), 7)


def test_max_t_bootstrap_uses_higher_quantile() -> None:
    values = tuple(float(value) for value in range(100))
    quantile = float(np.quantile(np.asarray(values), 0.95, method="higher"))
    assert quantile == 95.0


def test_final_estimate_stability_rules_with_synthetic_entries() -> None:
    entries = tuple(
        FinalResponseEntry(
            outcome_index=outcome,
            intervention_index=intervention,
            a_hat=1.0 if outcome == intervention else 0.0,
            standard_error=0.05,
            lower=0.8,
            upper=1.2,
            useful=True,
        )
        for outcome in range(3)
        for intervention in range(3)
    )
    stable = FinalResponseEstimate(
        entries=entries,
        critical_value=2.0,
        useful_intervention_columns=3,
        median_band_width_ratio=0.4,
        stability_rule_passed=True,
    )
    assert stable.stability_rule_passed
    unstable = FinalResponseEstimate(
        entries=entries,
        critical_value=2.0,
        useful_intervention_columns=1,
        median_band_width_ratio=0.4,
        stability_rule_passed=False,
    )
    assert not unstable.stability_rule_passed


def test_build_source_packet_round_trip_with_integrity() -> None:
    estimate = FinalResponseEstimate(
        entries=(
            FinalResponseEntry(0, 0, 0.3, 0.05, 0.1, 0.5, True),
            FinalResponseEntry(1, 0, -0.2, 0.04, -0.4, 0.0, False),
        ),
        critical_value=4.0,
        useful_intervention_columns=1,
        median_band_width_ratio=1.3,
        stability_rule_passed=True,
    )
    packet = build_source_packet(
        estimate,
        anonymous_fine_node_ids=("node-1", "node-2"),
        exposed_coarse_group_id="Disruption",
        per_node_train_support=(120, 90),
        per_node_meta_support=(30, 20),
        per_node_effective_replicate_count=(24, 24),
        source_checkpoint_sha256="a" * 64,
        response_configuration_sha256="b" * 64,
        creation_timestamp="2026-08-22T00:00:00Z",
    )
    assert packet.packet_integrity_sha256 == packet.compute_integrity_sha256()
    assert packet.packet_validity_state == "stable"
    assert packet.L == (0.1, -0.4)
    assert packet.U == (0.5, 0.0)
    packet.validate()
    corrupted = SourcePacket(
        anonymous_fine_node_ids=packet.anonymous_fine_node_ids,
        exposed_coarse_group_id=packet.exposed_coarse_group_id,
        L=(9.9,),
        U=packet.U,
        per_node_train_support=packet.per_node_train_support,
        per_node_meta_support=packet.per_node_meta_support,
        per_node_effective_replicate_count=packet.per_node_effective_replicate_count,
        packet_schema_metadata=packet.packet_schema_metadata,
        source_checkpoint_sha256=packet.source_checkpoint_sha256,
        response_configuration_sha256=packet.response_configuration_sha256,
        packet_integrity_sha256=packet.packet_integrity_sha256,
        packet_validity_state=packet.packet_validity_state,
        technical_creation_timestamp=packet.technical_creation_timestamp,
    )
    with pytest.raises(ValueError):
        corrupted.validate()


def test_final_config_values() -> None:
    config = load_fedorbit_config()
    final = config.scientific.source_response_final
    assert final.paired_replicates_per_intervention == 24
    assert final.simultaneous_confidence_level == 0.95
    assert final.max_t_bootstrap_resamples == 2000
    assert final.response_risk_denominator_floor == 1e-8
    assert final.response_standard_error_floor == 1e-12
    assert final.useful_response_magnitude_threshold == 0.005
    assert final.minimum_useful_intervention_columns == 2
