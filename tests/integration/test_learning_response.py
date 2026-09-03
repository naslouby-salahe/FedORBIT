from __future__ import annotations

from fedorbit.response.packet import build_source_packet
from fedorbit.response.uncertainty import FinalResponseEntry, FinalResponseEstimate


def test_response_pipeline_builds_valid_strict_packet() -> None:
    estimate = FinalResponseEstimate(
        entries=(FinalResponseEntry(0, 0, 0.1, 0.01, 0.08, 0.12, True),),
        critical_value=2.0,
        useful_intervention_columns=1,
        median_band_width_ratio=0.4,
        stability_rule_passed=True,
    )
    packet = build_source_packet(
        estimate,
        anonymous_fine_node_ids=("node-0001",),
        exposed_coarse_group_id="Disruption",
        per_node_train_support=(200,),
        per_node_meta_support=(40,),
        per_node_effective_replicate_count=(24,),
        source_checkpoint_sha256="a" * 64,
        response_configuration_sha256="b" * 64,
        creation_timestamp="2026-08-23T22:00:00Z",
    )
    packet.validate()
    assert packet.packet_integrity_sha256 == packet.compute_integrity_sha256()
