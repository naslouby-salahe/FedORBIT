from __future__ import annotations

import torch

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.models.architectures import (
    HostClassifier,
    NetworkFlowClassifier,
    classifier_for_modality,
)
from fedorbit.models.training import BaseCheckpoint
from fedorbit.response.final import (
    FinalResponseEntry,
    FinalResponseEstimate,
)
from fedorbit.response.orchestration import (
    PacketConstructionContext,
    anonymize_node_order,
    pad_absent_transfer_nodes,
)
from fedorbit.response.pilot import ResponseCandidate
from fedorbit.strict_interface.packet import SourcePacket


def test_classifier_for_modality_instantiates_per_modality() -> None:
    network = classifier_for_modality("network", 16, 3, 0.1)
    host = classifier_for_modality("host", 16, 3, 0.1)
    assert isinstance(network, NetworkFlowClassifier)
    assert isinstance(host, HostClassifier)


def test_classifier_dimensions_from_context() -> None:
    network = classifier_for_modality("network", 12, 4, 0.0)
    assert isinstance(network, NetworkFlowClassifier)
    assert int(network.input_dim) == 12
    assert int(network.n_classes) == 4


def test_anonymize_node_order_is_bijection_and_deterministic() -> None:
    context = _context()
    first = anonymize_node_order(context)
    second = anonymize_node_order(context)
    assert first == second
    assert tuple(sorted(first)) == tuple(sorted(context.fine_node_order))
    assert first != context.fine_node_order


def test_padding_adds_zero_entries_for_absent_nodes() -> None:
    estimate = FinalResponseEstimate(
        entries=(FinalResponseEntry(0, 0, 0.5, 0.1, 0.1, 0.9, True),),
        critical_value=2.0,
        useful_intervention_columns=1,
        median_band_width_ratio=0.5,
        stability_rule_passed=True,
    )
    padded = pad_absent_transfer_nodes(estimate, outcome_count=2, intervention_count=2)
    assert len(padded.entries) == 4
    present = {(entry.outcome_index, entry.intervention_index) for entry in padded.entries}
    assert present == {(0, 0), (0, 1), (1, 0), (1, 1)}
    zero = next(
        entry
        for entry in padded.entries
        if (entry.outcome_index, entry.intervention_index) == (1, 1)
    )
    assert zero.a_hat == 0.0
    assert zero.lower == 0.0
    assert zero.upper == 0.0
    assert not zero.useful


def test_packet_binds_dependency_digests() -> None:
    context = _context()
    from fedorbit.response.final import build_source_packet

    packet = build_source_packet(
        FinalResponseEstimate(
            entries=(FinalResponseEntry(0, 0, 0.5, 0.1, 0.1, 0.9, True),),
            critical_value=2.0,
            useful_intervention_columns=1,
            median_band_width_ratio=0.5,
            stability_rule_passed=True,
        ),
        anonymous_fine_node_ids=("n1",),
        exposed_coarse_group_id="Disruption",
        per_node_train_support=(10,),
        per_node_meta_support=(4,),
        per_node_effective_replicate_count=(24,),
        source_checkpoint_sha256=context.source_checkpoint_sha256,
        response_configuration_sha256=context.response_configuration_sha256,
        creation_timestamp="2026-08-22T00:00:00Z",
        preprocessing_state_sha256="c" * 64,
        transfer_node_manifest_sha256="d" * 64,
        response_seed=101,
    )
    assert packet.preprocessing_state_sha256 == "c" * 64
    assert packet.transfer_node_manifest_sha256 == "d" * 64
    assert packet.response_seed == 101
    assert packet.packet_integrity_sha256 == packet.compute_integrity_sha256()
    packet.validate()


def test_construct_source_packet_smoke() -> None:
    config = load_fedorbit_config()
    torch.manual_seed(0)
    model = NetworkFlowClassifier(8, 3, 0.0)
    train_features = torch.randn(24, 8)
    train_targets = torch.randint(0, 3, (24,))
    meta_features = torch.randn(6, 8)
    meta_targets = torch.randint(0, 3, (6,))
    weights = torch.ones(3)
    checkpoint = _random_checkpoint(model)
    context = _context()
    from fedorbit.response.orchestration import construct_source_packet

    constructed = construct_source_packet(
        config,
        context,
        checkpoint,
        model,
        train_features,
        train_targets,
        meta_features,
        meta_targets,
        intervention_classes=((0,), (1,)),
        outcome_native_class_sets=((0,), (1,), (2,)),
        base_class_weights=weights,
        selected_configuration=ResponseCandidate(0.05, 25),
        creation_timestamp="2026-08-22T00:00:00Z",
    )
    packet = constructed.packet
    assert isinstance(packet, SourcePacket)
    assert packet.packet_integrity_sha256 == packet.compute_integrity_sha256()
    assert packet.packet_validity_state in ("stable", "unstable")
    assert len(packet.anonymous_fine_node_ids) == len(context.fine_node_order)
    assert len(packet.L) == 3 * 2


def _context() -> PacketConstructionContext:
    return PacketConstructionContext(
        modality="network",
        input_dimension=8,
        n_classes=3,
        coarse_group_id="Disruption",
        fine_node_order=("a", "b", "c"),
        per_node_train_support=(8, 8, 8),
        per_node_meta_support=(2, 2, 2),
        source_checkpoint_sha256="a" * 64,
        preprocessing_state_sha256="b" * 64,
        transfer_node_manifest_sha256="c" * 64,
        response_configuration_sha256="d" * 64,
        seed=101,
        learning_rate=0.001,
        weight_decay=0.0,
        dropout_probability=0.0,
    )


def _random_checkpoint(model: torch.nn.Module) -> BaseCheckpoint:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    logits = model(torch.randn(4, 8))
    loss = logits.sum()
    loss.backward()
    optimizer.step()
    return BaseCheckpoint(
        epoch=0,
        valid_macro_cross_entropy=1.0,
        state_dict={key: value.detach().clone() for key, value in model.state_dict().items()},
        optimizer_state=optimizer.state_dict(),
        rng_state=torch.get_rng_state(),
    )
