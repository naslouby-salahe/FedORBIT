from __future__ import annotations

import math
from collections import OrderedDict
from typing import cast

import numpy as np
import pytest
import torch

from fedorbit.infrastructure.runtime import (
    RngNamespace,
    SeedDerivationRequest,
    derive_seed32,
)
from fedorbit.interface import (
    AnonymityCoordinate,
    AnonymityCoordinateEntry,
    anonymous_node_order,
)
from fedorbit.learning.pilot import create_classifier
from fedorbit.learning.training import (
    BaseCheckpoint,
    ClassWeights,
    ModelParameterState,
    OptimizerState,
    RngState,
    SelectedHyperparameters,
    make_adamw,
)
from fedorbit.response.estimation import (
    ShadowData,
    ShadowSettings,
    paired_shadow_derivative,
    run_shadow_pair,
    shadow_batch_schedule,
    standard_error,
)
from fedorbit.response.packet import (
    PacketConstructionContext,
    PacketField,
    SourcePacket,
    anonymized_response_estimate,
    build_source_packet,
    construct_source_packet,
)
from fedorbit.response.pilot import ResponseCandidate
from fedorbit.response.uncertainty import (
    FinalResponseEntry,
    FinalResponseEstimate,
    max_t_critical_value,
)
from fedorbit.types import (
    AnonymousNodeDisplayId,
    ClassIndex,
    ClientRole,
    CoarseGroup,
    DatasetId,
    ExposedCoarseGroupId,
    Rfc3339UtcTimestamp,
    Sha256Digest,
    StableJsonPayload,
)

_DEFAULT_TIMESTAMP = Rfc3339UtcTimestamp("2026-08-22T00:00:00Z")


def _estimate() -> FinalResponseEstimate:
    return FinalResponseEstimate(
        entries=(
            FinalResponseEntry(0, 0, 0.3, 0.05, 0.1, 0.5, True),
            FinalResponseEntry(1, 0, -0.2, 0.04, -0.4, -0.01, True),
            FinalResponseEntry(0, 1, 0.0, 0.0, 0.0, 0.0, False),
            FinalResponseEntry(1, 1, 0.0, 0.0, 0.0, 0.0, False),
        ),
        critical_value=4.0,
        useful_intervention_columns=1,
        median_band_width_ratio=1.3,
        stability_rule_passed=True,
    )


def _packet(timestamp: Rfc3339UtcTimestamp = _DEFAULT_TIMESTAMP) -> SourcePacket:
    return build_source_packet(
        _estimate(),
        anonymous_fine_node_ids=(
            AnonymousNodeDisplayId("node-0001"),
            AnonymousNodeDisplayId("node-0002"),
        ),
        exposed_coarse_group_id=ExposedCoarseGroupId("Disruption"),
        per_node_train_support=(120, 90),
        per_node_meta_support=(30, 20),
        per_node_effective_replicate_count=(24, 24),
        source_checkpoint_sha256=Sha256Digest("a" * 64),
        response_configuration_sha256=Sha256Digest("b" * 64),
        creation_timestamp=timestamp,
    )


def test_packet_uses_exact_anonymous_identifiers_and_integrity() -> None:
    packet = _packet()
    packet.validate()
    assert packet.anonymous_fine_node_ids == ("node-0001", "node-0002")
    assert packet.packet_integrity_sha256 == packet.compute_integrity_sha256()
    assert '"dtype":"float64"' in packet.integrity_payload()
    assert '"order":"C"' in packet.integrity_payload()


def test_timestamp_does_not_change_scientific_integrity() -> None:
    first = _packet(Rfc3339UtcTimestamp("2026-08-22T00:00:00Z"))
    second = _packet(Rfc3339UtcTimestamp("2026-08-23T00:00:00Z"))
    assert first.packet_integrity_sha256 == second.packet_integrity_sha256
    assert first.payload_sha256() != second.payload_sha256()


def test_packet_rejects_semantic_or_nonstable_node_ids() -> None:
    packet = _packet()
    invalid = SourcePacket(
        anonymous_fine_node_ids=(
            AnonymousNodeDisplayId("ddos"),
            AnonymousNodeDisplayId("ransomware"),
        ),
        exposed_coarse_group_id=packet.exposed_coarse_group_id,
        L=packet.L,
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
    with pytest.raises(PermissionError):
        invalid.validate()


def test_packet_rejects_invalid_timestamp_and_hashes() -> None:
    packet = _packet()
    invalid = SourcePacket(
        anonymous_fine_node_ids=packet.anonymous_fine_node_ids,
        exposed_coarse_group_id=packet.exposed_coarse_group_id,
        L=packet.L,
        U=packet.U,
        per_node_train_support=packet.per_node_train_support,
        per_node_meta_support=packet.per_node_meta_support,
        per_node_effective_replicate_count=packet.per_node_effective_replicate_count,
        packet_schema_metadata=packet.packet_schema_metadata,
        source_checkpoint_sha256=Sha256Digest("not-a-digest"),
        response_configuration_sha256=packet.response_configuration_sha256,
        packet_integrity_sha256=packet.packet_integrity_sha256,
        packet_validity_state=packet.packet_validity_state,
        technical_creation_timestamp=Rfc3339UtcTimestamp("2026-08-22 00:00:00"),
    )
    with pytest.raises(PermissionError):
        invalid.validate()


def test_paired_derivative_and_standard_error_match_hand_calculations() -> None:
    derivative = paired_shadow_derivative(8.0, 14.0, 4.0, 0.25, 1.0)
    values = (1.0, 2.0, 4.0)

    assert derivative == pytest.approx(3.0)
    assert standard_error(values) == pytest.approx(np.std(values, ddof=1) / math.sqrt(3))


def test_max_t_critical_value_uses_paired_indices_and_higher_quantile() -> None:
    derivatives = ((1.0, 3.0, 5.0), (2.0, 4.0, 8.0))
    seed = 101
    resamples = 7
    confidence = 0.8
    floor = 0.1
    critical = max_t_critical_value(derivatives, seed, resamples, confidence, floor)
    bootstrap_seed = derive_seed32(
        SeedDerivationRequest(
            seed,
            RngNamespace.RESPONSE_BOOTSTRAP,
            cast(
                StableJsonPayload,
                OrderedDict(
                    entries=len(derivatives),
                    replicates=len(derivatives[0]),
                    resamples=resamples,
                    confidence=confidence,
                ),
            ),
        )
    )
    indices = torch.randint(
        0,
        len(derivatives[0]),
        (resamples, len(derivatives[0])),
        generator=torch.Generator().manual_seed(bootstrap_seed),
    ).numpy()
    values = np.asarray(derivatives, dtype=np.float64)
    means = np.mean(values, axis=1)
    resampled = values[:, indices]
    standard_errors = np.std(resampled, axis=2, ddof=1) / math.sqrt(len(derivatives[0]))
    maxima = np.max(
        np.abs(np.mean(resampled, axis=2) - means[:, None]) / np.maximum(standard_errors, floor),
        axis=0,
    )

    assert critical == np.quantile(maxima, confidence, method="higher")


def test_shadow_pair_reuses_schedule_and_restores_the_shared_starting_state() -> None:
    torch.manual_seed(101)
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = make_adamw(model, 0.001, 0.0)
    base_state = ModelParameterState.capture(model)
    base_optimizer_state = OptimizerState.capture(optimizer)
    base_rng_state = RngState.capture()
    data = ShadowData(
        train_features=torch.tensor(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))),
        train_targets=torch.tensor((0, 1, 0, 1)),
        meta_features=torch.tensor(((1.0, 0.0), (0.0, 1.0))),
        meta_targets=torch.tensor((0, 1)),
        intervention_classes=(ClassIndex(0),),
        outcome_native_class_sets=((ClassIndex(0),), (ClassIndex(1),)),
        base_class_weights=ClassWeights(torch.ones(2)),
    )
    schedule_seed = 303
    first_schedule = shadow_batch_schedule(4, 2, torch.Generator().manual_seed(schedule_seed))
    second_schedule = shadow_batch_schedule(4, 2, torch.Generator().manual_seed(schedule_seed))
    assert torch.equal(next(first_schedule), next(second_schedule))
    assert torch.equal(next(first_schedule), next(second_schedule))

    risks = run_shadow_pair(
        model,
        base_state,
        base_optimizer_state,
        base_rng_state,
        data,
        ShadowSettings(0.25, 1, 0.001, 0.0),
        schedule_seed,
    )

    assert all(math.isfinite(value) for triple in risks for value in triple)
    restored = ModelParameterState.capture(model)
    assert all(
        torch.equal(actual.value, expected.value)
        for actual, expected in zip(restored.tensors, base_state.tensors, strict=True)
    )
    assert torch.equal(torch.get_rng_state(), base_rng_state.cpu)


def test_packet_response_entries_follow_the_registered_anonymous_order() -> None:
    estimate = FinalResponseEstimate(
        entries=tuple(
            FinalResponseEntry(
                outcome,
                intervention,
                float(outcome * 2 + intervention),
                0.0,
                float(outcome * 2 + intervention) + 0.1,
                0.0,
                True,
            )
            for outcome in range(2)
            for intervention in range(2)
        ),
        critical_value=4.0,
        useful_intervention_columns=2,
        median_band_width_ratio=1.3,
        stability_rule_passed=True,
    )
    order = anonymous_node_order(
        2,
        17,
        ClientRole.SOURCE,
        CoarseGroup.DISRUPTION,
        AnonymityCoordinate(
            (
                AnonymityCoordinateEntry("fixture_order", "first"),
                AnonymityCoordinateEntry("fixture_value", "second"),
            )
        ),
    )
    anonymized = anonymized_response_estimate(estimate, order)
    positions = order.anonymous_position_of_semantic_index()
    assert order.permutation != (0, 1)
    semantic = np.asarray([entry.lower for entry in estimate.entries], dtype=np.float64).reshape(
        (2, 2)
    )
    matrix = np.asarray([entry.lower for entry in anonymized.entries], dtype=np.float64).reshape(
        (2, 2)
    )
    assert np.array_equal(matrix, semantic[np.ix_(positions, positions)])


def test_packet_registered_node_order_reproduces_the_construction_permutation() -> None:
    order = anonymous_node_order(
        2,
        17,
        ClientRole.SOURCE,
        CoarseGroup.DISRUPTION,
        AnonymityCoordinate(
            (
                AnonymityCoordinateEntry(
                    PacketField.SOURCE_CHECKPOINT_SHA256,
                    Sha256Digest("a" * 64),
                ),
                AnonymityCoordinateEntry(
                    PacketField.RESPONSE_CONFIGURATION_SHA256,
                    Sha256Digest("b" * 64),
                ),
            )
        ),
    )
    assert _packet().registered_node_order(17).permutation == order.permutation


def test_constructed_packet_payload_is_entirely_in_the_registered_anonymous_order() -> None:
    semantic_node_names = ("DDoS", "Ransomware", "Backdoor")
    per_node_train_support = (120, 90, 45)
    per_node_meta_support = (30, 20, 10)
    context = PacketConstructionContext(
        dataset=DatasetId.TON_IOT_WINDOWS10_HOST,
        input_dimension=4,
        n_classes=3,
        coarse_group_id=CoarseGroup.DISRUPTION,
        fine_node_order=tuple(AnonymousNodeDisplayId(name) for name in semantic_node_names),
        per_node_train_support=per_node_train_support,
        per_node_meta_support=per_node_meta_support,
        source_checkpoint_sha256=Sha256Digest("a" * 64),
        response_configuration_sha256=Sha256Digest("b" * 64),
        seed=101,
    )
    model = create_classifier(DatasetId.TON_IOT_WINDOWS10_HOST, 4, 3, 0.0, context.seed)
    optimizer = make_adamw(model, 0.001, 0.0)
    checkpoint = BaseCheckpoint(
        epoch=0,
        valid_macro_cross_entropy=0.5,
        state_dict=ModelParameterState.capture(model),
        optimizer_state=OptimizerState.capture(optimizer),
        rng_state=RngState.capture(),
        selected_hyperparameters=SelectedHyperparameters(0.001, 0.0, 0.0),
        train_class_weights=ClassWeights(torch.ones(3)),
    )
    train_features = torch.tensor(
        (
            (1.0, 0.0, 0.5, 0.25),
            (0.0, 1.0, 0.25, 0.5),
            (1.0, 1.0, 0.0, 0.0),
            (0.5, 0.5, 1.0, 0.0),
            (0.25, 0.5, 1.0, 0.5),
            (0.75, 0.25, 0.5, 1.0),
        )
    )
    meta_features = torch.tensor(
        ((1.0, 0.0, 0.5, 0.25), (0.0, 1.0, 0.25, 0.5), (0.5, 0.0, 1.0, 0.5))
    )
    constructed = construct_source_packet(
        context,
        checkpoint,
        model,
        train_features,
        torch.tensor((0, 1, 2, 0, 1, 2)),
        meta_features,
        torch.tensor((0, 1, 2)),
        ((ClassIndex(0),), (ClassIndex(1),), (ClassIndex(2),)),
        ((ClassIndex(0),), (ClassIndex(1),), (ClassIndex(2),)),
        ClassWeights(torch.ones(3)),
        ResponseCandidate(0.25, 1),
        _DEFAULT_TIMESTAMP,
    )
    packet = constructed.packet
    order = packet.registered_node_order(context.seed)
    positions = order.anonymous_position_of_semantic_index()
    assert order.permutation != tuple(range(len(semantic_node_names)))
    assert (
        packet.anonymous_fine_node_ids
        == order.display_ids
        == (
            AnonymousNodeDisplayId("node-0001"),
            AnonymousNodeDisplayId("node-0002"),
            AnonymousNodeDisplayId("node-0003"),
        )
    )
    assert packet.per_node_train_support == order.reorder(per_node_train_support)
    assert packet.per_node_meta_support == order.reorder(per_node_meta_support)
    semantic_lower = np.asarray(
        [entry.lower for entry in constructed.estimate.entries], dtype=np.float64
    ).reshape((len(semantic_node_names), len(semantic_node_names)))
    semantic_upper = np.asarray(
        [entry.upper for entry in constructed.estimate.entries], dtype=np.float64
    ).reshape((len(semantic_node_names), len(semantic_node_names)))
    assert np.array_equal(packet.lower_matrix(), semantic_lower[np.ix_(positions, positions)])
    assert np.array_equal(packet.upper_matrix(), semantic_upper[np.ix_(positions, positions)])
    assert not np.array_equal(packet.lower_matrix(), semantic_lower)
    serialized = packet.serialized()
    for name in semantic_node_names:
        assert name not in serialized
    assert packet.registered_node_order(102).permutation != order.permutation
