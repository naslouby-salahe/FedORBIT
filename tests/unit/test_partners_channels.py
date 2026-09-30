from __future__ import annotations

import numpy as np
import pytest
import torch

from fedorbit.config.loading import load_config
from fedorbit.config.models import AutoencoderDetectorConfig, ConditionSpec
from fedorbit.detection.moments import summarise
from fedorbit.study import channels
from fedorbit.study.channels import (
    AutoencoderInputs,
    ChannelError,
    SharedSummaries,
    build_autoencoder_scorer,
    build_gaussian_scorer,
    standardisation_for,
)
from fedorbit.study.partners import (
    PartnerSelectionContext,
    PartnerSelectionError,
    scale_profile_distance,
    select_partners,
)
from fedorbit.types import DeviceName, FloatMatrix, PartnerPolicy, RandomSeed, ShareChannel


def context(seed: int = 1) -> PartnerSelectionContext:
    generator = np.random.default_rng(seed)
    partners = {
        DeviceName(name): summarise(generator.normal(0.0, scale, size=(200, 4)))
        for name, scale in (("narrow", 0.1), ("close", 1.0), ("wide", 10.0), ("wider", 20.0))
    }
    reference = np.ones(4)
    return PartnerSelectionContext(
        partners=partners,
        deployable_reference_scale=reference,
        oracle_reference_scale=reference,
        similarity_scale_floor=1e-3,
        seed=RandomSeed(seed),
    )


def spec(policy: PartnerPolicy, count: int | None = None) -> ConditionSpec:
    return ConditionSpec(
        channel=ShareChannel.SHARED_MARGINALS, partner_policy=policy, partner_count=count
    )


def test_all_policy_returns_every_partner_sorted() -> None:
    assert select_partners(spec(PartnerPolicy.ALL), context()) == (
        "close",
        "narrow",
        "wide",
        "wider",
    )


def test_nearest_policies_rank_by_scale_profile() -> None:
    assert select_partners(spec(PartnerPolicy.ORACLE_NEAREST, 2), context()) == ("close", "wide")
    assert select_partners(spec(PartnerPolicy.DEPLOYABLE_NEAREST, 1), context()) == ("close",)


def test_deployable_and_oracle_references_are_distinct_inputs() -> None:
    base = context()
    deployable = PartnerSelectionContext(
        partners=base.partners,
        deployable_reference_scale=np.full(4, 20.0),
        oracle_reference_scale=np.ones(4),
        similarity_scale_floor=1e-3,
        seed=RandomSeed(1),
    )
    assert select_partners(spec(PartnerPolicy.DEPLOYABLE_NEAREST, 1), deployable) == ("wider",)
    assert select_partners(spec(PartnerPolicy.ORACLE_NEAREST, 1), deployable) == ("close",)


def test_random_policy_is_deterministic_per_seed() -> None:
    first = select_partners(spec(PartnerPolicy.RANDOM, 2), context(seed=5))
    assert first == select_partners(spec(PartnerPolicy.RANDOM, 2), context(seed=5))
    assert len(set(first)) == 2
    outcomes = {select_partners(spec(PartnerPolicy.RANDOM, 2), context(seed=s)) for s in range(20)}
    assert len(outcomes) > 1


def test_partner_count_beyond_availability_is_rejected() -> None:
    with pytest.raises(PartnerSelectionError):
        select_partners(spec(PartnerPolicy.RANDOM, 9), context())


def test_scale_profile_distance_is_zero_for_identical_scales() -> None:
    scale = np.array([1.0, 2.0, 3.0])
    assert scale_profile_distance(scale, scale, 1e-3) == 0.0
    assert scale_profile_distance(scale * 2, scale, 1e-3) > 0.0


def test_local_channels_ignore_shared_summaries_and_shared_channels_use_them() -> None:
    config = load_config().detectors.gaussian
    generator = np.random.default_rng(2)
    rows = generator.normal(3.0, 2.0, size=(50, 4))
    partner = summarise(generator.normal(-1.0, 5.0, size=(500, 4)))
    shared = SharedSummaries((partner,))
    local = standardisation_for(
        ConditionSpec(channel=ShareChannel.LOCAL_STANDARDISED), rows, None, config
    )
    assert np.allclose(local.centre, rows.mean(axis=0))
    unscaled = standardisation_for(
        ConditionSpec(channel=ShareChannel.LOCAL_UNSCALED), rows, None, config
    )
    assert np.allclose(unscaled.scale, 1.0)
    floored = standardisation_for(
        ConditionSpec(channel=ShareChannel.LOCAL_STANDARD_DEVIATION_FLOOR),
        rows,
        None,
        config,
    )
    assert np.all(floored.scale >= local.scale - 1e-12)
    shared_standardisation = standardisation_for(spec(PartnerPolicy.ALL), rows, shared, config)
    assert np.allclose(shared_standardisation.centre, partner.mean)


def test_standard_deviation_floor_lifts_only_small_scales() -> None:
    config = load_config().detectors.gaussian
    generator = np.random.default_rng(3)
    rows = np.column_stack([generator.normal(0.0, 10.0, 60), generator.normal(0.0, 1e-4, 60)])
    floored = standardisation_for(
        ConditionSpec(channel=ShareChannel.LOCAL_STANDARD_DEVIATION_FLOOR), rows, None, config
    )
    local_scale = rows.std(axis=0) + config.minimum_scale
    assert floored.scale[0] == pytest.approx(local_scale[0])
    assert floored.scale[1] > local_scale[1]


def test_sharing_channels_without_summaries_fail_loudly() -> None:
    config = load_config().detectors.gaussian
    rows = np.ones((10, 3)) + np.arange(30).reshape(10, 3)
    with pytest.raises(ChannelError):
        standardisation_for(spec(PartnerPolicy.ALL), rows, None, config)
    covariance = ConditionSpec(
        channel=ShareChannel.SHARED_COVARIANCE, partner_policy=PartnerPolicy.ALL, dose_rows=10
    )
    with pytest.raises(ChannelError):
        build_gaussian_scorer(covariance, rows, None, config)


def test_local_full_support_equals_local_standardisation() -> None:
    config = load_config().detectors.gaussian
    rows = np.random.default_rng(4).normal(0.0, 1.0, size=(80, 3))
    full = standardisation_for(
        ConditionSpec(channel=ShareChannel.LOCAL_FULL_SUPPORT), rows, None, config
    )
    local = standardisation_for(
        ConditionSpec(channel=ShareChannel.LOCAL_STANDARDISED), rows, None, config
    )
    assert np.array_equal(full.scale, local.scale)


def test_federated_autoencoder_samples_selected_partner_rows_and_uses_shared_normalisation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    normaliser = config.detectors.gaussian
    autoencoder = config.detectors.autoencoder
    rows = np.arange(24, dtype=np.float64).reshape(12, 2)
    partner_rows = np.arange(80, dtype=np.float64).reshape(40, 2) + 100.0
    partner = DeviceName("partner")
    inputs = AutoencoderInputs(
        rows=rows,
        shared=SharedSummaries((summarise(partner_rows),)),
        partner_pools={partner: partner_rows},
        selected_partners=(partner,),
        initialisation_seed=RandomSeed(11),
        partner_sample_seed=RandomSeed(17),
    )
    captured: list[list[FloatMatrix]] = []

    def fake_training(
        clients: list[FloatMatrix],
        detector_config: AutoencoderDetectorConfig,
        seed: RandomSeed,
        device: torch.device,
    ) -> torch.nn.Module:
        captured.append(clients)
        assert detector_config is autoencoder
        assert seed == inputs.initialisation_seed
        assert device == torch.device("cpu")
        return torch.nn.Identity()

    monkeypatch.setattr(channels, "train_federated_autoencoder", fake_training)
    scorer = build_autoencoder_scorer(
        ConditionSpec(
            channel=ShareChannel.FEDERATED_AVERAGING,
            partner_policy=PartnerPolicy.ALL,
        ),
        inputs,
        normaliser,
        autoencoder,
        torch.device("cpu"),
    )

    assert len(captured) == 1
    assert len(captured[0]) == 2
    standardisation = standardisation_for(
        ConditionSpec(
            channel=ShareChannel.FEDERATED_AVERAGING,
            partner_policy=PartnerPolicy.ALL,
        ),
        rows,
        inputs.shared,
        normaliser,
    )
    assert np.allclose(captured[0][0], (rows - standardisation.centre) / standardisation.scale)
    sample_indices = np.random.default_rng(inputs.partner_sample_seed).choice(
        len(partner_rows), size=len(partner_rows), replace=False
    )
    expected_partner_rows = partner_rows[sample_indices]
    assert np.allclose(
        captured[0][1],
        (expected_partner_rows - standardisation.centre) / standardisation.scale,
    )
    assert np.array_equal(scorer.score(rows), np.zeros(len(rows)))
