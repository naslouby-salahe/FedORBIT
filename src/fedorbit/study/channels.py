from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from fedorbit.config.models import (
    AutoencoderDetectorConfig,
    ConditionSpec,
    GaussianDetectorConfig,
)
from fedorbit.detection.autoencoder import (
    AutoencoderScorer,
    train_federated_autoencoder,
    train_local_autoencoder,
)
from fedorbit.detection.gaussian import GaussianScorer, scorer_from_moments, scorer_from_rows
from fedorbit.detection.moments import (
    MomentSummary,
    aggregate_equal_weight,
    pool_with_partners,
    summarise,
)
from fedorbit.types import DeviceName, FloatMatrix, FloatVector, RandomSeed, ShareChannel


class ChannelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Standardisation:
    centre: FloatVector
    scale: FloatVector


@dataclass(frozen=True, slots=True)
class SharedSummaries:
    selected: tuple[MomentSummary, ...]

    @property
    def aggregate(self) -> MomentSummary:
        return aggregate_equal_weight(self.selected)


def standardisation_for(
    spec: ConditionSpec,
    rows: FloatMatrix,
    shared: SharedSummaries | None,
    config: GaussianDetectorConfig,
) -> Standardisation:
    channel = spec.channel
    local_scale = rows.std(axis=0) + config.minimum_scale
    if channel in {ShareChannel.LOCAL_STANDARDISED, ShareChannel.LOCAL_FULL_SUPPORT}:
        return Standardisation(rows.mean(axis=0), local_scale)
    if channel is ShareChannel.LOCAL_UNSCALED:
        return Standardisation(rows.mean(axis=0), np.ones(rows.shape[1]))
    if channel is ShareChannel.LOCAL_STANDARD_DEVIATION_FLOOR:
        floor = config.standard_deviation_floor_fraction * float(np.median(local_scale))
        return Standardisation(rows.mean(axis=0), np.maximum(local_scale, floor))
    if shared is None:
        raise ChannelError(f"channel {channel} needs shared summaries")
    aggregate = shared.aggregate
    if channel in {ShareChannel.SHARED_MARGINALS, ShareChannel.FEDERATED_AVERAGING}:
        return Standardisation(aggregate.mean, aggregate.standard_deviation + config.minimum_scale)
    raise ChannelError(f"channel {channel} has no standardisation")


def build_gaussian_scorer(
    spec: ConditionSpec,
    rows: FloatMatrix,
    shared: SharedSummaries | None,
    config: GaussianDetectorConfig,
) -> GaussianScorer:
    if spec.channel is ShareChannel.SHARED_COVARIANCE:
        if shared is None or spec.dose_rows is None:
            raise ChannelError("shared covariance needs shared summaries and a dose")
        pooled = pool_with_partners(summarise(rows), shared.aggregate, float(spec.dose_rows))
        return scorer_from_moments(pooled, config.minimum_scale)
    standardisation = standardisation_for(spec, rows, shared, config)
    return scorer_from_rows(rows, standardisation.centre, standardisation.scale)


@dataclass(frozen=True, slots=True)
class AutoencoderInputs:
    rows: FloatMatrix
    shared: SharedSummaries | None
    partner_pools: Mapping[DeviceName, FloatMatrix]
    selected_partners: tuple[DeviceName, ...]
    initialisation_seed: RandomSeed
    partner_sample_seed: RandomSeed


def build_autoencoder_scorer(
    spec: ConditionSpec,
    inputs: AutoencoderInputs,
    normaliser_config: GaussianDetectorConfig,
    config: AutoencoderDetectorConfig,
    device: torch.device,
) -> AutoencoderScorer:
    standardisation = standardisation_for(spec, inputs.rows, inputs.shared, normaliser_config)
    local_rows = (inputs.rows - standardisation.centre) / standardisation.scale
    if spec.channel is ShareChannel.FEDERATED_AVERAGING:
        generator = np.random.default_rng(inputs.partner_sample_seed)
        clients: list[FloatMatrix] = [local_rows]
        for partner in inputs.selected_partners:
            pool = inputs.partner_pools[partner]
            take = min(config.federated_partner_rows, len(pool))
            sample = pool[generator.choice(len(pool), size=take, replace=False)]
            clients.append((sample - standardisation.centre) / standardisation.scale)
        network = train_federated_autoencoder(clients, config, inputs.initialisation_seed, device)
    else:
        network = train_local_autoencoder(local_rows, config, inputs.initialisation_seed, device)
    return AutoencoderScorer(
        network=network,
        centre=standardisation.centre,
        scale=standardisation.scale,
        device=device,
    )
