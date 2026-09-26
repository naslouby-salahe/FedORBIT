from __future__ import annotations

import numpy as np
import pytest
import torch

from fedorbit.config.loading import load_config
from fedorbit.config.models import AutoencoderDetectorConfig, FedorbitConfig
from fedorbit.detection.autoencoder import (
    AutoencoderScorer,
    build_network,
    train_federated_autoencoder,
    train_local_autoencoder,
)
from fedorbit.detection.metrics import auroc
from fedorbit.infrastructure.runtime import ExecutionDeviceUnavailableError, require_cuda
from fedorbit.study.runner import run_experiment
from fedorbit.types import DeviceName, ExperimentId, RandomSeed
from tests.support import as_list, as_object, config_document, experiment_document, synthetic_device

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def small_config() -> AutoencoderDetectorConfig:
    return load_config().detectors.autoencoder.model_copy(
        update={"training_steps": 200, "federated_rounds": 3, "federated_local_steps": 20}
    )


def structured_rows(seed: int, count: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    latent = generator.normal(size=(count, 2))
    mixing = np.random.default_rng(99).normal(size=(2, 8))
    return latent @ mixing + generator.normal(0.0, 0.05, size=(count, 8))


def test_training_is_deterministic_for_a_seed() -> None:
    device = require_cuda()
    rows = structured_rows(1, 200)
    config = small_config()
    first = train_local_autoencoder(rows, config, RandomSeed(5), device)
    second = train_local_autoencoder(rows, config, RandomSeed(5), device)
    scorer_a = AutoencoderScorer(first, np.zeros(8), np.ones(8), device)
    scorer_b = AutoencoderScorer(second, np.zeros(8), np.ones(8), device)
    assert np.array_equal(scorer_a.score(rows), scorer_b.score(rows))


def test_autoencoder_separates_off_manifold_rows() -> None:
    device = require_cuda()
    config = small_config()
    train = structured_rows(2, 400)
    network = train_local_autoencoder(train, config, RandomSeed(3), device)
    scorer = AutoencoderScorer(network, np.zeros(8), np.ones(8), device)
    benign = structured_rows(3, 200)
    off_manifold = np.random.default_rng(4).normal(0.0, 2.0, size=(200, 8))
    assert auroc(scorer.score(benign), scorer.score(off_manifold)) > 0.9


def test_federated_training_returns_a_working_network() -> None:
    device = require_cuda()
    config = small_config()
    clients = [structured_rows(10 + index, 100 + 50 * index) for index in range(3)]
    network = train_federated_autoencoder(clients, config, RandomSeed(7), device)
    scorer = AutoencoderScorer(network, np.zeros(8), np.ones(8), device)
    scores = scorer.score(structured_rows(20, 50))
    assert np.all(np.isfinite(scores))
    fresh = build_network(8, config, RandomSeed(7)).to(device)
    untrained = AutoencoderScorer(fresh, np.zeros(8), np.ones(8), device).score(
        structured_rows(20, 50)
    )
    assert scores.mean() < untrained.mean()


def test_cuda_requirement_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ExecutionDeviceUnavailableError):
        require_cuda()


def test_deep_detector_experiment_runs_on_synthetic_devices() -> None:
    document = config_document()
    deep = experiment_document(document, ExperimentId.DEEP_DETECTOR)
    deep["support_sizes"] = [30]
    deep["replicates"] = 2
    deep["contrasts"] = [
        {**as_object(contrast), "support_sizes": [30]} for contrast in as_list(deep["contrasts"])
    ]
    autoencoder = as_object(as_object(document["detectors"])["autoencoder"])
    autoencoder.update({"training_steps": 100, "federated_rounds": 2, "federated_local_steps": 10})
    document["experiments"] = [deep]
    config = FedorbitConfig.model_validate(document)
    devices = {
        DeviceName(f"device_{index}"): synthetic_device(f"device_{index}", index, 1.0 + index)
        for index in range(3)
    }
    records = run_experiment(config, config.experiment(ExperimentId.DEEP_DETECTOR), devices)
    conditions = {getattr(record, "condition", None) for record in records}
    assert {"local-standardised", "shared-marginals", "federated-averaging"} <= conditions
