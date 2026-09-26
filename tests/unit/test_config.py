from __future__ import annotations

import pytest
from pydantic import ValidationError

from fedorbit.config.loading import DEFAULT_CONFIG_PATH, config_digest, load_config, section_digest
from fedorbit.config.models import ConditionSpec, FedorbitConfig
from fedorbit.types import (
    ContrastFamily,
    DatasetId,
    DetectorKind,
    ExperimentId,
    PartnerPolicy,
    ShareChannel,
)
from tests.support import as_list, as_object, config_document, experiment_document


def test_repository_configuration_declares_every_experiment_once() -> None:
    config = load_config()
    assert {experiment.id for experiment in config.experiments} == set(ExperimentId)
    assert DEFAULT_CONFIG_PATH.is_file()


def test_primary_contrasts_are_registered_before_execution() -> None:
    config = load_config()
    primary = [
        contrast
        for experiment in config.experiments
        for contrast in experiment.contrasts
        if contrast.family is ContrastFamily.PRIMARY
    ]
    assert primary
    assert all(contrast.support_sizes == (30,) for contrast in primary)


def test_datasets_map_to_expected_detectors() -> None:
    config = load_config()
    assert config.experiment(ExperimentId.DEEP_DETECTOR).detector is DetectorKind.AUTOENCODER
    assert config.experiment(ExperimentId.SIMULATED_BOUNDARY).dataset is DatasetId.GOTHAM


def test_unknown_fields_are_rejected() -> None:
    data = config_document()
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(data)


def test_configuration_is_immutable() -> None:
    config = load_config()
    with pytest.raises(ValidationError):
        type(config).__setattr__(config, "base_seed", 1)


def test_contrast_must_reference_declared_conditions() -> None:
    data = config_document()
    contrast = as_object(
        as_list(experiment_document(data, ExperimentId.COLD_START_LADDER)["contrasts"])[0]
    )
    contrast["treatment"] = "undeclared"
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(data)


def test_contrast_support_sizes_must_be_declared() -> None:
    data = config_document()
    contrast = as_object(
        as_list(experiment_document(data, ExperimentId.COLD_START_LADDER)["contrasts"])[0]
    )
    contrast["support_sizes"] = [7]
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(data)


def test_stratified_contrast_needs_reference_condition() -> None:
    data = config_document()
    del as_object(experiment_document(data, ExperimentId.COLD_START_LADDER)["conditions"])[
        "local-full-support"
    ]
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(data)


def test_duplicate_experiments_are_rejected() -> None:
    data = config_document()
    experiments = as_list(data["experiments"])
    experiments.append(experiments[0])
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(data)


def test_covariance_channel_is_gaussian_only() -> None:
    data = config_document()
    deep = experiment_document(data, ExperimentId.DEEP_DETECTOR)
    as_object(deep["conditions"])["shared-covariance"] = {
        "channel": "shared-covariance",
        "partner_policy": "all",
        "dose_rows": 10,
    }
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(data)


def test_federated_averaging_needs_the_autoencoder() -> None:
    data = config_document()
    as_object(experiment_document(data, ExperimentId.COLD_START_LADDER)["conditions"])["fedavg"] = {
        "channel": "federated-averaging",
        "partner_policy": "all",
    }
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(data)


def test_operating_point_needs_the_gaussian_detector() -> None:
    data = config_document()
    experiment_document(data, ExperimentId.DEEP_DETECTOR)["report_operating_point"] = True
    with pytest.raises(ValidationError):
        FedorbitConfig.model_validate(data)


@pytest.mark.parametrize(
    "payload",
    [
        {"channel": ShareChannel.SHARED_MARGINALS},
        {"channel": ShareChannel.LOCAL_STANDARDISED, "partner_policy": PartnerPolicy.ALL},
        {"channel": ShareChannel.SHARED_MARGINALS, "partner_policy": PartnerPolicy.RANDOM},
        {
            "channel": ShareChannel.SHARED_MARGINALS,
            "partner_policy": PartnerPolicy.ALL,
            "partner_count": 2,
        },
        {"channel": ShareChannel.SHARED_COVARIANCE, "partner_policy": PartnerPolicy.ALL},
        {
            "channel": ShareChannel.SHARED_MARGINALS,
            "partner_policy": PartnerPolicy.ALL,
            "dose_rows": 5,
        },
    ],
)
def test_inconsistent_conditions_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ConditionSpec.model_validate(payload)


def test_digest_changes_with_any_scientific_value() -> None:
    config = load_config()
    data = config_document()
    as_object(data["statistics"])["alpha"] = 0.01
    assert config_digest(FedorbitConfig.model_validate(data)) != config_digest(config)
    assert section_digest(config.detectors) == section_digest(config.detectors)
    assert section_digest(config.detectors) != section_digest(config.operating_point)
