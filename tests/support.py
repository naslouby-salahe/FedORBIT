from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import numpy as np

from fedorbit.config.loading import load_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.device_data import DeviceData
from fedorbit.datasets.nbaiot import ATTACK_DIRECTORIES, BENIGN_FILE, FEATURE_COUNT
from fedorbit.types import DatasetId, DeviceName, ExperimentId, FloatMatrix

GAFGYT_FILES = ("combo", "junk", "scan", "tcp", "udp")
MIRAI_FILES = ("ack", "scan", "syn", "udp", "udpplain")
FIXTURE_DEVICES = (
    DeviceName("device_alpha"),
    DeviceName("device_beta"),
    DeviceName("device_gamma"),
)
FIXTURE_FEATURES = 6

JsonObject = dict[str, object]


def as_object(value: object) -> JsonObject:
    return cast(JsonObject, value)


def as_list(value: object) -> list[object]:
    return cast(list[object], value)


def config_document() -> JsonObject:
    return as_object(copy.deepcopy(load_config().model_dump(mode="json")))


def experiment_document(document: JsonObject, identifier: ExperimentId) -> JsonObject:
    for entry in as_list(document["experiments"]):
        if as_object(entry)["id"] == identifier.value:
            return as_object(entry)
    raise KeyError(identifier)


def feature_header(width: int) -> str:
    return ",".join(f"feature_{index}" for index in range(width))


def write_matrix(path: Path, matrix: FloatMatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",", header=feature_header(matrix.shape[1]), comments="")


def synthetic_nbaiot_directory(root: Path, devices: tuple[DeviceName, ...], seed: int) -> Path:
    generator = np.random.default_rng(seed)
    dataset_root = root / "N-BaIoT"
    for position, device in enumerate(devices):
        scale = 1.0 + position
        benign = np.abs(generator.normal(5.0 * scale, scale, size=(900, FEATURE_COUNT)))
        write_matrix(dataset_root / device / BENIGN_FILE, benign)
        for directory, names in zip(ATTACK_DIRECTORIES, (GAFGYT_FILES, MIRAI_FILES), strict=True):
            for name in names:
                attack = np.abs(
                    generator.normal(12.0 * scale, 3.0 * scale, size=(120, FEATURE_COUNT))
                )
                write_matrix(dataset_root / device / directory / f"{name}.csv", attack)
    return root


def synthetic_config(devices: tuple[DeviceName, ...]) -> FedorbitConfig:
    document = config_document()
    nbaiot = as_object(as_object(document["datasets"])["nbaiot"])
    nbaiot["devices"] = list(devices)
    nbaiot["attack_rows_per_file"] = 40
    cold_start = experiment_document(document, ExperimentId.COLD_START_LADDER)
    support_sizes = [30, 60]
    cold_start["support_sizes"] = support_sizes
    cold_start["replicates"] = 4
    kept: list[object] = []
    for entry in as_list(cold_start["contrasts"]):
        contrast = as_object(entry)
        sizes = [size for size in as_list(contrast["support_sizes"]) if size in support_sizes]
        if sizes:
            kept.append({**contrast, "support_sizes": sizes})
    cold_start["contrasts"] = kept
    document["experiments"] = [cold_start]
    as_object(document["statistics"])["bootstrap_resamples"] = 100
    return FedorbitConfig.model_validate(document)


def synthetic_device(
    name: str,
    seed: int,
    scale: float,
    support_rows: int = 400,
    features: int = FIXTURE_FEATURES,
    attack_shift: float = 6.0,
) -> DeviceData:
    generator = np.random.default_rng(seed)
    return DeviceData(
        dataset=DatasetId.NBAIOT,
        device=DeviceName(name),
        support_pool=generator.normal(0.0, scale, size=(support_rows, features)),
        test_benign=generator.normal(0.0, scale, size=(300, features)),
        test_attack=generator.normal(attack_shift * scale, scale, size=(300, features)),
        purged_duplicate_rows=0,
    )
