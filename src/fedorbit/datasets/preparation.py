from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.device_data import DatasetValidationError, DeviceData
from fedorbit.datasets.gotham import device_name, gotham_source_files, load_gotham_device
from fedorbit.datasets.nbaiot import load_nbaiot_device, nbaiot_source_files
from fedorbit.infrastructure.artifacts import InputDigest, file_digest
from fedorbit.types import DatasetId, DeviceName, FloatMatrix

SUPPORT_KEY = "support_pool"
BENIGN_KEY = "test_benign"
ATTACK_KEY = "test_attack"
PURGED_KEY = "purged_duplicate_rows"


def device_sources(
    config: FedorbitConfig, raw_directory: Path, dataset: DatasetId
) -> dict[DeviceName, tuple[Path, ...]]:
    if dataset is DatasetId.NBAIOT:
        settings = config.datasets.nbaiot
        return {
            device: nbaiot_source_files(raw_directory / settings.relative_directory / device)
            for device in settings.devices
        }
    settings_gotham = config.datasets.gotham
    return {
        device_name(path): (path,)
        for path in gotham_source_files(raw_directory / settings_gotham.relative_directory)
    }


def source_digests(raw_directory: Path, files: tuple[Path, ...]) -> tuple[InputDigest, ...]:
    return tuple(
        InputDigest(name=path.relative_to(raw_directory).as_posix(), sha256=file_digest(path))
        for path in files
    )


def build_device(
    config: FedorbitConfig,
    raw_directory: Path,
    dataset: DatasetId,
    device: DeviceName,
    files: tuple[Path, ...],
) -> DeviceData:
    if dataset is DatasetId.NBAIOT:
        return load_nbaiot_device(raw_directory, device, config.datasets.nbaiot, config.base_seed)
    return load_gotham_device(files[0], config.datasets.gotham)


def serialise_device(data: DeviceData) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        support_pool=data.support_pool,
        test_benign=data.test_benign,
        test_attack=data.test_attack,
        purged_duplicate_rows=np.array([data.purged_duplicate_rows]),
    )
    return buffer.getvalue()


def parse_device(dataset: DatasetId, device: DeviceName, payload: bytes) -> DeviceData:
    with np.load(io.BytesIO(payload)) as archive:
        matrices: dict[str, FloatMatrix] = {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in (SUPPORT_KEY, BENIGN_KEY, ATTACK_KEY)
        }
        purged = int(archive[PURGED_KEY][0])
    if any(matrix.ndim != 2 for matrix in matrices.values()):
        raise DatasetValidationError(f"{device}: prepared arrays must be matrices")
    return DeviceData(
        dataset=dataset,
        device=device,
        support_pool=matrices[SUPPORT_KEY],
        test_benign=matrices[BENIGN_KEY],
        test_attack=matrices[ATTACK_KEY],
        purged_duplicate_rows=purged,
    )
