from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fedorbit.config.models import NBaIoTConfig
from fedorbit.datasets.device_data import (
    DatasetValidationError,
    DeviceData,
    chronological_split,
    purge_exact_duplicates,
)
from fedorbit.infrastructure.runtime import derive_seed
from fedorbit.types import DatasetId, DeviceName, FloatMatrix, RandomPurpose, RandomSeed

BENIGN_FILE = "benign_traffic.csv"
ATTACK_DIRECTORIES = ("gafgyt_attacks", "mirai_attacks")
FEATURE_COUNT = 115


def signed_log(rows: FloatMatrix) -> FloatMatrix:
    return np.sign(rows) * np.log1p(np.abs(rows))


def nbaiot_source_files(device_directory: Path) -> tuple[Path, ...]:
    files = [device_directory / BENIGN_FILE]
    for attack_directory in ATTACK_DIRECTORIES:
        files.extend(sorted((device_directory / attack_directory).glob("*.csv")))
    missing = [path for path in files if not path.is_file()]
    if missing or len(files) == 1:
        raise DatasetValidationError(f"{device_directory.name}: incomplete N-BaIoT files")
    return tuple(files)


def _read_features(path: Path) -> FloatMatrix:
    frame = pd.read_csv(path, dtype=np.float64)
    if frame.shape[1] != FEATURE_COUNT:
        raise DatasetValidationError(f"{path}: expected {FEATURE_COUNT} features")
    return frame.to_numpy(dtype=np.float64)


def load_nbaiot_device(
    raw_directory: Path, device: DeviceName, config: NBaIoTConfig, base_seed: RandomSeed
) -> DeviceData:
    device_directory = raw_directory / config.relative_directory / device
    files = nbaiot_source_files(device_directory)
    benign = signed_log(_read_features(files[0]))
    support_pool, test_benign = chronological_split(benign, config.benign_support_fraction)
    test_benign, purged = purge_exact_duplicates(support_pool, test_benign)
    attack_blocks: list[FloatMatrix] = []
    for path in files[1:]:
        rows = signed_log(_read_features(path))
        relative = path.relative_to(device_directory).as_posix()
        seed = derive_seed(
            base_seed, RandomPurpose.ATTACK_SAMPLE, DatasetId.NBAIOT, device, relative
        )
        generator = np.random.default_rng(seed)
        take = min(config.attack_rows_per_file, len(rows))
        attack_blocks.append(rows[generator.choice(len(rows), size=take, replace=False)])
    return DeviceData(
        dataset=DatasetId.NBAIOT,
        device=device,
        support_pool=support_pool,
        test_benign=test_benign,
        test_attack=np.vstack(attack_blocks),
        purged_duplicate_rows=purged,
    )
