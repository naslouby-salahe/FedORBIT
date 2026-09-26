from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedorbit.types import DatasetId, DeviceName, FloatMatrix


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DeviceData:
    dataset: DatasetId
    device: DeviceName
    support_pool: FloatMatrix
    test_benign: FloatMatrix
    test_attack: FloatMatrix
    purged_duplicate_rows: int

    def __post_init__(self) -> None:
        widths = {self.support_pool.shape[1], self.test_benign.shape[1], self.test_attack.shape[1]}
        if len(widths) != 1:
            raise DatasetValidationError(f"{self.device}: feature widths disagree")
        for name, rows in (
            ("support pool", self.support_pool),
            ("test benign", self.test_benign),
            ("test attack", self.test_attack),
        ):
            if not np.all(np.isfinite(rows)):
                raise DatasetValidationError(f"{self.device}: non-finite values in {name}")

    @property
    def is_evaluation_target(self) -> bool:
        return len(self.test_attack) > 0 and len(self.test_benign) > 0


def chronological_split(
    rows: FloatMatrix, support_fraction: float
) -> tuple[FloatMatrix, FloatMatrix]:
    boundary = int(len(rows) * support_fraction)
    return rows[:boundary], rows[boundary:]


def _row_keys(rows: FloatMatrix) -> set[bytes]:
    contiguous = np.ascontiguousarray(rows)
    return {contiguous[index].tobytes() for index in range(len(contiguous))}


def purge_exact_duplicates(
    support_pool: FloatMatrix, test_benign: FloatMatrix
) -> tuple[FloatMatrix, int]:
    support_keys = _row_keys(support_pool)
    contiguous = np.ascontiguousarray(test_benign)
    keep = np.fromiter(
        (contiguous[index].tobytes() not in support_keys for index in range(len(contiguous))),
        dtype=bool,
        count=len(contiguous),
    )
    return test_benign[keep], int((~keep).sum())
