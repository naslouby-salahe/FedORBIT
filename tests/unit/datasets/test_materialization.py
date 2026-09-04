from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.datasets.materialization import MaterializationError, require_safe_memory_budget
from fedorbit.types import DatasetId


class _FakeVirtualMemory:
    def __init__(self, available: int) -> None:
        self.available = available


def _write_file_of_size(path: Path, size_bytes: int) -> Path:
    path.write_bytes(b"0" * size_bytes)
    return path


def test_memory_guard_refuses_when_estimated_peak_exceeds_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big_file = _write_file_of_size(tmp_path / "big.csv", 10_000_000)
    monkeypatch.setattr(
        "fedorbit.datasets.materialization.psutil.virtual_memory",
        lambda: _FakeVirtualMemory(available=1_000_000),
    )
    with pytest.raises(MaterializationError):
        require_safe_memory_budget(DatasetId.TON_IOT_NETWORK, (big_file,))


def test_memory_guard_passes_when_estimated_peak_fits_the_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_file = _write_file_of_size(tmp_path / "small.csv", 1_000)
    monkeypatch.setattr(
        "fedorbit.datasets.materialization.psutil.virtual_memory",
        lambda: _FakeVirtualMemory(available=10_000_000_000),
    )
    require_safe_memory_budget(DatasetId.TON_IOT_WINDOWS10_HOST, (small_file,))
