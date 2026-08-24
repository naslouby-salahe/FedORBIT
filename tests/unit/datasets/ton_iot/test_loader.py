from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.datasets.ton_iot.components import component_for
from fedorbit.datasets.ton_iot.loader import (
    TonIotLoaderError,
    discover_ton_iot_component_files,
    inspect_ton_iot_component_files,
)
from fedorbit.domain.enums import DatasetId


def _write_csv(path: Path, header: str = "ts,label,type") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n1,0,normal\n", encoding="utf-8")


def test_linux_process_discovery_excludes_disk_and_memory_components(tmp_path: Path) -> None:
    _write_csv(tmp_path / "linux" / "process" / "process.csv")
    _write_csv(tmp_path / "linux" / "disk" / "disk.csv")
    _write_csv(tmp_path / "linux" / "memory" / "memory.csv")
    component = component_for(DatasetId.TON_IOT_LINUX_PROCESS_HOST)
    paths = discover_ton_iot_component_files(tmp_path, component)
    assert [path.name for path in paths] == ["process.csv"]


def test_windows10_discovery_excludes_windows7_paths(tmp_path: Path) -> None:
    _write_csv(tmp_path / "windows" / "10" / "host.csv")
    _write_csv(tmp_path / "windows7" / "host.csv")
    component = component_for(DatasetId.TON_IOT_WINDOWS10_HOST)
    paths = discover_ton_iot_component_files(tmp_path, component)
    assert [path.name for path in paths] == ["host.csv"]


def test_component_inspection_records_hashes_and_rejects_schema_drift(tmp_path: Path) -> None:
    component = component_for(DatasetId.TON_IOT_NETWORK)
    _write_csv(tmp_path / "network" / "a.csv")
    _write_csv(tmp_path / "network" / "b.csv", "ts,label,type,extra")
    with pytest.raises(TonIotLoaderError):
        inspect_ton_iot_component_files(tmp_path, component)


def test_component_discovery_fails_when_raw_selection_is_absent(tmp_path: Path) -> None:
    component = component_for(DatasetId.TON_IOT_NETWORK)
    with pytest.raises(TonIotLoaderError):
        discover_ton_iot_component_files(tmp_path, component)
