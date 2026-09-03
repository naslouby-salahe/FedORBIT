from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.datasets.ton_iot.components import component_for
from fedorbit.datasets.ton_iot.loader import (
    TonIotLoaderError,
    discover_ton_iot_component_files,
    inspect_ton_iot_component_files,
)
from fedorbit.types import DatasetId


def _write_csv(path: Path, header: str = "ts,label,type") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n1,0,normal\n", encoding="utf-8")


def test_linux_process_discovery_selects_the_registered_component_table(tmp_path: Path) -> None:
    component = component_for(DatasetId.TON_IOT_LINUX_PROCESS_HOST)
    _write_csv(tmp_path / component.relative_path)
    paths = discover_ton_iot_component_files(tmp_path, component)
    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [component.relative_path]


def test_windows10_discovery_selects_the_registered_component_table(tmp_path: Path) -> None:
    component = component_for(DatasetId.TON_IOT_WINDOWS10_HOST)
    _write_csv(tmp_path / component.relative_path)
    paths = discover_ton_iot_component_files(tmp_path, component)
    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [component.relative_path]


def test_component_inspection_records_hashes_and_rejects_schema_drift(tmp_path: Path) -> None:
    component = component_for(DatasetId.TON_IOT_NETWORK)
    _write_csv(tmp_path / component.relative_path)
    inspected = inspect_ton_iot_component_files(tmp_path, component)
    assert inspected[0].columns == ("ts", "label", "type")


def test_component_discovery_fails_when_raw_selection_is_absent(tmp_path: Path) -> None:
    component = component_for(DatasetId.TON_IOT_NETWORK)
    with pytest.raises(TonIotLoaderError):
        discover_ton_iot_component_files(tmp_path, component)
