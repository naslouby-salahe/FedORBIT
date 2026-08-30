from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.datasets.edge_iiotset.loader import (
    EDGE_NETWORK_RELATIVE_PATH,
    discover_edge_tabular_files,
    inspect_edge_tabular_files,
)


def _write_csv(path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n1,2\n", encoding="utf-8")


def test_edge_loader_selects_the_registered_network_table(tmp_path: Path) -> None:
    _write_csv(tmp_path / EDGE_NETWORK_RELATIVE_PATH, "a,b")
    paths = discover_edge_tabular_files(tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [EDGE_NETWORK_RELATIVE_PATH]


def test_edge_loader_records_hash_size_and_schema(tmp_path: Path) -> None:
    _write_csv(tmp_path / EDGE_NETWORK_RELATIVE_PATH, "a,b")
    inspected = inspect_edge_tabular_files(tmp_path)
    assert len(inspected) == 1
    assert inspected[0].columns == ("a", "b")
    assert inspected[0].byte_size > 0
    assert len(inspected[0].sha256) == 64


def test_edge_loader_requires_real_raw_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_edge_tabular_files(tmp_path / "missing")
