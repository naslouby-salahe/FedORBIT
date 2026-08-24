from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.datasets.edge_iiotset.loader import (
    EdgeLoaderError,
    discover_edge_tabular_files,
    inspect_edge_tabular_files,
)


def _write_csv(path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n1,2\n", encoding="utf-8")


def test_edge_loader_discovers_csvs_in_bytewise_relative_path_order(tmp_path: Path) -> None:
    _write_csv(tmp_path / "z" / "part.csv", "a,b")
    _write_csv(tmp_path / "a" / "part.csv", "a,b")
    paths = discover_edge_tabular_files(tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in paths] == ["a/part.csv", "z/part.csv"]


def test_edge_loader_records_hash_size_and_schema(tmp_path: Path) -> None:
    _write_csv(tmp_path / "part.csv", "a,b")
    inspected = inspect_edge_tabular_files(tmp_path)
    assert len(inspected) == 1
    assert inspected[0].columns == ("a", "b")
    assert inspected[0].byte_size > 0
    assert len(inspected[0].sha256) == 64


def test_edge_loader_rejects_cross_file_feature_set_mismatch(tmp_path: Path) -> None:
    _write_csv(tmp_path / "first.csv", "a,b")
    _write_csv(tmp_path / "second.csv", "a,c")
    with pytest.raises(EdgeLoaderError):
        inspect_edge_tabular_files(tmp_path)


def test_edge_loader_requires_real_raw_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_edge_tabular_files(tmp_path / "missing")
