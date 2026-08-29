from __future__ import annotations

from pathlib import Path

from fedorbit.datasets.materialization import (
    TabularMaterializationRequest,
    materialize_tabular_rows,
)


def test_materialization_preserves_schema_and_row_order(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("time,label\n1,normal\n", encoding="utf-8")
    second.write_text("time,label\n2,attack\n", encoding="utf-8")

    materialized = materialize_tabular_rows(TabularMaterializationRequest((first, second)))

    assert materialized.columns == ("time", "label")
    assert tuple(row.values["time"] for row in materialized.rows) == ("1", "2")
