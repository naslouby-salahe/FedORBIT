from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from fedorbit.infrastructure.storage import (
    PayloadIntegrityError,
    payload_sha256,
    promote_json,
    promote_staged,
    promote_table_payload,
    read_table_payload,
    sort_table,
    stage_bytes,
    staged_payload_path,
    verify_payload_bytes,
)
from fedorbit.types import (
    Sha256Digest,
    TabularColumnName,
    stable_json,
)

SORT_COLUMNS = (TabularColumnName("seed"), TabularColumnName("method"))
ZSTD_FRAME_MAGIC = b"\x28\xb5\x2f\xfd"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed": [3, 1, 2],
            "method": ["b", "a", "c"],
            "value": [0.3, 0.1, 0.2],
        }
    )


def test_tabular_payload_imposes_the_declared_scientific_row_order(tmp_path: Path) -> None:
    staging = tmp_path / "cache" / "staging"
    first = promote_table_payload(
        tmp_path / "payloads" / "data.parquet",
        _frame(),
        SORT_COLUMNS,
        staging,
    )
    shuffled = cast(pd.DataFrame, _frame().iloc[[2, 0, 1]])
    second = promote_table_payload(
        tmp_path / "payloads" / "other.parquet",
        shuffled,
        SORT_COLUMNS,
        staging,
    )
    assert first.read_bytes() == second.read_bytes()
    assert read_table_payload(first, SORT_COLUMNS)["seed"].tolist() == [1, 2, 3]


def test_tabular_row_group_layout_never_defines_scientific_order(tmp_path: Path) -> None:
    payload_path = tmp_path / "payloads" / "data.parquet"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    shuffled: pd.DataFrame = _frame().iloc[[2, 0, 1]].reset_index(drop=True)
    shuffled.to_parquet(
        payload_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
        row_group_size=1,
    )
    physical = pd.read_parquet(payload_path)
    assert physical["seed"].tolist() == [2, 3, 1]
    written = payload_path.read_bytes()
    assert written[:4] == b"PAR1"
    assert ZSTD_FRAME_MAGIC in written
    assert b"parquet-cpp-arrow" in written
    ordered = read_table_payload(payload_path, SORT_COLUMNS)
    assert ordered["seed"].tolist() == [1, 2, 3]
    assert ordered["method"].tolist() == ["a", "c", "b"]
    with pytest.raises(ValueError, match="scientific sort columns"):
        read_table_payload(payload_path, (TabularColumnName("absent"),))


def test_sort_table_rejects_payloads_without_declared_order() -> None:
    with pytest.raises(ValueError, match="scientific sort columns"):
        sort_table(_frame(), ())
    assert sort_table(_frame(), SORT_COLUMNS)["seed"].tolist() == [1, 2, 3]


def test_payload_verification_rejects_a_mismatched_digest() -> None:
    payload = b'{"a":1}\n'
    digest = verify_payload_bytes(payload_sha256(payload), payload)
    assert len(digest) == 64
    with pytest.raises(PayloadIntegrityError, match="payload checksum mismatch"):
        verify_payload_bytes(Sha256Digest("0" * 64), payload)


def test_staged_writes_never_appear_in_the_active_namespace_until_promotion(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "cache" / "staging"
    destination = tmp_path / "payloads" / "prepared" / "data.json"
    staged = stage_bytes(destination, b'{"state":"completed"}\n', staging)
    assert staged.is_file()
    assert staged_payload_path(destination, staging) == staged
    assert not destination.exists()
    promote_staged(staged, destination)
    assert destination.read_bytes() == b'{"state":"completed"}\n'
    assert not staged.is_file()


def test_promoted_json_payload_is_serialized_with_a_trailing_line_terminator(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "cache" / "staging"
    destination = tmp_path / "payloads" / "features" / "data.json"
    promote_json(destination, {"b": 1, "a": 2}, staging)
    text = destination.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == {"a": 2, "b": 1}
    assert text == stable_json({"a": 2, "b": 1}) + "\n"
    assert not tuple(staging.glob("*.tmp-*"))
