from __future__ import annotations

from pathlib import Path

from fedorbit.domain.enums import DatasetId
from fedorbit.execution.inventory import RawInventoryRequest, inspect_raw_inventory


def test_edge_raw_inventory_records_file_identity(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "traffic.csv").write_text("timestamp,label\n1,normal\n", encoding="utf-8")

    inventory = inspect_raw_inventory(RawInventoryRequest(DatasetId.EDGE_IIOTSET_NETWORK, raw))

    assert inventory.dataset == DatasetId.EDGE_IIOTSET_NETWORK
    assert inventory.files[0].relative_path == "traffic.csv"
    assert inventory.files[0].columns == ("timestamp", "label")
    assert len(inventory.files[0].sha256) == 64
    assert len(inventory.fingerprint()) == 64
