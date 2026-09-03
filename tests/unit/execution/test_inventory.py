from __future__ import annotations

from pathlib import Path

import pandas as pd

from fedorbit.datasets.edge_iiotset.loader import EDGE_NETWORK_RELATIVE_PATH
from fedorbit.infrastructure.workspace import (
    RawDuplicateReportRequest,
    RawInventoryPersistenceRequest,
    RawInventoryRequest,
    inspect_raw_inventory,
    persist_raw_duplicate_report,
    persist_raw_inventory,
)
from fedorbit.types import DatasetId, stable_json


def test_edge_raw_inventory_records_file_identity(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    selected = raw / "Edge-IIoTset" / EDGE_NETWORK_RELATIVE_PATH
    selected.parent.mkdir(parents=True)
    selected.write_text("timestamp,label\n1,normal\n", encoding="utf-8")

    inventory = inspect_raw_inventory(RawInventoryRequest(DatasetId.EDGE_IIOTSET_NETWORK, raw))

    assert inventory.dataset == DatasetId.EDGE_IIOTSET_NETWORK
    assert inventory.files[0].relative_path == EDGE_NETWORK_RELATIVE_PATH
    assert inventory.files[0].columns == ("timestamp", "label")
    assert len(inventory.files[0].sha256) == 64
    assert len(inventory.fingerprint()) == 64
    assert '"dataset":"edge_iiotset_network"' in stable_json(inventory.serialization_payload())

    path = persist_raw_inventory(RawInventoryPersistenceRequest(inventory, tmp_path / "outputs"))
    assert path.is_file()
    assert path.name == "manifest.json"
    assert path.parent.name == DatasetId.EDGE_IIOTSET_NETWORK.value
    assert path.parent.parent.name == "inventories"


def test_raw_duplicate_report_records_exact_duplicate_rows(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    selected = raw / "Edge-IIoTset" / EDGE_NETWORK_RELATIVE_PATH
    selected.parent.mkdir(parents=True)
    selected.write_text("timestamp,label\n1,normal\n1,normal\n2,attack\n", encoding="utf-8")

    path = persist_raw_duplicate_report(
        RawDuplicateReportRequest(DatasetId.EDGE_IIOTSET_NETWORK, raw, tmp_path / "outputs")
    )

    report = pd.read_parquet(path)
    assert tuple(report.columns) == (
        "raw_row_sha256",
        "occurrence_count",
        "duplicate_row_count",
    )
    assert report["occurrence_count"].tolist() == [2]
    assert report["duplicate_row_count"].tolist() == [1]
