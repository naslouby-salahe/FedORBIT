from __future__ import annotations

from pathlib import Path

from fedorbit.datasets.edge_iiotset.loader import EDGE_NETWORK_RELATIVE_PATH
from fedorbit.domain.enums import DatasetId
from fedorbit.domain.serialization import stable_json
from fedorbit.execution.inventory import (
    RawInventoryPersistenceRequest,
    RawInventoryRequest,
    inspect_raw_inventory,
    persist_raw_inventory,
)


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
    assert path.parent.name == "inventories"
