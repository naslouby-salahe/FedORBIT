from __future__ import annotations

import json
from pathlib import Path

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.infrastructure.preparation import load_cached_preparation
from fedorbit.infrastructure.workspace import (
    RawDatasetInventory,
    RawDuplicateReportRequest,
    RawInventoryPersistenceRequest,
    RawInventoryRequest,
    build_layout,
    inspect_raw_inventory,
    persist_raw_duplicate_report,
    persist_raw_inventory,
)
from fedorbit.types import (
    AcquisitionSourceDescriptor,
    DatasetId,
    DatasetReleaseIdentity,
    DatasetReleaseName,
    LicenseUseNote,
    Rfc3339UtcTimestamp,
    Sha256Digest,
    StorageLayoutSegment,
    stable_json,
)

DATASET = DatasetId.EDGE_IIOTSET_NETWORK
RELATIVE_PATH = load_fedorbit_config().scientific.datasets.edge_iiotset_network_relative_path
RELEASE_IDENTITY = DatasetReleaseIdentity(
    release=DatasetReleaseName("Edge-IIoTset-2022-07-01"),
    acquisition_source=AcquisitionSourceDescriptor("public mirror (documented URL)"),
    acquisition_timestamp=Rfc3339UtcTimestamp("2024-05-06T09:30:00Z"),
    license_note=LicenseUseNote("research use; cite the dataset publication"),
)


def _raw_root(tmp_path: Path, content: str) -> Path:
    raw = tmp_path / "raw"
    selected = raw / "Edge-IIoTset" / RELATIVE_PATH
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text(content, encoding="utf-8")
    return raw


def _inventory(tmp_path: Path, content: str, release: bool = True) -> RawDatasetInventory:
    return inspect_raw_inventory(
        RawInventoryRequest(
            DATASET,
            _raw_root(tmp_path, content),
            release_identity=RELEASE_IDENTITY if release else None,
        )
    )


def test_raw_inventory_records_the_registered_identity_fields(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, "timestamp,label\n1,normal\n")
    assert inventory.component == DATASET.value
    assert inventory.release_identity == RELEASE_IDENTITY
    payload = stable_json(inventory.serialization_payload())
    assert f'"dataset":"{DATASET.value}"' in payload
    assert '"release":"Edge-IIoTset-2022-07-01"' in payload
    assert '"acquisition_source":"public mirror (documented URL)"' in payload
    assert '"acquisition_timestamp":"2024-05-06T09:30:00Z"' in payload
    assert '"license_note":"research use; cite the dataset publication"' in payload
    assert f'"component":"{DATASET.value}"' in payload
    assert RELATIVE_PATH in payload


def test_persisted_raw_inventory_carries_the_registered_fields(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    inventory = _inventory(tmp_path, "timestamp,label\n1,normal\n")
    manifest_path = persist_raw_inventory(
        RawInventoryPersistenceRequest(
            inventory,
            layout.preprocessing,
            layout.staging,
        )
    )
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["dataset"] == DATASET.value
    assert persisted["component"] == DATASET.value
    assert persisted["release_identity"]["release"] == "Edge-IIoTset-2022-07-01"
    assert persisted["release_identity"]["license_note"].startswith("research use")
    assert persisted["files"][0]["relative_path"] == RELATIVE_PATH
    assert len(persisted["files"][0]["sha256"]) == 64
    assert not tuple(layout.staging.rglob("*.tmp-*"))


def test_changed_raw_checksum_produces_a_new_lineage(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    original = _inventory(tmp_path, "timestamp,label\n1,normal\n")
    changed = _inventory(tmp_path, "timestamp,label\n1,normal\n2,attack\n")
    assert original.files[0].sha256 != changed.files[0].sha256
    assert original.fingerprint() != changed.fingerprint()
    contract = Sha256Digest("c" * 64)
    record_path = (
        layout.preprocessing / StorageLayoutSegment.METADATA / DATASET.value / "preparation.json"
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "raw_inventory_fingerprint": original.fingerprint(),
                "preparation_contract_sha256": contract,
                "state": "resource_blocked",
                "resource_block_reason": "recorded",
            }
        ),
        encoding="utf-8",
    )
    validation = layout.preprocessing / StorageLayoutSegment.VALIDATION / DATASET.value
    validation.mkdir(parents=True, exist_ok=True)
    (validation / "validation.json").write_text("{}", encoding="utf-8")
    (validation / "duplicates.parquet").write_bytes(b"")
    assert load_cached_preparation(layout, changed.fingerprint(), contract, DATASET) is None


def test_release_identity_is_absent_until_the_config_section_is_wired(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, "timestamp,label\n1,normal\n", release=False)
    assert inventory.release_identity is None
    assert '"release_identity":null' in stable_json(inventory.serialization_payload())


def test_duplicate_report_is_written_through_staging(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    raw = _raw_root(tmp_path, "timestamp,label\n1,normal\n1,normal\n2,attack\n")
    path = persist_raw_duplicate_report(
        RawDuplicateReportRequest(DATASET, raw, layout.preprocessing, layout.staging)
    )
    assert path.is_file()
    assert path.parent.name == DATASET.value
    assert not tuple(layout.staging.rglob("*.tmp-*"))
