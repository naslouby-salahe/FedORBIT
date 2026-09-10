from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch

import fedorbit.infrastructure.preparation as execution
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.materialization import MaterializedClient
from fedorbit.infrastructure.preparation import persist_materialized_client
from fedorbit.infrastructure.workspace import (
    RawDuplicateReportRequest,
    RawInventoryPersistenceRequest,
    RawInventoryRequest,
    build_layout,
    inspect_raw_inventory,
    persist_raw_duplicate_report,
    persist_raw_inventory,
)
from fedorbit.types import DatasetId, OverwritePolicy, Split, stable_json

EDGE_NETWORK_RELATIVE_PATH = (
    load_fedorbit_config().scientific.datasets.edge_iiotset_network_relative_path
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


def test_materialized_client_persistence_writes_compressed_splits_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = build_layout(root=tmp_path)
    materialized = SimpleNamespace(
        dataset=DatasetId.EDGE_IIOTSET_NETWORK,
        splits=OrderedDict(
            (
                (
                    Split.TRAIN,
                    SimpleNamespace(
                        features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                        targets=torch.tensor([0, 1]),
                    ),
                ),
            )
        ),
        feature_names=("feature_a", "feature_b"),
        class_manifest=SimpleNamespace(class_names=("normal", "attack"), excluded_classes=()),
    )

    def model_dump(mode: str) -> dict[str, str]:
        del mode
        return {"dataset": "edge_iiotset_network"}

    manifest = SimpleNamespace(model_dump=model_dump)
    group = SimpleNamespace(
        concept=SimpleNamespace(value="DDoS"),
        native_class_indices=(1,),
        train_support=10,
        meta_support=4,
        source_eligible=True,
    )

    def fake_manifest(_materialized: MaterializedClient) -> Any:
        return manifest

    def fake_groups(_dataset: DatasetId, _materialized: MaterializedClient) -> Any:
        return (group,)

    monkeypatch.setattr(execution, "build_dataset_manifest", fake_manifest)
    monkeypatch.setattr(execution, "transfer_concept_groups", fake_groups)

    paths = persist_materialized_client(
        layout,
        cast(MaterializedClient, materialized),
        OverwritePolicy.REPLACE,
    )

    assert len(paths) == 1
    frame = pd.read_parquet(paths[0])
    assert tuple(frame.columns) == ("feature_a", "feature_b", "target")
    assert frame["target"].tolist() == [0, 1]
    assert (layout.preprocessing / "prepared" / "edge_iiotset_network" / "data.json").is_file()
    assert (layout.preprocessing / "features" / "edge_iiotset_network" / "data.json").is_file()
