from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch

import fedorbit.infrastructure.preparation as execution
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.common import (
    ChronologyValidationState,
    DatasetObservation,
    DatasetObservationPersistenceRequest,
    EventTimeInspection,
    persist_dataset_observation,
)
from fedorbit.datasets.materialization import MaterializedClient, PreprocessingFitAccess
from fedorbit.infrastructure.preparation import (
    load_cached_preparation,
    persist_materialized_client,
    persist_preparation_record,
)
from fedorbit.infrastructure.workspace import (
    RawDuplicateReportRequest,
    RawInventoryPersistenceRequest,
    RawInventoryRequest,
    build_layout,
    inspect_raw_inventory,
    persist_raw_duplicate_report,
    persist_raw_inventory,
)
from fedorbit.types import (
    ArtifactPath,
    DatasetId,
    DatasetPreprocessingState,
    OverwritePolicy,
    PreprocessingFitStage,
    ResourceLimitReason,
    Sha256Digest,
    Split,
    TabularColumnName,
    ValidationReason,
    stable_json,
)

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


@pytest.mark.parametrize(
    ("state", "requires_prepared", "resource_reason"),
    (
        (DatasetPreprocessingState.MATERIALIZED, True, None),
        (DatasetPreprocessingState.INVALID, False, None),
        (
            DatasetPreprocessingState.RESOURCE_BLOCKED,
            False,
            ResourceLimitReason("safe materialization budget exceeded"),
        ),
    ),
)
def test_cached_preparation_validates_terminal_states(
    tmp_path: Path,
    state: DatasetPreprocessingState,
    requires_prepared: bool,
    resource_reason: ResourceLimitReason | None,
) -> None:
    layout = build_layout(root=tmp_path)
    dataset = DatasetId.TON_IOT_WINDOWS10_HOST
    observation = DatasetObservation(
        dataset=dataset,
        row_count=1,
        observed_columns=(
            TabularColumnName("ts"),
            TabularColumnName("label"),
            TabularColumnName("type"),
        ),
        local_class_counts=(),
        binary_label_counts=(),
        inconsistent_binary_label_rows=0,
        event_time=EventTimeInspection(
            field=TabularColumnName("ts"),
            observed_row_count=1,
            timestamp_pattern_row_count=1,
            unusable_row_count=1 if state == DatasetPreprocessingState.INVALID else 0,
            state=(
                ChronologyValidationState.UNPARSEABLE_EVENT_TIME
                if state == DatasetPreprocessingState.INVALID
                else ChronologyValidationState.VALID
            ),
            reason=ValidationReason("fixture"),
        ),
    )
    validation_path = persist_dataset_observation(
        DatasetObservationPersistenceRequest(observation, layout.preprocessing)
    )
    duplicate_path = validation_path.parent / "duplicates.parquet"
    pd.DataFrame().to_parquet(duplicate_path, index=False)
    raw_fingerprint = Sha256Digest("a" * 64)
    contract_fingerprint = Sha256Digest("b" * 64)
    if requires_prepared:
        for path in (
            layout.preprocessing / "prepared" / dataset.value / "data.json",
            layout.preprocessing / "prepared" / dataset.value / "client.pt",
            layout.preprocessing / "features" / dataset.value / "data.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    persist_preparation_record(
        layout,
        raw_fingerprint,
        contract_fingerprint,
        observation,
        resource_reason,
    )

    cached = load_cached_preparation(layout, raw_fingerprint, contract_fingerprint, dataset)

    assert cached is not None
    (
        cached_observation,
        cached_validation,
        cached_duplicate,
        cached_reason,
        cached_invalid_reason,
    ) = cached
    assert cached_observation == observation
    assert cached_validation == ArtifactPath(validation_path)
    assert cached_duplicate == ArtifactPath(duplicate_path)
    assert cached_reason == resource_reason
    assert cached_invalid_reason is None


def test_cached_preparation_preserves_materialization_invalid_reason(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    dataset = DatasetId.TON_IOT_WINDOWS10_HOST
    observation = DatasetObservation(
        dataset=dataset,
        row_count=1,
        observed_columns=(TabularColumnName("ts"),),
        local_class_counts=(),
        binary_label_counts=(),
        inconsistent_binary_label_rows=0,
        event_time=EventTimeInspection(
            field=TabularColumnName("ts"),
            observed_row_count=1,
            timestamp_pattern_row_count=1,
            unusable_row_count=0,
            state=ChronologyValidationState.VALID,
            reason=ValidationReason("fixture"),
        ),
    )
    validation_path = persist_dataset_observation(
        DatasetObservationPersistenceRequest(observation, layout.preprocessing)
    )
    duplicate_path = validation_path.parent / "duplicates.parquet"
    pd.DataFrame().to_parquet(duplicate_path, index=False)
    raw_fingerprint = Sha256Digest("a" * 64)
    contract_fingerprint = Sha256Digest("b" * 64)
    invalid_reason = ValidationReason("conflicting duplicate labels")
    persist_preparation_record(
        layout,
        raw_fingerprint,
        contract_fingerprint,
        observation,
        None,
        invalid_reason,
    )

    cached = load_cached_preparation(layout, raw_fingerprint, contract_fingerprint, dataset)

    assert cached is not None
    assert cached[-1] == invalid_reason


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
        preprocessing_fit_accesses=(
            PreprocessingFitAccess(PreprocessingFitStage.FEATURE_QUALITY, (Split.TRAIN,)),
        ),
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
        confirm_support=4,
        test_support=4,
        target_eligible=True,
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
    features_path = layout.preprocessing / "features" / "edge_iiotset_network" / "data.json"
    assert features_path.is_file()
    feature_payload = json.loads(features_path.read_text())
    transfer_eligibility = feature_payload["transfer_eligibility"]
    assert transfer_eligibility == [
        {
            "concept": "DDoS",
            "native_class_indices": [1],
            "train_support": 10,
            "meta_support": 4,
            "source_eligible": True,
            "confirm_support": 4,
            "test_support": 4,
            "target_eligible": True,
        }
    ]
    assert feature_payload["preprocessing_fit_accesses"] == [
        {
            "stage": PreprocessingFitStage.FEATURE_QUALITY.value,
            "accessed_splits": [Split.TRAIN.value],
        }
    ]
