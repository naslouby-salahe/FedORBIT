from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

import fedorbit.infrastructure.execution as execution
from fedorbit.datasets.materialization import DatasetId
from fedorbit.infrastructure.evidence import TableScalar
from fedorbit.infrastructure.execution import ArtifactStore, training_protocol_rows
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.learning.pilot import HOST_DATASETS, NETWORK_DATASETS
from fedorbit.learning.training import SelectedHyperparameters


def test_training_protocol_rows_report_selected_hyperparameters_per_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)

    selections = {
        DatasetId.EDGE_IIOTSET_NETWORK: SelectedHyperparameters(1e-3, 1e-4, 0.1),
        DatasetId.TON_IOT_NETWORK: SelectedHyperparameters(3e-4, 1e-5, 0.2),
        DatasetId.TON_IOT_WINDOWS10_HOST: SelectedHyperparameters(1e-2, 1e-3, 0.3),
        DatasetId.TON_IOT_LINUX_PROCESS_HOST: SelectedHyperparameters(1e-2, 1e-3, 0.3),
    }

    def fake_selected(
        store_arg: ArtifactStore, dataset: DatasetId
    ) -> SelectedHyperparameters | None:
        del store_arg
        return selections.get(dataset)

    monkeypatch.setattr(execution, "_pilot_selected_hyperparameters", fake_selected)

    rows = training_protocol_rows(store)
    models: dict[str, Mapping[str, TableScalar]] = {str(row["model"]): row for row in rows}
    assert set(models) == {dataset.value for dataset in selections}

    network_row = models[DatasetId.EDGE_IIOTSET_NETWORK.value]
    assert network_row["architecture"] == "256-128-64 MLP (NetworkFlowClassifier)"
    assert network_row["normalization"] == "LayerNorm"
    assert network_row["activation"] == "GELU"
    assert network_row["initialization"] == "Xavier uniform"
    assert network_row["optimizer"] == "AdamW"
    assert network_row["selected_learning_rate"] == pytest.approx(1e-3)
    assert network_row["selected_weight_decay"] == pytest.approx(1e-4)
    assert network_row["selected_dropout"] == pytest.approx(0.1)

    host_row = models[DatasetId.TON_IOT_WINDOWS10_HOST.value]
    assert host_row["architecture"] == "192-96-48 MLP (HostClassifier)"
    assert host_row["normalization"] == "BatchNorm1d"
    assert host_row["activation"] == "ReLU"
    assert host_row["initialization"] == "Kaiming uniform"
    assert isinstance(host_row["stopping_rule"], str)
    assert "epochs" in host_row["stopping_rule"]

    for dataset in NETWORK_DATASETS:
        assert dataset.value in models
    for dataset in HOST_DATASETS:
        assert dataset.value in models


def test_training_protocol_rows_skips_clients_without_pilot_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)

    def no_selection(
        store_arg: ArtifactStore, dataset: DatasetId
    ) -> SelectedHyperparameters | None:
        del store_arg, dataset
        return None

    monkeypatch.setattr(execution, "_pilot_selected_hyperparameters", no_selection)
    assert training_protocol_rows(store) == ()
