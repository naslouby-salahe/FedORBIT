from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

import fedorbit.experiments.training as execution
from fedorbit.datasets.materialization import DatasetId
from fedorbit.experiments.training import training_protocol_rows
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.evidence import TableScalar
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.learning.pilot import HOST_DATASETS, NETWORK_DATASETS
from fedorbit.learning.training import SelectedHyperparameters
from fedorbit.types import ReportColumnName


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
    models: dict[str, Mapping[ReportColumnName, TableScalar]] = {
        str(row[ReportColumnName.MODEL]): row for row in rows
    }
    assert set(models) == {dataset.value for dataset in selections}

    network_row = models[DatasetId.EDGE_IIOTSET_NETWORK.value]
    assert network_row[ReportColumnName.ARCHITECTURE] == "256-128-64 MLP (NetworkFlowClassifier)"
    assert network_row[ReportColumnName.NORMALIZATION] == "LayerNorm"
    assert network_row[ReportColumnName.ACTIVATION] == "GELU"
    assert network_row[ReportColumnName.INITIALIZATION] == "Xavier uniform"
    assert network_row[ReportColumnName.OPTIMIZER] == "AdamW"
    assert network_row[ReportColumnName.SELECTED_LEARNING_RATE] == pytest.approx(1e-3)
    assert network_row[ReportColumnName.SELECTED_WEIGHT_DECAY] == pytest.approx(1e-4)
    assert network_row[ReportColumnName.SELECTED_DROPOUT] == pytest.approx(0.1)

    host_row = models[DatasetId.TON_IOT_WINDOWS10_HOST.value]
    assert host_row[ReportColumnName.ARCHITECTURE] == "192-96-48 MLP (HostClassifier)"
    assert host_row[ReportColumnName.NORMALIZATION] == "BatchNorm1d"
    assert host_row[ReportColumnName.ACTIVATION] == "ReLU"
    assert host_row[ReportColumnName.INITIALIZATION] == "Kaiming uniform"
    stopping_rule = host_row[ReportColumnName.STOPPING_RULE]
    assert isinstance(stopping_rule, str)
    assert "epochs" in stopping_rule

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
