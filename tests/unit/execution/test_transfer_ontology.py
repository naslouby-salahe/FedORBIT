from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest

import fedorbit.infrastructure.execution as execution
from fedorbit.config.loading import active_config
from fedorbit.datasets.common import AdapterSchema
from fedorbit.datasets.materialization import (
    DatasetProvenance,
    LocalClassManifest,
    MaterializationError,
    MaterializedClient,
)
from fedorbit.datasets.ontology import TonNativeLabel
from fedorbit.datasets.preprocessing import FeatureQualityReport
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.infrastructure.execution import (
    ArtifactStore,
    ExperimentExecutionRequest,
    transfer_ontology_and_null_padding_rows,
)
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import (
    ClientComponentName,
    DatasetId,
    ExperimentName,
    FeatureName,
    FineLabel,
    OverwritePolicy,
    Split,
    TabularColumnName,
)


def _client(
    dataset: DatasetId, class_row_counts: dict[FineLabel, dict[Split, int]]
) -> MaterializedClient:
    class_names = tuple(class_row_counts)
    return MaterializedClient(
        dataset=dataset,
        schema=AdapterSchema(dataset_id=dataset, feature_order=(TabularColumnName("feature_0"),)),
        class_manifest=LocalClassManifest(class_names=class_names, excluded_classes=()),
        feature_names=(FeatureName("feature_0"),),
        splits=OrderedDict(),
        feature_quality=FeatureQualityReport(
            candidate_features=(), dropped_feature_count=0, client_invalid=False
        ),
        class_row_counts=OrderedDict(class_row_counts),
        provenance=DatasetProvenance(
            component=ClientComponentName("test-component"),
            raw_files=(),
            accepted_timestamp_column=TabularColumnName("ts"),
            timestamp_range=(0.0, 1.0),
            duplicate_group_count=0,
            conflicting_duplicate_group_count=0,
        ),
    )


def _zero_counts() -> dict[Split, int]:
    return dict.fromkeys(Split, 0)


def test_dataset_client_validation_persists_transfer_ontology_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pair = active_config().scientific.datasets.primary_directed_pairs[0]

    source_counts = _zero_counts()
    source_counts[Split.TRAIN] = 1000
    source_counts[Split.META] = 1000
    source_client = _client(pair.source, {FineLabel(TonNativeLabel.DDOS): source_counts})

    target_counts = _zero_counts()
    target_counts[Split.META] = 1000
    target_counts[Split.CONFIRM] = 1000
    target_counts[Split.TEST] = 1000
    target_client = _client(pair.target, {FineLabel(TonNativeLabel.DDOS): target_counts})

    clients = {pair.source: source_client, pair.target: target_client}

    def fake_materialize_client(dataset: DatasetId, raw_root: Path) -> MaterializedClient:
        del raw_root
        client = clients.get(dataset)
        if client is None:
            raise MaterializationError(f"no synthetic fixture for {dataset.value}")
        return client

    monkeypatch.setattr(execution, "materialize_client", fake_materialize_client)
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = ExperimentExecutionRequest(
        ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION,
        build_catalogue().definition(ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION),
        OverwritePolicy.REPLACE,
    )
    execution.execute_dataset_client_and_resource_validation(store, layout, request)

    rows = transfer_ontology_and_null_padding_rows(store)
    assert rows
    expected_pair = f"{pair.source.value} -> {pair.target.value}"
    ddos_rows = [
        row for row in rows if row["pair"] == expected_pair and row["candidate_concept"] == "DDoS"
    ]
    assert len(ddos_rows) == 1
    ddos_row = ddos_rows[0]
    assert ddos_row["source_real_or_null"] == "real"
    assert ddos_row["target_real_or_null"] == "real"
    assert ddos_row["action_eligibility"] is True
    assert ddos_row["null_reason"] is None

    other_rows = [
        row for row in rows if row["pair"] == expected_pair and row["candidate_concept"] != "DDoS"
    ]
    assert other_rows
    for row in other_rows:
        assert row["source_real_or_null"] == "null"
        assert row["target_real_or_null"] == "null"
        assert row["null_reason"] is not None
