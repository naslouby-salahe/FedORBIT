from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.materialization import materialize_client
from fedorbit.infrastructure.execution import build_dataset_manifest
from fedorbit.types import DatasetId

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.skipif(not RAW_ROOT.is_dir(), reason="real raw datasets are unavailable")
def test_dataset_manifest_reflects_real_windows10_materialization() -> None:
    load_fedorbit_config()
    materialized = materialize_client(DatasetId.TON_IOT_WINDOWS10_HOST, RAW_ROOT)
    manifest = build_dataset_manifest(materialized)
    assert manifest.dataset == DatasetId.TON_IOT_WINDOWS10_HOST
    assert manifest.raw_files
    assert len(manifest.raw_sha256) == 64
    assert sum(manifest.raw_counts.values()) > 0
    assert manifest.adapter_feature_order == materialized.schema.feature_order
    assert manifest.timestamp_field == manifest.accepted_schema_aliases[0]
    assert float(manifest.timestamp_range[0]) <= float(manifest.timestamp_range[1])
    assert manifest.duplicate_counts["total"] >= 0
    assert manifest.conflicting_duplicate_counts["total"] >= 0
    assert set(manifest.local_class_counts) == set(materialized.class_manifest.class_names)
    assert manifest.preprocessing_state == "materialized"
    assert len(manifest.dependency_fingerprint_sha256) == 64
    assert len(manifest.producer_code_sha256) == 64
