from __future__ import annotations

import pytest
from pydantic import ValidationError

from fedorbit.datasets.manifests import (
    DatasetManifest,
    EligibilityCopyKind,
    SemanticCellManifest,
    TransferEligibilityManifest,
    eligibility_copy,
)

DATASET_FIELDS = {
    "dataset": "edge_iiotset_network",
    "component": "network",
    "raw_files": ("ML-EdgeIIoT-dataset.csv",),
    "raw_sha256": "a" * 64,
    "raw_counts": {"rows": 100},
    "schema": "1.0",
    "adapter_feature_order": ("tcp.ack",),
    "adapter_feature_roles": {"tcp.ack": "behavioral_numeric"},
    "accepted_schema_aliases": (),
    "adapter_adaptations": (),
    "timestamp_field": "frame.time",
    "timestamp_range": ("2020-01-01", "2020-01-02"),
    "duplicate_counts": {"group-a": 2},
    "conflicting_duplicate_counts": {},
    "local_class_counts": {"normal": 50, "ddos_tcp": 50},
    "transfer_candidate_counts": {"DDoS": 50},
    "feature_quality": {"dropped": 0},
    "preprocessing_state": "completed",
    "dependency_fingerprint_sha256": "b" * 64,
    "producer_code_sha256": "c" * 64,
}

ELIGIBILITY_FIELDS = {
    "client": "edge_iiotset_network",
    "seed": 1103,
    "coarse_group": "Disruption",
    "anonymous_node_id": "node-42",
    "native_local_class_ids": ("ddos_tcp",),
    "present": True,
    "train_count": 40,
    "meta_count": 5,
    "confirm_count": 0,
    "test_count": 5,
    "source_eligible": True,
    "target_eligible": False,
    "null_reason": None,
    "fine_concept": "DDoS",
}

CELL_FIELDS = {
    "experiment": "Primary Strict Cross-Telemetry Transfer",
    "dataset": "edge_iiotset_network",
    "source_client": "edge_iiotset_network",
    "target_client": "ton_iot_windows10_host",
    "directed_pair": "edge_iiotset_network -> ton_iot_windows10_host",
    "method": "LIVE",
    "condition": "two_way",
    "support": 2,
    "seed": 1103,
    "scientific_configuration_sha256": "d" * 64,
    "dependency_fingerprint_sha256": "e" * 64,
    "producer_stage": "response",
}


def test_dataset_manifest_requires_each_field() -> None:
    for field_name in DATASET_FIELDS:
        missing = {key: value for key, value in DATASET_FIELDS.items() if key != field_name}
        if field_name == "schema":
            missing.pop("schema", None)
        with pytest.raises(ValidationError):
            DatasetManifest.model_validate(missing)


def test_dataset_manifest_round_trips() -> None:
    manifest = DatasetManifest.model_validate(DATASET_FIELDS)
    dumped = manifest.model_dump(mode="json", by_alias=True)
    assert "schema" in dumped
    assert dumped["schema"] == "1.0"
    assert DatasetManifest.model_validate(dumped) == manifest


def test_dataset_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate({**DATASET_FIELDS, "invented": 1})


def test_eligibility_manifest_requires_each_field() -> None:
    optional_fields = {"null_reason", "fine_concept"}
    for field_name in ELIGIBILITY_FIELDS:
        if field_name in optional_fields:
            continue
        missing = {key: value for key, value in ELIGIBILITY_FIELDS.items() if key != field_name}
        with pytest.raises(ValidationError):
            TransferEligibilityManifest.model_validate(missing)


def test_eligibility_manifest_round_trips() -> None:
    manifest = TransferEligibilityManifest.model_validate(ELIGIBILITY_FIELDS)
    dumped = manifest.model_dump(mode="json")
    assert TransferEligibilityManifest.model_validate(dumped) == manifest


def test_method_readable_copy_omits_class_ids_and_fine_concept() -> None:
    manifest = TransferEligibilityManifest.model_validate(ELIGIBILITY_FIELDS)
    readable = eligibility_copy(manifest, EligibilityCopyKind.METHOD_READABLE)
    assert readable.native_local_class_ids == ()
    assert readable.fine_concept is None
    payload = readable.model_dump(mode="json", exclude={"native_local_class_ids", "fine_concept"})
    assert "native_local_class_ids" not in payload
    assert "fine_concept" not in payload


def test_oracle_copy_keeps_fine_concept_without_class_ids() -> None:
    manifest = TransferEligibilityManifest.model_validate(ELIGIBILITY_FIELDS)
    oracle = eligibility_copy(manifest, EligibilityCopyKind.ORACLE)
    assert oracle.fine_concept == "DDoS"
    assert oracle.native_local_class_ids == ()


def test_builder_copy_retains_class_ids() -> None:
    manifest = TransferEligibilityManifest.model_validate(ELIGIBILITY_FIELDS)
    builder = eligibility_copy(manifest, EligibilityCopyKind.BUILDER)
    assert builder.native_local_class_ids == ("ddos_tcp",)


def test_oracle_copy_requires_fine_concept() -> None:
    fields = dict(ELIGIBILITY_FIELDS)
    fields["fine_concept"] = None
    manifest = TransferEligibilityManifest.model_validate(fields)
    with pytest.raises(ValueError):
        eligibility_copy(manifest, EligibilityCopyKind.ORACLE)


def test_semantic_cell_manifest_requires_each_field() -> None:
    for field_name in CELL_FIELDS:
        missing = {key: value for key, value in CELL_FIELDS.items() if key != field_name}
        with pytest.raises(ValidationError):
            SemanticCellManifest.model_validate(missing)


def test_semantic_cell_manifest_round_trips() -> None:
    manifest = SemanticCellManifest.model_validate(CELL_FIELDS)
    dumped = manifest.model_dump(mode="json")
    assert SemanticCellManifest.model_validate(dumped) == manifest


def test_one_eligibility_row_per_endpoint_seed_concept() -> None:
    rows = [
        TransferEligibilityManifest.model_validate(
            {**ELIGIBILITY_FIELDS, "seed": 1103, "coarse_group": "Disruption"}
        ),
        TransferEligibilityManifest.model_validate(
            {**ELIGIBILITY_FIELDS, "seed": 5531, "coarse_group": "Disruption"}
        ),
        TransferEligibilityManifest.model_validate(
            {**ELIGIBILITY_FIELDS, "seed": 1103, "coarse_group": "Exploitation"}
        ),
    ]
    keys = {(row.client, row.seed, row.coarse_group) for row in rows}
    assert len(keys) == 3
