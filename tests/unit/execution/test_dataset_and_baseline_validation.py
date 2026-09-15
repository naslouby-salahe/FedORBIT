from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import cast

import pytest

from fedorbit.config.loading import active_config
from fedorbit.experiments import validation as experiment_validation
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.synthesis import completed_experiment_metric_records
from fedorbit.experiments.validation import (
    DatasetValidationInvalidity,
    dataset_client_pair_seed_units,
    execute_dataset_client_and_resource_validation,
    oracle_information_accessed,
    oracle_validation_context,
    pair_strict_resource_validity,
    persist_real_pair_baseline_validation,
    registered_method_resource_validity,
)
from fedorbit.infrastructure.artifacts import ArtifactStore
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.interface import validate_oracle_acl_isolation
from fedorbit.types import (
    ArtifactState,
    ClientRole,
    DatasetId,
    ExperimentName,
    MetricId,
    OverwritePolicy,
    StrictResourceViolationError,
    TransferMethod,
    ValidationReason,
)


def _request(experiment: ExperimentName) -> ExperimentExecutionRequest:
    catalogue = build_catalogue()
    return ExperimentExecutionRequest(
        experiment=experiment,
        definition=catalogue.definition(experiment),
        overwrite_policy=OverwritePolicy.REPLACE,
    )


def _no_blocked_datasets() -> OrderedDict[DatasetId, ValidationReason]:
    return OrderedDict()


def _unavailable_client(*_arguments: object, **_keywords: object) -> None:
    from fedorbit.datasets.materialization import MaterializationError

    raise MaterializationError("fixture materialization is unavailable")


def _primary_clients() -> tuple[DatasetId, ...]:
    return tuple(
        dataset
        for dataset, client in active_config().scientific.datasets.clients.items()
        if client.role is ClientRole.PRIMARY
    )


def test_dataset_validation_covers_every_registered_client_pair_seed_unit() -> None:
    clients = active_config().scientific.datasets.clients
    primary = _primary_clients()
    pairs = active_config().scientific.datasets.primary_directed_pairs
    seeds = active_config().scientific.randomness.confirmatory_seeds
    units = dataset_client_pair_seed_units(
        OrderedDict((dataset, None) for dataset in clients), primary, pairs, seeds
    )
    assert len(units) == len(primary) * len(pairs) * len(seeds) == 180
    assert len({cast(int, unit["seed"]) for unit in units}) == len(seeds)
    for unit in units:
        assert unit["state"] == ArtifactState.INVALID.value
        assert unit["invalid_reason"] == (DatasetValidationInvalidity.CLIENT_NOT_MATERIALIZED.value)
        assert unit["strict_resource_valid"] is False
        assert cast(str, unit["client"]) in {dataset.value for dataset in primary}
    endpoint_units = tuple(
        unit for unit in units if unit["is_source_endpoint"] or unit["is_target_endpoint"]
    )
    assert len(endpoint_units) == 2 * len(pairs) * len(seeds)


def test_pair_strict_resource_validity_reports_typed_reasons() -> None:
    validity = pair_strict_resource_validity(None, None)
    assert validity.valid is False
    assert validity.reason == DatasetValidationInvalidity.PAIR_ENDPOINT_NOT_MATERIALIZED.value


def test_dataset_validation_persists_typed_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION)
    monkeypatch.setattr(experiment_validation, "load_or_materialize_client", _unavailable_client)
    monkeypatch.setattr(experiment_validation, "_chronology_block_reasons", _no_blocked_datasets)
    manifest = execute_dataset_client_and_resource_validation(store, layout, request)
    payload = json.loads(Path(manifest.payload_paths[0]).read_text(encoding="utf-8"))
    assert payload["unit_count"] == 180
    assert len(payload["units"]) == 180
    assert all(unit["invalid_reason"] for unit in payload["units"])
    assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED


def test_real_pair_baseline_validation_covers_every_registered_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION)
    methods = tuple(
        method for method in request.definition.methods if isinstance(method, TransferMethod)
    )
    assert len(methods) == 6
    seeds = active_config().scientific.randomness.confirmatory_seeds
    validation_seeds = (seeds[0], seeds[4])
    monkeypatch.setattr(experiment_validation, "load_or_materialize_client", _unavailable_client)
    persist_real_pair_baseline_validation(store, layout, request, validation_seeds)
    records = tuple(
        record
        for record in completed_experiment_metric_records(
            store, ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION
        )
        if record.metric_name is MetricId.STRICT_RESOURCE_VALIDITY
    )
    pairs = active_config().scientific.datasets.primary_directed_pairs
    assert len(records) == len(pairs) * len(validation_seeds) * len(methods) == 72
    assert {record.method for record in records} == set(methods)
    for record in records:
        assert not record.valid
        assert record.metric_value is None
        assert (
            record.invalid_reason
            == DatasetValidationInvalidity.PAIR_ENDPOINT_NOT_MATERIALIZED.value
        )


def test_registered_method_resource_manifests_are_permitted_by_the_policy() -> None:
    for method in TransferMethod:
        validity = registered_method_resource_validity(method)
        assert validity.valid, method
        assert validity.reason is None


def test_oracle_access_is_rejected_outside_a_registered_validation_context() -> None:
    oracle_methods: tuple[TransferMethod, ...] = (TransferMethod.EXACT_MAP_ORACLE,)
    assert oracle_validation_context(
        ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION, oracle_methods
    )
    assert not oracle_validation_context(
        ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER, oracle_methods
    )
    assert oracle_information_accessed(TransferMethod.EXACT_MAP_ORACLE)
    assert not oracle_information_accessed(TransferMethod.LOCAL_ONLY)
    with pytest.raises(StrictResourceViolationError):
        validate_oracle_acl_isolation(
            False, oracle_information_accessed(TransferMethod.EXACT_MAP_ORACLE)
        )
    validate_oracle_acl_isolation(
        oracle_validation_context(
            ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION, oracle_methods
        ),
        oracle_information_accessed(TransferMethod.EXACT_MAP_ORACLE),
    )
