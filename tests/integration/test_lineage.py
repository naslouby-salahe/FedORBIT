from __future__ import annotations

from pathlib import Path

from fedorbit.infrastructure.artifacts import file_digest, read_manifest
from fedorbit.pipeline.execution import execute_experiment, prepare_dataset
from fedorbit.types import DatasetId, ExperimentId, OverwritePolicy
from tests.e2e.test_pipeline import build
from tests.support import FIXTURE_DEVICES

COLD = ExperimentId.COLD_START_LADDER


def test_records_provenance_binds_the_exact_prepared_payloads(tmp_path: Path) -> None:
    config, context = build(tmp_path)
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    execute_experiment(config, context, COLD, OverwritePolicy.REUSE)
    manifest = read_manifest(context.workspace.records_path(COLD))
    assert manifest is not None
    recorded = {item.name: item.sha256 for item in manifest.provenance.inputs}
    for device in FIXTURE_DEVICES:
        prepared = context.workspace.prepared_path(DatasetId.NBAIOT, device)
        assert recorded[f"nbaiot/{device}"] == file_digest(prepared)


def test_prepared_manifest_records_every_raw_source_file(tmp_path: Path) -> None:
    config, context = build(tmp_path)
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    manifest = read_manifest(context.workspace.prepared_path(DatasetId.NBAIOT, FIXTURE_DEVICES[0]))
    assert manifest is not None
    names = {item.name for item in manifest.provenance.inputs}
    assert len(names) == 11
    assert all(name.startswith("N-BaIoT/") for name in names)
