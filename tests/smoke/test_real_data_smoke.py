from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.config.loading import REPOSITORY_ROOT, load_config
from fedorbit.infrastructure.workspace import Workspace
from fedorbit.pipeline.execution import (
    ExecutionContext,
    analysis_state,
    execute_analysis,
    execute_experiment,
    prepare_dataset,
)
from fedorbit.pipeline.smoke import smoke_config
from fedorbit.types import DatasetId, EvidenceState, OverwritePolicy

RAW = REPOSITORY_ROOT / load_config().datasets.raw_directory
NBAIOT = RAW / load_config().datasets.nbaiot.relative_directory


@pytest.mark.skipif(not NBAIOT.is_dir(), reason="raw N-BaIoT data is not available")
def test_smoke_pipeline_on_real_devices(tmp_path: Path) -> None:
    config = smoke_config(load_config())
    context = ExecutionContext(
        repository_root=REPOSITORY_ROOT,
        raw_directory=RAW,
        workspace=Workspace(root=tmp_path / "workspace", results_directory=tmp_path / "results"),
        package_directory=REPOSITORY_ROOT / "src" / "fedorbit",
    )
    prepare_dataset(config, context, DatasetId.NBAIOT, OverwritePolicy.REUSE)
    for experiment in config.experiments:
        assert (
            execute_experiment(config, context, experiment.id, OverwritePolicy.REUSE)
            is EvidenceState.VALID
        )
    analysis = execute_analysis(config, context, OverwritePolicy.REUSE)
    assert analysis_state(config, context) is EvidenceState.VALID
    assert analysis.contrasts
    assert all(
        device.delta == device.delta
        for result in analysis.contrasts
        for device in result.device_deltas
    )
