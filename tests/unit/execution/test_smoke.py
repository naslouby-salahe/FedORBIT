from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.experiments import dispatch
from fedorbit.experiments.dispatch import run_smoke_validation
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.types import OverwritePolicy, SmokeArtifactFileName, StorageLayoutSegment


def test_smoke_validation_is_isolated_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatch, "build_layout", lambda: build_layout(tmp_path))
    run_smoke_validation(OverwritePolicy.REUSE)
    completion = (
        build_layout(tmp_path).execution_root
        / StorageLayoutSegment.SMOKE
        / SmokeArtifactFileName.EXECUTION_JSON
    )
    assert completion.is_file()
    first = completion.read_bytes()
    run_smoke_validation(OverwritePolicy.REUSE)
    assert completion.read_bytes() == first
