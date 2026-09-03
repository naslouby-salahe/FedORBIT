from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.types import ArtifactPath


def test_artifact_path_requires_absolute_location() -> None:
    with pytest.raises(ValueError, match="absolute"):
        ArtifactPath(Path("outputs/preprocessing/validation.json"))


def test_artifact_path_retains_absolute_location(tmp_path: Path) -> None:
    location = tmp_path / "validation.json"
    assert ArtifactPath(location).value == location
