from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from fedorbit.config.context import active_config
from fedorbit.config.loading import repository_root
from fedorbit.domain.enums import ExperimentName


class WorkspaceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    execution_root: Path
    manuscript_root: Path
    preprocessing: Path
    artifacts: Path
    experiments: Path
    cache: Path
    staging: Path
    results_experiments: Path
    project_summary: Path


def safe_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    if not slug:
        raise WorkspaceError("descriptive name does not produce a filesystem slug")
    return slug


def build_layout(root: Path | None = None) -> WorkspaceLayout:
    layout = active_config().runtime.artifact_layout
    base = root if root is not None else repository_root()
    execution_root = base / layout.execution_root
    manuscript_root = base / layout.manuscript_root
    return WorkspaceLayout(
        execution_root=execution_root,
        manuscript_root=manuscript_root,
        preprocessing=execution_root / "preprocessing",
        artifacts=execution_root / "artifacts",
        experiments=execution_root / "experiments",
        cache=execution_root / "cache",
        staging=execution_root / "cache" / "staging",
        results_experiments=manuscript_root / "experiments",
        project_summary=manuscript_root / "project_summary",
    )


def experiment_workspace(layout: WorkspaceLayout, experiment: ExperimentName) -> Path:
    return layout.experiments / safe_slug(experiment.value)


def results_workspace(layout: WorkspaceLayout, experiment: ExperimentName) -> Path:
    return layout.results_experiments / safe_slug(experiment.value)


def leaf_path(
    layout: WorkspaceLayout,
    workspace: Path,
    semantic_coordinates: str,
    fingerprint_sha256: str,
    suffix: str,
) -> Path:
    if not workspace.is_absolute():
        workspace = layout.execution_root / workspace
    semantic_slug = safe_slug(semantic_coordinates)
    return workspace / f"{semantic_slug}.{fingerprint_sha256[:16]}{suffix}"


def enforce_workspace_boundary(layout: WorkspaceLayout, path: Path) -> None:
    resolved = path.resolve()
    execution = layout.execution_root.resolve()
    manuscript = layout.manuscript_root.resolve()
    if resolved in (execution, manuscript):
        raise WorkspaceError(f"path is a workspace root, not an artifact: {path}")
    if execution not in resolved.parents and manuscript not in resolved.parents:
        raise WorkspaceError(f"path outside stable workspace: {path}")
