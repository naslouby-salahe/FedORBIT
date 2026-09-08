from __future__ import annotations

from pathlib import Path

from fedorbit.infrastructure.workspace import build_layout


def test_reporting_workspace_is_terminal_and_separate_from_execution(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    assert layout.project_summary == tmp_path / "results" / "project_summary"
    assert layout.project_summary != layout.execution_root
