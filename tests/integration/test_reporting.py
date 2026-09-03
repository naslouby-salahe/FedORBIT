from __future__ import annotations

from pathlib import Path

from fedorbit.infrastructure.workspace import build_layout


def test_reporting_workspace_is_terminal_and_separate_from_execution(tmp_path: Path) -> None:
    layout = build_layout(root=tmp_path)
    assert layout.manuscript_root == tmp_path / "results"
    assert layout.manuscript_root != layout.execution_root
