from __future__ import annotations

from tests.architecture.scan import REPOSITORY_ROOT


def test_ruff_line_length_configured() -> None:
    text = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "line-length = 100" in text
