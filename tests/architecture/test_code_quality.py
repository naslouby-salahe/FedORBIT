from __future__ import annotations

import subprocess

from tests.architecture.scan import REPOSITORY_ROOT


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_ruff_lint_clean() -> None:
    result = _run(["uv", "run", "ruff", "check", "src", "tests"])
    assert result.returncode == 0, result.stdout + result.stderr


def test_ruff_format_clean() -> None:
    result = _run(["uv", "run", "ruff", "format", "--check", "src", "tests"])
    assert result.returncode == 0, result.stdout + result.stderr


def test_ruff_line_length_configured() -> None:
    text = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "line-length = 100" in text
