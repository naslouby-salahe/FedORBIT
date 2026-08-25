from __future__ import annotations

import subprocess
import sys

from tests.architecture.scan import REPOSITORY_ROOT

ROADMAP_LOCKED_NOT_YET_CONSUMED = {
    "scikit-learn",
    "pandas",
    "pyarrow",
    "highspy",
    "pyscipopt",
    "typer",
}


def test_deptry_reports_only_roadmap_locked_unconsumed_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deptry", "src/fedorbit"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "Success" in combined or "No dependency issues" in combined


def test_deptry_ignored_set_matches_expected_unconsumed() -> None:
    text = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.deptry]" in text
    for package in ROADMAP_LOCKED_NOT_YET_CONSUMED:
        assert package in text, f"deptry ignore set missing roadmap-locked package {package}"
