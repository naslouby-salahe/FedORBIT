from __future__ import annotations

from tests.architecture.scan import REPOSITORY_ROOT

ROADMAP_LOCKED_NOT_YET_CONSUMED = {
    "scikit-learn",
    "pandas",
    "pyarrow",
    "highspy",
    "pyscipopt",
    "typer",
}


def test_deptry_ignored_set_matches_expected_unconsumed() -> None:
    text = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.deptry]" in text
    for package in ROADMAP_LOCKED_NOT_YET_CONSUMED:
        assert package in text, f"deptry ignore set missing roadmap-locked package {package}"
