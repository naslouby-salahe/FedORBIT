from __future__ import annotations

import subprocess

from tests.architecture.scan import REPOSITORY_ROOT, SRC_ROOT, iter_test_files, relative_module


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_vulture_finds_no_unused_production_code() -> None:
    whitelist = REPOSITORY_ROOT / "vulture_whitelist.py"
    command = ["uv", "run", "vulture"]
    if whitelist.is_file():
        command.append(str(whitelist))
    command.extend(["src/fedorbit", "--min-confidence", "80"])
    result = _run(command)
    assert result.returncode == 0, result.stdout + result.stderr


def test_vulture_whitelist_is_committed() -> None:
    whitelist = REPOSITORY_ROOT / "vulture_whitelist.py"
    assert whitelist.is_file(), "vulture_whitelist.py must be committed"


def test_no_production_module_referenced_only_by_tests() -> None:
    console_script_modules = {"fedorbit.cli.main"}
    production_modules = [
        relative_module(path) for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in path.parts
    ]
    production_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    production_text += "\n" + (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    test_text = "".join(path.read_text(encoding="utf-8") for path in iter_test_files())
    for module in production_modules:
        if module.endswith("__init__"):
            continue
        if module not in test_text:
            continue
        if module not in production_text and module not in console_script_modules:
            raise AssertionError(f"production module referenced only by tests: {module}")
