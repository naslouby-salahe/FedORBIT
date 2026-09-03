from __future__ import annotations

import ast
import subprocess
import sys

from tests.architecture.scan import (
    REPOSITORY_ROOT,
    SRC_ROOT,
    iter_source_files,
    iter_test_files,
    parse_module,
    relative_module,
)


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
    command = [sys.executable, "-m", "vulture"]
    if whitelist.is_file():
        command.append(str(whitelist))
    command.extend(["src/fedorbit", "--min-confidence", "80"])
    result = _run(command)
    assert result.returncode == 0, result.stdout + result.stderr


def test_vulture_whitelist_is_committed() -> None:
    whitelist = REPOSITORY_ROOT / "vulture_whitelist.py"
    assert whitelist.is_file(), "vulture_whitelist.py must be committed"


def test_no_production_module_referenced_only_by_tests() -> None:
    console_script_modules = {"fedorbit.cli"}
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


def test_no_production_function_used_only_from_tests() -> None:
    console_script_modules = {"fedorbit.cli"}
    production_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    production_text += "\n" + (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    test_text = "".join(path.read_text(encoding="utf-8") for path in iter_test_files())
    for path in iter_source_files():
        module = relative_module(path)
        if module.endswith("__init__"):
            continue
        if module not in production_text and module not in console_script_modules:
            if module in test_text:
                raise AssertionError(f"production module imported only from tests: {module}")
            continue
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            if node.name in production_text:
                continue
            if node.name in test_text:
                raise AssertionError(
                    f"production function {module}.{node.name} used only from tests"
                )
