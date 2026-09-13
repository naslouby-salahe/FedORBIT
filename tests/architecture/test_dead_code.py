from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.architecture.scan import (
    REPOSITORY_ROOT,
    SRC_ROOT,
    iter_source_files,
    iter_test_files,
    parse_module,
    relative_module,
)


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
        if (
            module not in production_text
            and module not in console_script_modules
            and module in test_text
        ):
            raise AssertionError(f"production module imported only from tests: {module}")
    offenders: list[str] = []
    for path in iter_source_files():
        module = relative_module(path)
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            if _production_references(node.name) or not _test_references(node.name):
                continue
            offenders.append(f"{module}.{node.name}")
    assert not offenders, f"production functions used only from tests: {sorted(offenders)}"


def _definition_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"\s*(?:def|async def|class)\s+{re.escape(name)}\b")


def _name_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(name)}\b")


def _references(name: str, paths: tuple[Path, ...]) -> int:
    name_pattern = _name_pattern(name)
    definition_pattern = _definition_pattern(name)
    return sum(
        1
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if name_pattern.search(line) and not definition_pattern.match(line)
    )


def _production_references(name: str) -> int:
    return _references(name, iter_source_files())


def _test_references(name: str) -> int:
    return _references(name, iter_test_files())
