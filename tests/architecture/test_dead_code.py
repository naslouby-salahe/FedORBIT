from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from tests.architecture.scan import (
    REPOSITORY_ROOT,
    SRC_ROOT,
    iter_source_files,
    iter_test_files,
    parse_module,
    relative_module,
)

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEFINITION = re.compile(r"^\s*(?:def|async def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)


def _declared_public_api() -> frozenset[str]:
    whitelist = REPOSITORY_ROOT / "vulture_whitelist.py"
    if not whitelist.is_file():
        return frozenset()
    tree = ast.parse(whitelist.read_text(encoding="utf-8"))
    return frozenset(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))


FRAMEWORK_DECORATOR_FRAGMENTS = ("validator", "property", "app.command", "app.callback")
FRAMEWORK_CALLBACK_NAMES = frozenset({"forward"})


def test_vulture_whitelist_is_committed() -> None:
    whitelist = REPOSITORY_ROOT / "vulture_whitelist.py"
    assert whitelist.is_file(), "vulture_whitelist.py must be committed"


def _production_files() -> tuple[Path, ...]:
    return tuple(path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _identifier_counts(paths: tuple[Path, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        counts.update(IDENTIFIER.findall(path.read_text(encoding="utf-8")))
    return counts


def _declared_counts(paths: tuple[Path, ...]) -> Counter[str]:
    declared: Counter[str] = Counter()
    for path in paths:
        declared.update(DEFINITION.findall(path.read_text(encoding="utf-8")))
    return declared


def _framework_registered(node: ast.FunctionDef | ast.ClassDef) -> bool:
    if node.name in FRAMEWORK_CALLBACK_NAMES:
        return True
    decorators = [ast.unparse(decorator) for decorator in getattr(node, "decorator_list", [])]
    return any(
        fragment in decorator
        for decorator in decorators
        for fragment in FRAMEWORK_DECORATOR_FRAGMENTS
    )


def _public_symbols(path: Path) -> list[ast.FunctionDef | ast.ClassDef]:
    declared = _declared_public_api()
    return [
        node
        for node in ast.walk(parse_module(path))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
        and not _framework_registered(node)
        and node.name not in declared
    ]


def test_no_production_module_referenced_only_by_tests() -> None:
    console_script_modules = {"fedorbit.cli"}
    production_modules = [relative_module(path) for path in _production_files()]
    production_text = "\n".join(path.read_text(encoding="utf-8") for path in _production_files())
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


def test_no_production_function_used_only_from_tests() -> None:
    production = _identifier_counts(_production_files())
    declared = _declared_counts(_production_files())
    tests = _identifier_counts(iter_test_files())
    offenders = sorted(
        f"{relative_module(path)}.{node.name}"
        for path in iter_source_files()
        for node in _public_symbols(path)
        if production[node.name] - declared[node.name] <= 0 and tests[node.name] > 0
    )
    assert not offenders, f"production functions used only from tests: {offenders}"


def test_no_unreferenced_production_symbols() -> None:
    production = _identifier_counts(_production_files())
    declared = _declared_counts(_production_files())
    tests = _identifier_counts(iter_test_files())
    offenders = sorted(
        f"{relative_module(path)}.{node.name}"
        for path in iter_source_files()
        for node in _public_symbols(path)
        if production[node.name] - declared[node.name] <= 0 and tests[node.name] == 0
    )
    assert not offenders, f"production symbols with no reference anywhere: {offenders}"
