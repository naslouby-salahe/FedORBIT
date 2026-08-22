from __future__ import annotations

import ast

from tests.architecture.scan import (
    SRC_ROOT,
    iter_source_files,
    iter_test_files,
    parse_module,
    relative_module,
)


def test_no_production_function_used_only_from_tests() -> None:
    production_text = "".join(
        path.read_text(encoding="utf-8")
        for path in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    test_text = "".join(path.read_text(encoding="utf-8") for path in iter_test_files())
    for path in iter_source_files():
        module = relative_module(path)
        if module.endswith("__init__"):
            continue
        if module not in production_text:
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
