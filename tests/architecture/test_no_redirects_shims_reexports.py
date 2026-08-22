from __future__ import annotations

import ast

from tests.architecture.scan import (
    iter_source_files,
    parse_module,
    reexport_only_module,
    relative_module,
)


def test_no_reexport_only_modules() -> None:
    for path in iter_source_files():
        module = relative_module(path)
        if module.endswith("__init__"):
            continue
        tree = parse_module(path)
        assert not reexport_only_module(tree), f"re-export-only module: {path}"


def test_no_legacy_alias_imports() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for alias in ("import fedorbit as", "from fedorbit import fedorbit"):
            assert alias not in text, f"legacy alias in {path}"


def test_no_shadowing_aliases() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") and (
                stripped.endswith(" as util") or stripped.endswith(" as common")
            ):
                raise AssertionError(f"shadowing alias in {path}: {stripped}")


def test_no_compatibility_shim_classes() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in {
                        "DeprecatedClass",
                        "LegacyAdapter",
                        "CompatibilityWrapper",
                    }:
                        raise AssertionError(f"compatibility shim in {path}: {node.name}")
