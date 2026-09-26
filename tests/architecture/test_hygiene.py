from __future__ import annotations

import ast
import re

from tests.architecture.scan import (
    BANNED_NAME_FRAGMENTS,
    RESIDUE_MARKERS,
    STALE_TERMS,
    VAGUE_MODULE_NAMES,
    comment_tokens,
    docstring_nodes,
    iter_source_files,
    iter_test_files,
    parse_module,
)


def test_python_files_contain_no_comments() -> None:
    for path in [*iter_source_files(), *iter_test_files()]:
        assert not comment_tokens(path), f"comment in {path}"


def test_python_files_contain_no_docstrings() -> None:
    for path in [*iter_source_files(), *iter_test_files()]:
        assert not docstring_nodes(parse_module(path)), f"docstring in {path}"


def test_no_development_residue_markers() -> None:
    for path in [*iter_source_files(), *iter_test_files()]:
        text = path.read_text(encoding="utf-8")
        if path.name in {"scan.py", "test_hygiene.py"}:
            continue
        for marker in RESIDUE_MARKERS:
            assert re.search(rf"\b{marker}\b", text) is None, f"{marker} in {path}"


def test_no_vague_module_names() -> None:
    for path in iter_source_files():
        assert path.stem not in VAGUE_MODULE_NAMES, f"vague module name {path}"


def test_no_banned_name_fragments_in_identifiers() -> None:
    for path in iter_source_files():
        for node in ast.walk(parse_module(path)):
            names: list[str] = []
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            for name in names:
                lowered = name.lower()
                for fragment in BANNED_NAME_FRAGMENTS:
                    assert fragment not in lowered.split("_") and not lowered.endswith(fragment), (
                        f"banned fragment {fragment!r} in {name} ({path})"
                    )


def test_no_stale_terminology() -> None:
    for path in iter_source_files():
        lowered = path.read_text(encoding="utf-8").lower()
        for term in STALE_TERMS:
            assert term not in lowered, f"stale term {term!r} in {path}"


def test_no_any_or_object_annotations_in_source() -> None:
    for path in iter_source_files():
        for node in ast.walk(parse_module(path)):
            if isinstance(node, ast.Name) and node.id in {"Any", "object"}:
                raise AssertionError(f"{node.id} used in {path}:{node.lineno}")


def test_no_type_suppressions() -> None:
    for path in [*iter_source_files(), *iter_test_files()]:
        text = path.read_text(encoding="utf-8")
        if path.name in {"test_hygiene.py"}:
            continue
        assert "type: ignore" not in text and "pyright: ignore" not in text, (
            f"suppression in {path}"
        )
        assert "noqa" not in text, f"lint suppression in {path}"
