from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.architecture.scan import (
    TODO_MARKERS,
    comments_and_docstrings,
    iter_source_files,
    iter_test_files,
    parse_module,
)


def _python_files() -> list[Path]:
    return [*iter_source_files(), *iter_test_files()]


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_no_comments_in_python_source() -> None:
    for path in _python_files():
        for lineno, line in enumerate(_lines(path), start=1):
            assert not line.lstrip().startswith("#"), f"comment in {path}:{lineno}"


def test_no_docstrings_in_python_source() -> None:
    for path in _python_files():
        tree = parse_module(path)
        flagged = comments_and_docstrings(tree)
        assert not flagged, f"docstring in {path} at lines {flagged}"


def test_no_type_ignore_comments() -> None:
    for path in _python_files():
        if path.parent.name == "architecture":
            continue
        for lineno, line in enumerate(_lines(path), start=1):
            marker = "type:" + " ignore"
            assert marker not in line, f"suppression in {path}:{lineno}"


def test_no_pragma_or_suppression_comments() -> None:
    for path in _python_files():
        if "tests/architecture" in str(path):
            continue
        for lineno, line in enumerate(_lines(path), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#") and any(
                marker in stripped.lower()
                for marker in ("noqa", "pylint", "pyright:", "pragma", "type:" + " ignore")
            ):
                raise AssertionError(f"suppression comment in {path}:{lineno}")


def test_no_shebang_lines_in_python_source() -> None:
    for path in _python_files():
        lines = _lines(path)
        assert not lines or not lines[0].startswith("#!"), f"shebang in {path}"


def test_source_uses_no_forbidden_abstractions() -> None:
    forbidden_terms = ("canonical", "claim")
    for path in iter_source_files():
        source = path.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in source, f"forbidden implementation term {term!r} in {path}"


def test_no_todo_markers_in_python_source() -> None:
    for path in iter_source_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for marker in TODO_MARKERS:
                assert marker not in line, f"{marker} marker in {path}:{lineno}"


def test_no_debug_residue_in_production() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for marker in ("breakpoint()", "pdb."):
            assert marker not in text, f"debug residue in {path}: {marker}"
        assert not re.search(r"(?<![A-Za-z])print\(", text), f"print residue in {path}"
        assert not re.search(r"(?<![A-Za-z])pprint\(", text), f"pprint residue in {path}"


def test_no_pass_only_function_bodies_in_production() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
            ):
                raise AssertionError(f"pass-only function in {path}: {node.name}")
            if (
                isinstance(node, ast.ClassDef)
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
                and not _exception_class(node)
            ):
                raise AssertionError(f"pass-only class in {path}: {node.name}")


def _exception_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and (base.id == "Exception" or base.id.endswith("Error")):
            return True
    return False


def test_no_commented_out_code() -> None:
    for path in iter_source_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#") and any(
                token in stripped for token in ("import ", "def ", "class ", "return ")
            ):
                raise AssertionError(f"commented-out code in {path}:{lineno}")


def test_no_unreachable_dead_markers() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for marker in ("if False:", "if __debug__ and False"):
            assert marker not in text, f"unreachable marker in {path}"
