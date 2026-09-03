from __future__ import annotations

from pathlib import Path

from tests.architecture.scan import (
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
