from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.architecture.scan import REPOSITORY_ROOT, SRC_ROOT


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _pyright() -> str:
    suffix = ".exe" if sys.platform == "win32" else ""
    return str(Path(sys.executable).with_name(f"pyright{suffix}"))


def test_strict_pyright_passes_on_production() -> None:
    result = _run([_pyright(), "src"])
    assert result.returncode == 0, result.stdout + result.stderr


def test_strict_pyright_passes_on_tests() -> None:
    result = _run([_pyright(), "tests"])
    assert result.returncode == 0, result.stdout + result.stderr


def test_pyright_config_is_strict() -> None:
    from pathlib import Path

    pyproject = Path(REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'typeCheckingMode = "strict"' in pyproject
    assert 'include = ["src", "tests"]' in pyproject


def test_no_source_file_uses_untyped_def() -> None:
    import ast

    from tests.architecture.scan import parse_module

    for path in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for argument in node.args.args:
                    if argument.arg in {"self", "cls"}:
                        continue
                    if argument.annotation is None:
                        raise AssertionError(
                            f"untyped parameter in {path}:{node.name}({argument.arg})"
                        )
