from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterator
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src" / "fedorbit"
TESTS_ROOT = REPOSITORY_ROOT / "tests"

ALLOWED_ROOT_ENTRIES = {
    ".agents",
    ".claude",
    ".codex",
    ".coverage",
    ".env",
    ".git",
    ".github",
    ".gitignore",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "CLAUDE.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "__pycache__",
    "configs",
    "data",
    "docs",
    "graphify-out",
    "noxfile.py",
    "outputs",
    "pyproject.toml",
    "results",
    "src",
    "tests",
    "uv.lock",
}

PACKAGE_LAYERS: dict[str, int] = {
    "types": 0,
    "config": 1,
    "infrastructure": 2,
    "datasets": 3,
    "detection": 3,
    "study": 4,
    "analysis": 5,
    "pipeline": 6,
    "reporting": 7,
    "cli": 8,
}

VAGUE_MODULE_NAMES = {
    "utils",
    "helpers",
    "common",
    "manager",
    "processor",
    "base",
    "misc",
    "tools",
    "shared",
    "util",
    "helper",
}

BANNED_NAME_FRAGMENTS = (
    "v2",
    "final2",
    "_new",
    "_old",
    "copy2",
    "tmp",
    "dummy",
    "placeholder",
    "wip",
)
STALE_TERMS = ("orchestrator", "workflow engine", "response operator", "orbit map")
RESIDUE_MARKERS = ("TODO", "FIXME", "HACK", "XXX")


def iter_source_files() -> Iterator[Path]:
    yield from sorted(SRC_ROOT.rglob("*.py"))


def iter_test_files() -> Iterator[Path]:
    yield from sorted(TESTS_ROOT.rglob("*.py"))


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def comment_tokens(path: Path) -> list[tokenize.TokenInfo]:
    stream = io.StringIO(path.read_text(encoding="utf-8"))
    return [
        token
        for token in tokenize.generate_tokens(stream.readline)
        if token.type == tokenize.COMMENT
    ]


def docstring_nodes(tree: ast.Module) -> list[ast.AST]:
    found: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found.append(body[0])
    return found


def module_package(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT)
    return relative.parts[0].removesuffix(".py")


def imported_packages(tree: ast.Module) -> set[str]:
    packages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("fedorbit."):
            packages.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("fedorbit."):
                    packages.add(alias.name.split(".")[1])
    return packages
