from __future__ import annotations

from pathlib import Path

from tests.architecture.scan import (
    ALLOWED_ROOT_ENTRIES,
    PACKAGE_LAYERS,
    REPOSITORY_ROOT,
    SRC_ROOT,
    imported_packages,
    iter_source_files,
    module_package,
    parse_module,
)

REQUIRED_ROOT_FILES = {
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "noxfile.py",
    "Makefile",
    ".gitignore",
}
REQUIRED_DIRECTORIES = {
    "src/fedorbit/config",
    "src/fedorbit/datasets",
    "src/fedorbit/detection",
    "src/fedorbit/study",
    "src/fedorbit/analysis",
    "src/fedorbit/pipeline",
    "src/fedorbit/infrastructure",
    "tests/architecture",
    "tests/unit",
    "tests/scientific",
    "tests/integration",
    "tests/e2e",
    "tests/smoke",
    "configs",
}


def test_dependency_direction_is_one_way() -> None:
    for path in iter_source_files():
        package = module_package(path)
        if package not in PACKAGE_LAYERS:
            continue
        for imported in imported_packages(parse_module(path)):
            if imported == package:
                continue
            assert imported in PACKAGE_LAYERS, f"unlayered package {imported} imported by {path}"
            assert PACKAGE_LAYERS[imported] < PACKAGE_LAYERS[package], (
                f"{path} ({package}) must not import {imported}"
            )


def test_every_source_package_is_layered() -> None:
    packages = {
        child.name for child in SRC_ROOT.iterdir() if child.is_dir() and child.name != "__pycache__"
    }
    modules = {child.stem for child in SRC_ROOT.glob("*.py") if child.stem != "__init__"}
    assert (packages | modules) == set(PACKAGE_LAYERS)


def test_types_module_imports_no_project_code() -> None:
    assert imported_packages(parse_module(SRC_ROOT / "types.py")) == set()


def test_required_layout_present() -> None:
    for name in REQUIRED_ROOT_FILES:
        assert (REPOSITORY_ROOT / name).is_file(), name
    for relative in REQUIRED_DIRECTORIES:
        assert (REPOSITORY_ROOT / relative).is_dir(), relative
    assert (REPOSITORY_ROOT / "configs" / "fedorbit.yaml").is_file()


def test_no_unexpected_root_entries() -> None:
    entries = {path.name for path in REPOSITORY_ROOT.iterdir()}
    assert entries <= ALLOWED_ROOT_ENTRIES, entries - ALLOWED_ROOT_ENTRIES


def test_no_redirect_or_reexport_only_modules() -> None:
    for path in iter_source_files():
        if path.name == "__init__.py":
            assert path.read_text(encoding="utf-8").strip() == "", f"non-empty package init {path}"
            continue
        tree = parse_module(path)
        body = [node for node in tree.body if not _is_future_import(node)]
        assert body and not all(_is_import(node) for node in body), f"re-export-only module {path}"


def _is_future_import(node: object) -> bool:
    import ast

    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


def _is_import(node: object) -> bool:
    import ast

    return isinstance(node, ast.Import | ast.ImportFrom)


def test_workspace_outputs_are_not_tracked_paths() -> None:
    gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
    for directory in ("outputs", "results", "docs/audit/pocs"):
        assert directory in gitignore
    assert isinstance(Path(REPOSITORY_ROOT), Path)
