from __future__ import annotations

import subprocess
from pathlib import Path

from tests.architecture.scan import (
    ALLOWED_ROOT_ENTRIES,
    REPOSITORY_ROOT,
    SRC_ROOT,
    TESTS_ROOT,
    iter_source_files,
)

REQUIRED_ROOT_FILES = {
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "noxfile.py",
    "Makefile",
    ".gitignore",
}

REQUIRED_PACKAGE_DIRECTORIES = {
    "src/fedorbit/domain",
    "src/fedorbit/config",
    "src/fedorbit/datasets",
    "src/fedorbit/datasets/edge_iiotset",
    "src/fedorbit/datasets/ton_iot",
    "src/fedorbit/models",
    "src/fedorbit/training",
    "src/fedorbit/response",
    "src/fedorbit/strict_interface",
    "src/fedorbit/orbit",
    "src/fedorbit/solvers",
    "src/fedorbit/transfer",
    "src/fedorbit/baselines",
    "src/fedorbit/oracle",
    "src/fedorbit/synthetic",
    "src/fedorbit/experiments",
    "src/fedorbit/evaluation",
    "src/fedorbit/analysis",
    "src/fedorbit/artifacts",
    "src/fedorbit/execution",
    "src/fedorbit/runtime",
    "src/fedorbit/reporting",
    "src/fedorbit/cli",
    "src/fedorbit/cli/commands",
}

REQUIRED_TEST_DIRECTORIES = {
    "tests/architecture",
    "tests/unit",
    "tests/unit/config",
    "tests/unit/runtime",
    "tests/scientific",
    "tests/integration",
    "tests/e2e",
    "tests/smoke",
}

REQUIRED_CONFIG_FILES = {
    "configs/fedorbit.yaml",
    "configs/tests.yml",
    "configs/smoke.yml",
    "configs/scientific_contract_snapshot.json",
}

REQUIRED_SYNTHETIC_MODULES = {
    "src/fedorbit/synthetic/generators.py",
    "src/fedorbit/synthetic/exactness.py",
    "src/fedorbit/synthetic/mechanisms.py",
    "src/fedorbit/synthetic/scalability.py",
}

REQUIRED_TRANSFER_MODULES = {
    "src/fedorbit/transfer/curriculum.py",
}


def test_repository_root_entries_are_allowed() -> None:
    machine_local = {
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".nox",
        "__pycache__",
        ".scannerwork",
        ".import_linter_cache",
        "outputs",
        "results",
    }
    actual = {
        entry.name
        for entry in REPOSITORY_ROOT.iterdir()
        if entry.name != ".git" and entry.name not in machine_local
    }
    unexpected = actual - ALLOWED_ROOT_ENTRIES
    assert not unexpected, f"unexpected repository-root entries: {sorted(unexpected)}"


def test_required_root_files_exist() -> None:
    missing = {name for name in REQUIRED_ROOT_FILES if not (REPOSITORY_ROOT / name).is_file()}
    assert not missing, f"missing required root files: {sorted(missing)}"


def test_required_package_directories_exist() -> None:
    missing = {
        relative
        for relative in REQUIRED_PACKAGE_DIRECTORIES
        if not (REPOSITORY_ROOT / relative).is_dir()
    }
    assert not missing, f"missing required package directories: {sorted(missing)}"


def test_every_package_directory_has_init() -> None:
    for relative in REQUIRED_PACKAGE_DIRECTORIES:
        assert (REPOSITORY_ROOT / relative / "__init__.py").is_file(), (
            f"missing __init__.py in {relative}"
        )


def test_required_test_directories_exist() -> None:
    missing = {
        relative
        for relative in REQUIRED_TEST_DIRECTORIES
        if not (REPOSITORY_ROOT / relative).is_dir()
    }
    assert not missing, f"missing required test directories: {sorted(missing)}"


def test_required_config_files_exist() -> None:
    missing = {
        relative for relative in REQUIRED_CONFIG_FILES if not (REPOSITORY_ROOT / relative).is_file()
    }
    assert not missing, f"missing required config files: {sorted(missing)}"


def test_roadmap_required_synthetic_modules_exist() -> None:
    missing = {
        relative
        for relative in REQUIRED_SYNTHETIC_MODULES
        if not (REPOSITORY_ROOT / relative).is_file()
    }
    assert not missing, f"missing roadmap synthetic modules: {sorted(missing)}"


def test_roadmap_required_transfer_modules_exist() -> None:
    missing = {
        relative
        for relative in REQUIRED_TRANSFER_MODULES
        if not (REPOSITORY_ROOT / relative).is_file()
    }
    assert not missing, f"missing roadmap transfer modules: {sorted(missing)}"


def test_data_raw_is_symlink_to_immutable_external_tree() -> None:
    link = REPOSITORY_ROOT / "data" / "raw"
    if link.is_symlink():
        target = link.readlink()
    else:
        target = Path(link.read_text(encoding="utf-8").strip())
    assert not target.is_absolute(), "data/raw must use a repository-relative external-data link"
    assert target.as_posix().endswith("datp-shared-data/raw")


def test_no_markdown_planning_documents_in_repo_root() -> None:
    root_markdown = {path.name for path in REPOSITORY_ROOT.glob("*.md")}
    allowed = {"README.md", "CLAUDE.md"}
    unexpected = root_markdown - allowed
    assert not unexpected, f"unexpected markdown at repository root: {sorted(unexpected)}"


def test_no_generated_workspaces_committed() -> None:
    for workspace in ("outputs", "results"):
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", workspace],
            cwd=REPOSITORY_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert tracked.returncode != 0, f"generated workspace is committed: {workspace}"


def test_no_temp_planning_directory() -> None:
    temp = REPOSITORY_ROOT / "docs" / "temp"
    assert not temp.exists(), "docs/temp/ must not exist in a clean repository"


def test_no_unexpected_files_under_src() -> None:
    suspicious_extensions = {".csv", ".json", ".yaml", ".yml", ".ipynb", ".txt", ".log"}
    for path in SRC_ROOT.rglob("*"):
        if path.is_file() and path.suffix in suspicious_extensions:
            raise AssertionError(f"unexpected file under src: {path}")


def test_no_unexpected_files_under_tests() -> None:
    forbidden = {".ipynb", ".log", ".tmp", ".bak"}
    for path in TESTS_ROOT.rglob("*"):
        if path.is_file() and path.suffix in forbidden:
            raise AssertionError(f"unexpected file under tests: {path}")


def test_no_scratch_or_cache_directories_in_source_tree() -> None:
    for path in iter_source_files():
        assert "__pycache__" not in path.parts
        assert ".pytest_cache" not in path.parts


def test_configs_directory_contains_only_contract_files() -> None:
    configs = REPOSITORY_ROOT / "configs"
    actual = {path.name for path in configs.iterdir() if path.is_file()}
    expected = {"fedorbit.yaml", "tests.yml", "smoke.yml", "scientific_contract_snapshot.json"}
    assert actual == expected, f"unexpected config files: {actual - expected}"


def test_no_agent_instruction_files_created_in_src() -> None:
    for name in ("AGENTS.md", "CLAUDE.md", "agent_notes.md", "progress.md"):
        for root in (SRC_ROOT, TESTS_ROOT):
            assert not (root / name).exists(), f"forbidden planning file: {root / name}"
