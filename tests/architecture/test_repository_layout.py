from __future__ import annotations

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


def test_repository_root_entries_are_allowed() -> None:
    machine_local = {
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".nox",
        "__pycache__",
        ".scannerwork",
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


def test_data_raw_is_symlink_to_immutable_external_tree() -> None:
    link = REPOSITORY_ROOT / "data" / "raw"
    assert link.is_symlink(), "data/raw must be a symlink to the immutable raw-data tree"
    target = link.readlink()
    assert not target.is_absolute(), "data/raw must use a repository-relative external-data link"
    assert target.as_posix().endswith("datp-shared-data/raw")


def test_no_markdown_planning_documents_in_repo_root() -> None:
    root_markdown = {path.name for path in REPOSITORY_ROOT.glob("*.md")}
    allowed = {"README.md", "CLAUDE.md"}
    unexpected = root_markdown - allowed
    assert not unexpected, f"unexpected markdown at repository root: {sorted(unexpected)}"


def test_no_generated_workspaces_committed() -> None:
    assert not (REPOSITORY_ROOT / "outputs").exists(), "outputs/ must not be committed"
    assert not (REPOSITORY_ROOT / "results").exists(), "results/ must not be committed"


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
