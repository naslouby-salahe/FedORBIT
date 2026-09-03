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
    "src/fedorbit/config",
    "src/fedorbit/datasets",
    "src/fedorbit/datasets/edge_iiotset",
    "src/fedorbit/datasets/ton_iot",
    "src/fedorbit/learning",
    "src/fedorbit/response",
    "src/fedorbit/optimization",
    "src/fedorbit/methods",
    "src/fedorbit/experiments",
    "src/fedorbit/analysis",
    "src/fedorbit/infrastructure",
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
}

REQUIRED_SOURCE_FILES = {
    "src/fedorbit/types.py",
    "src/fedorbit/interface.py",
    "src/fedorbit/oracle.py",
    "src/fedorbit/reporting.py",
    "src/fedorbit/cli.py",
    "src/fedorbit/config/loading.py",
    "src/fedorbit/config/models.py",
    "src/fedorbit/config/validation.py",
    "src/fedorbit/datasets/common.py",
    "src/fedorbit/datasets/preprocessing.py",
    "src/fedorbit/datasets/splitting.py",
    "src/fedorbit/datasets/ontology.py",
    "src/fedorbit/datasets/edge_iiotset/loader.py",
    "src/fedorbit/datasets/edge_iiotset/schema.py",
    "src/fedorbit/datasets/edge_iiotset/validation.py",
    "src/fedorbit/datasets/ton_iot/loader.py",
    "src/fedorbit/datasets/ton_iot/components.py",
    "src/fedorbit/datasets/ton_iot/validation.py",
    "src/fedorbit/learning/models.py",
    "src/fedorbit/learning/pilot.py",
    "src/fedorbit/learning/scoring.py",
    "src/fedorbit/learning/training.py",
    "src/fedorbit/response/pilot.py",
    "src/fedorbit/response/estimation.py",
    "src/fedorbit/response/uncertainty.py",
    "src/fedorbit/response/packet.py",
    "src/fedorbit/optimization/correspondence.py",
    "src/fedorbit/optimization/objective.py",
    "src/fedorbit/optimization/diagnostics.py",
    "src/fedorbit/optimization/assignment.py",
    "src/fedorbit/optimization/exact_sparse.py",
    "src/fedorbit/optimization/exact_qap.py",
    "src/fedorbit/optimization/dense_ccp.py",
    "src/fedorbit/optimization/certificates.py",
    "src/fedorbit/methods/target.py",
    "src/fedorbit/methods/confirmation.py",
    "src/fedorbit/methods/assimilation.py",
    "src/fedorbit/methods/baselines.py",
    "src/fedorbit/experiments/catalogue.py",
    "src/fedorbit/experiments/cells.py",
    "src/fedorbit/experiments/synthetic.py",
    "src/fedorbit/analysis/records.py",
    "src/fedorbit/analysis/metrics.py",
    "src/fedorbit/analysis/comparisons.py",
    "src/fedorbit/analysis/statistics.py",
    "src/fedorbit/infrastructure/workspace.py",
    "src/fedorbit/infrastructure/manifests.py",
    "src/fedorbit/infrastructure/provenance.py",
    "src/fedorbit/infrastructure/planner.py",
    "src/fedorbit/infrastructure/reuse.py",
    "src/fedorbit/infrastructure/execution.py",
    "src/fedorbit/infrastructure/runtime.py",
    "src/fedorbit/infrastructure/environment.py",
    "src/fedorbit/infrastructure/failures.py",
}

REQUIRED_SYNTHETIC_MODULES = {
    "src/fedorbit/experiments/synthetic.py",
}

REQUIRED_TRANSFER_MODULES = {
    "src/fedorbit/methods/target.py",
    "src/fedorbit/methods/confirmation.py",
    "src/fedorbit/methods/assimilation.py",
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


def test_target_source_files_exist() -> None:
    missing = {
        relative for relative in REQUIRED_SOURCE_FILES if not (REPOSITORY_ROOT / relative).is_file()
    }
    assert not missing, f"missing target source files: {sorted(missing)}"


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
    expected = {"fedorbit.yaml"}
    assert actual == expected, f"unexpected config files: {actual - expected}"


def test_no_agent_instruction_files_created_in_src() -> None:
    for name in ("AGENTS.md", "CLAUDE.md", "agent_notes.md", "progress.md"):
        for root in (SRC_ROOT, TESTS_ROOT):
            assert not (root / name).exists(), f"forbidden planning file: {root / name}"
