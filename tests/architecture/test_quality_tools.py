from __future__ import annotations

import ast
import subprocess
import sys
from enum import StrEnum

from fedorbit import types as domain_types
from fedorbit.cli import app
from fedorbit.config.loading import load_config
from fedorbit.types import CliCommand, DetectorKind, ExperimentId, ShareChannel
from tests.architecture.scan import REPOSITORY_ROOT, SRC_ROOT, iter_source_files, parse_module


def run_tool(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_vulture_reports_no_dead_production_code() -> None:
    result = run_tool("vulture")
    assert result.returncode == 0, result.stdout + result.stderr


def test_deptry_reports_no_dependency_problems() -> None:
    result = run_tool("deptry", ".")
    assert result.returncode == 0, result.stdout + result.stderr


def enum_members() -> dict[str, str]:
    members: dict[str, str] = {}
    for name in dir(domain_types):
        candidate = getattr(domain_types, name)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, StrEnum)
            and candidate is not StrEnum
        ):
            for member in candidate:
                members[f"{name}.{member.name}"] = member.value
    return members


def test_every_enum_member_is_used_in_source_or_configuration() -> None:
    attribute_uses: set[tuple[str, str]] = set()
    for path in iter_source_files():
        for node in ast.walk(parse_module(path)):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                attribute_uses.add((node.value.id, node.attr))
    configuration = (REPOSITORY_ROOT / "configs" / "fedorbit.yaml").read_text(encoding="utf-8")
    for qualified, value in enum_members().items():
        owner, member = qualified.split(".")
        used_in_code = (owner, member) in attribute_uses
        used_in_configuration = value in configuration
        assert used_in_code or used_in_configuration, f"{qualified} is unused"


def test_enum_typed_domain_values_are_not_free_form_strings() -> None:
    forbidden = {
        member.value for enum in (ExperimentId, ShareChannel, DetectorKind) for member in enum
    }
    for path in iter_source_files():
        if path.name == "types.py":
            continue
        for node in ast.walk(parse_module(path)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in forbidden, f"free-form enum value in {path}"


def test_every_configured_experiment_is_runnable_from_the_cli() -> None:
    commands = {command.name for command in app.registered_commands}
    assert {command.value for command in CliCommand} == commands
    assert {experiment.id for experiment in load_config().experiments} == set(ExperimentId)


def test_source_root_contains_only_declared_top_level_modules() -> None:
    modules = {path.stem for path in SRC_ROOT.glob("*.py") if path.stem != "__init__"}
    assert modules == {"types", "cli", "reporting"}
