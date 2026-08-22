from __future__ import annotations

import ast

from tests.architecture.scan import (
    LOCKED_VALUE_CONSTANT_PATTERN,
    iter_source_files,
    module_level_constants,
    parse_module,
)

CONFIRMATORY_SEED_VALUES = {1103, 2207, 3319, 4421, 5531, 6653, 7753, 8861, 9973, 11027}
PILOT_SEED_VALUES = {101, 202, 303}
LOCKED_NUMERIC_LITERALS = {0.5, 0.25, 0.01, 0.005, 0.05, 200, 40, 512, 0.95, 0.999}


def test_no_locked_config_constants_redeclared_at_module_level() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for assignment in module_level_constants(tree):
            for target in assignment.targets:
                if isinstance(target, ast.Name):
                    assert target.id not in LOCKED_VALUE_CONSTANT_PATTERN, (
                        f"locked config value redeclared as constant {target.id} in {path}"
                    )


def test_no_seed_values_hardcoded_in_production() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, int)
                and node.value in (CONFIRMATORY_SEED_VALUES | PILOT_SEED_VALUES)
            ):
                raise AssertionError(f"hardcoded seed {node.value} in {path}:{node.lineno}")


def test_no_locked_numeric_literals_as_module_constants() -> None:
    for path in iter_source_files():
        tree = parse_module(path)
        for assignment in module_level_constants(tree):
            value = assignment.value
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, (int, float))
                and value.value in LOCKED_NUMERIC_LITERALS
            ):
                raise AssertionError(f"locked numeric literal {value.value} as constant in {path}")


def test_no_governed_values_in_cli_defaults() -> None:
    cli_paths = [path for path in iter_source_files() if "cli" in path.parts]
    for path in cli_paths:
        text = path.read_text(encoding="utf-8")
        for literal in ("0.5", "0.25", "0.01", "0.005", "200", "40", "512"):
            assert f"default={literal}" not in text, f"governed value as CLI default in {path}"


def test_no_observed_data_facts_hardcoded() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for marker in ("row_count", "file_count", "sha256_of_raw", "observed_rows"):
            assert marker not in text, f"observed-data fact hardcoded in {path}"
