from __future__ import annotations

import ast

from tests.architecture.scan import (
    LOCKED_VALUE_CONSTANT_PATTERN,
    iter_source_files,
    module_level_constants,
    parse_module,
)

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

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


ROADMAP_LOCKED_ARCHITECTURE_VALUES = frozenset({0.1})
STRUCTURAL_IDENTITY_VALUES = frozenset({0.0, 1.0})


def _config_numeric_values() -> frozenset[float]:
    import json

    from fedorbit.config.loading import load_fedorbit_config

    config = load_fedorbit_config()
    collected: set[float] = set()
    raw = json.loads(config.model_dump_json())

    def walk(value: JsonValue) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            collected.add(float(value))
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(raw)
    return frozenset(collected)


def test_no_governed_config_values_literalized_in_production() -> None:
    governed = _config_numeric_values()
    for path in iter_source_files():
        tree = parse_module(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, float)
                and float(node.value) in governed
                and node.value not in ROADMAP_LOCKED_ARCHITECTURE_VALUES
                and node.value not in STRUCTURAL_IDENTITY_VALUES
            ):
                raise AssertionError(
                    f"governed config value {node.value} literalized in {path}:{node.lineno}"
                )


ROADMAP_LOCKED_STRING_VALUES = frozenset(
    {
        "normal",
        "preprocessing",
        "scientific",
        "solvers",
        "generators",
        "experiments",
        "runtime",
        "environment",
        "exact_orbit",
        "outputs",
        "results",
        "inventories",
        "validation",
        "prepared",
        "splits",
        "features",
        "metadata",
        "cache",
        "staging",
        "artifacts",
        "project_summary",
        "models",
        "scores",
        "fitted",
        "baselines",
        "derived",
        "evaluation",
        "analysis",
        "diagnostics",
        "data",
        "eligibility",
        "pilot_selection",
        "training",
        "scoring",
        "response",
        "target_importance",
        "correspondence",
        "confirmation",
        "multi_source_selection",
        "statistics",
        "reporting",
        "frame.time",
    }
)


def _config_string_values() -> frozenset[str]:
    import json

    from fedorbit.config.loading import load_fedorbit_config

    config = load_fedorbit_config()
    collected: set[str] = set()
    raw = json.loads(config.model_dump_json())

    def walk(value: JsonValue) -> None:
        if isinstance(value, str) and len(value) > 2:
            collected.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(raw)
    return frozenset(collected)


def test_no_governed_config_strings_literalized_in_production() -> None:
    governed = _config_string_values()
    for path in iter_source_files():
        if path.name == "enums.py":
            continue
        tree = parse_module(path)
        dict_key_nodes = {
            key
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if key is not None
        }
        for node in ast.walk(tree):
            if node in dict_key_nodes:
                continue
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and len(node.value) > 2
                and node.value in governed
                and node.value not in ROADMAP_LOCKED_STRING_VALUES
            ):
                raise AssertionError(
                    f"governed config string {node.value!r} literalized in {path}:{node.lineno}"
                )
