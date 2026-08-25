from __future__ import annotations

import re
from typing import cast

import yaml

from tests.architecture.scan import REPOSITORY_ROOT, iter_source_files

CONFIG_PATH = REPOSITORY_ROOT / "configs" / "fedorbit.yaml"
SCIENTIFIC_SECTION_NAMES = {
    "action",
    "materiality",
    "transfer_support",
    "datasets",
    "split",
    "preprocessing",
    "training",
    "base_model_pilot",
    "source_response_pilot",
    "source_response_final",
    "target_response_diagnostic",
    "confirmation",
    "target_optimizer_budget",
    "baselines",
    "target_importance",
    "randomness",
    "statistics",
    "evaluation_criteria",
    "metrics",
    "multi_source_selection",
    "simplification_rules",
}


def _config_keys() -> set[str]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    keys: set[str] = set()
    for section in raw:
        if not isinstance(raw[section], dict):
            continue
        for key in raw[section]:
            keys.add(key)
    return keys


def test_configuration_is_single_source_of_scientific_values() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    scientific = raw["scientific"]
    assert set(scientific.keys()) == SCIENTIFIC_SECTION_NAMES


def test_no_parallel_configuration_files() -> None:
    configs = REPOSITORY_ROOT / "configs"
    allowed = {"fedorbit.yaml", "tests.yml", "smoke.yml", "scientific_contract_snapshot.json"}
    for path in configs.iterdir():
        assert path.name in allowed, f"unexpected configuration file: {path.name}"


def test_no_scientific_keys_in_test_or_smoke_configs() -> None:
    for name in ("tests.yml", "smoke.yml"):
        path = REPOSITORY_ROOT / "configs" / name
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        assert isinstance(raw, dict)
        for key in cast(dict[str, object], raw):
            assert key not in SCIENTIFIC_SECTION_NAMES, (
                f"forbidden scientific section in {name}: {key}"
            )


def test_no_configuration_key_duplicated_in_production_code() -> None:
    configured_keys = _config_keys()
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for key in configured_keys:
            pattern = rf"\b{re.escape(key)}\s*=\s*[0-9]"
            if re.search(pattern, text):
                raise AssertionError(f"configuration key {key!r} assigned a value in {path}")


def test_no_environment_variable_scientific_overrides() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for marker in ("os.environ[", "os.getenv(", "getenv("):
            assert marker not in text, f"environment override in {path}: {marker}"


def test_no_environment_yml_or_toml_scientific_override() -> None:
    allowed_root_configs = {"pyproject.toml", "uv.lock"}
    for path in REPOSITORY_ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".yml", ".yaml", ".toml"}:
            if (
                "configs" in path.parts
                or ".github" in path.parts
                or ".venv" in path.parts
                or ".git" in path.parts
            ):
                continue
            if (
                path.name in allowed_root_configs
                and len(path.parts) == len(REPOSITORY_ROOT.parts) + 1
            ):
                continue
            raise AssertionError(f"unexpected configuration file: {path}")


def test_snapshot_is_generated_not_hand_edited() -> None:
    snapshot = REPOSITORY_ROOT / "configs" / "scientific_contract_snapshot.json"
    text = snapshot.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\n\n" not in text


def test_every_configured_top_level_section_has_a_typed_consumer() -> None:
    from fedorbit.config.loading import load_fedorbit_config

    with CONFIG_PATH.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert isinstance(raw, dict)
    config = load_fedorbit_config()
    for section in cast(dict[str, object], raw):
        assert hasattr(config, section), f"configured section {section!r} has no typed consumer"
