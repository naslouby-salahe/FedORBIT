from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

from fedorbit.config.loading import repository_root
from fedorbit.config.testing import (
    FORBIDDEN_PRODUCTION_SECTIONS,
    REGISTERED_FIXTURE_KEYS,
    FixtureConfigError,
    FixtureFixtureConfig,
    load_smoke_config,
    load_tests_config,
)


def test_tests_config_loads_with_exact_locked_values() -> None:
    config = load_tests_config()
    assert config.fixture_seed == 0
    assert config.synthetic_instances_per_case == 3
    assert config.tiny_rows_per_class == 64
    assert config.tiny_optimizer_steps == 2
    assert config == FixtureFixtureConfig(
        fixture_seed=0,
        synthetic_instances_per_case=3,
        tiny_rows_per_class=64,
        tiny_optimizer_steps=2,
    )


def test_smoke_config_loads_with_exact_locked_values() -> None:
    config = load_smoke_config()
    assert config.fixture_seed == 0
    assert config.synthetic_instances_per_case == 2
    assert config.tiny_rows_per_class == 64
    assert config.tiny_optimizer_steps == 2
    assert config == FixtureFixtureConfig(
        fixture_seed=0,
        synthetic_instances_per_case=2,
        tiny_rows_per_class=64,
        tiny_optimizer_steps=2,
    )


def test_tests_and_smoke_differ_only_in_instance_count() -> None:
    tests = load_tests_config().model_dump()
    smoke = load_smoke_config().model_dump()
    for key in REGISTERED_FIXTURE_KEYS:
        if key == "synthetic_instances_per_case":
            assert tests[key] != smoke[key]
        else:
            assert tests[key] == smoke[key]


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tests.yml"
    path.write_text(
        "fixture_seed: 0\nsynthetic_instances_per_case: 3\ntiny_rows_per_class: 64\n"
        "tiny_optimizer_steps: 2\ninvented_control: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(FixtureConfigError):
        load_tests_config(path)


def test_missing_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tests.yml"
    path.write_text("fixture_seed: 0\nsynthetic_instances_per_case: 3\n", encoding="utf-8")
    with pytest.raises(FixtureConfigError):
        load_tests_config(path)


def test_scientific_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "smoke.yml"
    path.write_text(
        "fixture_seed: 0\nsynthetic_instances_per_case: 2\ntiny_rows_per_class: 64\n"
        "tiny_optimizer_steps: 2\nscientific:\n  action:\n    principal_sparse_support: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(FixtureConfigError):
        load_smoke_config(path)


def test_no_production_section_shadowing_in_committed_files() -> None:
    for name in ("tests.yml", "smoke.yml"):
        path = repository_root() / "configs" / name
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        assert isinstance(raw, dict)
        present = set(cast(dict[str, object], raw))
        assert present <= REGISTERED_FIXTURE_KEYS
        assert not present & FORBIDDEN_PRODUCTION_SECTIONS


def test_committed_smoke_and_tests_values_are_locked() -> None:
    tests = load_tests_config()
    smoke = load_smoke_config()
    assert tests.synthetic_instances_per_case == 3
    assert smoke.synthetic_instances_per_case == 2
    for config in (tests, smoke):
        assert config.fixture_seed == 0
        assert config.tiny_rows_per_class == 64
        assert config.tiny_optimizer_steps == 2


def test_missing_fixture_config_fails() -> None:
    with pytest.raises(FileNotFoundError):
        load_tests_config(Path("/nonexistent/tests.yml"))
