from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict

from fedorbit.config.loading import repository_root
from fedorbit.domain.serialization import StableJsonPayload


class FixtureConfigError(ValueError):
    pass


class FixtureFixtureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_seed: int
    synthetic_instances_per_case: int
    tiny_rows_per_class: int
    tiny_optimizer_steps: int


REGISTERED_FIXTURE_KEYS = frozenset(
    {"fixture_seed", "synthetic_instances_per_case", "tiny_rows_per_class", "tiny_optimizer_steps"}
)

FORBIDDEN_PRODUCTION_SECTIONS = frozenset(
    {
        "scientific",
        "solvers",
        "generators",
        "experiments",
        "runtime",
        "environment",
        "reporting",
    }
)


def _load_fixture(path: Path) -> FixtureFixtureConfig:
    if not path.is_file():
        raise FileNotFoundError(f"fixture configuration missing: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise FixtureConfigError(f"fixture configuration must be a mapping: {path}")
    present = set(cast(Mapping[str, StableJsonPayload], raw))
    unknown = present - REGISTERED_FIXTURE_KEYS
    if unknown:
        raise FixtureConfigError(f"unregistered fixture keys in {path.name}: {sorted(unknown)}")
    missing = REGISTERED_FIXTURE_KEYS - present
    if missing:
        raise FixtureConfigError(f"missing fixture keys in {path.name}: {sorted(missing)}")
    return FixtureFixtureConfig.model_validate(raw)


def load_tests_config(path: Path | None = None) -> FixtureFixtureConfig:
    config_path = path if path is not None else repository_root() / "configs" / "tests.yml"
    return _load_fixture(config_path)


def load_smoke_config(path: Path | None = None) -> FixtureFixtureConfig:
    config_path = path if path is not None else repository_root() / "configs" / "smoke.yml"
    return _load_fixture(config_path)
