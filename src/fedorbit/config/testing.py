from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict

from fedorbit.config.loading import repository_root


class NonclaimConfigError(ValueError):
    pass


class NonclaimFixtureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_seed: int
    synthetic_instances_per_case: int
    tiny_rows_per_class: int
    tiny_optimizer_steps: int


REGISTERED_NONCLAIM_KEYS = frozenset(
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


def _load_nonclaim(path: Path) -> NonclaimFixtureConfig:
    if not path.is_file():
        raise FileNotFoundError(f"nonclaim configuration missing: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise NonclaimConfigError(f"nonclaim configuration must be a mapping: {path}")
    present = set(cast(dict[str, object], raw))
    unknown = present - REGISTERED_NONCLAIM_KEYS
    if unknown:
        raise NonclaimConfigError(f"unregistered nonclaim keys in {path.name}: {sorted(unknown)}")
    missing = REGISTERED_NONCLAIM_KEYS - present
    if missing:
        raise NonclaimConfigError(f"missing nonclaim keys in {path.name}: {sorted(missing)}")
    return NonclaimFixtureConfig.model_validate(raw)


def load_tests_config(path: Path | None = None) -> NonclaimFixtureConfig:
    config_path = path if path is not None else repository_root() / "configs" / "tests.yml"
    return _load_nonclaim(config_path)


def load_smoke_config(path: Path | None = None) -> NonclaimFixtureConfig:
    config_path = path if path is not None else repository_root() / "configs" / "smoke.yml"
    return _load_nonclaim(config_path)
