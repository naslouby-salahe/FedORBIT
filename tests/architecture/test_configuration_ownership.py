from __future__ import annotations

import ast
import re

from fedorbit.config import models
from fedorbit.config.loading import load_config
from fedorbit.types import FrozenModel
from tests.architecture.scan import SRC_ROOT, iter_source_files, parse_module


def _model_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    for name in dir(models):
        candidate = getattr(models, name)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, FrozenModel)
            and candidate is not FrozenModel
        ):
            for field in candidate.model_fields:
                fields[field] = name
    return fields


def test_every_configuration_field_is_consumed_by_production_code() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in iter_source_files()
        if path.parts[-3:-1] != ("fedorbit", "config") or path.name == "loading.py"
    )
    for field, owner in _model_fields().items():
        assert re.search(rf"\.{field}\b|{field}=", text), f"{owner}.{field} is never consumed"


def test_configured_seed_is_not_hardcoded_in_source() -> None:
    seed = load_config().base_seed
    for path in iter_source_files():
        for node in ast.walk(parse_module(path)):
            if isinstance(node, ast.Constant) and node.value == seed:
                raise AssertionError(f"configured seed hardcoded in {path}")


def test_governed_values_are_not_redeclared_in_source() -> None:
    config = load_config()
    governed = {
        config.statistics.saturation_headroom,
        config.statistics.bootstrap_resamples,
        config.operating_point.nominal_false_positive_rate,
        config.detectors.autoencoder.training_steps,
        config.datasets.nbaiot.attack_rows_per_file,
    }
    for path in SRC_ROOT.rglob("*.py"):
        if path.parts[-2] == "config":
            continue
        for node in ast.walk(parse_module(path)):
            if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
                assert node.value not in governed or node.value in {0, 1}, (
                    f"governed value {node.value} appears in {path}:{node.lineno}"
                )
