from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import pytest
import yaml

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from tests.typed_access import ConfigDocument


@pytest.fixture(scope="session")
def fedorbit_config() -> FedorbitConfig:
    return load_fedorbit_config()


@pytest.fixture(scope="session")
def config_dict() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "configs" / "fedorbit.yaml"
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    return cast(dict[str, object], raw)


@pytest.fixture()
def mutable_config(config_dict: dict[str, object]) -> ConfigDocument:
    return ConfigDocument(copy.deepcopy(config_dict))
