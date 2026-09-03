from __future__ import annotations

import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import cache
from pathlib import Path

import yaml

from fedorbit.config.models import FedorbitConfig
from fedorbit.config.validation import validate_cross_field_contract

_bound_config: ContextVar[FedorbitConfig | None] = ContextVar("fedorbit_config", default=None)


def repository_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("FedORBIT repository root not found from package location")


def default_config_path() -> Path:
    return repository_root() / "configs" / "fedorbit.yaml"


def load_fedorbit_config(path: Path | None = None) -> FedorbitConfig:
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Configuration file must contain a mapping: {config_path}")
    config = FedorbitConfig.model_validate(raw)
    validate_cross_field_contract(config)
    return config


@cache
def application_config() -> FedorbitConfig:
    return load_fedorbit_config()


def active_config() -> FedorbitConfig:
    bound = _bound_config.get()
    return bound if bound is not None else application_config()


@contextmanager
def configured(config: FedorbitConfig) -> Generator[None]:
    token: Token[FedorbitConfig | None] = _bound_config.set(config)
    try:
        yield
    finally:
        _bound_config.reset(token)


def stable_json(config: FedorbitConfig) -> str:
    payload = config.model_dump(mode="json")
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
