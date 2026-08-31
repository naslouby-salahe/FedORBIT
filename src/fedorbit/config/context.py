from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import cache

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig

_bound_config: ContextVar[FedorbitConfig | None] = ContextVar("fedorbit_config", default=None)


@cache
def application_config() -> FedorbitConfig:
    return load_fedorbit_config()


def active_config() -> FedorbitConfig:
    bound = _bound_config.get()
    if bound is not None:
        return bound
    return application_config()


@contextmanager
def configured(config: FedorbitConfig) -> Generator[None]:
    token: Token[FedorbitConfig | None] = _bound_config.set(config)
    try:
        yield
    finally:
        _bound_config.reset(token)
