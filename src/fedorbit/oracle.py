from __future__ import annotations

from dataclasses import dataclass

from fedorbit.types import (
    ExperimentName,
    MethodName,
    TransferMethod,
)


class OracleAccessError(RuntimeError):
    pass


ORACLE_METHOD = TransferMethod.EXACT_MAP_ORACLE


@dataclass(frozen=True, slots=True)
class OracleAccessToken:
    experiment: ExperimentName


def authorize_oracle_access(
    experiment: ExperimentName,
    registered_methods: tuple[MethodName, ...],
) -> OracleAccessToken:
    if ORACLE_METHOD not in registered_methods:
        raise OracleAccessError(f"{experiment} is not a registered oracle-method experiment")
    return OracleAccessToken(experiment)
