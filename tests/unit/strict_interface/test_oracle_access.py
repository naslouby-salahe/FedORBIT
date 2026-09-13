from __future__ import annotations

import pytest

from fedorbit.experiments.catalogue import CatalogueMethodLabel, build_catalogue
from fedorbit.oracle import (
    OracleAccessError,
    authorize_oracle_access,
)
from fedorbit.types import ExperimentName


def _authorize(experiment: ExperimentName):
    methods = build_catalogue().definition(experiment).methods
    registered_methods = tuple(
        method for method in methods if not isinstance(method, CatalogueMethodLabel)
    )
    return authorize_oracle_access(experiment, registered_methods)


def test_authorize_oracle_access_accepts_registered_oracle_experiments() -> None:
    token = _authorize(ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION)
    assert token.experiment == ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION


def test_authorize_oracle_access_rejects_non_oracle_experiments() -> None:
    with pytest.raises(OracleAccessError):
        _authorize(ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION)
