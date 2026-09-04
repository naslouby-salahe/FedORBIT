from __future__ import annotations

import numpy as np
import pytest

from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.optimization.correspondence import (
    BlockCorrespondence,
    build_padded_block_structure,
)
from fedorbit.optimization.objective import build_robust_action_problem
from fedorbit.oracle import (
    OracleAccessError,
    OracleCorrespondence,
    authorize_oracle_access,
    exact_map_action,
)
from fedorbit.types import CoarseGroup, DatasetId, ExperimentName


def _authorize(experiment: ExperimentName):
    methods = build_catalogue().definition(experiment).methods
    return authorize_oracle_access(experiment, methods)


def _problem():
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION,), {CoarseGroup.DISRUPTION: 2}, {CoarseGroup.DISRUPTION: 2}
    )
    matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    return build_robust_action_problem(blocks, matrix, matrix, np.array([0.5, 0.5]), (0, 1))


def test_authorize_oracle_access_accepts_registered_oracle_experiments() -> None:
    token = _authorize(ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION)
    assert token.experiment == ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION


def test_authorize_oracle_access_rejects_non_oracle_experiments() -> None:
    with pytest.raises(OracleAccessError):
        _authorize(ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION)


def test_exact_map_action_requires_an_authorized_token() -> None:
    problem = _problem()
    blocks = problem.blocks
    correspondence = OracleCorrespondence(
        DatasetId.TON_IOT_WINDOWS10_HOST,
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        BlockCorrespondence.identity(blocks),
    )
    token = _authorize(ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION)
    outcome = exact_map_action(token, problem, correspondence)
    assert np.isfinite(outcome.objective_value)
