from __future__ import annotations

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import TransferMethod
from fedorbit.oracle.mapping import OracleCorrespondence
from fedorbit.orbit.objective import CurriculumAction, RobustActionProblem


class OracleAccessError(RuntimeError):
    pass


def exact_map_action(
    problem: RobustActionProblem,
    oracle_correspondence: OracleCorrespondence,
    config: FedorbitConfig,
    support_limit: int | None = None,
) -> tuple[CurriculumAction, float]:
    from fedorbit.baselines.local import optimize_against_fixed_matrix

    committed = oracle_correspondence.correspondence.permute_response_matrix(
        problem.lower_response_matrix
    )
    solution = optimize_against_fixed_matrix(problem, committed, config, support_limit)
    return solution.selected_action, solution.objective_value


ORACLE_METHOD_NAME = TransferMethod.EXACT_MAP_ORACLE.value
