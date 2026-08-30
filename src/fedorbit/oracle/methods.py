from __future__ import annotations

from dataclasses import dataclass

from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import TransferMethod
from fedorbit.oracle.mapping import OracleCorrespondence
from fedorbit.orbit.objective import CurriculumAction, RobustActionProblem


class OracleAccessError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExactMapActionOutcome:
    selected_action: CurriculumAction
    objective_value: float


def exact_map_action(
    problem: RobustActionProblem,
    oracle_correspondence: OracleCorrespondence,
    config: FedorbitConfig,
    support_limit: int | None = None,
) -> ExactMapActionOutcome:
    from fedorbit.baselines.local import optimize_against_fixed_matrix

    committed = oracle_correspondence.correspondence.permute_response_matrix(
        problem.lower_response_matrix
    )
    solution = optimize_against_fixed_matrix(problem, committed, config, support_limit)
    return ExactMapActionOutcome(solution.selected_action, solution.objective_value)


ORACLE_METHOD_NAME = TransferMethod.EXACT_MAP_ORACLE.value
