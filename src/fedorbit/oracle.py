from __future__ import annotations

from dataclasses import dataclass

from fedorbit.optimization.correspondence import BlockCorrespondence
from fedorbit.optimization.objective import CurriculumAction, RobustActionProblem
from fedorbit.types import DatasetId, TransferMethod


class OracleMappingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OracleCorrespondence:
    source_client: DatasetId
    target_client: DatasetId
    correspondence: BlockCorrespondence

    def __post_init__(self) -> None:
        if self.source_client == self.target_client:
            raise OracleMappingError(
                "oracle correspondence requires distinct source and target clients"
            )


class OracleAccessError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExactMapActionOutcome:
    selected_action: CurriculumAction
    objective_value: float


def exact_map_action(
    problem: RobustActionProblem,
    oracle_correspondence: OracleCorrespondence,
    support_limit: int | None = None,
) -> ExactMapActionOutcome:
    from fedorbit.methods.baselines import optimize_against_fixed_matrix

    committed = oracle_correspondence.correspondence.permute_response_matrix(
        problem.lower_response_matrix
    )
    solution = optimize_against_fixed_matrix(problem, committed, support_limit)
    return ExactMapActionOutcome(solution.selected_action, solution.objective_value)


ORACLE_METHOD_NAME = TransferMethod.EXACT_MAP_ORACLE.value
