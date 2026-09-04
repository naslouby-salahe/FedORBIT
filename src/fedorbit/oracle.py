from __future__ import annotations

from dataclasses import dataclass

from fedorbit.optimization.correspondence import BlockCorrespondence
from fedorbit.optimization.objective import CurriculumAction, RobustActionProblem
from fedorbit.types import DatasetId, ExperimentName, TransferMethod


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


ORACLE_METHOD_NAME = TransferMethod.EXACT_MAP_ORACLE.value


@dataclass(frozen=True, slots=True)
class OracleAccessToken:
    experiment: ExperimentName


def authorize_oracle_access(
    experiment: ExperimentName,
    registered_methods: tuple[str, ...],
) -> OracleAccessToken:
    if ORACLE_METHOD_NAME not in registered_methods:
        raise OracleAccessError(f"{experiment.value} is not a registered oracle-method experiment")
    return OracleAccessToken(experiment)


@dataclass(frozen=True, slots=True)
class ExactMapActionOutcome:
    selected_action: CurriculumAction
    objective_value: float


def exact_map_action(
    access: OracleAccessToken,
    problem: RobustActionProblem,
    oracle_correspondence: OracleCorrespondence,
    support_limit: int | None = None,
) -> ExactMapActionOutcome:
    del access
    from fedorbit.methods.baselines import optimize_against_fixed_matrix

    committed = oracle_correspondence.correspondence.permute_response_matrix(
        problem.lower_response_matrix
    )
    solution = optimize_against_fixed_matrix(problem, committed, support_limit)
    return ExactMapActionOutcome(solution.selected_action, solution.objective_value)
