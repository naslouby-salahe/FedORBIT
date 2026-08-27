from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fedorbit.orbit.objective import ActionSpaceError, CurriculumAction, RobustActionProblem


@dataclass(frozen=True, slots=True)
class CurriculumWeights:
    base_class_weights: NDArray[np.float64]
    action: CurriculumAction

    def __post_init__(self) -> None:
        if self.base_class_weights.shape != self.action.coordinates.shape:
            raise ActionSpaceError("class weights and curriculum action must share coordinates")
        if np.any(self.base_class_weights < 0.0):
            raise ActionSpaceError("base class weights must be nonnegative")

    def multipliers(self) -> NDArray[np.float64]:
        return np.asarray(1.0 + self.action.coordinates, dtype=np.float64)

    def applied(self) -> NDArray[np.float64]:
        return self.base_class_weights * self.multipliers()


def validate_curriculum_action(action: CurriculumAction, support_limit: int) -> None:
    if support_limit < 0:
        raise ActionSpaceError("support limit must be nonnegative")
    if not action.is_within_budget():
        raise ActionSpaceError("curriculum action exceeds total budget")
    if not action.is_support_limited(support_limit):
        raise ActionSpaceError("curriculum action exceeds support limit")


def apply_curriculum(
    problem: RobustActionProblem,
    action: CurriculumAction,
    base_class_weights: NDArray[np.float64],
    support_limit: int,
) -> CurriculumWeights:
    if action.problem is not problem:
        raise ActionSpaceError("curriculum action belongs to a different robust-action problem")
    validate_curriculum_action(action, support_limit)
    return CurriculumWeights(base_class_weights=base_class_weights.copy(), action=action)
