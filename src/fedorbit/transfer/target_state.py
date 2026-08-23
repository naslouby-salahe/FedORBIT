from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fedorbit.config.models import FedorbitConfig


class TargetImportanceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TransferNodeRisk:
    node_index: int
    is_actionable: bool
    meta_class_risk: float

    def __post_init__(self) -> None:
        if self.node_index < 0:
            raise TargetImportanceError(f"negative node index: {self.node_index}")
        if not math.isfinite(self.meta_class_risk):
            raise TargetImportanceError(
                f"node {self.node_index} META class risk is not finite: {self.meta_class_risk}"
            )
        if self.meta_class_risk < 0.0:
            raise TargetImportanceError(
                f"node {self.node_index} META class risk must be nonnegative"
            )


@dataclass(frozen=True, slots=True)
class TargetImportance:
    weights_by_node_index: Mapping[int, float]

    def __post_init__(self) -> None:
        for node_index, weight in self.weights_by_node_index.items():
            if not math.isfinite(weight):
                raise TargetImportanceError(f"node {node_index} importance is not finite")
            if weight < 0.0:
                raise TargetImportanceError(f"node {node_index} importance must be nonnegative")

    def weight_of(self, node_index: int) -> float:
        return self.weights_by_node_index[node_index]

    def as_vector(self, size: int) -> NDArray[np.float64]:
        vector = np.zeros(size, dtype=np.float64)
        for node_index, weight in self.weights_by_node_index.items():
            if node_index >= size:
                raise TargetImportanceError(f"node {node_index} outside vector size {size}")
            vector[node_index] = weight
        return vector

    @property
    def actionable_total(self) -> float:
        return sum(self.weights_by_node_index.values())


def build_target_importance(
    config: FedorbitConfig,
    node_risks: tuple[TransferNodeRisk, ...],
) -> TargetImportance:
    floor = config.scientific.target_importance.class_risk_floor
    if floor <= 0.0:
        raise TargetImportanceError("class risk floor must be positive")
    seen: set[int] = set()
    floored: dict[int, float] = {}
    for node_risk in node_risks:
        if node_risk.node_index in seen:
            raise TargetImportanceError(f"node {node_risk.node_index} reported more than once")
        seen.add(node_risk.node_index)
        if node_risk.is_actionable:
            floored[node_risk.node_index] = max(node_risk.meta_class_risk, floor)
    if not floored:
        raise TargetImportanceError(
            "no actionable target nodes with META risk; target importance undefined"
        )
    total = sum(floored.values())
    weights = {node_index: value / total for node_index, value in sorted(floored.items())}
    zero_nodes = {
        node_risk.node_index: 0.0 for node_risk in node_risks if not node_risk.is_actionable
    }
    return TargetImportance(weights_by_node_index={**zero_nodes, **weights})
