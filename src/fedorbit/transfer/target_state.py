from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from fedorbit.config.models import FedorbitConfig
from fedorbit.models.training import BaseCheckpoint
from fedorbit.response.estimation import ShadowSettings
from fedorbit.response.pilot import PilotData
from fedorbit.response.uncertainty import FinalResponseEstimate, estimate_response_bands


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
            if node_index < 0:
                raise TargetImportanceError(f"negative node index: {node_index}")
            if not math.isfinite(weight):
                raise TargetImportanceError(f"node {node_index} importance is not finite")
            if weight < 0.0:
                raise TargetImportanceError(f"node {node_index} importance must be nonnegative")
        if self.weights_by_node_index and not math.isclose(
            sum(self.weights_by_node_index.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise TargetImportanceError("target importance weights must sum to one")

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
    zero_nodes: dict[int, float] = {}
    for node_risk in node_risks:
        if node_risk.node_index in seen:
            raise TargetImportanceError(f"node {node_risk.node_index} reported more than once")
        seen.add(node_risk.node_index)
        if node_risk.is_actionable:
            floored[node_risk.node_index] = max(node_risk.meta_class_risk, floor)
        else:
            zero_nodes[node_risk.node_index] = 0.0
    if not floored:
        raise TargetImportanceError(
            "no actionable target nodes with META risk; target importance undefined"
        )
    total = sum(floored.values())
    weights = {node_index: value / total for node_index, value in sorted(floored.items())}
    ordered = {
        node_index: ({**zero_nodes, **weights})[node_index]
        for node_index in sorted(seen)
    }
    return TargetImportance(weights_by_node_index=ordered)


def estimate_target_response_diagnostic(
    config: FedorbitConfig,
    model: torch.nn.Module,
    checkpoint: BaseCheckpoint,
    data: PilotData,
    intervention_classes: tuple[int, ...],
    seed: int,
) -> FinalResponseEstimate:
    diagnostic = config.scientific.target_response_diagnostic
    settings = ShadowSettings(
        diagnostic.intervention_magnitude,
        diagnostic.shadow_optimizer_steps,
        data.learning_rate,
        data.weight_decay,
    )
    return estimate_response_bands(
        config,
        model,
        checkpoint,
        data,
        (intervention_classes,),
        settings,
        seed,
        replicate_count=diagnostic.paired_replicates,
        bootstrap_resamples=diagnostic.simultaneous_bootstrap_resamples,
        confidence_level=diagnostic.confidence_level,
        seed_stage="target-local-diagnostic",
    )
