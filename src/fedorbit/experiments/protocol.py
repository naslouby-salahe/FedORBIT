from __future__ import annotations

from dataclasses import dataclass

from fedorbit.experiments.catalogue import ExperimentDefinition
from fedorbit.types import ExperimentName, OverwritePolicy


@dataclass(frozen=True, slots=True)
class ExperimentExecutionRequest:
    experiment: ExperimentName
    definition: ExperimentDefinition
    overwrite_policy: OverwritePolicy
