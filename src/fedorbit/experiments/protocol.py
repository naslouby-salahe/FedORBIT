from __future__ import annotations

from dataclasses import dataclass

from fedorbit.experiments.catalogue import ExperimentDefinition
from fedorbit.types import ExperimentName, OverwritePolicy


@dataclass(frozen=True, slots=True)
class ExperimentExecutionRequest: #TODO: move to an already existing module and delete this protocol.py. No need for a whole file just for this
    experiment: ExperimentName
    definition: ExperimentDefinition
    overwrite_policy: OverwritePolicy
