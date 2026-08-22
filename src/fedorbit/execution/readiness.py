from __future__ import annotations

from dataclasses import dataclass

from fedorbit.artifacts.reuse import ArtifactStore
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import ArtifactState, ExperimentName
from fedorbit.execution.layers import PROGRAMME_PREREQUISITES
from fedorbit.runtime.environment import EnvironmentMismatchError, validate_environment


class ReadinessError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PrerequisiteState:
    step_index: int
    name: str
    satisfied: bool
    owning_experiment: ExperimentName | None = None
    reason: str | None = None


class ExecutionReadiness:
    def __init__(self, store: ArtifactStore, config: FedorbitConfig | None = None) -> None:
        self._store = store
        self._config = config if config is not None else load_fedorbit_config()

    def prerequisite_states(self) -> tuple[PrerequisiteState, ...]:
        states: list[PrerequisiteState] = []
        for index, (name, owning_experiment) in enumerate(PROGRAMME_PREREQUISITES):
            states.append(self._state(index, name, owning_experiment))
        return tuple(states)

    def _state(
        self, index: int, name: str, owning_experiment: ExperimentName | None
    ) -> PrerequisiteState:
        if name == "environment diagnosis":
            try:
                validate_environment(self._config, strict=True)
                return PrerequisiteState(index, name, True, None)
            except EnvironmentMismatchError as error:
                return PrerequisiteState(index, name, False, None, str(error))
        if owning_experiment is None:
            return PrerequisiteState(index, name, True, None)
        evidence = self._experiment_evidence(owning_experiment)
        if evidence is None:
            return PrerequisiteState(index, name, False, owning_experiment, "no completed evidence")
        return PrerequisiteState(index, name, True, owning_experiment)

    def _experiment_evidence(self, experiment: ExperimentName) -> object | None:
        manifest_dir = self._store.manifest_dir()
        if not manifest_dir.is_dir():
            return None
        for path in sorted(manifest_dir.glob("*.json")):
            manifest = self._store.read_reusable(path.stem)
            if manifest.state != ArtifactState.COMPLETED:
                continue
            coordinates = manifest.semantic_producer_coordinates
            if experiment.value in coordinates:
                try:
                    self._store.resolve(path.stem)
                    return manifest
                except Exception:
                    continue
        return None

    def first_blocked(self) -> PrerequisiteState | None:
        for state in self.prerequisite_states():
            if not state.satisfied:
                return state
        return None

    def programme_ready(self) -> bool:
        return self.first_blocked() is None
