from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.artifacts.storage import ArtifactStore
from fedorbit.domain.enums import ArtifactState, ExperimentClassification, ExperimentName
from fedorbit.domain.records import ArtifactIdentifier
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.runtime.environment import EnvironmentMismatchError, validate_environment

EXECUTION_LAYERS = (
    "inputs",
    "preprocessing / splits",
    "training / checkpoint selection",
    "scoring and source/target risk derivation",
    "response-packet construction",
    "correspondence / action optimization",
    "target confirmation and live assimilation",
    "TEST evaluation",
    "statistical analysis",
    "reporting",
)

PROGRAMME_PREREQUISITES = (
    ("environment diagnosis", None),
    ("raw-data identity", None),
    ("preprocessing", ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION),
    ("smoke validation", None),
    ("mathematical primitive validation", ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION),
    ("exact-sparse theorem validation", ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION),
    (
        "dataset/client/resource validation",
        ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION,
    ),
    ("base-model pilot", ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT),
    ("base-model checkpoint selection", ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT),
    ("source-response pilot", ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT),
    ("final source-response bands", ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION),
    ("baseline/oracle validation", ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION),
    ("exact-sparse solver benchmark", ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK),
    ("synthetic coupling mechanism", ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION),
    ("real-packet coupling mechanism", ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION),
    ("unresolved-map action diagnostics", ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP),
    ("principal strict transfer", ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER),
    ("multi-source diagnostic", ExperimentName.MULTI_SOURCE_SELECTION_VALIDATION),
    ("mechanism ablations", ExperimentName.MECHANISM_ABLATIONS),
    ("sparsity/dense sensitivity", ExperimentName.SPARSITY_AND_DENSE_FALLBACK),
    ("confirmation/portability", ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY),
    ("secondary generalization", ExperimentName.SECONDARY_CROSS_MODALITY_GENERALIZATION),
    ("semantic sufficiency boundary", ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER),
    (
        "weak-signal/support/heterogeneity boundaries",
        ExperimentName.WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES,
    ),
    ("map applicability audit", ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT),
    ("scalability/efficiency", ExperimentName.SCALABILITY_AND_EFFICIENCY),
    ("statistical synthesis", ExperimentName.STATISTICAL_SYNTHESIS),
    ("manuscript evidence export", None),
)


@dataclass(frozen=True, slots=True)
class PlanRow:
    experiment: ExperimentName
    classification: ExperimentClassification
    planned_cells: int
    prerequisites: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrerequisiteState:
    step_index: int
    name: str
    satisfied: bool
    owning_experiment: ExperimentName | None = None
    reason: str | None = None


def layer_index(layer: str) -> int:
    for index, candidate in enumerate(EXECUTION_LAYERS):
        if candidate == layer:
            return index
    raise ValueError(f"unknown execution layer: {layer}")


def build_plan() -> tuple[PlanRow, ...]:
    catalogue = build_catalogue()
    return tuple(
        PlanRow(
            experiment=name,
            classification=catalogue.definition(name).classification,
            planned_cells=catalogue.definition(name).derived_planned_cells,
            prerequisites=catalogue.definition(name).prerequisites,
        )
        for name in catalogue.registered_names()
    )


class ExecutionReadiness:
    def __init__(
        self,
        store: ArtifactStore,
        raw_root: Path = Path("data/raw"),
    ) -> None:
        self._store = store
        self._raw_root = raw_root

    def prerequisite_states(self) -> tuple[PrerequisiteState, ...]:
        return tuple(
            self._state(index, name, owner)
            for index, (name, owner) in enumerate(PROGRAMME_PREREQUISITES)
        )

    def _state(
        self,
        index: int,
        name: str,
        owner: ExperimentName | None,
    ) -> PrerequisiteState:
        if name == "environment diagnosis":
            try:
                validate_environment(strict=True)
                return PrerequisiteState(index, name, True)
            except EnvironmentMismatchError as error:
                return PrerequisiteState(index, name, False, reason=str(error))
        if name == "raw-data identity":
            if not self._raw_root.is_dir():
                return PrerequisiteState(index, name, False, reason="raw-data root is unavailable")
            return PrerequisiteState(index, name, True)
        if owner is None:
            return PrerequisiteState(index, name, True)
        manifest = self._experiment_evidence(owner)
        if manifest is None:
            return PrerequisiteState(index, name, False, owner, "no completed evidence")
        return PrerequisiteState(index, name, True, owner)

    def _experiment_evidence(self, experiment: ExperimentName) -> ReusableArtifactManifest | None:
        for manifest in self._store.all_manifests():
            if manifest.state != ArtifactState.COMPLETED:
                continue
            if experiment.value not in manifest.semantic_producer_coordinates:
                continue
            try:
                return self._store.resolve(ArtifactIdentifier(manifest.artifact_id))
            except ValueError:
                continue
        return None

    def first_blocked(self) -> PrerequisiteState | None:
        return next((state for state in self.prerequisite_states() if not state.satisfied), None)

    def programme_ready(self) -> bool:
        return self.first_blocked() is None
