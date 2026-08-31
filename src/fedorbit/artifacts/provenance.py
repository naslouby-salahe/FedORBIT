from __future__ import annotations

import ast
import hashlib
import importlib.metadata
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fedorbit.artifacts.manifests import ReusableArtifactManifest
from fedorbit.config.context import active_config
from fedorbit.config.loading import repository_root
from fedorbit.domain.enums import ArtifactStage, SemanticCoordinate
from fedorbit.domain.records import SemanticCell
from fedorbit.domain.serialization import StableJsonPayload, stable_json
from fedorbit.runtime.environment import environment_snapshot
from fedorbit.runtime.reproducibility import current_code_revision

JsonValue = str | int | float | bool | None | list["JsonValue"] | Mapping[str, "JsonValue"]

STAGE_DEPENDENCIES: Mapping[ArtifactStage, tuple[ArtifactStage, ...]] = OrderedDict(
    (
        (ArtifactStage.RAW, ()),
        (ArtifactStage.PREPROCESSING, (ArtifactStage.RAW,)),
        (ArtifactStage.ELIGIBILITY, (ArtifactStage.PREPROCESSING,)),
        (
            ArtifactStage.PILOT_SELECTION,
            (ArtifactStage.PREPROCESSING, ArtifactStage.ELIGIBILITY),
        ),
        (
            ArtifactStage.TRAINING,
            (
                ArtifactStage.PREPROCESSING,
                ArtifactStage.ELIGIBILITY,
                ArtifactStage.PILOT_SELECTION,
            ),
        ),
        (ArtifactStage.SCORING, (ArtifactStage.TRAINING, ArtifactStage.PREPROCESSING)),
        (ArtifactStage.RESPONSE, (ArtifactStage.PREPROCESSING, ArtifactStage.SCORING)),
        (
            ArtifactStage.TARGET_IMPORTANCE,
            (ArtifactStage.TRAINING, ArtifactStage.SCORING),
        ),
        (
            ArtifactStage.CORRESPONDENCE,
            (ArtifactStage.RESPONSE, ArtifactStage.TARGET_IMPORTANCE),
        ),
        (ArtifactStage.CONFIRMATION, (ArtifactStage.CORRESPONDENCE, ArtifactStage.RESPONSE)),
        (
            ArtifactStage.MULTI_SOURCE_SELECTION,
            (ArtifactStage.CONFIRMATION, ArtifactStage.CORRESPONDENCE),
        ),
        (ArtifactStage.EVALUATION, (ArtifactStage.CONFIRMATION, ArtifactStage.SCORING)),
        (ArtifactStage.STATISTICS, (ArtifactStage.EVALUATION,)),
        (ArtifactStage.REPORTING, (ArtifactStage.STATISTICS,)),
    )
)

RUNTIME_COMPONENTS: Mapping[ArtifactStage, tuple[str, ...]] = OrderedDict(
    (
        (ArtifactStage.RAW, ("numpy", "pandas")),
        (ArtifactStage.PREPROCESSING, ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn")),
        (ArtifactStage.ELIGIBILITY, ("numpy",)),
        (ArtifactStage.PILOT_SELECTION, ("numpy", "scipy")),
        (ArtifactStage.TRAINING, ("torch", "numpy", "torch-cuda")),
        (ArtifactStage.SCORING, ("torch", "numpy")),
        (ArtifactStage.RESPONSE, ("numpy", "scipy", "torch")),
        (ArtifactStage.TARGET_IMPORTANCE, ("numpy", "torch")),
        (ArtifactStage.CORRESPONDENCE, ("highspy", "pyscipopt", "numpy", "scipy")),
        (ArtifactStage.CONFIRMATION, ("numpy", "scipy", "torch")),
        (ArtifactStage.MULTI_SOURCE_SELECTION, ("numpy", "scipy")),
        (ArtifactStage.EVALUATION, ("numpy", "scipy", "scikit-learn")),
        (ArtifactStage.STATISTICS, ("numpy", "scipy")),
        (ArtifactStage.REPORTING, ()),
    )
)


class ProvenanceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeFingerprint:
    components: tuple[str, ...]
    versions: tuple[tuple[str, str], ...]
    digest: str

    @property
    def sha256(self) -> str:
        return self.digest


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    artifact_id: str
    created_git_commit: str
    dependency_lock_sha256: str
    operating_system: str
    hardware: str
    driver: str
    environment_sha256: str


def _resolve_module_path(module_name: str) -> Path:
    module_path = repository_root() / "src" / Path(*module_name.split(".")).with_suffix(".py")
    if not module_path.is_file():
        module_path = module_path.with_name(module_path.stem) / "__init__.py"
    if not module_path.is_file():
        raise ProvenanceError(f"module not found: {module_name}")
    return module_path


def _local_imported_modules(tree: ast.Module) -> tuple[str, ...]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("fedorbit"):
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names if alias.name.startswith("fedorbit"))
    return tuple(imported)


def _module_source_digest(module_name: str, visited: set[str]) -> str:
    if module_name in visited:
        return ""
    visited.add(module_name)
    module_path = _resolve_module_path(module_name)
    digest = hashlib.sha256(module_path.read_bytes())
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for dependency in _local_imported_modules(tree):
        digest.update(_module_source_digest(dependency, visited).encode("utf-8"))
    return digest.hexdigest()


def implementation_fingerprint(producer_module: str) -> str:
    if not producer_module.startswith("fedorbit"):
        raise ProvenanceError(f"producer must be a fedorbit module: {producer_module}")
    return _module_source_digest(producer_module, set())


def runtime_fingerprint(stage: ArtifactStage) -> RuntimeFingerprint:
    if stage not in STAGE_DEPENDENCIES:
        raise ProvenanceError(f"unknown stage: {stage}")
    components = RUNTIME_COMPONENTS[stage]
    versions: list[tuple[str, str]] = []
    for distribution in components:
        if distribution == "torch-cuda":
            import torch

            versions.append(("torch-cuda", torch.version.cuda or "unknown"))
            continue
        try:
            versions.append((distribution, importlib.metadata.version(distribution)))
        except importlib.metadata.PackageNotFoundError:
            raise ProvenanceError(f"runtime component not installed: {distribution}") from None
    payload = stable_json(
        cast(StableJsonPayload, OrderedDict(components=components, versions=versions))
    )
    return RuntimeFingerprint(
        components=components,
        versions=tuple(versions),
        digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


def _section_extractors() -> Mapping[str, Callable[[], JsonValue]]:
    config = active_config()
    scientific = config.scientific
    return OrderedDict(
        generators=lambda: config.generators.model_dump(mode="json"),
        action=lambda: scientific.action.model_dump(mode="json"),
        models=lambda: OrderedDict(
            training=scientific.training.model_dump(mode="json"),
            base_model_pilot=scientific.base_model_pilot.model_dump(mode="json"),
        ),
        response=lambda: OrderedDict(
            source_response_pilot=scientific.source_response_pilot.model_dump(mode="json"),
            source_response_final=scientific.source_response_final.model_dump(mode="json"),
            target_response_diagnostic=scientific.target_response_diagnostic.model_dump(
                mode="json"
            ),
        ),
        confirmation=lambda: scientific.confirmation.model_dump(mode="json"),
        evaluation=lambda: OrderedDict(
            metrics=scientific.metrics.model_dump(mode="json"),
            statistics=scientific.statistics.model_dump(mode="json"),
        ),
        statistics=lambda: scientific.statistics.model_dump(mode="json"),
        experiments=lambda: config.experiments.model_dump(mode="json"),
        simplification_rules=lambda: scientific.simplification_rules.model_dump(mode="json"),
    )


def configuration_subset_digest(relevant_sections: frozenset[str]) -> str:
    extractors = _section_extractors()
    values: OrderedDict[str, JsonValue] = OrderedDict()
    for section in sorted(relevant_sections):
        extractor = extractors.get(section)
        if extractor is not None:
            values[section] = extractor()
    return hashlib.sha256(stable_json(values).encode("utf-8")).hexdigest()


def stage_dependency_fingerprint(
    stage: ArtifactStage,
    cell: SemanticCell,
    relevance: frozenset[SemanticCoordinate],
    upstream_artifact_ids: tuple[str, ...],
    config_sections: frozenset[str],
    producer_module: str,
) -> str:
    payload = stable_json(
        cast(
            StableJsonPayload,
            OrderedDict(
                stage=stage.value,
                semantic_coordinates=cell.identity_json(relevance),
                upstream_artifact_ids=list(upstream_artifact_ids),
                configuration_sha256=configuration_subset_digest(config_sections),
                implementation_sha256=implementation_fingerprint(producer_module),
                runtime_sha256=runtime_fingerprint(stage).sha256,
            ),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provenance_record(manifest: ReusableArtifactManifest) -> ProvenanceRecord:
    environment = environment_snapshot()
    revision = current_code_revision()
    return ProvenanceRecord(
        artifact_id=manifest.artifact_id,
        created_git_commit=revision.commit,
        dependency_lock_sha256=manifest.created_environment_sha256,
        operating_system=environment.hardware.os_release,
        hardware=environment.hardware.cpu_name,
        driver=environment.hardware.driver_cuda_version or "unknown",
        environment_sha256=environment.fingerprint_sha256,
    )
