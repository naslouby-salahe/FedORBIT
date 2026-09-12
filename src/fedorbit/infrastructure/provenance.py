from __future__ import annotations

import hashlib
import importlib.metadata
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from fedorbit.config.loading import active_config
from fedorbit.types import (
    ArtifactStage,
    ConfigurationSection,
    SemanticCell,
    SemanticCoordinate,
    Sha256Digest,
    StableJsonPayload,
    stable_json,
)

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
    digest: Sha256Digest

    @property
    def sha256(self) -> Sha256Digest:
        return self.digest


def implementation_fingerprint(producer_module: str) -> Sha256Digest:
    if not producer_module.startswith("fedorbit"):
        raise ProvenanceError(f"producer must be a fedorbit module: {producer_module}")
    return Sha256Digest(hashlib.sha256(producer_module.encode("utf-8")).hexdigest())


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
        digest=Sha256Digest(hashlib.sha256(payload.encode("utf-8")).hexdigest()),
    )


def _section_extractors() -> Mapping[ConfigurationSection, Callable[[], JsonValue]]:
    config = active_config()
    scientific = config.scientific
    return OrderedDict(
        (
            (
                ConfigurationSection.ACTION,
                lambda: scientific.action.model_dump(mode="json"),
            ),
            (
                ConfigurationSection.GENERATORS,
                lambda: config.generators.model_dump(mode="json"),
            ),
            (
                ConfigurationSection.METRICS,
                lambda: OrderedDict(
                    metrics=scientific.metrics.model_dump(mode="json"),
                    statistics=scientific.statistics.model_dump(mode="json"),
                ),
            ),
            (
                ConfigurationSection.MODELS,
                lambda: OrderedDict(
                    training=scientific.training.model_dump(mode="json"),
                    base_model_pilot=scientific.base_model_pilot.model_dump(mode="json"),
                ),
            ),
            (
                ConfigurationSection.RESPONSE,
                lambda: OrderedDict(
                    source_response_pilot=scientific.source_response_pilot.model_dump(mode="json"),
                    source_response_final=scientific.source_response_final.model_dump(mode="json"),
                    target_response_diagnostic=scientific.target_response_diagnostic.model_dump(
                        mode="json"
                    ),
                ),
            ),
            (
                ConfigurationSection.SOLVERS,
                lambda: config.solvers.model_dump(mode="json"),
            ),
        )
    )


def configuration_subset_digest(
    relevant_sections: frozenset[ConfigurationSection],
) -> str:
    extractors = _section_extractors()
    values: OrderedDict[ConfigurationSection, JsonValue] = OrderedDict()
    for section in sorted(relevant_sections):
        values[section] = extractors[section]()
    return hashlib.sha256(stable_json(values).encode("utf-8")).hexdigest()


def stage_dependency_fingerprint(
    stage: ArtifactStage,
    cell: SemanticCell,
    relevance: frozenset[SemanticCoordinate],
    upstream_artifact_ids: tuple[str, ...],
    config_sections: frozenset[ConfigurationSection],
    producer_module: str,
) -> Sha256Digest:
    del producer_module
    payload = stable_json(
        cast(
            StableJsonPayload,
            OrderedDict(
                stage=stage.value,
                semantic_coordinates=cell.identity_json(relevance),
                upstream_artifact_ids=list(upstream_artifact_ids),
                configuration_sha256=configuration_subset_digest(config_sections),
            ),
        )
    )
    return Sha256Digest(hashlib.sha256(payload.encode("utf-8")).hexdigest())
