from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue

from fedorbit.config.loading import active_config
from fedorbit.types import (
    ArtifactIdentifiers,
    ArtifactStage,
    ArtifactType,
    ConfigurationSection,
    ImplementationIdentity,
    MetricId,
    ScientificSubsystem,
    SemanticCell,
    SemanticCoordinate,
    SemanticCoordinateText,
    Sha256Digest,
    StableJsonPayload,
    stable_json,
)

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


class ProvenanceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactFamilyRegistration:
    stage: ArtifactStage | None
    cache_key_material: frozenset[ConfigurationSection]
    implementation_identity: ImplementationIdentity


ARTIFACT_FAMILY_REGISTRATION: Mapping[ArtifactType, ArtifactFamilyRegistration] = OrderedDict(
    (
        (
            ArtifactType.PREPARED_SPLIT,
            ArtifactFamilyRegistration(
                ArtifactStage.PREPROCESSING,
                frozenset(),
                ImplementationIdentity.DATASET_MATERIALIZATION_V3,
            ),
        ),
        (
            ArtifactType.CHECKPOINT,
            ArtifactFamilyRegistration(
                ArtifactStage.TRAINING,
                frozenset({ConfigurationSection.MODELS}),
                ImplementationIdentity.TRAINING_V1,
            ),
        ),
        (
            ArtifactType.PREDICTION,
            ArtifactFamilyRegistration(
                ArtifactStage.EVALUATION,
                frozenset({ConfigurationSection.MODELS, ConfigurationSection.METRICS}),
                ImplementationIdentity.SCORING_V1,
            ),
        ),
        (
            ArtifactType.RESPONSE_PACKET,
            ArtifactFamilyRegistration(
                ArtifactStage.RESPONSE,
                frozenset({ConfigurationSection.RESPONSE}),
                ImplementationIdentity.TRAINING_V1,
            ),
        ),
        (
            ArtifactType.TARGET_IMPORTANCE,
            ArtifactFamilyRegistration(
                ArtifactStage.TARGET_IMPORTANCE,
                frozenset({ConfigurationSection.MODELS, ConfigurationSection.TARGET_IMPORTANCE}),
                ImplementationIdentity.SCORING_V1,
            ),
        ),
        (
            ArtifactType.SOLVER_RESULT,
            ArtifactFamilyRegistration(
                ArtifactStage.CORRESPONDENCE,
                frozenset({ConfigurationSection.SOLVERS, ConfigurationSection.ACTION}),
                ImplementationIdentity.SOLVERS_V1,
            ),
        ),
        (
            ArtifactType.CONFIRMATION_INPUT,
            ArtifactFamilyRegistration(
                ArtifactStage.CONFIRMATION,
                frozenset({ConfigurationSection.CONFIRMATION, ConfigurationSection.MODELS}),
                ImplementationIdentity.SCORING_V1,
            ),
        ),
        (
            ArtifactType.STATISTICAL_RESULT,
            ArtifactFamilyRegistration(
                ArtifactStage.STATISTICS,
                frozenset({ConfigurationSection.METRICS}),
                ImplementationIdentity.SYNTHESIS_V1,
            ),
        ),
        (
            ArtifactType.EVIDENCE,
            ArtifactFamilyRegistration(
                None,
                frozenset({ConfigurationSection.METRICS}),
                ImplementationIdentity.CLASSIFICATION_V1,
            ),
        ),
        (
            ArtifactType.OTHER,
            ArtifactFamilyRegistration(
                None,
                frozenset(),
                ImplementationIdentity.DATASET_MATERIALIZATION_V3,
            ),
        ),
    )
)

SUBSYSTEM_REGISTRATION: Mapping[ScientificSubsystem, ArtifactStage] = OrderedDict(
    (
        (ScientificSubsystem.RAW_INVENTORY, ArtifactStage.RAW),
        (ScientificSubsystem.PREPARATION, ArtifactStage.PREPROCESSING),
        (ScientificSubsystem.ELIGIBILITY, ArtifactStage.ELIGIBILITY),
        (ScientificSubsystem.BASE_MODEL_PILOT, ArtifactStage.PILOT_SELECTION),
        (ScientificSubsystem.BASE_MODEL_TRAINING, ArtifactStage.TRAINING),
        (ScientificSubsystem.SCORING, ArtifactStage.SCORING),
        (ScientificSubsystem.SOURCE_RESPONSE, ArtifactStage.RESPONSE),
        (ScientificSubsystem.TARGET_IMPORTANCE, ArtifactStage.TARGET_IMPORTANCE),
        (ScientificSubsystem.CORRESPONDENCE, ArtifactStage.CORRESPONDENCE),
        (ScientificSubsystem.CONFIRMATION, ArtifactStage.CONFIRMATION),
        (ScientificSubsystem.STATISTICS, ArtifactStage.STATISTICS),
        (ScientificSubsystem.REPORTING, ArtifactStage.REPORTING),
    )
)


def artifact_family_registration(artifact_type: ArtifactType) -> ArtifactFamilyRegistration:
    registration = ARTIFACT_FAMILY_REGISTRATION.get(artifact_type)
    if registration is None:
        raise ProvenanceError(f"artifact family is not registered: {artifact_type.value}")
    return registration


def subsystem_stage(subsystem: ScientificSubsystem) -> ArtifactStage:
    stage = SUBSYSTEM_REGISTRATION.get(subsystem)
    if stage is None:
        raise ProvenanceError(f"scientific subsystem is not registered: {subsystem.value}")
    return stage


def validate_artifact_family_registry() -> None:
    missing = tuple(
        artifact_type.value
        for artifact_type in ArtifactType
        if artifact_type not in ARTIFACT_FAMILY_REGISTRATION
    )
    if missing:
        raise ProvenanceError(f"artifact families without a registration: {sorted(missing)}")
    unregistered_subsystems = tuple(
        subsystem.value
        for subsystem in ScientificSubsystem
        if subsystem not in SUBSYSTEM_REGISTRATION
    )
    if unregistered_subsystems:
        raise ProvenanceError(
            f"scientific subsystems without a stage registration: {sorted(unregistered_subsystems)}"
        )


def implementation_fingerprint(identity: ImplementationIdentity) -> Sha256Digest:
    payload = stable_json(cast(StableJsonPayload, OrderedDict(identity=identity.value)))
    return Sha256Digest(hashlib.sha256(payload.encode("utf-8")).hexdigest())


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
            (
                ConfigurationSection.CONFIRMATION,
                lambda: OrderedDict(
                    confirmation=scientific.confirmation.model_dump(mode="json"),
                    target_optimizer_budget=scientific.target_optimizer_budget.model_dump(
                        mode="json"
                    ),
                ),
            ),
            (
                ConfigurationSection.TARGET_IMPORTANCE,
                lambda: scientific.target_importance.model_dump(mode="json"),
            ),
            (
                ConfigurationSection.MULTI_SOURCE_SELECTION,
                lambda: scientific.multi_source_selection.model_dump(mode="json"),
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


def artifact_payload_fingerprint(
    stage: ArtifactStage,
    semantic_coordinates: SemanticCoordinateText,
    upstream_artifact_ids: ArtifactIdentifiers,
    config_sections: frozenset[ConfigurationSection],
    implementation_identity: ImplementationIdentity,
) -> Sha256Digest:
    payload = stable_json(
        cast(
            StableJsonPayload,
            OrderedDict(
                stage=stage.value,
                recorded_coordinates=semantic_coordinates,
                upstream_artifact_ids=[identifier.value for identifier in upstream_artifact_ids],
                configuration_sha256=configuration_subset_digest(config_sections),
                implementation_sha256=implementation_fingerprint(implementation_identity),
            ),
        )
    )
    return Sha256Digest(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def stage_dependency_fingerprint(
    stage: ArtifactStage,
    cell: SemanticCell,
    relevance: frozenset[SemanticCoordinate],
    upstream_artifact_ids: ArtifactIdentifiers,
    config_sections: frozenset[ConfigurationSection],
    implementation_identity: ImplementationIdentity,
    metric_name: MetricId | None = None,
) -> Sha256Digest:
    artifact_ids: list[str] = [identifier.value for identifier in upstream_artifact_ids]
    if metric_name is not None:
        artifact_ids.append(metric_name.value)
    payload = stable_json(
        cast(
            StableJsonPayload,
            OrderedDict(
                stage=stage.value,
                semantic_coordinates=cell.identity_json(relevance),
                upstream_artifact_ids=artifact_ids,
                configuration_sha256=configuration_subset_digest(config_sections),
                implementation_sha256=implementation_fingerprint(implementation_identity),
            ),
        )
    )
    return Sha256Digest(hashlib.sha256(payload.encode("utf-8")).hexdigest())
