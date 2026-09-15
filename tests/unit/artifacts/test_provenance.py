from __future__ import annotations

import pytest

from fedorbit.experiments.cells import experiment_relevance
from fedorbit.infrastructure.provenance import (
    configuration_subset_digest,
    implementation_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactStage,
    ConfigurationSection,
    DatasetId,
    DirectedPair,
    ExperimentName,
    ExperimentSeed,
    ImplementationIdentity,
    SemanticCell,
    SemanticCoordinate,
    SupportSize,
    TransferMethod,
)

PAIR = DirectedPair(DatasetId.EDGE_IIOTSET_NETWORK, DatasetId.TON_IOT_NETWORK)
PRIMARY_CELL = SemanticCell(
    experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
    directed_pair=PAIR,
    method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
    support=SupportSize(2),
    seed=ExperimentSeed(1103),
)


def test_semantic_identity_is_stable_and_seed_sensitive() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    first = PRIMARY_CELL.identity_json(relevance)
    assert PRIMARY_CELL.identity_json(relevance) == first
    changed_seed = SemanticCell(
        experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        directed_pair=PAIR,
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        support=SupportSize(2),
        seed=ExperimentSeed(5531),
    )
    assert changed_seed.identity_json(relevance) != first


def test_semantic_identity_excludes_irrelevant_dimensions() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    assert SemanticCoordinate.CONDITION not in relevance
    identity = PRIMARY_CELL.identity_json(relevance)
    assert "condition" not in identity
    assert "support" in identity
    assert "seed" in identity


def test_identity_contains_no_nonscientific_identifiers() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    identity = PRIMARY_CELL.identity_json(relevance)
    for banned in ("uuid", "timestamp", "run_number", "random", "hash"):
        assert banned not in identity.lower()


def test_relevance_covers_registered_experiments() -> None:
    for experiment in ExperimentName:
        relevance = experiment_relevance(experiment)
        assert SemanticCoordinate.EXPERIMENT in relevance
        assert len(relevance) >= 2


def test_implementation_fingerprint_is_stable_for_unchanged_identity() -> None:
    baseline = implementation_fingerprint(ImplementationIdentity.TRAINING_V1)
    assert implementation_fingerprint(ImplementationIdentity.TRAINING_V1) == baseline
    assert implementation_fingerprint(ImplementationIdentity.SCORING_V1) != baseline


def test_invented_stage_rejected() -> None:
    with pytest.raises(ValueError):
        ArtifactStage("invented_stage")


def test_stage_dependency_fingerprint_composes_all_material_inputs() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    arguments = (
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (ArtifactIdentifier("upstream-1"),),
        frozenset({ConfigurationSection.MODELS, ConfigurationSection.GENERATORS}),
        ImplementationIdentity.TRAINING_V1,
    )
    assert stage_dependency_fingerprint(*arguments) == stage_dependency_fingerprint(*arguments)


def test_stage_dependency_fingerprint_sensitive_to_upstreams() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    base = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (ArtifactIdentifier("upstream-1"),),
        frozenset({ConfigurationSection.MODELS}),
        ImplementationIdentity.TRAINING_V1,
    )
    changed = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (ArtifactIdentifier("upstream-2"),),
        frozenset({ConfigurationSection.MODELS}),
        ImplementationIdentity.TRAINING_V1,
    )
    assert changed != base


def test_stage_dependency_fingerprint_sensitive_to_config_subset() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    base = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (),
        frozenset({ConfigurationSection.MODELS}),
        ImplementationIdentity.TRAINING_V1,
    )
    changed = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (),
        frozenset({ConfigurationSection.MODELS, ConfigurationSection.GENERATORS}),
        ImplementationIdentity.TRAINING_V1,
    )
    assert changed != base


def test_stage_dependency_fingerprint_is_sensitive_to_implementation_identity() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    base = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (),
        frozenset({ConfigurationSection.MODELS}),
        ImplementationIdentity.TRAINING_V1,
    )
    changed = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (),
        frozenset({ConfigurationSection.MODELS}),
        ImplementationIdentity.SCORING_V1,
    )
    assert changed != base


def test_every_configuration_section_contributes_to_the_dependency_digest() -> None:
    empty = configuration_subset_digest(frozenset())
    per_section = {
        section: configuration_subset_digest(frozenset({section}))
        for section in ConfigurationSection
    }
    absent = [section for section, digest in per_section.items() if digest == empty]
    assert not absent, f"configuration sections absent from the dependency digest: {absent}"
    assert len(set(per_section.values())) == len(ConfigurationSection)
