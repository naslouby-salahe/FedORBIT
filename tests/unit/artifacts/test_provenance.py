from __future__ import annotations

import pytest

from fedorbit.experiments.cells import experiment_relevance
from fedorbit.infrastructure.provenance import (
    ProvenanceError,
    implementation_fingerprint,
    runtime_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.types import (
    ArtifactStage,
    DatasetId,
    DirectedPair,
    ExperimentName,
    ExperimentSeed,
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


def test_implementation_fingerprint_is_stage_local() -> None:
    baseline = implementation_fingerprint("fedorbit.infrastructure.manifests")
    assert implementation_fingerprint("fedorbit.infrastructure.manifests") == baseline
    assert implementation_fingerprint("fedorbit.config.loading") != baseline


def test_implementation_fingerprint_follows_transitive_imports() -> None:
    producer = implementation_fingerprint("fedorbit.infrastructure.manifests")
    consumer = implementation_fingerprint("fedorbit.infrastructure.reuse")
    assert producer != consumer


def test_implementation_fingerprint_rejects_non_fedorbit_producer() -> None:
    with pytest.raises(ProvenanceError):
        implementation_fingerprint("os.path")


def test_runtime_fingerprint_is_stage_local() -> None:
    training = runtime_fingerprint(ArtifactStage.TRAINING)
    assert "torch" in training.components
    assert "torch-cuda" in training.components
    preprocessing = runtime_fingerprint(ArtifactStage.PREPROCESSING)
    assert preprocessing.digest != training.digest
    assert "matplotlib" not in preprocessing.components


def test_runtime_fingerprint_excludes_plotting_dependencies() -> None:
    for stage in (ArtifactStage.EVALUATION, ArtifactStage.REPORTING, ArtifactStage.STATISTICS):
        assert "matplotlib" not in runtime_fingerprint(stage).components


def test_runtime_fingerprint_unknown_stage_rejected() -> None:
    with pytest.raises(ValueError):
        ArtifactStage("invented_stage")


def test_stage_dependency_fingerprint_composes_all_material_inputs() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    arguments = (
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        ("upstream-1",),
        frozenset({"models", "generators"}),
        "fedorbit.infrastructure.manifests",
    )
    assert stage_dependency_fingerprint(*arguments) == stage_dependency_fingerprint(*arguments)


def test_stage_dependency_fingerprint_sensitive_to_upstreams() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    base = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        ("upstream-1",),
        frozenset({"models"}),
        "fedorbit.infrastructure.manifests",
    )
    changed = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        ("upstream-2",),
        frozenset({"models"}),
        "fedorbit.infrastructure.manifests",
    )
    assert changed != base


def test_stage_dependency_fingerprint_sensitive_to_config_subset() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    base = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (),
        frozenset({"models"}),
        "fedorbit.infrastructure.manifests",
    )
    changed = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (),
        frozenset({"models", "generators"}),
        "fedorbit.infrastructure.manifests",
    )
    assert changed != base


def test_stage_dependency_fingerprint_sensitive_to_producer_code() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    base = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (),
        frozenset({"models"}),
        "fedorbit.infrastructure.manifests",
    )
    changed = stage_dependency_fingerprint(
        ArtifactStage.TRAINING,
        PRIMARY_CELL,
        relevance,
        (),
        frozenset({"models"}),
        "fedorbit.infrastructure.reuse",
    )
    assert changed != base
