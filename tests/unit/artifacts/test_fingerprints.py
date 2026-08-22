from __future__ import annotations

import pytest

from fedorbit.artifacts.fingerprints import (
    FingerprintError,
    implementation_fingerprint,
    runtime_fingerprint,
    stage_dependency_fingerprint,
)
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.domain.enums import DatasetId, ExperimentName, TransferMethod
from fedorbit.domain.records import DirectedPair, SemanticCell
from fedorbit.experiments.cells import experiment_relevance

PAIR = DirectedPair(DatasetId.EDGE_IIOTSET_NETWORK, DatasetId.TON_IOT_NETWORK)

PRIMARY_CELL = SemanticCell(
    experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
    directed_pair=PAIR,
    method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
    support=2,
    seed=1103,
)


def test_semantic_identity_is_stable_and_seed_sensitive() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    first = PRIMARY_CELL.identity_json(relevance)
    second = PRIMARY_CELL.identity_json(relevance)
    assert first == second
    changed_seed = SemanticCell(
        experiment=ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER,
        directed_pair=PAIR,
        method=TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        support=2,
        seed=5531,
    )
    assert changed_seed.identity_json(relevance) != first


def test_semantic_identity_excludes_irrelevant_dimensions() -> None:
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    assert "condition" not in relevance
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
        assert "experiment" in relevance
        assert len(relevance) >= 2


def test_implementation_fingerprint_is_stage_local() -> None:
    baseline = implementation_fingerprint("fedorbit.artifacts.manifests")
    again = implementation_fingerprint("fedorbit.artifacts.manifests")
    assert baseline == again
    unrelated = implementation_fingerprint("fedorbit.config.loading")
    assert unrelated != baseline


def test_implementation_fingerprint_follows_transitive_imports() -> None:
    producer = implementation_fingerprint("fedorbit.artifacts.manifests")
    consumer = implementation_fingerprint("fedorbit.artifacts.reuse")
    assert producer != consumer


def test_implementation_fingerprint_rejects_non_fedorbit_producer() -> None:
    with pytest.raises(FingerprintError):
        implementation_fingerprint("os.path")


def test_runtime_fingerprint_is_stage_local() -> None:
    training = runtime_fingerprint("training")
    assert "torch" in training.components
    assert "torch-cuda" in training.components
    preprocessing = runtime_fingerprint("preprocessing")
    assert preprocessing.digest != training.digest
    assert "matplotlib" not in preprocessing.components


def test_runtime_fingerprint_excludes_plotting_dependencies() -> None:
    for stage in ("evaluation", "reporting", "statistics"):
        components = runtime_fingerprint(stage).components
        assert "matplotlib" not in components


def test_runtime_fingerprint_unknown_stage_rejected() -> None:
    with pytest.raises(FingerprintError):
        runtime_fingerprint("invented_stage")


def test_stage_dependency_fingerprint_composes_all_material_inputs() -> None:
    config = load_fedorbit_config()
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    first = stage_dependency_fingerprint(
        "training",
        PRIMARY_CELL,
        relevance,
        ("upstream-1",),
        config,
        frozenset({"models", "generators"}),
        "fedorbit.artifacts.manifests",
    )
    second = stage_dependency_fingerprint(
        "training",
        PRIMARY_CELL,
        relevance,
        ("upstream-1",),
        config,
        frozenset({"models", "generators"}),
        "fedorbit.artifacts.manifests",
    )
    assert first == second


def test_stage_dependency_fingerprint_sensitive_to_upstreams() -> None:
    config = load_fedorbit_config()
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    base = stage_dependency_fingerprint(
        "training",
        PRIMARY_CELL,
        relevance,
        ("upstream-1",),
        config,
        frozenset({"models"}),
        "fedorbit.artifacts.manifests",
    )
    changed = stage_dependency_fingerprint(
        "training",
        PRIMARY_CELL,
        relevance,
        ("upstream-2",),
        config,
        frozenset({"models"}),
        "fedorbit.artifacts.manifests",
    )
    assert changed != base


def test_stage_dependency_fingerprint_sensitive_to_config_subset() -> None:
    config = load_fedorbit_config()
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    base = stage_dependency_fingerprint(
        "training",
        PRIMARY_CELL,
        relevance,
        (),
        config,
        frozenset({"models"}),
        "fedorbit.artifacts.manifests",
    )
    changed = stage_dependency_fingerprint(
        "training",
        PRIMARY_CELL,
        relevance,
        (),
        config,
        frozenset({"models", "generators"}),
        "fedorbit.artifacts.manifests",
    )
    assert changed != base


def test_stage_dependency_fingerprint_sensitive_to_producer_code() -> None:
    config = load_fedorbit_config()
    relevance = experiment_relevance(ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER)
    base = stage_dependency_fingerprint(
        "training",
        PRIMARY_CELL,
        relevance,
        (),
        config,
        frozenset({"models"}),
        "fedorbit.artifacts.manifests",
    )
    changed = stage_dependency_fingerprint(
        "training",
        PRIMARY_CELL,
        relevance,
        (),
        config,
        frozenset({"models"}),
        "fedorbit.artifacts.reuse",
    )
    assert changed != base
