from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.infrastructure.execution import ArtifactStore
from fedorbit.infrastructure.manifests import ReusableArtifactManifest, artifact_id, file_sha256
from fedorbit.infrastructure.planner import (
    EXECUTION_LAYERS,
    PROGRAMME_PREREQUISITES,
    ExecutionReadiness,
    layer_index,
)
from fedorbit.types import ArtifactState, ExperimentName

COORDINATES = {"experiment": "Mathematical Primitive Validation"}


def test_execution_layers_in_roadmap_order() -> None:
    assert EXECUTION_LAYERS == (
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
    assert len(EXECUTION_LAYERS) == 10


def test_layer_index() -> None:
    assert layer_index("inputs") == 0
    assert layer_index("reporting") == 9
    with pytest.raises(ValueError):
        layer_index("invented")


def test_programme_prerequisites_in_roadmap_order() -> None:
    names = [name for name, _ in PROGRAMME_PREREQUISITES]
    assert names[0] == "environment diagnosis"
    assert names[1] == "raw-data identity"
    assert names[2] == "preprocessing"
    assert names[3] == "smoke validation"
    assert names[-1] == "manuscript evidence export"
    assert len(PROGRAMME_PREREQUISITES) == 28


def test_readiness_empty_store_blocks_before_preprocessing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "outputs")
    readiness = ExecutionReadiness(store, raw_root=tmp_path / "raw")
    states = readiness.prerequisite_states()
    assert len(states) == 28
    assert not readiness.programme_ready()
    blocked = readiness.first_blocked()
    assert blocked is not None
    assert blocked.step_index <= 1


def _manifest(payload: Path, fingerprint: str, coordinates: str) -> ReusableArtifactManifest:
    return ReusableArtifactManifest.model_validate(
        {
            "artifact_id": artifact_id("other", COORDINATES, fingerprint),
            "artifact_type": "other",
            "semantic_producer_coordinates": coordinates,
            "producer_stage": "preprocessing",
            "dependency_fingerprint_sha256": fingerprint,
            "upstream_artifact_ids": (),
            "applicable_configuration_sha256": "c" * 64,
            "relevant_code_sha256": "d" * 64,
            "material_runtime_sha256": "e" * 64,
            "payload_paths": (str(payload),),
            "payload_sha256": file_sha256(payload),
            "schema_version": "1.0",
            "created_git_commit": "a" * 40,
            "created_environment_sha256": "g" * 64,
            "state": ArtifactState.COMPLETED,
            "completion_manifest_sha256": "f" * 64,
        }
    )


def test_readiness_satisfies_gates_from_completed_evidence(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "evidence.bin"
    payload.write_bytes(b"evidence")
    coordinates = '{"experiment":"Mathematical Primitive Validation","seed":101}'
    manifest = _manifest(payload, "fp-math", coordinates)
    store.write_reusable(manifest)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    readiness = ExecutionReadiness(store, raw_root=raw_root)

    states = {state.name: state for state in readiness.prerequisite_states()}
    assert states["mathematical primitive validation"].satisfied
    assert states["mathematical primitive validation"].owning_experiment == (
        ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION
    )


def test_readiness_prefix_property(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = tmp_path / "evidence.bin"
    payload.write_bytes(b"evidence")
    coordinates = '{"experiment":"Primary Strict Cross-Telemetry Transfer","seed":1103}'
    store.write_reusable(_manifest(payload, "fp-transfer", coordinates))
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    readiness = ExecutionReadiness(store, raw_root=raw_root)
    states = {state.name: state for state in readiness.prerequisite_states()}
    assert states["principal strict transfer"].satisfied
    blocked = readiness.first_blocked()
    assert blocked is not None
    assert blocked.step_index < 17


def test_readiness_incomplete_evidence_not_satisfied(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    missing = tmp_path / "missing.bin"
    missing.write_bytes(b"x")
    coordinates = '{"experiment":"Mathematical Primitive Validation"}'
    manifest = _manifest(missing, "fp-math", coordinates)
    missing.unlink()
    store.write_reusable(manifest)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    readiness = ExecutionReadiness(store, raw_root=raw_root)
    states = {state.name: state for state in readiness.prerequisite_states()}
    assert not states["mathematical primitive validation"].satisfied
