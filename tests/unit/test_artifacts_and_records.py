from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.infrastructure.artifacts import (
    InputDigest,
    Provenance,
    bytes_digest,
    classify_artifact,
    manifest_path,
    publish_artifact,
    write_atomic,
)
from fedorbit.infrastructure.runtime import derive_seed, source_digest
from fedorbit.study.records import (
    CellRecord,
    IneligibleDeviceRecord,
    InfeasibleSupportRecord,
    parse_records,
    serialise_records,
)
from fedorbit.types import (
    ConditionName,
    DatasetId,
    DeviceName,
    EvidenceState,
    ExperimentId,
    RandomPurpose,
    RandomSeed,
    ReplicateIndex,
    Sha256Digest,
    SupportSize,
)

BASE = Provenance(
    config_digest=bytes_digest(b"config"),
    source_digest=bytes_digest(b"source"),
    inputs=(InputDigest(name="prepared", sha256=bytes_digest(b"input")),),
)


def publish(path: Path, payload: bytes = b"payload", records: int = 3) -> None:
    publish_artifact(path, payload, records, "revision", BASE)


def test_valid_artifact_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    publish(path)
    assert classify_artifact(path, BASE, 3, compare_inputs=True) is EvidenceState.VALID


def test_missing_artifact(tmp_path: Path) -> None:
    assert (
        classify_artifact(tmp_path / "none.bin", BASE, None, compare_inputs=True)
        is EvidenceState.MISSING
    )


def test_payload_corruption_is_malformed(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    publish(path)
    path.write_bytes(b"tampered")
    assert classify_artifact(path, BASE, 3, compare_inputs=True) is EvidenceState.MALFORMED


def test_manifest_removal_is_malformed(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    publish(path)
    manifest_path(path).unlink()
    assert classify_artifact(path, BASE, 3, compare_inputs=True) is EvidenceState.MALFORMED


def test_garbage_manifest_is_malformed(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    publish(path)
    manifest_path(path).write_text("not json", encoding="utf-8")
    assert classify_artifact(path, BASE, 3, compare_inputs=True) is EvidenceState.MALFORMED


def test_changed_provenance_is_stale(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    publish(path)
    changed_config = BASE.model_copy(update={"config_digest": bytes_digest(b"other")})
    changed_input = BASE.model_copy(
        update={"inputs": (InputDigest(name="prepared", sha256=bytes_digest(b"new")),)}
    )
    assert classify_artifact(path, changed_config, 3, compare_inputs=True) is EvidenceState.STALE
    assert classify_artifact(path, changed_input, 3, compare_inputs=True) is EvidenceState.STALE
    assert classify_artifact(path, changed_input, 3, compare_inputs=False) is EvidenceState.VALID


def test_record_count_mismatch_is_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    publish(path)
    assert classify_artifact(path, BASE, 9, compare_inputs=True) is EvidenceState.INCOMPLETE


def test_atomic_write_leaves_no_staging_files(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "file.bin"
    write_atomic(path, b"data")
    assert path.read_bytes() == b"data"
    assert [item.name for item in path.parent.iterdir()] == ["file.bin"]


def test_atomic_write_failure_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "directory"
    target.mkdir()
    with pytest.raises(OSError):
        write_atomic(target, b"data")
    assert [item.name for item in tmp_path.iterdir()] == ["directory"]


def test_seed_derivation_is_stable_and_coordinate_sensitive() -> None:
    first = derive_seed(RandomSeed(1), RandomPurpose.SUPPORT_WINDOW, "a", "1")
    assert first == derive_seed(RandomSeed(1), RandomPurpose.SUPPORT_WINDOW, "a", "1")
    assert first != derive_seed(RandomSeed(1), RandomPurpose.SUPPORT_WINDOW, "a", "2")
    assert first != derive_seed(RandomSeed(1), RandomPurpose.PARTNER_SAMPLE, "a", "1")
    assert first != derive_seed(RandomSeed(2), RandomPurpose.SUPPORT_WINDOW, "a", "1")


def test_source_digest_tracks_content(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("x = 1\n", encoding="utf-8")
    before: Sha256Digest = source_digest(tmp_path, ("module.py",))
    (tmp_path / "module.py").write_text("x = 2\n", encoding="utf-8")
    assert source_digest(tmp_path, ("module.py",)) != before


def test_records_round_trip_preserves_both_kinds() -> None:
    cell = CellRecord(
        experiment=ExperimentId.COLD_START_LADDER,
        dataset=DatasetId.NBAIOT,
        device=DeviceName("d"),
        support_size=SupportSize(30),
        replicate=ReplicateIndex(2),
        condition=ConditionName("shared-marginals"),
        auroc=0.97,
        false_positive_rate=0.02,
        true_positive_rate=0.9,
        scale_mismatch=0.4,
        split_half_instability=0.1,
    )
    infeasible = InfeasibleSupportRecord(
        experiment=ExperimentId.COLD_START_LADDER,
        dataset=DatasetId.NBAIOT,
        device=DeviceName("d"),
        support_size=SupportSize(1000),
        available_rows=40,
    )
    ineligible = IneligibleDeviceRecord(
        experiment=ExperimentId.SIMULATED_BOUNDARY,
        dataset=DatasetId.GOTHAM,
        device=DeviceName("g"),
        benign_rows=3,
        attack_rows=500,
        worst_case_standard_error=0.1,
    )
    assert parse_records(serialise_records([cell, infeasible, ineligible])) == [
        cell,
        infeasible,
        ineligible,
    ]


def test_record_rejects_out_of_range_auroc() -> None:
    with pytest.raises(ValueError):
        CellRecord(
            experiment=ExperimentId.COLD_START_LADDER,
            dataset=DatasetId.NBAIOT,
            device=DeviceName("d"),
            support_size=SupportSize(30),
            replicate=ReplicateIndex(0),
            condition=ConditionName("c"),
            auroc=1.5,
            scale_mismatch=0.0,
            split_half_instability=0.0,
        )
