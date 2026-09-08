from __future__ import annotations

import pytest

from fedorbit.analysis.comparisons import (
    PairContrastEvidence,
    PairContrastEvidenceSet,
    PairedObservation,
    PairingError,
    PairingLineage,
    validate_paired_observations,
)
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.types import (
    ArtifactIdentifier,
    DirectedPairName,
    Sha256Digest,
    Split,
    StrictResourceValidity,
    TransferMethod,
)


@pytest.fixture
def config() -> FedorbitConfig:
    return load_fedorbit_config()


def test_pair_evidence_set_rejects_duplicate_directed_pairs() -> None:
    evidence = PairContrastEvidence(
        DirectedPairName("pair"),
        0.1,
        0.01,
        0.01,
        StrictResourceValidity(True),
        10,
    )
    with pytest.raises(ValueError):
        PairContrastEvidenceSet((evidence, evidence))


def _lineage(seed: int = 1103, source_packet: str | None = "packet-1") -> PairingLineage:
    return PairingLineage(
        raw_dataset_lineage_sha256=Sha256Digest("a" * 64),
        directed_pair=DirectedPairName("Edge→Windows"),
        seed=seed,
        split=Split.TEST,
        target_pre_transfer_checkpoint_artifact_id=ArtifactIdentifier("checkpoint-1"),
        target_importance_artifact_id=ArtifactIdentifier("importance-1"),
        source_packet_artifact_id=(
            ArtifactIdentifier(source_packet) if source_packet is not None else None
        ),
        action_budget=0.5,
        support_budget=2,
        confirmation_budget=200,
        environment_lineage_sha256=Sha256Digest("b" * 64),
    )


def test_pairing_engine_accepts_only_identical_registered_lineage() -> None:
    method = (
        PairedObservation(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, 0.2, _lineage(1103)),
        PairedObservation(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, 0.3, _lineage(2089)),
    )
    reference = (
        PairedObservation(TransferMethod.LOCAL_ONLY, 0.1, _lineage(2089)),
        PairedObservation(TransferMethod.LOCAL_ONLY, 0.15, _lineage(1103)),
    )
    paired = validate_paired_observations(method, reference)
    assert paired.directed_pair == "Edge→Windows"
    assert paired.seeds == (1103, 2089)
    assert paired.method_values == pytest.approx((0.2, 0.3))
    assert paired.reference_values == pytest.approx((0.15, 0.1))


def test_pairing_engine_rejects_source_packet_mismatch() -> None:
    method = (PairedObservation(TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER, 0.2, _lineage()),)
    reference = (
        PairedObservation(TransferMethod.LOCAL_ONLY, 0.1, _lineage(source_packet="packet-2")),
    )
    with pytest.raises(PairingError, match="lineage mismatch"):
        validate_paired_observations(method, reference)
