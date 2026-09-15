from __future__ import annotations

import pytest

from fedorbit.analysis.comparisons import (
    PairContrastEvidence,
    PairContrastEvidenceSet,
    PairedObservation,
    PairingField,
    PairingLineage,
    PairingMismatchError,
    require_matching_lineage,
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


def _observation(
    *,
    method: TransferMethod = TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
    value: float = 0.1,
    raw_dataset_lineage: str = "a" * 64,
    directed_pair: str = "client-a -> client-b",
    seed: int = 7,
    split: Split = Split.TEST,
    checkpoint: str = "checkpoint-a",
    importance: str = "importance-a",
    source_packet: str | None = "packet-a",
    action_budget: float = 0.5,
    support_budget: int = 2,
    confirmation_budget: int = 1,
    environment_lineage: str = "b" * 64,
) -> PairedObservation:
    return PairedObservation(
        method=method,
        value=value,
        lineage=PairingLineage(
            raw_dataset_lineage_sha256=Sha256Digest(raw_dataset_lineage),
            directed_pair=DirectedPairName(directed_pair),
            seed=seed,
            split=split,
            target_pre_transfer_checkpoint_artifact_id=ArtifactIdentifier(checkpoint),
            target_importance_artifact_id=ArtifactIdentifier(importance),
            source_packet_artifact_id=(
                None if source_packet is None else ArtifactIdentifier(source_packet)
            ),
            action_budget=action_budget,
            support_budget=support_budget,
            confirmation_budget=confirmation_budget,
            environment_lineage_sha256=Sha256Digest(environment_lineage),
        ),
    )


def test_matching_lineage_is_accepted(config: FedorbitConfig) -> None:
    del config
    reference = _observation()
    counterpart = _observation(
        method=TransferMethod.LOCAL_ONLY,
        value=0.4,
        source_packet=None,
    )
    reference_without_packet = _observation(source_packet=None)
    require_matching_lineage(reference, reference)
    require_matching_lineage(reference_without_packet, reference_without_packet)
    with pytest.raises(PairingMismatchError) as raised:
        require_matching_lineage(reference, counterpart)
    assert raised.value.field is PairingField.SOURCE_PACKET


@pytest.mark.parametrize(
    ("field", "counterpart"),
    (
        (
            PairingField.RAW_DATASET_LINEAGE,
            _observation(raw_dataset_lineage="c" * 64),
        ),
        (PairingField.DIRECTED_PAIR, _observation(directed_pair="client-b -> client-a")),
        (PairingField.SEED, _observation(seed=8)),
        (PairingField.SPLIT, _observation(split=Split.CONFIRM)),
        (PairingField.TARGET_PRE_TRANSFER_CHECKPOINT, _observation(checkpoint="checkpoint-b")),
        (PairingField.TARGET_IMPORTANCE, _observation(importance="importance-b")),
        (PairingField.SOURCE_PACKET, _observation(source_packet="packet-b")),
        (PairingField.ACTION_BUDGET, _observation(action_budget=0.25)),
        (PairingField.SUPPORT_BUDGET, _observation(support_budget=3)),
        (PairingField.CONFIRMATION_BUDGET, _observation(confirmation_budget=2)),
        (PairingField.ENVIRONMENT_LINEAGE, _observation(environment_lineage="d" * 64)),
    ),
)
def test_single_field_lineage_mismatch_is_rejected(
    config: FedorbitConfig,
    field: PairingField,
    counterpart: PairedObservation,
) -> None:
    del config
    with pytest.raises(PairingMismatchError) as raised:
        require_matching_lineage(_observation(), counterpart)
    assert raised.value.field is field
    assert field.value in str(raised.value)
