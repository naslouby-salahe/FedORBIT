from __future__ import annotations

import pytest

from fedorbit.analysis.comparisons import (
    PairContrastEvidence,
    PairContrastEvidenceSet,
)
from fedorbit.config.loading import load_fedorbit_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.types import (
    DirectedPairName,
    StrictResourceValidity,
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
