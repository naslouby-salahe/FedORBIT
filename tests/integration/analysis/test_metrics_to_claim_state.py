from __future__ import annotations

from fedorbit.analysis.claims import evaluate_transfer_style_criteria
from fedorbit.analysis.comparisons import PairContrastEvidence, PairContrastEvidenceSet
from fedorbit.analysis.statistics import NamedPValue, PValueSet, exact_sign_flip_test, holm_step_down
from fedorbit.config.loading import load_fedorbit_config


def test_metrics_to_claim_state_uses_typed_deterministic_statistics() -> None:
    config = load_fedorbit_config()
    result = exact_sign_flip_test(config, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    assert 0.0 <= result.p_value <= 1.0
    adjusted = holm_step_down(
        PValueSet((NamedPValue("a", 0.01), NamedPValue("b", 0.02), NamedPValue("c", 0.5)))
    )
    assert adjusted.value_of("a") is not None
    assert adjusted.value_of("b") is not None
    assert adjusted.value_of("c") is not None
    evidence = PairContrastEvidenceSet(
        (
            PairContrastEvidence("p1", 0.05, 0.01, 0.02, True, 10),
            PairContrastEvidence("p2", 0.04, 0.01, 0.02, True, 10),
            PairContrastEvidence("p3", 0.03, 0.01, 0.02, True, 10),
            PairContrastEvidence("p4", 0.0, 0.5, -0.01, True, 10),
        )
    )
    decision = evaluate_transfer_style_criteria(config, evidence, False)
    assert decision.supported
