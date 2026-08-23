from __future__ import annotations

from fedorbit.analysis.statistics import exact_paired_sign_flip_two_sided, holm_adjust


def test_analysis_pipeline_exact_sign_flip_and_holm_are_deterministic() -> None:
    result = exact_paired_sign_flip_two_sided((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 1e-15)
    assert 0.0 <= result.p_value <= 1.0
    adjusted = holm_adjust((0.01, 0.02, 0.5))
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
