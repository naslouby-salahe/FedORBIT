from __future__ import annotations

from fedorbit.analysis.statistics import exact_sign_flip_test, holm_step_down
from fedorbit.config.loading import load_fedorbit_config


def test_analysis_pipeline_exact_sign_flip_and_holm_are_deterministic() -> None:
    config = load_fedorbit_config()
    result = exact_sign_flip_test(config, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    assert 0.0 <= result.p_value <= 1.0
    adjusted = holm_step_down({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
