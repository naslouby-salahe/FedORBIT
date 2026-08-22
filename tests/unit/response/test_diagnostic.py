from __future__ import annotations

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.response.bootstrap import max_t_critical_value


def test_diagnostic_config_values() -> None:
    config = load_fedorbit_config()
    diagnostic = config.scientific.target_response_diagnostic
    assert diagnostic.intervention_magnitude == 0.10
    assert diagnostic.shadow_optimizer_steps == 25
    assert diagnostic.paired_replicates == 8
    assert diagnostic.simultaneous_bootstrap_resamples == 1000
    assert diagnostic.confidence_level == 0.95


def test_bootstrap_accepts_parameter_override() -> None:
    config = load_fedorbit_config()
    entries = ((1.0, 1.5, 2.0), (0.5, 1.0, 1.5))
    default = max_t_critical_value(config, entries, 7)
    overridden = max_t_critical_value(
        config,
        entries,
        7,
        resamples=1000,
        confidence_level=0.95,
        standard_error_floor=1e-12,
    )
    assert overridden > 0
    assert default > 0
