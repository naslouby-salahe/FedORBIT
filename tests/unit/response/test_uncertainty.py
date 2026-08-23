from __future__ import annotations

import math

import numpy as np
import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.response.uncertainty import (
    FinalResponseEntry,
    FinalResponseEstimate,
    ResponseUncertaintyError,
    max_t_critical_value,
    standard_error,
)


def test_max_t_bootstrap_is_deterministic() -> None:
    config = load_fedorbit_config()
    entries = ((1.0, 1.5, 2.0), (0.5, 1.0, 1.5))
    assert max_t_critical_value(config, entries, 7) == max_t_critical_value(config, entries, 7)


def test_max_t_bootstrap_uses_higher_quantile() -> None:
    values = np.asarray(tuple(float(value) for value in range(100)), dtype=np.float64)
    assert float(np.quantile(values, 0.95, method="higher")) == 95.0


def test_max_t_bootstrap_rejects_invalid_replication() -> None:
    config = load_fedorbit_config()
    with pytest.raises(ResponseUncertaintyError):
        max_t_critical_value(config, (), 7)
    with pytest.raises(ResponseUncertaintyError):
        max_t_critical_value(config, ((1.0,),), 7)
    with pytest.raises(ResponseUncertaintyError):
        max_t_critical_value(config, ((1.0, 2.0), (1.0,)), 7)


def test_standard_error_uses_sample_sd() -> None:
    values = (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
    assert standard_error(values) == pytest.approx(math.sqrt(32.0 / 7.0) / math.sqrt(8.0))
    assert math.isnan(standard_error((1.0,)))


def test_final_response_records_are_unclipped() -> None:
    entries = (
        FinalResponseEntry(0, 0, -2.0, 0.5, -3.5, -0.5, True),
        FinalResponseEntry(1, 0, 2.0, 0.5, 0.5, 3.5, True),
    )
    estimate = FinalResponseEstimate(entries, 3.0, 1, 1.5, False)
    assert estimate.entries[0].lower == -3.5
    assert estimate.entries[1].upper == 3.5
