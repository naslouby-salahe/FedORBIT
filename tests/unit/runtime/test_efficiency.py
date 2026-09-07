from __future__ import annotations

import time

import pytest
import torch

from fedorbit.infrastructure.runtime import measure_efficiency


def test_measure_efficiency_reports_positive_wall_time_and_rss() -> None:
    with measure_efficiency() as handle:
        time.sleep(0.01)
        _ = [0] * 100_000
    assert handle.result.wall_time_seconds >= 0.01
    assert handle.result.peak_host_rss_mib > 0.0
    assert handle.result.peak_cuda_allocated_bytes >= 0


def test_measure_efficiency_reports_zero_cuda_bytes_without_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with measure_efficiency() as handle:
        pass
    assert handle.result.peak_cuda_allocated_bytes == 0
