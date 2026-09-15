from __future__ import annotations

import re
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import cast

import pytest
import torch

import fedorbit.infrastructure.runtime as runtime_module
from fedorbit.config.loading import active_config
from fedorbit.infrastructure.environment import (
    environment_snapshot,
    observed_hardware,
    reference_gpu_matches,
)
from fedorbit.infrastructure.runtime import (
    ExecutionDeviceUnavailableError,
    PrincipalDeterminismError,
    apply_deterministic_backend,
    assert_float32_training,
    deterministic_backend_state,
    execution_device,
    measure_efficiency,
    principal_determinism,
    require_cuda,
    synchronize_cuda,
)
from fedorbit.types import RuntimeDeviceType

SHA256_DIGEST_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
IEEE_PRECISION = "ieee"
HIGHEST_FLOAT32_MATMUL_PRECISION = "highest"
CUDA_SYNCHRONIZATIONS_PER_TIMED_REGION = 2
NON_FLOAT32_DTYPES = (torch.float64, torch.float16, torch.bfloat16)


@contextmanager
def _restored_deterministic_backend() -> Generator[None]:
    before = deterministic_backend_state()
    try:
        yield
    finally:
        cast(Callable[[bool], None], torch.use_deterministic_algorithms)(
            before.deterministic_algorithms
        )
        torch.backends.cudnn.benchmark = before.cudnn_benchmark
        torch.backends.cudnn.deterministic = before.cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = before.matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = before.cudnn_allow_tf32


class CudaSynchronizationRecorder:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> None:
        self.count += 1


def test_registered_execution_device_is_cuda_without_a_silent_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = active_config().runtime.execution_device
    assert configured is RuntimeDeviceType.CUDA
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ExecutionDeviceUnavailableError):
        execution_device()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert execution_device() == torch.device(configured)


def test_principal_determinism_guard_fails_closed_without_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    before = deterministic_backend_state()
    with pytest.raises(PrincipalDeterminismError):
        require_cuda()
    with pytest.raises(PrincipalDeterminismError):
        apply_deterministic_backend()
    assert deterministic_backend_state() == before
    with pytest.raises(PrincipalDeterminismError), principal_determinism():
        pass
    assert deterministic_backend_state() == before


def test_cpu_smoke_path_applies_every_deterministic_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with _restored_deterministic_backend():
        apply_deterministic_backend(require_cuda_device=False)
        state = deterministic_backend_state()
        assert state.deterministic_algorithms is True
        assert state.cudnn_benchmark is False
        assert state.cudnn_deterministic is True
        assert state.matmul_allow_tf32 is False
        assert state.cudnn_allow_tf32 is False
        assert state.matmul_fp32_precision == IEEE_PRECISION
        assert state.conv_fp32_precision == IEEE_PRECISION
        assert state.stochastic_rounding is False
        assert state.default_dtype == str(torch.float32)
        assert state.float32_matmul_precision == HIGHEST_FLOAT32_MATMUL_PRECISION
        assert torch.is_autocast_enabled() is False


def test_cpu_context_requires_the_explicit_cuda_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with _restored_deterministic_backend():
        with pytest.raises(PrincipalDeterminismError):
            apply_deterministic_backend(require_cuda_device=True)
        apply_deterministic_backend(require_cuda_device=False)
        assert deterministic_backend_state().deterministic_algorithms is True


def test_assert_float32_training_rejects_every_other_dtype() -> None:
    assert_float32_training(torch.float32)
    for dtype in NON_FLOAT32_DTYPES:
        with pytest.raises(PrincipalDeterminismError):
            assert_float32_training(dtype)


def test_timed_region_synchronizes_cuda_before_and_after(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert synchronize_cuda() is None
    synchronizations = CudaSynchronizationRecorder()
    monkeypatch.setattr(runtime_module, "synchronize_cuda", synchronizations)
    assert synchronizations.count == 0
    with measure_efficiency() as measurement:
        assert measurement.result.wall_time_seconds == 0.0
    assert synchronizations.count == CUDA_SYNCHRONIZATIONS_PER_TIMED_REGION
    assert measurement.result.wall_time_seconds >= 0.0
    assert measurement.result.peak_host_rss_mib > 0.0
    assert measurement.result.peak_cuda_allocated_bytes == 0


def test_environment_snapshot_reports_the_cuda_availability_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = environment_snapshot()
    assert snapshot.hardware.cuda_available is torch.cuda.is_available()
    assert SHA256_DIGEST_PATTERN.match(snapshot.fingerprint_sha256) is not None
    assert environment_snapshot() == snapshot
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cpu_only = environment_snapshot()
    assert cpu_only.hardware.cuda_available is False
    assert cpu_only.hardware.gpu_name is None
    assert cpu_only.hardware.gpu_memory_bytes is None
    assert observed_hardware().cuda_available is False
    assert reference_gpu_matches() is False
    with pytest.raises(ExecutionDeviceUnavailableError):
        execution_device()
