from __future__ import annotations

import pytest
import torch

from fedorbit.runtime.determinism import (
    PrincipalDeterminismError,
    apply_deterministic_backend,
    assert_float32_training,
    deterministic_backend_state,
    principal_determinism,
    require_cuda,
    synchronize_cuda,
)
from fedorbit.runtime.determinism import (
    test_determinism as cpu_test_determinism,
)


def test_apply_deterministic_backend_sets_all_flags() -> None:
    apply_deterministic_backend(require_cuda_device=False)
    state = deterministic_backend_state()
    assert state.deterministic_algorithms
    assert not state.cudnn_benchmark
    assert state.cudnn_deterministic
    assert not state.matmul_allow_tf32
    assert not state.cudnn_allow_tf32
    assert state.matmul_fp32_precision == "ieee"
    assert state.conv_fp32_precision == "ieee"
    assert not state.stochastic_rounding
    assert state.default_dtype == "torch.float32"
    assert state.float32_matmul_precision == "highest"


def test_require_cuda_raises_without_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PrincipalDeterminismError):
        require_cuda()


def test_principal_determinism_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PrincipalDeterminismError), principal_determinism():
        pass


def test_test_determinism_allows_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with cpu_test_determinism():
        assert torch.are_deterministic_algorithms_enabled()


def test_principal_determinism_applies_flags() -> None:
    with principal_determinism():
        state = deterministic_backend_state()
        assert state.deterministic_algorithms
        assert not state.matmul_allow_tf32


def test_synchronize_cuda_noop_without_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    synchronize_cuda()


def test_assert_float32_training_accepts_float32() -> None:
    assert_float32_training(torch.float32)


def test_assert_float32_training_rejects_float64() -> None:
    with pytest.raises(PrincipalDeterminismError):
        assert_float32_training(torch.float64)


def test_state_reflects_manual_flag_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_deterministic_backend(require_cuda_device=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    state = deterministic_backend_state()
    assert state.cudnn_benchmark
