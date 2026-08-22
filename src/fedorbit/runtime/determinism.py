from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import torch

_set_deterministic_algorithms = cast(Callable[[bool], None], torch.use_deterministic_algorithms)


class PrincipalDeterminismError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeterministicBackendState:
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    matmul_fp32_precision: str
    conv_fp32_precision: str
    stochastic_rounding: bool
    default_dtype: str
    float32_matmul_precision: str


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise PrincipalDeterminismError(
            "principal execution requires CUDA; no CUDA device is available"
        )


def apply_deterministic_backend(require_cuda_device: bool = True) -> None:
    if require_cuda_device:
        require_cuda()
    _set_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    matmul_precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if matmul_precision is not None:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    conv = getattr(torch.backends.cudnn, "conv", None)
    if conv is not None:
        conv_precision = getattr(conv, "fp32_precision", None)
        if conv_precision is not None:
            conv.fp32_precision = "ieee"
    stochastic_rounding = getattr(torch.backends.cuda.matmul, "stochastic_rounding", None)
    if stochastic_rounding is not None:
        torch.backends.cuda.matmul.stochastic_rounding = False
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")


@contextmanager
def principal_determinism() -> Generator[None]:
    apply_deterministic_backend(require_cuda_device=True)
    yield


@contextmanager
def test_determinism() -> Generator[None]:
    apply_deterministic_backend(require_cuda_device=False)
    yield


def _conv_fp32_precision() -> str:
    conv = getattr(torch.backends.cudnn, "conv", None)
    if conv is not None:
        precision = getattr(conv, "fp32_precision", None)
        if precision is not None:
            return str(precision)
    return "absent"


def _matmul_fp32_precision() -> str:
    precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if precision is not None:
        return str(precision)
    return "absent"


def _stochastic_rounding() -> bool:
    if hasattr(torch.backends.cuda.matmul, "stochastic_rounding"):
        return bool(torch.backends.cuda.matmul.stochastic_rounding)
    return False


def deterministic_backend_state() -> DeterministicBackendState:
    return DeterministicBackendState(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        matmul_fp32_precision=_matmul_fp32_precision(),
        conv_fp32_precision=_conv_fp32_precision(),
        stochastic_rounding=_stochastic_rounding(),
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def assert_float32_training(dtype: torch.dtype) -> None:
    if dtype != torch.float32:
        raise PrincipalDeterminismError(f"principal training must remain float32; found {dtype}")
