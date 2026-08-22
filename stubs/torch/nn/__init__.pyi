from typing import Iterator

from torch import Tensor

from . import init, utils


class Parameter(Tensor): ...


class Module:
    def __call__(self, *args: object) -> Tensor: ...
    def reset_parameters(self) -> None: ...
    def parameters(self) -> Iterator[Parameter]: ...
    def named_parameters(self) -> Iterator[tuple[str, Parameter]]: ...
    def zero_grad(self, set_to_none: bool = True) -> None: ...
    def train(self, mode: bool = True) -> Module: ...
    def eval(self) -> Module: ...
    def modules(self) -> Iterator[Module]: ...
    def state_dict(self) -> dict[str, Tensor]: ...


class Sequential(Module):
    def __init__(self, *modules: Module) -> None: ...
    def __iter__(self) -> Iterator[Module]: ...


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.weight: Tensor
        self.bias: Tensor
        self.in_features: int
        self.out_features: int

    def forward(self, input: Tensor) -> Tensor: ...


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        self.eps: float
        self.elementwise_affine: bool

    def forward(self, input: Tensor) -> Tensor: ...


class GELU(Module):
    def __init__(self, approximate: str = "none") -> None:
        self.approximate: str

    def forward(self, input: Tensor) -> Tensor: ...


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        self.p: float

    def forward(self, input: Tensor) -> Tensor: ...


class BatchNorm1d(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        self.eps: float
        self.momentum: float | None
        self.affine: bool
        self.track_running_stats: bool

    def forward(self, input: Tensor) -> Tensor: ...


class ReLU(Module):
    def __init__(self, inplace: bool = False) -> None:
        self.inplace: bool

    def forward(self, input: Tensor) -> Tensor: ...


class CrossEntropyLoss(Module):
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: object | None = None,
        ignore_index: int = -100,
        reduce: object | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...
