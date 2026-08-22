from typing import Iterator

from torch.nn import Parameter


class AdamW:
    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None: ...
    def step(self, closure: object | None = None) -> None: ...
    def zero_grad(self, set_to_none: bool = True) -> None: ...
