from typing import Iterator

from torch import Tensor
from torch.nn import Parameter


def clip_grad_norm_(parameters: Iterator[Parameter], max_norm: float) -> Tensor: ...
