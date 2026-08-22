from torch import Tensor


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> None: ...
def zeros_(tensor: Tensor) -> None: ...
def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> None: ...
