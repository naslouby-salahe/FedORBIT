class matmul:
    allow_tf32: bool
    fp32_precision: str
    stochastic_rounding: bool


def is_available() -> bool: ...
def get_device_name(device: object | None = None) -> str: ...
