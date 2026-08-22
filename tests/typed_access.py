from __future__ import annotations

from typing import cast


def as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def as_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)


def as_float(value: object) -> float:
    assert isinstance(value, (int, float))
    return float(value)


def as_int(value: object) -> int:
    assert isinstance(value, int)
    return value


def as_str(value: object) -> str:
    assert isinstance(value, str)
    return value


def as_bool(value: object) -> bool:
    assert isinstance(value, bool)
    return value
