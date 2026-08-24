from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import cast


class CanonicalSerializationError(ValueError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":"))


def _canonical_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(mapping.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence = cast(Sequence[object], value)
        return [_canonical_value(item) for item in sequence]
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalSerializationError(f"non-finite canonical value: {value}")
        return value
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    raise CanonicalSerializationError(f"unsupported canonical value: {type(value).__name__}")
