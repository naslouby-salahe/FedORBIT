from __future__ import annotations

from collections.abc import Mapping
from typing import cast


class ConfigDocument:
    def __init__(self, root: Mapping[str, object]) -> None:
        self._root = dict(root)

    def as_dict(self) -> dict[str, object]:
        return self._root

    def section(self, *path: str) -> ConfigDocument:
        current: object = self._root
        for key in path:
            current = _mapping(current)[key]
        return ConfigDocument(_mapping(current))

    def list(self, *path: str) -> list[object]:
        current: object = self._root
        for key in path:
            current = _mapping(current)[key]
        return _list(current)

    def set_value(self, *path: str | int, value: object) -> None:
        current: object = self._root
        for segment in path[:-1]:
            if isinstance(segment, int):
                current = _list(current)[segment]
            else:
                current = _mapping(current)[segment]
        final = path[-1]
        if isinstance(final, int):
            _list(current)[final] = value
        else:
            _mapping(current)[final] = value


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _list(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)
