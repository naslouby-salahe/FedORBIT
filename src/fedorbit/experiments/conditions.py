from __future__ import annotations

from dataclasses import dataclass

ConditionEntry = str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegisteredConditions:
    entries: tuple[ConditionEntry, ...]

    def __post_init__(self) -> None:
        for entry in self.entries:
            values = (entry,) if isinstance(entry, str) else entry
            if not values or any(not value for value in values):
                raise ConditionRegistrationError("registered conditions must be non-empty")

    def __len__(self) -> int:
        return len(self.entries)


class ConditionRegistrationError(ValueError):
    pass
