from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConditionLabel:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ConditionRegistrationError("registered conditions must be non-empty")


@dataclass(frozen=True, slots=True)
class RegisteredCondition:
    labels: tuple[ConditionLabel, ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ConditionRegistrationError("registered condition must contain at least one label")


@dataclass(frozen=True, slots=True)
class RegisteredConditions:
    entries: tuple[RegisteredCondition, ...]

    def __post_init__(self) -> None:
        if len(set(self.entries)) != len(self.entries):
            raise ConditionRegistrationError("registered conditions must be distinct")

    def __len__(self) -> int:
        return len(self.entries)


class ConditionRegistrationError(ValueError):
    pass
