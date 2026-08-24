from __future__ import annotations

from dataclasses import dataclass


class FigureError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FigureSeries:
    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise FigureError("figure series name must be non-empty")
        if len(self.x) != len(self.y):
            raise FigureError("figure series coordinates differ in length")


@dataclass(frozen=True, slots=True)
class EvidenceFigurePayload:
    x_label: str
    y_label: str
    series: tuple[FigureSeries, ...]


@dataclass(frozen=True, slots=True)
class EvidenceFigure:
    x_label: str
    y_label: str
    series: tuple[FigureSeries, ...]

    def __post_init__(self) -> None:
        if not self.x_label or not self.y_label:
            raise FigureError("figure axes must be named")
        if not self.series:
            raise FigureError("evidence figure requires at least one series")

    def payload(self) -> EvidenceFigurePayload:
        return EvidenceFigurePayload(self.x_label, self.y_label, self.series)
