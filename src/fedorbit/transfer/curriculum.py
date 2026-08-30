from __future__ import annotations

from dataclasses import dataclass

import torch


class CurriculumError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CurriculumMultipliers:
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.values.numel() == 0:
            raise CurriculumError("curriculum multipliers must be a non-empty vector")
        if not bool(torch.isfinite(self.values).all()) or bool((self.values < 0.0).any()):
            raise CurriculumError("curriculum multipliers must be finite and nonnegative")
