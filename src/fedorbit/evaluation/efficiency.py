from __future__ import annotations

import math
from dataclasses import dataclass


class EfficiencyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EfficiencyRecord:
    wall_time_seconds: float
    peak_host_rss_mib: float
    peak_cuda_allocated_bytes: int
    packet_serialized_byte_count: int
    source_response_optimizer_steps: int
    target_confirmation_optimizer_steps: int
    live_assimilation_optimizer_steps: int
    timeout_indicator: bool
    resource_limit_indicator: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds < 0.0:
            raise EfficiencyError("wall time must be finite and nonnegative")
        if not math.isfinite(self.peak_host_rss_mib) or self.peak_host_rss_mib < 0.0:
            raise EfficiencyError("peak host RSS must be finite and nonnegative")
        for name, value in (
            ("CUDA bytes", self.peak_cuda_allocated_bytes),
            ("packet bytes", self.packet_serialized_byte_count),
            ("source response steps", self.source_response_optimizer_steps),
            ("confirmation steps", self.target_confirmation_optimizer_steps),
            ("assimilation steps", self.live_assimilation_optimizer_steps),
        ):
            if value < 0:
                raise EfficiencyError(f"{name} must be nonnegative")
