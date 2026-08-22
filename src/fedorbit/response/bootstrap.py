from __future__ import annotations

import math
import statistics

import numpy as np
import torch

from fedorbit.config.models import FedorbitConfig
from fedorbit.runtime.seeds import RngNamespace, derive_seed32


class BootstrapError(ValueError):
    pass


def max_t_critical_value(
    config: FedorbitConfig,
    entry_derivatives: tuple[tuple[float, ...], ...],
    seed: int,
) -> float:
    final = config.scientific.source_response_final
    means = tuple(statistics.fmean(values) for values in entry_derivatives)
    if not means:
        raise BootstrapError("no response entries for bootstrap")
    rng = torch.Generator().manual_seed(
        derive_seed32(seed, RngNamespace.STATISTICAL_BOOTSTRAP, "max-t")
    )
    replicate_count = len(entry_derivatives[0])
    statistics_values: list[float] = []
    for _ in range(final.max_t_bootstrap_resamples):
        indices = tuple(
            int(torch.randint(0, replicate_count, (1,), generator=rng)[0])
            for _ in range(replicate_count)
        )
        studentized: list[float] = []
        for entry_index, values in enumerate(entry_derivatives):
            resampled = tuple(values[index] for index in indices)
            bootstrap_mean = statistics.fmean(resampled)
            bootstrap_se = _bootstrap_se(resampled)
            studentized.append(
                abs(bootstrap_mean - means[entry_index])
                / max(bootstrap_se, final.response_standard_error_floor)
            )
        statistics_values.append(max(studentized))
    if not statistics_values:
        raise BootstrapError("no bootstrap resamples produced statistics")
    return float(
        np.quantile(
            np.asarray(statistics_values, dtype=np.float64),
            final.simultaneous_confidence_level,
            method="higher",
        )
    )


def _bootstrap_se(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values) / math.sqrt(len(values))
