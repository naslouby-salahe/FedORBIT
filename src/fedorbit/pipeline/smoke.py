from __future__ import annotations

from fedorbit.config.models import FedorbitConfig
from fedorbit.types import ExperimentId

SMOKE_DEVICE_COUNT = 2
SMOKE_REPLICATES = 2
SMOKE_BOOTSTRAP_RESAMPLES = 200


def smoke_config(config: FedorbitConfig) -> FedorbitConfig:
    experiment = config.experiment(ExperimentId.COLD_START_LADDER)
    smallest = min(experiment.support_sizes)
    contrasts = tuple(
        contrast.model_copy(update={"support_sizes": (smallest,)})
        for contrast in experiment.contrasts
        if smallest in contrast.support_sizes
    )
    reduced = experiment.model_copy(
        update={
            "support_sizes": (smallest,),
            "replicates": SMOKE_REPLICATES,
            "contrasts": contrasts,
        }
    )
    document = config.model_dump()
    document["datasets"]["nbaiot"]["devices"] = config.datasets.nbaiot.devices[:SMOKE_DEVICE_COUNT]
    document["experiments"] = [reduced.model_dump()]
    document["statistics"]["bootstrap_resamples"] = SMOKE_BOOTSTRAP_RESAMPLES
    return FedorbitConfig.model_validate(document)
