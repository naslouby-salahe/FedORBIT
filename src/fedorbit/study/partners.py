from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedorbit.config.models import ConditionSpec
from fedorbit.detection.moments import MomentSummary
from fedorbit.types import DeviceName, FloatVector, PartnerPolicy, RandomSeed


class PartnerSelectionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PartnerSelectionContext:
    partners: dict[DeviceName, MomentSummary]
    deployable_reference_scale: FloatVector
    oracle_reference_scale: FloatVector
    similarity_scale_floor: float
    seed: RandomSeed


def scale_profile_distance(
    partner_scale: FloatVector, reference_scale: FloatVector, floor: float
) -> float:
    return float(np.mean(np.abs(np.log(partner_scale + floor) - np.log(reference_scale + floor))))


def _nearest(
    context: PartnerSelectionContext, reference_scale: FloatVector, count: int
) -> tuple[DeviceName, ...]:
    distances = {
        device: scale_profile_distance(
            summary.standard_deviation, reference_scale, context.similarity_scale_floor
        )
        for device, summary in context.partners.items()
    }
    ordered = sorted(distances, key=lambda device: (distances[device], device))
    return tuple(ordered[:count])


def select_partners(
    spec: ConditionSpec, context: PartnerSelectionContext
) -> tuple[DeviceName, ...]:
    policy = spec.partner_policy
    if policy is None:
        raise PartnerSelectionError("condition does not share")
    if policy is PartnerPolicy.ALL:
        return tuple(sorted(context.partners))
    count = spec.partner_count
    if count is None or count > len(context.partners):
        raise PartnerSelectionError("partner count exceeds the available partners")
    if policy is PartnerPolicy.DEPLOYABLE_NEAREST:
        return _nearest(context, context.deployable_reference_scale, count)
    if policy is PartnerPolicy.ORACLE_NEAREST:
        return _nearest(context, context.oracle_reference_scale, count)
    generator = np.random.default_rng(context.seed)
    devices = sorted(context.partners)
    chosen = generator.choice(len(devices), size=count, replace=False)
    return tuple(devices[int(index)] for index in sorted(chosen))
