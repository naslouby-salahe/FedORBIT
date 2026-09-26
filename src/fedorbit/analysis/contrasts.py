from __future__ import annotations

from collections import defaultdict
from enum import StrEnum

import numpy as np
from pydantic import Field

from fedorbit.analysis.statistics import (
    cluster_bootstrap_interval,
    exact_sign_test,
    exact_signed_rank_test,
    spearman,
)
from fedorbit.config.models import ContrastSpec, ExperimentConfig, FedorbitConfig
from fedorbit.infrastructure.runtime import derive_seed
from fedorbit.study.records import AnyRecord, CellRecord
from fedorbit.types import (
    ConditionName,
    ContrastDirection,
    ContrastFamily,
    ContrastName,
    DeviceName,
    DeviceStratum,
    ExperimentId,
    FloatVector,
    FrozenModel,
    RandomPurpose,
    ReplicateIndex,
    ShareChannel,
    SupportSize,
)

CellKey = tuple[DeviceName, SupportSize, ConditionName]
CellIndex = dict[CellKey, dict[ReplicateIndex, CellRecord]]


class ContrastStatus(StrEnum):
    ESTIMATED = "estimated"
    NO_ELIGIBLE_DEVICES = "no-eligible-devices"


class DeviceDelta(FrozenModel):
    device: DeviceName
    delta: float
    replicate_lower: float
    replicate_upper: float
    control_auroc: float
    treatment_auroc: float
    reference_auroc: float | None
    scale_mismatch: float
    split_half_instability: float
    recovery: float | None


class PredictorCorrelation(FrozenModel):
    predictor: str
    spearman: float | None
    lower: float | None
    upper: float | None


class ContrastResult(FrozenModel):
    experiment: ExperimentId
    contrast: ContrastName
    family: ContrastFamily
    treatment: ConditionName
    control: ConditionName
    support_size: SupportSize
    stratum: DeviceStratum
    direction: ContrastDirection
    status: ContrastStatus
    device_deltas: tuple[DeviceDelta, ...]
    mean_delta: float | None
    interval_lower: float | None
    interval_upper: float | None
    positive_devices: int = Field(ge=0)
    negative_devices: int = Field(ge=0)
    tied_devices: int = Field(ge=0)
    signed_rank_p: float | None
    sign_p: float | None
    holm_adjusted_p: float | None = None
    mean_recovery: float | None
    recovery_devices: int = Field(ge=0)
    predictors: tuple[PredictorCorrelation, ...]


class ConditionSummary(FrozenModel):
    experiment: ExperimentId
    condition: ConditionName
    support_size: SupportSize
    devices: int
    mean_auroc: float
    interval_lower: float
    interval_upper: float
    mean_false_positive_rate: float | None
    mean_true_positive_rate: float | None


def index_cells(records: list[AnyRecord]) -> CellIndex:
    index: CellIndex = defaultdict(dict)
    for record in records:
        if isinstance(record, CellRecord):
            index[(record.device, record.support_size, record.condition)][record.replicate] = record
    return dict(index)


def reference_condition(experiment: ExperimentConfig) -> ConditionName | None:
    for name, spec in sorted(experiment.conditions.items()):
        if spec.channel is ShareChannel.LOCAL_FULL_SUPPORT:
            return name
    return None


def reference_aurocs(experiment: ExperimentConfig, index: CellIndex) -> dict[DeviceName, float]:
    condition = reference_condition(experiment)
    if condition is None:
        return {}
    return {
        device: float(np.mean([record.auroc for record in replicates.values()]))
        for (device, _, name), replicates in index.items()
        if name == condition
    }


def devices_with_condition(
    index: CellIndex, support_size: SupportSize, condition: ConditionName
) -> list[DeviceName]:
    return sorted(
        device for (device, size, name) in index if size == support_size and name == condition
    )


def _in_stratum(
    device: DeviceName,
    stratum: DeviceStratum,
    references: dict[DeviceName, float],
    saturation_headroom: float,
) -> bool:
    if stratum is DeviceStratum.ALL:
        return True
    reference = references.get(device)
    if reference is None:
        return False
    saturated = reference >= 1.0 - saturation_headroom
    return saturated if stratum is DeviceStratum.SATURATED else not saturated


def _device_delta(
    config: FedorbitConfig,
    experiment: ExperimentConfig,
    spec: ContrastSpec,
    support_size: SupportSize,
    device: DeviceName,
    index: CellIndex,
    references: dict[DeviceName, float],
) -> DeviceDelta | None:
    treatment = index.get((device, support_size, spec.treatment))
    control = index.get((device, support_size, spec.control))
    if not treatment or not control:
        return None
    shared = sorted(set(treatment) & set(control))
    if not shared:
        return None
    deltas = np.array([treatment[r].auroc - control[r].auroc for r in shared])
    seed = derive_seed(
        config.base_seed,
        RandomPurpose.BOOTSTRAP,
        experiment.id,
        spec.name,
        device,
        str(support_size),
    )
    lower, upper = cluster_bootstrap_interval(
        deltas, config.statistics.bootstrap_resamples, config.statistics.confidence_level, seed
    )
    control_mean = float(np.mean([control[r].auroc for r in shared]))
    treatment_mean = float(np.mean([treatment[r].auroc for r in shared]))
    reference = references.get(device)
    recovery = None
    if (
        reference is not None
        and reference - control_mean >= config.statistics.minimum_recovery_headroom
    ):
        recovery = (treatment_mean - control_mean) / (reference - control_mean)
    return DeviceDelta(
        device=device,
        delta=float(deltas.mean()),
        replicate_lower=lower,
        replicate_upper=upper,
        control_auroc=control_mean,
        treatment_auroc=treatment_mean,
        reference_auroc=reference,
        scale_mismatch=float(np.mean([control[r].scale_mismatch for r in shared])),
        split_half_instability=float(np.mean([control[r].split_half_instability for r in shared])),
        recovery=recovery,
    )


def _predictors(
    config: FedorbitConfig, seed_parts: tuple[str, ...], deltas: tuple[DeviceDelta, ...]
) -> tuple[PredictorCorrelation, ...]:
    values = np.array([delta.delta for delta in deltas])
    candidates: dict[str, FloatVector] = {
        "control-auroc": np.array([delta.control_auroc for delta in deltas]),
        "scale-mismatch": np.array([delta.scale_mismatch for delta in deltas]),
        "split-half-instability": np.array([delta.split_half_instability for delta in deltas]),
    }
    results: list[PredictorCorrelation] = []
    generator = np.random.default_rng(
        derive_seed(config.base_seed, RandomPurpose.BOOTSTRAP, "predictors", *seed_parts)
    )
    for name, predictor in candidates.items():
        observed = spearman(predictor, values)
        if observed is None:
            results.append(
                PredictorCorrelation(predictor=name, spearman=None, lower=None, upper=None)
            )
            continue
        draws: list[float] = []
        for _ in range(config.statistics.bootstrap_resamples):
            chosen = generator.integers(0, len(values), size=len(values))
            resampled = spearman(predictor[chosen], values[chosen])
            if resampled is not None:
                draws.append(resampled)
        tail = (1.0 - config.statistics.confidence_level) / 2.0
        lower = float(np.quantile(draws, tail)) if draws else None
        upper = float(np.quantile(draws, 1.0 - tail)) if draws else None
        results.append(
            PredictorCorrelation(predictor=name, spearman=observed, lower=lower, upper=upper)
        )
    return tuple(results)


def analyse_contrast(
    config: FedorbitConfig,
    experiment: ExperimentConfig,
    spec: ContrastSpec,
    support_size: SupportSize,
    index: CellIndex,
) -> ContrastResult:
    references = reference_aurocs(experiment, index)
    eligible = [
        device
        for device in devices_with_condition(index, support_size, spec.treatment)
        if _in_stratum(device, spec.stratum, references, config.statistics.saturation_headroom)
    ]
    deltas = tuple(
        delta
        for device in eligible
        if (
            delta := _device_delta(
                config, experiment, spec, support_size, device, index, references
            )
        )
        is not None
    )
    if not deltas:
        return ContrastResult(
            experiment=experiment.id,
            contrast=spec.name,
            family=spec.family,
            treatment=spec.treatment,
            control=spec.control,
            support_size=support_size,
            stratum=spec.stratum,
            direction=spec.direction,
            status=ContrastStatus.NO_ELIGIBLE_DEVICES,
            device_deltas=(),
            mean_delta=None,
            interval_lower=None,
            interval_upper=None,
            positive_devices=0,
            negative_devices=0,
            tied_devices=0,
            signed_rank_p=None,
            sign_p=None,
            mean_recovery=None,
            recovery_devices=0,
            predictors=(),
        )
    values = np.array([delta.delta for delta in deltas])
    seed = derive_seed(
        config.base_seed,
        RandomPurpose.BOOTSTRAP,
        experiment.id,
        spec.name,
        "population",
        str(support_size),
    )
    lower, upper = cluster_bootstrap_interval(
        values, config.statistics.bootstrap_resamples, config.statistics.confidence_level, seed
    )
    recoveries = [delta.recovery for delta in deltas if delta.recovery is not None]
    return ContrastResult(
        experiment=experiment.id,
        contrast=spec.name,
        family=spec.family,
        treatment=spec.treatment,
        control=spec.control,
        support_size=support_size,
        stratum=spec.stratum,
        direction=spec.direction,
        status=ContrastStatus.ESTIMATED,
        device_deltas=deltas,
        mean_delta=float(values.mean()),
        interval_lower=lower,
        interval_upper=upper,
        positive_devices=int((values > 0).sum()),
        negative_devices=int((values < 0).sum()),
        tied_devices=int((values == 0).sum()),
        signed_rank_p=exact_signed_rank_test(values, spec.direction),
        sign_p=exact_sign_test(values, spec.direction),
        mean_recovery=float(np.mean(recoveries)) if recoveries else None,
        recovery_devices=len(recoveries),
        predictors=_predictors(config, (experiment.id, spec.name, str(support_size)), deltas),
    )


def summarise_conditions(
    config: FedorbitConfig, experiment: ExperimentConfig, index: CellIndex
) -> list[ConditionSummary]:
    grouped: dict[tuple[ConditionName, SupportSize], list[DeviceName]] = defaultdict(list)
    for device, size, condition in index:
        grouped[(condition, size)].append(device)
    summaries: list[ConditionSummary] = []
    reference = reference_condition(experiment)
    for (condition, size), devices in sorted(grouped.items()):
        if condition == reference:
            continue
        aurocs: list[float] = []
        false_positive_rates: list[float] = []
        true_positive_rates: list[float] = []
        for device in sorted(devices):
            replicates = index[(device, size, condition)].values()
            aurocs.append(float(np.mean([record.auroc for record in replicates])))
            rates = [record.false_positive_rate for record in replicates]
            if all(rate is not None for rate in rates):
                false_positive_rates.append(float(np.mean([r for r in rates if r is not None])))
                true_positive_rates.append(
                    float(
                        np.mean(
                            [
                                record.true_positive_rate
                                for record in replicates
                                if record.true_positive_rate is not None
                            ]
                        )
                    )
                )
        values = np.array(aurocs)
        seed = derive_seed(
            config.base_seed,
            RandomPurpose.BOOTSTRAP,
            experiment.id,
            condition,
            "summary",
            str(size),
        )
        lower, upper = cluster_bootstrap_interval(
            values, config.statistics.bootstrap_resamples, config.statistics.confidence_level, seed
        )
        summaries.append(
            ConditionSummary(
                experiment=experiment.id,
                condition=condition,
                support_size=size,
                devices=len(devices),
                mean_auroc=float(values.mean()),
                interval_lower=lower,
                interval_upper=upper,
                mean_false_positive_rate=float(np.mean(false_positive_rates))
                if false_positive_rates
                else None,
                mean_true_positive_rate=float(np.mean(true_positive_rates))
                if true_positive_rates
                else None,
            )
        )
    return summaries
