from __future__ import annotations

from collections.abc import Mapping

from fedorbit.analysis.contrasts import (
    ConditionSummary,
    ContrastResult,
    analyse_contrast,
    index_cells,
    reference_aurocs,
    summarise_conditions,
)
from fedorbit.analysis.statistics import holm_adjust
from fedorbit.config.models import FedorbitConfig
from fedorbit.study.records import AnyRecord, IneligibleDeviceRecord, InfeasibleSupportRecord
from fedorbit.types import ContrastFamily, DeviceName, ExperimentId, FrozenModel

EvidenceRows = list[AnyRecord]


class DeviceReference(FrozenModel):
    experiment: ExperimentId
    device: DeviceName
    reference_auroc: float
    saturated: bool


class StudyAnalysis(FrozenModel):
    contrasts: tuple[ContrastResult, ...]
    conditions: tuple[ConditionSummary, ...]
    references: tuple[DeviceReference, ...]
    infeasible: tuple[InfeasibleSupportRecord, ...]
    ineligible: tuple[IneligibleDeviceRecord, ...]


def _p_value(result: ContrastResult) -> float:
    if result.signed_rank_p is None:
        raise ValueError("confirmatory contrasts need a p-value")
    return result.signed_rank_p


def _apply_holm(results: list[ContrastResult]) -> list[ContrastResult]:
    confirmatory = [
        position
        for position, result in enumerate(results)
        if result.family is ContrastFamily.PRIMARY and result.signed_rank_p is not None
    ]
    adjusted = holm_adjust([_p_value(results[position]) for position in confirmatory])
    updated = list(results)
    for position, value in zip(confirmatory, adjusted, strict=True):
        updated[position] = results[position].model_copy(update={"holm_adjusted_p": value})
    return updated


def analyse_study(
    config: FedorbitConfig, records: Mapping[ExperimentId, EvidenceRows]
) -> StudyAnalysis:
    contrasts: list[ContrastResult] = []
    conditions: list[ConditionSummary] = []
    references: list[DeviceReference] = []
    infeasible: list[InfeasibleSupportRecord] = []
    ineligible: list[IneligibleDeviceRecord] = []
    for experiment in config.experiments:
        rows = records[experiment.id]
        index = index_cells(rows)
        infeasible.extend(row for row in rows if isinstance(row, InfeasibleSupportRecord))
        ineligible.extend(row for row in rows if isinstance(row, IneligibleDeviceRecord))
        for spec in experiment.contrasts:
            for support_size in spec.support_sizes:
                contrasts.append(analyse_contrast(config, experiment, spec, support_size, index))
        conditions.extend(summarise_conditions(config, experiment, index))
        for device, value in sorted(reference_aurocs(experiment, index).items()):
            references.append(
                DeviceReference(
                    experiment=experiment.id,
                    device=device,
                    reference_auroc=value,
                    saturated=value >= 1.0 - config.statistics.saturation_headroom,
                )
            )
    return StudyAnalysis(
        contrasts=tuple(_apply_holm(contrasts)),
        conditions=tuple(conditions),
        references=tuple(references),
        infeasible=tuple(infeasible),
        ineligible=tuple(ineligible),
    )
