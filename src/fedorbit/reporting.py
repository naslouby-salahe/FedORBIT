from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from fedorbit.analysis.contrasts import ConditionSummary, ContrastResult
from fedorbit.analysis.synthesis import StudyAnalysis
from fedorbit.config.models import FedorbitConfig
from fedorbit.infrastructure.artifacts import Provenance, publish_artifact
from fedorbit.study.records import IneligibleDeviceRecord, InfeasibleSupportRecord
from fedorbit.types import ContrastFamily, ExperimentId

CONTRAST_COLUMNS = (
    "experiment",
    "contrast",
    "family",
    "treatment",
    "control",
    "support_size",
    "stratum",
    "direction",
    "status",
    "devices",
    "mean_delta",
    "interval_lower",
    "interval_upper",
    "positive_devices",
    "negative_devices",
    "tied_devices",
    "signed_rank_p",
    "sign_p",
    "holm_adjusted_p",
    "holm_significant",
    "mean_recovery",
    "recovery_devices",
)
DEVICE_COLUMNS = (
    "experiment",
    "contrast",
    "support_size",
    "device",
    "delta",
    "replicate_lower",
    "replicate_upper",
    "control_auroc",
    "treatment_auroc",
    "reference_auroc",
    "scale_mismatch",
    "split_half_instability",
    "recovery",
)
CONDITION_COLUMNS = (
    "experiment",
    "condition",
    "support_size",
    "devices",
    "mean_auroc",
    "interval_lower",
    "interval_upper",
    "mean_false_positive_rate",
    "mean_true_positive_rate",
)
INELIGIBLE_COLUMNS = (
    "experiment",
    "dataset",
    "device",
    "benign_rows",
    "attack_rows",
    "worst_case_standard_error",
)
INFEASIBLE_COLUMNS = ("experiment", "dataset", "device", "support_size", "available_rows")
PREDICTOR_COLUMNS = (
    "experiment",
    "contrast",
    "support_size",
    "predictor",
    "spearman",
    "lower",
    "upper",
)
FIGURE_METADATA = {"Date": None}
TableCell = str | int | float | bool | None
TableRow = tuple[TableCell, ...]


def _cell(value: TableCell) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _csv_bytes(columns: tuple[str, ...], rows: list[TableRow]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(value) for value in row])
    return buffer.getvalue().encode("utf-8")


def holm_significant(result: ContrastResult, alpha: float) -> bool | None:
    if result.holm_adjusted_p is None:
        return None
    return result.holm_adjusted_p < alpha


def contrast_rows(results: tuple[ContrastResult, ...], alpha: float) -> list[TableRow]:
    return [
        (
            result.experiment,
            result.contrast,
            result.family,
            result.treatment,
            result.control,
            result.support_size,
            result.stratum,
            result.direction,
            result.status,
            len(result.device_deltas),
            result.mean_delta,
            result.interval_lower,
            result.interval_upper,
            result.positive_devices,
            result.negative_devices,
            result.tied_devices,
            result.signed_rank_p,
            result.sign_p,
            result.holm_adjusted_p,
            holm_significant(result, alpha),
            result.mean_recovery,
            result.recovery_devices,
        )
        for result in results
    ]


def device_rows(results: tuple[ContrastResult, ...]) -> list[TableRow]:
    return [
        (
            result.experiment,
            result.contrast,
            result.support_size,
            delta.device,
            delta.delta,
            delta.replicate_lower,
            delta.replicate_upper,
            delta.control_auroc,
            delta.treatment_auroc,
            delta.reference_auroc,
            delta.scale_mismatch,
            delta.split_half_instability,
            delta.recovery,
        )
        for result in results
        for delta in result.device_deltas
    ]


def predictor_rows(results: tuple[ContrastResult, ...]) -> list[TableRow]:
    return [
        (
            result.experiment,
            result.contrast,
            result.support_size,
            predictor.predictor,
            predictor.spearman,
            predictor.lower,
            predictor.upper,
        )
        for result in results
        for predictor in result.predictors
    ]


def infeasible_rows(records: tuple[InfeasibleSupportRecord, ...]) -> list[TableRow]:
    return [
        (
            record.experiment,
            record.dataset,
            record.device,
            record.support_size,
            record.available_rows,
        )
        for record in records
    ]


def ineligible_rows(records: tuple[IneligibleDeviceRecord, ...]) -> list[TableRow]:
    return [
        (
            record.experiment,
            record.dataset,
            record.device,
            record.benign_rows,
            record.attack_rows,
            record.worst_case_standard_error,
        )
        for record in records
    ]


def condition_rows(summaries: tuple[ConditionSummary, ...]) -> list[TableRow]:
    return [
        (
            summary.experiment,
            summary.condition,
            summary.support_size,
            summary.devices,
            summary.mean_auroc,
            summary.interval_lower,
            summary.interval_upper,
            summary.mean_false_positive_rate,
            summary.mean_true_positive_rate,
        )
        for summary in summaries
    ]


def _svg_bytes(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", metadata=FIGURE_METADATA)
    plt.close(figure)
    return buffer.getvalue()


def dose_response_figure(
    summaries: tuple[ConditionSummary, ...], experiment: ExperimentId
) -> bytes:
    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    conditions = sorted({item.condition for item in summaries if item.experiment is experiment})
    for condition in conditions:
        points = sorted(
            (
                item
                for item in summaries
                if item.experiment is experiment and item.condition == condition
            ),
            key=lambda item: item.support_size,
        )
        axes.errorbar(
            [item.support_size for item in points],
            [item.mean_auroc for item in points],
            yerr=[
                [item.mean_auroc - item.interval_lower for item in points],
                [item.interval_upper - item.mean_auroc for item in points],
            ],
            marker="o",
            capsize=2,
            label=condition,
        )
    axes.set_xscale("log")
    axes.set_xlabel("local benign support (windows)")
    axes.set_ylabel("mean AUROC across devices")
    axes.legend(fontsize=7)
    return _svg_bytes(figure)


def device_delta_figure(results: tuple[ContrastResult, ...], experiment: ExperimentId) -> bytes:
    primary = [
        result
        for result in results
        if result.experiment is experiment and result.family is ContrastFamily.PRIMARY
    ]
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    for result in primary:
        axes.plot(
            [delta.device for delta in result.device_deltas],
            [delta.delta for delta in result.device_deltas],
            marker="o",
            linestyle="none",
            label=f"{result.contrast} (n={result.support_size})",
        )
    axes.axhline(0.0, color="black", linewidth=0.6)
    axes.set_ylabel("paired AUROC difference")
    axes.tick_params(axis="x", labelrotation=60, labelsize=6)
    axes.legend(fontsize=7)
    return _svg_bytes(figure)


def write_report(
    config: FedorbitConfig,
    analysis: StudyAnalysis,
    directory: Path,
    provenance: Provenance,
    revision: str,
) -> tuple[Path, ...]:
    outputs: dict[str, tuple[bytes, int]] = {
        "contrasts.csv": (
            _csv_bytes(
                CONTRAST_COLUMNS, contrast_rows(analysis.contrasts, config.statistics.alpha)
            ),
            len(analysis.contrasts),
        ),
        "devices.csv": (
            _csv_bytes(DEVICE_COLUMNS, device_rows(analysis.contrasts)),
            len(device_rows(analysis.contrasts)),
        ),
        "conditions.csv": (
            _csv_bytes(CONDITION_COLUMNS, condition_rows(analysis.conditions)),
            len(analysis.conditions),
        ),
        "infeasible_support.csv": (
            _csv_bytes(INFEASIBLE_COLUMNS, infeasible_rows(analysis.infeasible)),
            len(analysis.infeasible),
        ),
        "ineligible_devices.csv": (
            _csv_bytes(INELIGIBLE_COLUMNS, ineligible_rows(analysis.ineligible)),
            len(analysis.ineligible),
        ),
        "predictors.csv": (
            _csv_bytes(PREDICTOR_COLUMNS, predictor_rows(analysis.contrasts)),
            len(predictor_rows(analysis.contrasts)),
        ),
    }
    for experiment in config.experiments:
        outputs[f"dose_response_{experiment.id.value}.svg"] = (
            dose_response_figure(analysis.conditions, experiment.id),
            len(analysis.conditions),
        )
        outputs[f"device_deltas_{experiment.id.value}.svg"] = (
            device_delta_figure(analysis.contrasts, experiment.id),
            len(analysis.contrasts),
        )
    written: list[Path] = []
    for name, (payload, count) in sorted(outputs.items()):
        path = directory / name
        publish_artifact(path, payload, count, revision, provenance)
        written.append(path)
    return tuple(written)
