from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol, cast

import pandas as pd
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict
from reportlab.pdfgen.canvas import Canvas

from fedorbit.analysis.records import MetricRecord
from fedorbit.config.loading import active_config
from fedorbit.infrastructure.manifests import ReusableArtifactManifest
from fedorbit.infrastructure.storage import atomic_write_bytes, atomic_write_json
from fedorbit.infrastructure.workspace import WorkspaceLayout, results_workspace
from fedorbit.types import (
    ArtifactIdentifier,
    ExperimentName,
    ProjectSummaryColumn,
    ReportArtifactName,
    ReportAxisLabel,
    ReportColumns,
    ReportCoordinates,
    ReportingPathSegment,
    ReportSeriesName,
    StableJsonPayload,
    stable_json,
)


class FigureError(ValueError):
    pass


REPORT_FIGURE_WIDTH = 480
REPORT_FIGURE_HEIGHT = 180
REPORT_METRIC_SCALE = 4.0
REPORT_BAR_COLOR = "#2a6fbb"


@dataclass(frozen=True, slots=True)
class FigureSeries:
    name: ReportSeriesName
    x: ReportCoordinates
    y: ReportCoordinates

    def __post_init__(self) -> None:
        if not self.name:
            raise FigureError("figure series name must be non-empty")
        if len(self.x) != len(self.y):
            raise FigureError("figure series coordinates differ in length")


@dataclass(frozen=True, slots=True)
class EvidenceFigure:
    x_label: ReportAxisLabel
    y_label: ReportAxisLabel
    series: tuple[FigureSeries, ...]

    def __post_init__(self) -> None:
        if not self.x_label or not self.y_label:
            raise FigureError("figure axes must be named")
        if not self.series:
            raise FigureError("evidence figure requires at least one series")


class TableError(ValueError):
    pass


TableScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class EvidenceTablePayload:
    columns: ReportColumns
    rows: tuple[tuple[TableScalar, ...], ...]


@dataclass(frozen=True, slots=True)
class EvidenceTable:
    columns: ReportColumns
    rows: tuple[tuple[TableScalar, ...], ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise TableError("evidence table requires at least one column")
        if len(set(self.columns)) != len(self.columns):
            raise TableError("evidence table columns must be unique")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise TableError("evidence table row width differs from column count")

    def payload(self) -> EvidenceTablePayload:
        return EvidenceTablePayload(self.columns, self.rows)


class EvidenceExportError(ValueError):
    pass


class MetricArtifactPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric_record: MetricRecord | None = None


class ArtifactResolver(Protocol):
    def resolve(self, artifact_id: ArtifactIdentifier) -> ReusableArtifactManifest: ...


def _metric_tabular_values(metric: MetricRecord) -> tuple[tuple[str, ...], tuple[str, ...]]:
    serialized = metric.model_dump(mode="json")
    return tuple(serialized), tuple(str(value) for value in serialized.values())


class VerifiedEvidenceWriter:
    def __init__(self, store: ArtifactResolver, layout: WorkspaceLayout) -> None:
        self._store = store
        self._layout = layout

    def write(
        self,
        experiment: ExperimentName,
        artifact_id: ArtifactIdentifier,
        evidence: StableJsonPayload,
        overwrite: bool = False,
    ) -> Path:
        try:
            self._store.resolve(artifact_id)
        except ValueError as error:
            raise EvidenceExportError(
                f"evidence requires a verified completed artifact: {error}"
            ) from error
        workspace = results_workspace(self._layout, experiment)
        destination = workspace / f"{experiment.value}{ReportingPathSegment.EVIDENCE_SUFFIX.value}"
        rendered = (stable_json(evidence) + "\n").encode("utf-8")
        if destination.is_file():
            if destination.read_bytes() == rendered:
                return destination
            if not overwrite:
                raise EvidenceExportError(
                    "evidence export already exists with different content; use --overwrite"
                )
        atomic_write_bytes(destination, rendered)
        return destination

    def write_metric_exports(
        self,
        experiment: ExperimentName,
        artifact_id: ArtifactIdentifier,
    ) -> tuple[Path, ...]:
        metric = self.metric_record(artifact_id)
        if metric is None:
            return ()
        workspace = results_workspace(self._layout, experiment)
        serialized = cast(StableJsonPayload, metric.model_dump(mode="json"))
        summary = (
            workspace
            / ReportingPathSegment.METRICS
            / _experiment_metric_summary_directory()
            / ReportingPathSegment.SUMMARY_JSON
        )
        atomic_write_json(summary, serialized)
        columns, row = _metric_tabular_values(metric)
        csv_path = (
            workspace
            / ReportingPathSegment.TABLES
            / _experiment_supplementary_table_directory()
            / ReportingPathSegment.METRIC_RECORDS_CSV
        )
        tex_path = (
            workspace
            / ReportingPathSegment.TABLES
            / _experiment_supplementary_table_directory()
            / ReportingPathSegment.METRIC_RECORDS_TEX
        )
        atomic_write_bytes(csv_path, _csv_bytes(columns, (row,)))
        atomic_write_bytes(tex_path, _tex_bytes(columns, (row,)))
        figure_paths = self.write_metric_figure(experiment, artifact_id, metric)
        return (summary, csv_path, tex_path, *figure_paths)

    def write_metric_figure(
        self,
        experiment: ExperimentName,
        artifact_id: ArtifactIdentifier,
        metric: MetricRecord,
    ) -> tuple[Path, Path]:
        self._store.resolve(artifact_id)
        if metric.metric_value is None:
            raise EvidenceExportError("metric figure requires a valid finite metric value")
        destination = (
            results_workspace(self._layout, experiment)
            / ReportingPathSegment.FIGURES
            / _experiment_main_figure_directory()
        )
        label = f"{metric.metric_name.value}: {metric.metric_value:g} {metric.metric_unit}"
        svg_path = destination / ReportingPathSegment.METRIC_VALUE_SVG
        pdf_path = destination / ReportingPathSegment.METRIC_VALUE_PDF
        atomic_write_bytes(svg_path, _metric_svg_bytes(label, metric.metric_value))
        atomic_write_bytes(pdf_path, _metric_pdf_bytes(label, metric.metric_value))
        return (svg_path, pdf_path)

    def write_project_summary(
        self,
        manifests: tuple[ReusableArtifactManifest, ...],
        metrics: tuple[MetricRecord, ...],
    ) -> tuple[Path, ...]:
        if not manifests:
            return ()
        summary = self._layout.project_summary
        manifest_rows = tuple(
            (
                manifest.artifact_id.value,
                str(manifest.semantic_producer_coordinates),
                manifest.producer_stage.value,
                str(manifest.dependency_fingerprint_sha256),
            )
            for manifest in manifests
        )
        experiments = (
            summary
            / ReportingPathSegment.TABLES
            / _project_main_table_directory()
            / ReportingPathSegment.EXPERIMENTS_CSV
        )
        atomic_write_bytes(
            experiments,
            _csv_bytes(
                (
                    ProjectSummaryColumn.ARTIFACT_ID,
                    ProjectSummaryColumn.SEMANTIC_PRODUCER_COORDINATES,
                    ProjectSummaryColumn.PRODUCER_STAGE,
                    ProjectSummaryColumn.DEPENDENCY_FINGERPRINT_SHA256,
                ),
                manifest_rows,
            ),
        )
        metric_tabular_values = tuple(_metric_tabular_values(metric) for metric in metrics)
        metric_rows = tuple(row for _, row in metric_tabular_values)
        metric_columns = (
            metric_tabular_values[0][0] if metric_tabular_values else ("metric_record",)
        )
        evidence_summary = (
            summary
            / ReportingPathSegment.TABLES
            / _project_main_table_directory()
            / ReportingPathSegment.EVIDENCE_SUMMARY_CSV
        )
        atomic_write_bytes(evidence_summary, _csv_bytes(metric_columns, metric_rows))
        metrics_summary = (
            summary
            / ReportingPathSegment.METRICS
            / _project_metric_summary_directory()
            / ReportingPathSegment.SUMMARY_JSON
        )
        atomic_write_json(
            metrics_summary,
            cast(
                StableJsonPayload,
                OrderedDict(
                    metric_records=tuple(metric.model_dump(mode="json") for metric in metrics)
                ),
            ),
        )
        configuration = (
            summary
            / ReportingPathSegment.REPRODUCIBILITY
            / _project_configuration_reproducibility_directory()
            / ReportingPathSegment.SCIENTIFIC_CONFIGURATION_JSON
        )
        atomic_write_json(
            configuration,
            cast(
                StableJsonPayload,
                OrderedDict(
                    configuration_sha256=tuple(
                        manifest.applicable_configuration_sha256 for manifest in manifests
                    )
                ),
            ),
        )
        execution = (
            summary
            / ReportingPathSegment.REPRODUCIBILITY
            / _project_execution_reproducibility_directory()
            / ReportingPathSegment.EXECUTION_JSON
        )
        atomic_write_json(
            execution,
            cast(
                StableJsonPayload,
                OrderedDict(
                    completed_artifact_ids=tuple(manifest.artifact_id for manifest in manifests),
                    dependency_fingerprints=tuple(
                        manifest.dependency_fingerprint_sha256 for manifest in manifests
                    ),
                ),
            ),
        )
        return (experiments, evidence_summary, metrics_summary, configuration, execution)

    def write_project_evidence_table(
        self,
        table: EvidenceTable,
        name: ReportArtifactName,
    ) -> Path:
        destination = (
            self._layout.project_summary
            / ReportingPathSegment.TABLES
            / _project_main_table_directory()
            / f"{name}{ReportingPathSegment.TABLE_SUFFIX.value}"
        )
        atomic_write_json(destination, table.payload())
        return destination

    def write_project_evidence_figure(
        self,
        figure: EvidenceFigure,
        name: ReportArtifactName,
    ) -> Path:
        destination = (
            self._layout.project_summary
            / ReportingPathSegment.FIGURES
            / _project_main_figure_directory()
            / f"{name}{ReportingPathSegment.FIGURE_SUFFIX.value}"
        )
        atomic_write_bytes(destination, _evidence_figure_svg_bytes(figure))
        return destination

    def write_figure(
        self,
        experiment: ExperimentName,
        artifact_id: ArtifactIdentifier,
        figure: EvidenceFigure,
        name: ReportArtifactName,
    ) -> Path:
        self._store.resolve(artifact_id)
        destination = (
            results_workspace(self._layout, experiment)
            / f"{name}{ReportingPathSegment.FIGURE_SUFFIX.value}"
        )
        atomic_write_bytes(destination, _evidence_figure_svg_bytes(figure))
        return destination

    def metric_record(self, artifact_id: ArtifactIdentifier) -> MetricRecord | None:
        manifest = self._store.resolve(artifact_id)
        if len(manifest.payload_paths) != 1:
            raise EvidenceExportError("metric export requires exactly one verified payload")
        payload_path = Path(manifest.payload_paths[0])
        if not payload_path.is_file():
            raise EvidenceExportError(f"verified payload is missing: {payload_path}")
        try:
            payload = MetricArtifactPayload.model_validate_json(
                payload_path.read_text(encoding="utf-8")
            )
        except ValueError as error:
            raise EvidenceExportError(f"verified metric payload is invalid: {error}") from error
        return payload.metric_record


def _tabular_frame(columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def _csv_bytes(columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> bytes:
    return _tabular_frame(columns, rows).to_csv(index=False, lineterminator="\n").encode("utf-8")


def _tex_bytes(columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> bytes:
    return _tabular_frame(columns, rows).to_latex(index=False, escape=True).encode("utf-8")


def _experiment_metric_summary_directory() -> str:
    directories = active_config().runtime.artifact_layout.reporting_output_directories
    return directories.manuscript_metric_summary


def _project_metric_summary_directory() -> str:
    return (
        active_config().runtime.artifact_layout.reporting_output_directories.project_metric_summary
    )


def _experiment_supplementary_table_directory() -> str:
    directories = active_config().runtime.artifact_layout.reporting_output_directories
    return directories.manuscript_supplementary_table


def _experiment_main_figure_directory() -> str:
    return (
        active_config().runtime.artifact_layout.reporting_output_directories.manuscript_main_figure
    )


def _project_main_table_directory() -> str:
    return active_config().runtime.artifact_layout.reporting_output_directories.project_main_table


def _project_main_figure_directory() -> str:
    return active_config().runtime.artifact_layout.reporting_output_directories.project_main_figure


def _project_configuration_reproducibility_directory() -> str:
    directories = active_config().runtime.artifact_layout.reporting_output_directories
    return directories.project_configuration_reproducibility


def _project_execution_reproducibility_directory() -> str:
    directories = active_config().runtime.artifact_layout.reporting_output_directories
    return directories.project_execution_reproducibility


def _metric_svg_bytes(label: str, value: float) -> bytes:
    figure = Figure(
        figsize=(REPORT_FIGURE_WIDTH / 72, REPORT_FIGURE_HEIGHT / 72),
        dpi=72,
        layout="constrained",
    )
    axes = figure.subplots()
    axes.barh(
        ("metric",), (min(REPORT_METRIC_SCALE, max(0.0, abs(value))),), color=REPORT_BAR_COLOR
    )
    axes.set_xlim(0.0, REPORT_METRIC_SCALE)
    axes.set_title(label)
    axes.set_yticks(())
    buffer = BytesIO()
    FigureCanvasSVG(figure).print_svg(buffer)
    return buffer.getvalue()


def _evidence_figure_svg_bytes(figure: EvidenceFigure) -> bytes:
    plot = Figure(
        figsize=(REPORT_FIGURE_WIDTH / 72, REPORT_FIGURE_HEIGHT / 72),
        dpi=72,
        layout="constrained",
    )
    axes = plot.subplots()
    for series in figure.series:
        axes.plot(series.x, series.y, marker="o", label=str(series.name))
    axes.set_xlabel(str(figure.x_label))
    axes.set_ylabel(str(figure.y_label))
    if len(figure.series) > 1:
        axes.legend()
    buffer = BytesIO()
    FigureCanvasSVG(plot).print_svg(buffer)
    return buffer.getvalue()


def _metric_pdf_bytes(label: str, value: float) -> bytes:
    bar_width = min(
        float(REPORT_FIGURE_WIDTH - 80),
        max(0.0, abs(value) * (REPORT_FIGURE_WIDTH - 80) / REPORT_METRIC_SCALE),
    )
    buffer = BytesIO()
    document = Canvas(
        buffer,
        pagesize=(REPORT_FIGURE_WIDTH, REPORT_FIGURE_HEIGHT),
        pageCompression=1,
        invariant=1,
    )
    document.setFont("Helvetica", 14)
    document.drawString(40, 150, label)
    document.setFillColor(REPORT_BAR_COLOR)
    document.rect(40, 70, bar_width, 40, fill=1, stroke=0)
    document.setStrokeColor("black")
    document.line(40, 70, REPORT_FIGURE_WIDTH - 40, 70)
    document.save()
    return buffer.getvalue()
