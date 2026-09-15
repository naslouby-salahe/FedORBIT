from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol, cast

import matplotlib
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from fedorbit.analysis.records import MetricRecord
from fedorbit.config.loading import active_config
from fedorbit.config.models import ReportingPrecisionConfig
from fedorbit.infrastructure.environment import environment_snapshot
from fedorbit.infrastructure.manifests import ReusableArtifactManifest
from fedorbit.infrastructure.runtime import (
    CodeRevision,
    GitRevision,
    ReproducibilityIdentity,
    build_reproducibility_identity,
    deterministic_backend_state,
)
from fedorbit.infrastructure.storage import atomic_write_bytes, atomic_write_json
from fedorbit.infrastructure.workspace import WorkspaceLayout, results_workspace
from fedorbit.types import (
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactStage,
    ConfidenceIntervalText,
    ExperimentName,
    FieldDescription,
    JsonValue,
    MetricId,
    MetricUnit,
    ReferenceLineCoordinate,
    ReportArtifactName,
    ReportAxisLabel,
    ReportColumnName,
    ReportColumns,
    ReportCoordinates,
    ReportingPathSegment,
    ReportSeriesName,
    Sha256Digest,
    StableJsonPayload,
    stable_json,
)


class FigureError(ValueError):
    pass


REPORT_FIGURE_WIDTH = 480
REPORT_FIGURE_HEIGHT = 180
REPORT_METRIC_SCALE = 4.0
REPORT_BAR_COLOR = "#2a6fbb"
SVG_HASH_SALT = "fedorbit"
METRIC_VALUE_FIELD = "metric_value"
TABLE_EXPORT_SUFFIX = ".csv"
UNAVAILABLE_CELL_TEXT = "NA"
EXPORT_LEDGER_KEY = "report_exports"
SVG_EXPORT_DESCRIPTION_PREFIX = "dependency-fingerprint-sha256"
NONDETERMINISTIC_SVG_METADATA: Mapping[str, None] = OrderedDict((("Date", None), ("Creator", None)))


def export_dependency_fingerprint(
    name: ReportArtifactName,
    dependency_artifact_ids: ArtifactIdentifiers = (),
) -> Sha256Digest:
    payload = stable_json(
        cast(
            StableJsonPayload,
            OrderedDict(
                export=name.value,
                producer_stage=ArtifactStage.REPORTING.value,
                dependency_artifact_ids=tuple(
                    identifier.value for identifier in dependency_artifact_ids
                ),
            ),
        )
    )
    return Sha256Digest(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def _reporting_precision() -> ReportingPrecisionConfig:
    return active_config().reporting.precision


def format_scientific_metric(value: float) -> str:
    return f"{value:.{_reporting_precision().scientific_metric_decimals}f}"


def format_macro_f1(value: float) -> str:
    return f"{value:.{_reporting_precision().macro_f1_decimals}f}"


def format_balanced_accuracy(value: float) -> str:
    return f"{value:.{_reporting_precision().balanced_accuracy_decimals}f}"


def format_p_value(value: float) -> str:
    precision = _reporting_precision()
    if value < precision.p_value_less_than_threshold:
        return f"<{precision.p_value_less_than_threshold:.{precision.p_value_decimals}f}"
    return f"{value:.{precision.p_value_decimals}f}"


def format_runtime_seconds(value: float) -> str:
    return f"{value:.{_reporting_precision().runtime_seconds_decimals}f}"


def format_memory_mib(value: float) -> str:
    return f"{value:.{_reporting_precision().memory_decimals}f}"


def format_integer(value: float) -> str:
    return str(round(value))


def format_interval_estimate(
    estimate: float,
    low: float | None,
    high: float | None,
) -> ConfidenceIntervalText:
    rendered_estimate = format_scientific_metric(estimate)
    if low is None or high is None:
        return ConfidenceIntervalText(rendered_estimate)
    rendered_low = format_scientific_metric(low)
    rendered_high = format_scientific_metric(high)
    return ConfidenceIntervalText(f"{rendered_estimate} [{rendered_low}, {rendered_high}]")


def format_metric_value(metric: MetricRecord) -> str:
    value = metric.metric_value
    if value is None:
        return ""
    if metric.metric_unit in (MetricUnit.BOOLEAN, MetricUnit.COUNT, MetricUnit.BYTES):
        return format_integer(value)
    if metric.metric_unit == MetricUnit.MEBIBYTES:
        return format_memory_mib(value)
    if metric.metric_unit == MetricUnit.SECONDS:
        return format_runtime_seconds(value)
    if metric.metric_name == MetricId.MACRO_F1:
        return format_macro_f1(value)
    if metric.metric_name == MetricId.BALANCED_ACCURACY:
        return format_balanced_accuracy(value)
    return format_scientific_metric(value)


@dataclass(frozen=True, slots=True)
class FigureSeries:
    name: ReportSeriesName
    x: ReportCoordinates
    y: ReportCoordinates
    y_low: ReportCoordinates | None = None
    y_high: ReportCoordinates | None = None
    x_low: ReportCoordinates | None = None
    x_high: ReportCoordinates | None = None
    marker_sizes: ReportCoordinates | None = None
    arrow_x: ReportCoordinates | None = None
    arrow_y: ReportCoordinates | None = None
    panel: ReportSeriesName | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise FigureError("figure series name must be non-empty")
        if len(self.x) != len(self.y):
            raise FigureError("figure series coordinates differ in length")
        if self.y_low is not None and len(self.y_low) != len(self.y):
            raise FigureError("figure interval bounds differ in length")
        if self.y_high is not None and len(self.y_high) != len(self.y):
            raise FigureError("figure interval bounds differ in length")
        if self.x_low is not None and len(self.x_low) != len(self.x):
            raise FigureError("figure interval bounds differ in length")
        if self.x_high is not None and len(self.x_high) != len(self.x):
            raise FigureError("figure interval bounds differ in length")
        if self.marker_sizes is not None and len(self.marker_sizes) != len(self.x):
            raise FigureError("figure marker sizes differ in length")
        if self.arrow_x is not None and len(self.arrow_x) != len(self.x):
            raise FigureError("figure arrow coordinates differ in length")
        if self.arrow_y is not None and len(self.arrow_y) != len(self.y):
            raise FigureError("figure arrow coordinates differ in length")


@dataclass(frozen=True, slots=True)
class EvidenceFigure:
    x_label: ReportAxisLabel
    y_label: ReportAxisLabel
    series: tuple[FigureSeries, ...]
    vertical_reference_lines: tuple[ReferenceLineCoordinate, ...] = ()
    horizontal_reference_lines: tuple[ReferenceLineCoordinate, ...] = ()
    log_x: bool = False
    log_y: bool = False
    draw_unit_diagonal: bool = False
    separate_panels: bool = False
    y_tick_labels: tuple[ReportSeriesName, ...] | None = None

    def __post_init__(self) -> None:
        if not self.x_label or not self.y_label:
            raise FigureError("figure axes must be named")
        if not self.series:
            raise FigureError("evidence figure requires at least one series")


class TableError(ValueError):
    pass


TableScalar = str | int | float | bool | None


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


class EvidenceExportError(ValueError):
    pass


class MetricArtifactPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric_record: MetricRecord | None = None


class ArtifactResolver(Protocol):
    def resolve(self, artifact_id: ArtifactIdentifier) -> ReusableArtifactManifest: ...


def _metric_tabular_values(
    metric: MetricRecord,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    serialized = metric.model_dump(mode="json")
    return tuple(serialized), tuple(
        format_metric_value(metric) if column == METRIC_VALUE_FIELD else str(value)
        for column, value in serialized.items()
    )


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
        atomic_write_bytes(csv_path, _csv_bytes(columns, (row,)))
        figure_paths = self.write_metric_figure(experiment, artifact_id, metric)
        return (summary, csv_path, *figure_paths)

    def write_metric_figure(
        self,
        experiment: ExperimentName,
        artifact_id: ArtifactIdentifier,
        metric: MetricRecord,
    ) -> tuple[Path, ...]:
        self._store.resolve(artifact_id)
        if metric.metric_value is None:
            raise EvidenceExportError("metric figure requires a valid finite metric value")
        destination = (
            results_workspace(self._layout, experiment)
            / ReportingPathSegment.FIGURES
            / _experiment_main_figure_directory()
        )
        label = f"{metric.metric_name.value}: {format_metric_value(metric)} {metric.metric_unit}"
        svg_path = destination / ReportingPathSegment.METRIC_VALUE_SVG
        atomic_write_bytes(svg_path, _metric_svg_bytes(label, metric.metric_value))
        return (svg_path,)

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
                    ReportColumnName.ARTIFACT_ID,
                    ReportColumnName.SEMANTIC_PRODUCER_COORDINATES,
                    ReportColumnName.PRODUCER_STAGE,
                    ReportColumnName.DEPENDENCY_FINGERPRINT_SHA256,
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
        identity = build_reproducibility_identity(environment_snapshot())
        atomic_write_json(
            execution,
            cast(
                StableJsonPayload,
                OrderedDict(
                    completed_artifact_ids=tuple(manifest.artifact_id for manifest in manifests),
                    dependency_fingerprints=tuple(
                        manifest.dependency_fingerprint_sha256 for manifest in manifests
                    ),
                    reproducibility_identity=_identity_payload(identity),
                    deterministic_backend=deterministic_backend_state(),
                ),
            ),
        )
        return (experiments, evidence_summary, metrics_summary, configuration, execution)

    def write_project_evidence_table(
        self,
        table: EvidenceTable,
        name: ReportArtifactName,
        dependency_artifact_ids: ArtifactIdentifiers = (),
    ) -> Path:
        destination = (
            self._layout.project_summary
            / ReportingPathSegment.TABLES
            / _project_main_table_directory()
            / f"{name}{TABLE_EXPORT_SUFFIX}"
        )
        fingerprint = export_dependency_fingerprint(name, dependency_artifact_ids)
        promoted = self._promote_export(destination, _table_csv_bytes(table))
        self._record_export_fingerprint(name, fingerprint)
        return promoted

    def write_project_evidence_figure(
        self,
        figure: EvidenceFigure,
        name: ReportArtifactName,
        dependency_artifact_ids: ArtifactIdentifiers = (),
    ) -> Path:
        destination = (
            self._layout.project_summary
            / ReportingPathSegment.FIGURES
            / _project_main_figure_directory()
            / f"{name}{ReportingPathSegment.FIGURE_SUFFIX.value}"
        )
        fingerprint = export_dependency_fingerprint(name, dependency_artifact_ids)
        rendered = figure_svg_bytes(
            figure,
            FieldDescription(f"{SVG_EXPORT_DESCRIPTION_PREFIX}: {fingerprint}"),
        )
        promoted = self._promote_export(destination, rendered)
        self._record_export_fingerprint(name, fingerprint)
        return promoted

    def _promote_export(self, destination: Path, rendered: bytes) -> Path:
        if destination.is_file() and destination.read_bytes() == rendered:
            return destination
        atomic_write_bytes(destination, rendered)
        return destination

    def _record_export_fingerprint(
        self,
        name: ReportArtifactName,
        fingerprint: Sha256Digest,
    ) -> None:
        ledger = (
            self._layout.project_summary
            / ReportingPathSegment.REPRODUCIBILITY
            / _project_execution_reproducibility_directory()
            / ReportingPathSegment.EXECUTION_JSON
        )
        payload: OrderedDict[str, JsonValue] = OrderedDict()
        if ledger.is_file():
            parsed = json.loads(ledger.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                payload.update(cast(Mapping[str, JsonValue], parsed))
        recorded = payload.get(EXPORT_LEDGER_KEY)
        exports: OrderedDict[str, JsonValue] = OrderedDict()
        if isinstance(recorded, Mapping):
            exports.update(cast(Mapping[str, JsonValue], recorded))
        exports[name.value] = fingerprint
        payload[EXPORT_LEDGER_KEY] = exports
        atomic_write_json(ledger, cast(StableJsonPayload, payload))

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


def _table_csv_bytes(table: EvidenceTable) -> bytes:
    rendered = tuple(
        tuple(
            UNAVAILABLE_CELL_TEXT if cell is None else cell if isinstance(cell, str) else str(cell)
            for cell in row
        )
        for row in table.rows
    )
    return _csv_bytes(table.columns, rendered)


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


def _render_svg_bytes(figure: Figure, description: FieldDescription | None = None) -> bytes:
    metadata: OrderedDict[str, str | None] = OrderedDict(NONDETERMINISTIC_SVG_METADATA)
    if description is not None:
        metadata["Description"] = description
    buffer = BytesIO()
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASH_SALT}):
        FigureCanvasSVG(figure).print_svg(buffer, metadata=metadata)
    return buffer.getvalue()


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
    return _render_svg_bytes(figure)


def _draw_series(axes: Axes, series: FigureSeries) -> None:
    if series.marker_sizes is not None:
        axes.scatter(series.x, series.y, s=series.marker_sizes, label=str(series.name))
    else:
        axes.plot(series.x, series.y, marker="o", label=str(series.name))
    if series.y_low is not None and series.y_high is not None:
        axes.vlines(series.x, series.y_low, series.y_high)
    if series.x_low is not None and series.x_high is not None:
        axes.errorbar(
            series.x,
            series.y,
            xerr=(
                tuple(mid - low for mid, low in zip(series.x, series.x_low, strict=True)),
                tuple(high - mid for mid, high in zip(series.x, series.x_high, strict=True)),
            ),
            fmt="none",
        )
    if series.arrow_x is not None and series.arrow_y is not None:
        for start_x, start_y, end_x, end_y in zip(
            series.x, series.y, series.arrow_x, series.arrow_y, strict=True
        ):
            axes.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops=OrderedDict(arrowstyle="->", color="black"),
            )


def _style_axes(axes: Axes, figure: EvidenceFigure, series_group: tuple[FigureSeries, ...]) -> None:
    for x_value in figure.vertical_reference_lines:
        axes.axvline(x_value, color="black", linewidth=1.0)
    for y_value in figure.horizontal_reference_lines:
        axes.axhline(y_value, color="black", linewidth=1.0)
    if figure.draw_unit_diagonal:
        xs = [value for series in series_group for value in series.x]
        ys = [value for series in series_group for value in series.y]
        if xs and ys:
            low = min(min(xs), min(ys))
            high = max(max(xs), max(ys))
            axes.plot((low, high), (low, high), color="black", linewidth=1.0)
    if figure.log_x:
        axes.set_xscale("log")
    if figure.log_y:
        axes.set_yscale("log")
    axes.set_xlabel(str(figure.x_label))
    axes.set_ylabel(str(figure.y_label))
    if figure.y_tick_labels is not None:
        axes.set_yticks(range(len(figure.y_tick_labels)))
        axes.set_yticklabels([str(label) for label in figure.y_tick_labels])
    if len(series_group) > 1:
        axes.legend()


def _panel_groups(
    series: tuple[FigureSeries, ...],
) -> tuple[tuple[str, tuple[FigureSeries, ...]], ...]:
    grouped: OrderedDict[str, list[FigureSeries]] = OrderedDict()
    for item in series:
        key = str(item.panel) if item.panel is not None else str(item.name)
        grouped.setdefault(key, []).append(item)
    return tuple((key, tuple(items)) for key, items in grouped.items())


def figure_svg_bytes(
    figure: EvidenceFigure,
    export_description: FieldDescription | None = None,
) -> bytes:
    panels = _panel_groups(figure.series) if figure.separate_panels else ()
    panel_count = len(panels) if figure.separate_panels else 1
    plot = Figure(
        figsize=(REPORT_FIGURE_WIDTH / 72 * max(1, panel_count / 2), REPORT_FIGURE_HEIGHT / 72),
        dpi=72,
        layout="constrained",
    )
    if figure.separate_panels:
        axes_grid = plot.subplots(1, panel_count, squeeze=False)
        for index, (title, panel_series) in enumerate(panels):
            axes = axes_grid[0][index]
            for series in panel_series:
                _draw_series(axes, series)
            _style_axes(axes, figure, panel_series)
            axes.set_title(title)
    else:
        axes = plot.subplots()
        for series in figure.series:
            _draw_series(axes, series)
        _style_axes(axes, figure, figure.series)
    return _render_svg_bytes(plot, export_description)


def _identity_payload(identity: ReproducibilityIdentity) -> OrderedDict[str, JsonValue]:
    return OrderedDict(
        config_digest=identity.config_digest,
        seed_digest=identity.seed_digest,
        environment_fingerprint=identity.environment_fingerprint,
        code_revision=identity.code_revision.commit,
        statistical_identity_digest=identity.statistical_identity_digest,
    )


def recorded_execution_identity(layout: WorkspaceLayout) -> ReproducibilityIdentity | None:
    execution = (
        layout.project_summary
        / ReportingPathSegment.REPRODUCIBILITY
        / _project_execution_reproducibility_directory()
        / ReportingPathSegment.EXECUTION_JSON
    )
    if not execution.is_file():
        return None
    payload = cast(Mapping[str, JsonValue], json.loads(execution.read_text(encoding="utf-8")))
    recorded = payload.get("reproducibility_identity")
    if not isinstance(recorded, Mapping):
        return None
    return ReproducibilityIdentity(
        config_digest=Sha256Digest(str(recorded["config_digest"])),
        seed_digest=Sha256Digest(str(recorded["seed_digest"])),
        environment_fingerprint=Sha256Digest(str(recorded["environment_fingerprint"])),
        code_revision=CodeRevision(GitRevision(str(recorded["code_revision"]))),
        statistical_identity_digest=Sha256Digest(str(recorded["statistical_identity_digest"])),
    )
