from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

import pandas as pd
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, JsonValue
from reportlab.pdfgen.canvas import Canvas

from fedorbit.analysis.records import MetricRecord, PairedComparisonRecord
from fedorbit.config.loading import active_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.infrastructure.execution import ArtifactStore
from fedorbit.infrastructure.manifests import DatasetManifest, ReusableArtifactManifest
from fedorbit.infrastructure.storage import atomic_write_bytes, atomic_write_json
from fedorbit.infrastructure.workspace import WorkspaceLayout, results_workspace
from fedorbit.types import (
    ArtifactIdentifier,
    ClientRole,
    ExperimentName,
    FedorbitConfigSection,
    MetricId,
    ProjectSummaryColumn,
    ReportArtifactName,
    ReportAxisLabel,
    ReportColumnName,
    ReportColumns,
    ReportCoordinates,
    ReportingPathSegment,
    ReportSeriesName,
    RiskReductionColumn,
    StableJsonPayload,
    TransferMethod,
    stable_json,
)


class FigureError(ValueError):
    pass


REPORT_FIGURE_WIDTH = 480
REPORT_FIGURE_HEIGHT = 180
REPORT_METRIC_SCALE = 4.0
REPORT_BAR_COLOR = "#2a6fbb"


@dataclass(frozen=True, slots=True)
class FigureSeries:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete
    name: ReportSeriesName
    x: ReportCoordinates
    y: ReportCoordinates

    def __post_init__(self) -> None:
        if not self.name:
            raise FigureError("figure series name must be non-empty")
        if len(self.x) != len(self.y):
            raise FigureError("figure series coordinates differ in length")


@dataclass(frozen=True, slots=True)
class EvidenceFigurePayload:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete
    x_label: ReportAxisLabel
    y_label: ReportAxisLabel
    series: tuple[FigureSeries, ...]


@dataclass(frozen=True, slots=True)
class EvidenceFigure:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete
    x_label: ReportAxisLabel
    y_label: ReportAxisLabel
    series: tuple[FigureSeries, ...]

    def __post_init__(self) -> None:
        if not self.x_label or not self.y_label:
            raise FigureError("figure axes must be named")
        if not self.series:
            raise FigureError("evidence figure requires at least one series")

    def payload(
        self,
    ) -> EvidenceFigurePayload:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete
        return EvidenceFigurePayload(self.x_label, self.y_label, self.series)


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


def _report_columns(columns: Sequence[str]) -> ReportColumns:
    return tuple(ReportColumnName(column) for column in columns)


def _metric_tabular_values(metric: MetricRecord) -> tuple[tuple[str, ...], tuple[str, ...]]:
    serialized = metric.model_dump(mode="json")
    return tuple(serialized), tuple(str(value) for value in serialized.values())


class VerifiedEvidenceWriter:
    def __init__(self, store: ArtifactStore, layout: WorkspaceLayout) -> None:
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

    def write_table(  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete
        self,
        experiment: ExperimentName,
        artifact_id: ArtifactIdentifier,
        table: EvidenceTable,
        name: ReportArtifactName,
    ) -> Path:
        self._store.resolve(artifact_id)
        destination = (
            results_workspace(self._layout, experiment)
            / f"{name}{ReportingPathSegment.TABLE_SUFFIX.value}"
        )
        atomic_write_json(destination, table.payload())
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
        atomic_write_json(destination, figure.payload())
        return destination

    def write_figure(  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete #TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
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
        atomic_write_json(destination, figure.payload())
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


def _leaf_scalars(prefix: str, value: JsonValue) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            rows.extend(_leaf_scalars(f"{prefix}.{key}", item))
        return tuple(rows)
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_leaf_scalars(f"{prefix}[{index}]", item))
        return tuple(rows)
    return ((prefix, str(value)),)


def numerical_constants_and_seeds_table(
    config: FedorbitConfig | None = None,
) -> EvidenceTable:
    resolved = config if config is not None else active_config()
    dumped = cast(Mapping[str, JsonValue], resolved.model_dump(mode="json"))
    rows = tuple(
        sorted(
            _leaf_scalars(
                FedorbitConfigSection.SCIENTIFIC,
                dumped.get(FedorbitConfigSection.SCIENTIFIC, OrderedDict()),
            )
        )
        + sorted(
            _leaf_scalars(
                FedorbitConfigSection.SOLVERS,
                dumped.get(FedorbitConfigSection.SOLVERS, OrderedDict()),
            )
        )
    )
    return EvidenceTable(
        columns=_report_columns(("configuration_path", "value")),
        rows=tuple((path, value) for path, value in rows),
    )


def experiment_matrix_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "experiment",
            "classification",
            "datasets_or_pairs",
            "methods",
            "registered_seeds",
            "conditions",
            "derived_planned_cells",
            "prerequisites",
            "evidence_relationship",
        ),
        rows,
    )


def dataset_and_client_protocol_table(
    manifests: Sequence[DatasetManifest],
    modality_by_dataset: Mapping[str, str],
    role_by_dataset: Mapping[str, ClientRole],
    excluded_class_counts: Mapping[str, int],
) -> EvidenceTable:
    columns = (
        "dataset_component",
        "modality",
        "observed_raw_rows",
        "retained_rows",
        "timestamp_range",
        "local_prediction_classes",
        "feature_count",
        "transfer_candidates",
        "exclusions",
        "scientific_role",
        "raw_manifest_hash",
    )
    rows = tuple(
        (
            manifest.component,
            modality_by_dataset.get(manifest.dataset.value, ""),
            sum(manifest.raw_counts.values()),
            sum(manifest.local_class_counts.values()),
            f"{manifest.timestamp_range[0]}..{manifest.timestamp_range[1]}",
            len(manifest.local_class_counts),
            len(manifest.adapter_feature_order),
            len(manifest.transfer_candidate_counts),
            excluded_class_counts.get(manifest.dataset.value, 0),
            role_by_dataset.get(manifest.dataset.value, ClientRole.PRIMARY).value,
            manifest.raw_sha256,
        )
        for manifest in manifests
    )
    return EvidenceTable(columns=_report_columns(columns), rows=rows)


def _metric_value(
    records: Sequence[MetricRecord],
    pair: str,
    method: TransferMethod,
    metric_name: MetricId,
) -> float | None:
    matches = [
        record
        for record in records
        if record.pair == pair
        and record.method == method
        and record.metric_name == metric_name
        and record.valid
    ]
    if not matches:
        return None
    return sum(record.metric_value for record in matches if record.metric_value is not None) / len(
        matches
    )


def primary_strict_transfer_results_table(
    metric_records: Sequence[MetricRecord],
    comparison_records: Sequence[PairedComparisonRecord],
) -> EvidenceTable:
    columns = (
        "pair",
        "method",
        "valid_seeds",
        "test_macro_ce",
        "macro_f1",
        "balanced_accuracy",
        "gain_vs_local",
        "bca_ci_low",
        "bca_ci_high",
        "raw_p",
        "holm_p",
        "strict_validity",
        "confirmation_coverage",
    )
    pairs = sorted({record.pair for record in metric_records})
    method_order = (
        TransferMethod.LOCAL_ONLY,
        TransferMethod.LOCAL_SIR,
        TransferMethod.MATCHED_RESOURCE_RECTANGULAR,
        TransferMethod.POINT_CORRESPONDENCE_COMMITMENT,
        TransferMethod.GENERIC_EXACT_QAP,
        TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER,
        TransferMethod.EXACT_MAP_ORACLE,
    )
    rows: list[tuple[TableScalar, ...]] = []
    for pair in pairs:
        for method in method_order:
            comparison = next(
                (
                    record
                    for record in comparison_records
                    if record.pair == pair and record.method_a == method
                ),
                None,
            )
            rows.append(
                (
                    pair,
                    method.value,
                    comparison.paired_seed_count if comparison is not None else None,
                    _metric_value(metric_records, pair, method, MetricId.MACRO_CROSS_ENTROPY),
                    _metric_value(metric_records, pair, method, MetricId.MACRO_F1),
                    _metric_value(metric_records, pair, method, MetricId.BALANCED_ACCURACY),
                    comparison.mean_difference if comparison is not None else None,
                    comparison.bca_ci_low if comparison is not None else None,
                    comparison.bca_ci_high if comparison is not None else None,
                    comparison.raw_p if comparison is not None else None,
                    comparison.holm_p if comparison is not None else None,
                    comparison.decision.value if comparison is not None else None,
                    _metric_value(metric_records, pair, method, MetricId.COVERAGE_CONFIRM),
                )
            )
    return EvidenceTable(columns=_report_columns(columns), rows=tuple(rows))


def _rows_table(
    columns: tuple[str, ...], rows: Sequence[Mapping[str, TableScalar]]
) -> EvidenceTable:
    return EvidenceTable(
        columns=_report_columns(columns),
        rows=tuple(tuple(row[column] for column in columns) for row in rows),
    )


def transfer_ontology_and_null_padding_table(  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "coarse_group",
            "source_real_or_null",
            "target_real_or_null",
            "support_counts",
            "action_eligibility",
            "null_reason",
        ),
        rows,
    )


def model_and_training_protocol_table(  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "model",
            "architecture",
            "normalization",
            "activation",
            "initialization",
            "optimizer",
            "batch",
            "selected_learning_rate",
            "selected_weight_decay",
            "selected_dropout",
            "stopping_rule",
        ),
        rows,
    )


def information_resource_matrix_table(  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "method",
            "target_raw_data",
            "anonymous_source_nodes",
            "coarse_groups",
            "source_response",
            "target_local_response",
            "fine_names",
            "exact_map",
            "confirmation",
            "predecision_test_access",
            "strict_compatibility",
        ),
        rows,
    )


def coupling_mechanism_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "condition_or_pair",
            "valid_units",
            "fixed_action_gap",
            "robust_coupling_gap",
            "fraction_above_materiality",
            "ci",
            "holm_p",
            "coupling_destruction_retained_gain_fraction",
        ),
        rows,
    )


def exact_solver_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "k",
            "block_pattern",
            "support",
            "truth_availability",
            "exact_mismatches",
            "maximum_absolute_error",
            "runtime_median",
            "runtime_p95",
            "qap_runtime",
            "dense_runtime",
            "timeouts",
            "memory",
            "active_images",
            "lap_calls",
        ),
        rows,
    )


def ablation_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "ablation",
            "pair",
            "realized_gain",
            "difference_vs_full",
            "equivalence",
            "retained_gain",
            "confirmation_safety",
        ),
        rows,
    )


def sparsity_and_dense_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "support_or_dense_condition",
            "pair",
            "realized_gain",
            "certified_value",
            "runtime",
            "memory",
            "confirmation_coverage",
            "dense_minus_sparse_difference",
        ),
        rows,
    )


def confirmation_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "pair",
            "proposals",
            "accepted",
            "harmful_accepted_rate",
            "useful_accepted_rate",
            "beneficial_rejected_rate",
            "coverage",
            "no_confirm_harmful_rate",
            RiskReductionColumn.ABSOLUTE_RISK_REDUCTION,
            RiskReductionColumn.RELATIVE_RISK_REDUCTION,
            "ci",
            "p",
        ),
        rows,
    )


def generalization_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "pair",
            "method",
            "valid_seeds",
            "test_macro_ce",
            "macro_f1",
            "balanced_accuracy",
            "gain_vs_local",
            "bca_ci_low",
            "bca_ci_high",
            "raw_p",
            "holm_p",
            "strict_validity",
            "confirmation_coverage",
            "is_secondary_pair",
        ),
        rows,
    )


def failure_boundary_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete
    return _rows_table(
        (
            "boundary_dimension",
            "setting",
            "pair",
            "method",
            "certified_value",
            "realized_gain",
            "abstention",
            "null_node_count",
            "confirmation_coverage",
            "state",
        ),
        rows,
    )


def scalability_results_table(
    rows: Sequence[Mapping[str, TableScalar]],
) -> EvidenceTable:
    return _rows_table(
        (
            "k",
            "block",
            "support",
            "method",
            "n_s",
            "lap_calls",
            "cuts",
            "runtime_median",
            "runtime_p95",
            "rss",
            "cuda_memory",
            "timeout",
            "exactness_status",
        ),
        rows,
    )


def _figure(
    x_label: str, y_label: str, series: Sequence[FigureSeries]
) -> (
    EvidenceFigure
):  # TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
    return EvidenceFigure(
        x_label=ReportAxisLabel(x_label), y_label=ReportAxisLabel(y_label), series=tuple(series)
    )


def real_transfer_gain_forest_plot(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("paired mean relative macro-CE gain vs local", "primary directed pair", series)


def baseline_paired_difference_plot(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete #TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
    return _figure("primary directed pair", "seed-level paired difference", series)


def coupling_gap_phase_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete #TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
    return _figure("coupling factor combination", "predicted structural zero/strict state", series)


def predicted_vs_realized_transfer_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete #TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
    return _figure("certified robust predicted value", "TEST relative macro-CE gain", series)


def sparsity_utility_efficiency_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete #TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
    return _figure("runtime", "realized gain", series)


def confirmation_safety_coverage_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete #TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
    return _figure("confirmation coverage", "harmful accepted rate", series)


def semantic_sufficiency_frontier_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete #TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
    return _figure("log|orbit|", "realized gain", series)


def failure_boundary_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete #TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
    return _figure("boundary setting", "certified value / realized gain", series)


def scalability_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete #TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
    return _figure(
        "N_S * sum(n_g^3) (log scale)",
        "runtime (log scale)",
        series,
    )


def map_value_bound_figure(
    series: Sequence[FigureSeries],
) -> EvidenceFigure:  # TODO: roadmap §23 report table/figure surface (evidence-onboarding): wire into the report flow; do not delete #TODO: use matplotlib to emit canonical SVG figures instead of hand-built SVG markup
    return _figure("orbit-radius bound", "exact map action value", series)
