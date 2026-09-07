from __future__ import annotations

import csv
import html
import io
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from fedorbit.analysis.records import MetricRecord, PairedComparisonRecord
from fedorbit.config.loading import active_config
from fedorbit.config.models import FedorbitConfig
from fedorbit.infrastructure.execution import ArtifactStore, atomic_write_bytes, atomic_write_json
from fedorbit.infrastructure.manifests import DatasetManifest, ReusableArtifactManifest
from fedorbit.infrastructure.workspace import WorkspaceLayout, results_workspace
from fedorbit.types import (
    ArtifactIdentifier,
    ClientRole,
    ExperimentName,
    MetricId,
    StableJsonPayload,
    TransferMethod,
    stable_json,
)


class FigureError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FigureSeries:
    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise FigureError("figure series name must be non-empty")
        if len(self.x) != len(self.y):
            raise FigureError("figure series coordinates differ in length")


@dataclass(frozen=True, slots=True)
class EvidenceFigurePayload:
    x_label: str
    y_label: str
    series: tuple[FigureSeries, ...]


@dataclass(frozen=True, slots=True)
class EvidenceFigure:
    x_label: str
    y_label: str
    series: tuple[FigureSeries, ...]

    def __post_init__(self) -> None:
        if not self.x_label or not self.y_label:
            raise FigureError("figure axes must be named")
        if not self.series:
            raise FigureError("evidence figure requires at least one series")

    def payload(self) -> EvidenceFigurePayload:
        return EvidenceFigurePayload(self.x_label, self.y_label, self.series)


class TableError(ValueError):
    pass


TableScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class EvidenceTablePayload:
    columns: tuple[str, ...]
    rows: tuple[tuple[TableScalar, ...], ...]


@dataclass(frozen=True, slots=True)
class EvidenceTable:
    columns: tuple[str, ...]
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
        destination = workspace / f"{experiment.value}.evidence.json"
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

    def write_table(
        self,
        experiment: ExperimentName,
        artifact_id: ArtifactIdentifier,
        table: EvidenceTable,
        name: str,
    ) -> Path:
        self._store.resolve(artifact_id)
        destination = results_workspace(self._layout, experiment) / f"{name}.table.json"
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
        summary = workspace / "metrics" / _experiment_metric_summary_directory() / "summary.json"
        atomic_write_json(summary, serialized)
        columns = tuple(metric.model_dump(mode="json").keys())
        row = tuple(str(value) for value in metric.model_dump(mode="json").values())
        csv_path = (
            workspace
            / "tables"
            / _experiment_supplementary_table_directory()
            / "metric_records.csv"
        )
        tex_path = (
            workspace
            / "tables"
            / _experiment_supplementary_table_directory()
            / "metric_records.tex"
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
            / "figures"
            / _experiment_main_figure_directory()
        )
        label = f"{metric.metric_name.value}: {metric.metric_value:g} {metric.metric_unit}"
        svg_path = destination / "metric_value.svg"
        pdf_path = destination / "metric_value.pdf"
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
                manifest.artifact_id,
                manifest.semantic_producer_coordinates,
                manifest.producer_stage.value,
                manifest.dependency_fingerprint_sha256,
            )
            for manifest in manifests
        )
        experiments = summary / "tables" / _project_main_table_directory() / "experiments.csv"
        atomic_write_bytes(
            experiments,
            _csv_bytes(
                (
                    "artifact_id",
                    "semantic_producer_coordinates",
                    "producer_stage",
                    "dependency_fingerprint_sha256",
                ),
                manifest_rows,
            ),
        )
        metric_rows = tuple(
            tuple(str(value) for value in metric.model_dump(mode="json").values())
            for metric in metrics
        )
        metric_columns = (
            tuple(metrics[0].model_dump(mode="json").keys()) if metrics else ("metric_record",)
        )
        evidence_summary = (
            summary / "tables" / _project_main_table_directory() / "evidence_summary.csv"
        )
        atomic_write_bytes(evidence_summary, _csv_bytes(metric_columns, metric_rows))
        metrics_summary = summary / "metrics" / _project_metric_summary_directory() / "summary.json"
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
            / "reproducibility"
            / _project_configuration_reproducibility_directory()
            / "scientific_configuration.json"
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
            / "reproducibility"
            / _project_execution_reproducibility_directory()
            / "execution.json"
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

    def write_figure(
        self,
        experiment: ExperimentName,
        artifact_id: ArtifactIdentifier,
        figure: EvidenceFigure,
        name: str,
    ) -> Path:
        self._store.resolve(artifact_id)
        destination = results_workspace(self._layout, experiment) / f"{name}.figure.json"
        atomic_write_json(destination, figure.payload())
        return destination

    def metric_record(self, artifact_id: ArtifactIdentifier) -> MetricRecord | None:
        manifest = self._store.resolve(artifact_id)
        if len(manifest.payload_paths) != 1:
            raise EvidenceExportError("metric export requires exactly one verified payload")
        payload_path = Path(manifest.payload_paths[0])
        if not payload_path.is_file():
            raise EvidenceExportError(f"verified payload is missing: {payload_path}")
        raw = json.loads(payload_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise EvidenceExportError("verified payload must be a JSON object")
        metric_payload = raw.get("metric_record")
        if metric_payload is None:
            return None
        try:
            return MetricRecord.model_validate(metric_payload)
        except ValueError as error:
            raise EvidenceExportError(f"verified metric payload is invalid: {error}") from error


def _csv_bytes(columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _tex_bytes(columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> bytes:
    escaped_columns = tuple(_tex_escape(value) for value in columns)
    escaped_rows = tuple(tuple(_tex_escape(value) for value in row) for row in rows)
    lines = (
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        " & ".join(escaped_columns) + r" \\",
        r"\hline",
        *(" & ".join(row) + r" \\" for row in escaped_rows),
        r"\end{tabular}",
        "",
    )
    return "\n".join(lines).encode("utf-8")


def _tex_escape(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_")


def _experiment_metric_summary_directory() -> str:
    return active_config().runtime.artifact_layout.manuscript_experiment_subdirectories.metrics[-1]


def _project_metric_summary_directory() -> str:
    return active_config().runtime.artifact_layout.project_summary_subdirectories.metrics[-1]


def _experiment_supplementary_table_directory() -> str:
    return active_config().runtime.artifact_layout.manuscript_experiment_subdirectories.tables[-1]


def _experiment_main_figure_directory() -> str:
    return active_config().runtime.artifact_layout.manuscript_experiment_subdirectories.figures[0]


def _project_main_table_directory() -> str:
    return active_config().runtime.artifact_layout.project_summary_subdirectories.tables[0]


def _project_configuration_reproducibility_directory() -> str:
    return active_config().runtime.artifact_layout.project_summary_subdirectories.reproducibility[0]


def _project_execution_reproducibility_directory() -> str:
    return active_config().runtime.artifact_layout.project_summary_subdirectories.reproducibility[
        -1
    ]


def _metric_svg_bytes(label: str, value: float) -> bytes:
    bar_width = min(float(480 - 80), max(0.0, abs(value) * (480 - 80) / 4))
    text = html.escape(label)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="180" '
        'viewBox="0 0 480 180"><rect width="480" height="180" fill="white"/>'
        f'<text x="40" y="45" font-family="sans-serif" font-size="18">{text}</text>'
        '<line x1="40" y1="130" x2="440" y2="130" stroke="black"/>'
        f'<rect x="40" y="80" width="{bar_width:g}" height="50" fill="#2a6fbb"/>'
        "</svg>\n"
    )
    return svg.encode("utf-8")


def _metric_pdf_bytes(label: str, value: float) -> bytes:
    bar_width = min(float(480 - 80), max(0.0, abs(value) * (480 - 80) / 4))
    escaped = label.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = (
        f"BT /F1 14 Tf 40 150 Td ({escaped}) Tj ET\n"
        "0.16 0.44 0.73 rg\n"
        f"40 70 {bar_width:g} 40 re f\n"
        "0 0 0 RG\n40 70 m 440 70 l S\n"
    ).encode("ascii", errors="replace")
    objects = (
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n",
        (
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 480 180]"
            b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        ),
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
        b"5 0 obj<</Length "
        + str(len(stream)).encode("ascii")
        + b">>stream\n"
        + stream
        + b"endstream\nendobj\n",
    )
    document = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for item in objects:
        offsets.append(len(document))
        document.extend(item)
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        f"trailer<</Size {len(offsets)}/Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(document)


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


def numerical_constants_and_seeds_table(config: FedorbitConfig | None = None) -> EvidenceTable:
    resolved = config if config is not None else active_config()
    dumped = cast(Mapping[str, JsonValue], resolved.model_dump(mode="json"))
    rows = tuple(
        sorted(_leaf_scalars("scientific", dumped.get("scientific", OrderedDict())))
        + sorted(_leaf_scalars("solvers", dumped.get("solvers", OrderedDict())))
    )
    return EvidenceTable(
        columns=("configuration_path", "value"),
        rows=tuple((path, value) for path, value in rows),
    )


def experiment_matrix_table(rows: Sequence[Mapping[str, TableScalar]]) -> EvidenceTable:
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
    return EvidenceTable(columns=columns, rows=rows)


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
                    comparison.paired_seed_count if comparison is not None else 0,
                    _metric_value(metric_records, pair, method, MetricId.MACRO_CROSS_ENTROPY),
                    _metric_value(metric_records, pair, method, MetricId.MACRO_F1),
                    _metric_value(metric_records, pair, method, MetricId.BALANCED_ACCURACY),
                    comparison.mean_difference if comparison is not None else None,
                    comparison.bca_ci_low if comparison is not None else None,
                    comparison.bca_ci_high if comparison is not None else None,
                    comparison.raw_p if comparison is not None else None,
                    comparison.holm_p if comparison is not None else None,
                    comparison.decision.value if comparison is not None else "",
                    _metric_value(metric_records, pair, method, MetricId.COVERAGE_CONFIRM),
                )
            )
    return EvidenceTable(columns=columns, rows=tuple(rows))


def _rows_table(
    columns: tuple[str, ...], rows: Sequence[Mapping[str, TableScalar]]
) -> EvidenceTable:
    return EvidenceTable(
        columns=columns,
        rows=tuple(tuple(row[column] for column in columns) for row in rows),
    )


def transfer_ontology_and_null_padding_table(
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


def model_and_training_protocol_table(
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


def information_resource_matrix_table(
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


def exact_solver_results_table(rows: Sequence[Mapping[str, TableScalar]]) -> EvidenceTable:
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


def ablation_results_table(rows: Sequence[Mapping[str, TableScalar]]) -> EvidenceTable:
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


def sparsity_and_dense_results_table(rows: Sequence[Mapping[str, TableScalar]]) -> EvidenceTable:
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


def confirmation_results_table(rows: Sequence[Mapping[str, TableScalar]]) -> EvidenceTable:
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
            "arr",
            "rrr",
            "ci",
            "p",
        ),
        rows,
    )


def generalization_results_table(rows: Sequence[Mapping[str, TableScalar]]) -> EvidenceTable:
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


def failure_boundary_results_table(rows: Sequence[Mapping[str, TableScalar]]) -> EvidenceTable:
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


def scalability_results_table(rows: Sequence[Mapping[str, TableScalar]]) -> EvidenceTable:
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


def _figure(x_label: str, y_label: str, series: Sequence[FigureSeries]) -> EvidenceFigure:
    return EvidenceFigure(x_label=x_label, y_label=y_label, series=tuple(series))


def real_transfer_gain_forest_plot(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("paired mean relative macro-CE gain vs local", "primary directed pair", series)


def baseline_paired_difference_plot(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("primary directed pair", "seed-level paired difference", series)


def coupling_gap_phase_figure(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("coupling factor combination", "predicted structural zero/strict state", series)


def predicted_vs_realized_transfer_figure(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("certified robust predicted value", "TEST relative macro-CE gain", series)


def sparsity_utility_efficiency_figure(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("runtime", "realized gain", series)


def confirmation_safety_coverage_figure(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("confirmation coverage", "harmful accepted rate", series)


def semantic_sufficiency_frontier_figure(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("log|orbit|", "realized gain", series)


def failure_boundary_figure(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("boundary setting", "certified value / realized gain", series)


def scalability_figure(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure(
        "N_S * sum(n_g^3) (log scale)",
        "runtime (log scale)",
        series,
    )


def map_value_bound_figure(series: Sequence[FigureSeries]) -> EvidenceFigure:
    return _figure("orbit-radius bound", "exact map action value", series)
