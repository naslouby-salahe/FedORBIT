from __future__ import annotations

import csv
import html
import io
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fedorbit.analysis.evidence import EvidenceClassificationInputs, classify_all_propositions
from fedorbit.analysis.records import MetricRecord
from fedorbit.config.loading import active_config
from fedorbit.infrastructure.execution import ArtifactStore, atomic_write_bytes, atomic_write_json
from fedorbit.infrastructure.manifests import ReusableArtifactManifest
from fedorbit.infrastructure.workspace import WorkspaceLayout, results_workspace
from fedorbit.types import ArtifactIdentifier, ExperimentName, StableJsonPayload, stable_json


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

    def write_evidence_classification_table(
        self,
        inputs: EvidenceClassificationInputs,
    ) -> tuple[Path, Path]:
        statuses = classify_all_propositions(inputs)
        columns = ("proposition", "status")
        rows = tuple((proposition.value, status.value) for proposition, status in statuses.items())
        destination = self._layout.project_summary / "tables" / _project_main_table_directory()
        csv_path = destination / "evidence_classification.csv"
        tex_path = destination / "evidence_classification.tex"
        atomic_write_bytes(csv_path, _csv_bytes(columns, rows))
        atomic_write_bytes(tex_path, _tex_bytes(columns, rows))
        return (csv_path, tex_path)

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
