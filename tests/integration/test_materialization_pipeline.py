from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.unit.artifacts.artifact_fixtures import artifact_completion, artifact_manifest

from fedorbit.config.loading import active_config
from fedorbit.datasets.common import (
    AdapterContract,
    DatasetAdapter,
    FieldRole,
    ObservedColumnSamples,
)
from fedorbit.datasets.preprocessing import (
    NormalizedFeatureVector,
    NormalizedRow,
    RowNormalizationError,
    TrainingFeatureValues,
    evaluate_feature_quality,
    fit_numeric_preprocessor,
    normalize_and_split_training_rows,
    transform_numeric,
)
from fedorbit.datasets.splitting import DuplicateGroupId, DuplicateGroupIdentifier
from fedorbit.infrastructure.artifacts import ArtifactStore, StagedPayload
from fedorbit.infrastructure.reuse import ArtifactValidationError
from fedorbit.optimization.correspondence import build_padded_block_structure
from fedorbit.types import (
    ArtifactStage,
    CoarseGroup,
    DatasetId,
    FineLabel,
    NormalizedGroupIdentifier,
    RawCellText,
    Sha256Digest,
    Split,
    TabularColumnName,
)

_TIMESTAMP = TabularColumnName("ts")
_LABEL = TabularColumnName("label")
_BINARY = TabularColumnName("binary_label")
_DURATION = TabularColumnName("duration")
_SERVICE = TabularColumnName("service")


def _adapter() -> DatasetAdapter:
    return DatasetAdapter(
        AdapterContract(
            DatasetId.TON_IOT_NETWORK,
            (_TIMESTAMP,),
            (_LABEL,),
            (_BINARY,),
        )
    )


def _schema():
    columns = (_TIMESTAMP, _LABEL, _BINARY, _DURATION, _SERVICE)
    samples = ObservedColumnSamples(
        {
            _DURATION: tuple(RawCellText(value) for value in ("0.5", "1.5", "2.5")),
            _SERVICE: (RawCellText("http"), RawCellText("dns")),
        }
    )
    return _adapter().resolve_schema(
        columns,
        timestamp_parse_success_fraction=1.0,
        timestamp_alias_minimum=active_config().scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum,
        observed_value_samples=samples,
    )


def _row(
    timestamp: float,
    label: str,
    duration: str,
    service: str,
) -> NormalizedRow:
    features = NormalizedFeatureVector(
        {
            _DURATION: RawCellText(duration),
            _SERVICE: RawCellText(service),
        }
    )
    return NormalizedRow(
        features=features,
        label=FineLabel(label),
        timestamp_fraction=timestamp,
        group_id=NormalizedGroupIdentifier(""),
    )


def test_adapter_schema_assigns_registered_roles_and_infers_types_from_observed_values() -> None:
    schema = _schema()
    assert schema.timestamp_column == _TIMESTAMP
    assert schema.multiclass_label_column == _LABEL
    assert schema.binary_label_column == _BINARY
    assert schema.role_of(_DURATION) == FieldRole.BEHAVIORAL_NUMERIC
    assert schema.role_of(_SERVICE) == FieldRole.BEHAVIORAL_CATEGORICAL
    assert schema.feature_order == (_DURATION, _SERVICE)


def test_duplicate_safe_chronological_split_keeps_groups_indivisible_and_orders_by_time() -> None:
    schema = _schema()
    rows = (
        _row(1.0, "normal", "1.0", "http"),
        _row(2.0, "normal", "2.0", "http"),
        _row(3.0, "normal", "3.0", "dns"),
        _row(4.0, "normal", "3.0", "dns"),
        _row(5.0, "attack", "5.0", "http"),
        _row(6.0, "attack", "6.0", "http"),
        _row(7.0, "attack", "7.0", "dns"),
        _row(8.0, "attack", "8.0", "dns"),
    )
    result = normalize_and_split_training_rows(schema, rows)
    grouped_sizes = sorted(len(members) for _, members in result.duplicate_groups.groups)
    assert grouped_sizes == [1, 1, 1, 1, 1, 1, 2]
    duplicated = tuple(
        (candidate_id, members)
        for candidate_id, members in result.duplicate_groups.groups
        if len(members) == 2
    )
    assert len(duplicated) == 1
    _duplicate_id, members = duplicated[0]
    assert {row.label for row in members} == {FineLabel("normal")}
    assert {row.timestamp_fraction for row in members} == {3.0, 4.0}
    for candidate_group, _member_rows in result.duplicate_groups.groups:
        assigned = result.split_assignment.split_of(
            DuplicateGroupId(DuplicateGroupIdentifier(candidate_group))
        )
        assert assigned in {Split.TRAIN, Split.META, Split.VALID, Split.CONFIRM, Split.TEST}
    with pytest.raises(RowNormalizationError):
        normalize_and_split_training_rows(
            schema,
            (
                _row(1.0, "normal", "9.0", "http"),
                _row(2.0, "attack", "9.0", "http"),
            ),
        )


def test_feature_quality_and_numeric_fitting_use_train_rows_only() -> None:
    preprocessing = active_config().scientific.preprocessing
    retained_feature = TabularColumnName("service")
    stable_numerics = tuple(
        TabularColumnName(name) for name in ("src_bytes", "dst_bytes", "src_pkts", "dst_pkts")
    )
    feature_names = (_DURATION, *stable_numerics, retained_feature)
    train_values = TrainingFeatureValues(
        {
            _DURATION: np.asarray([1.0, 2.0, 3.0, np.nan, 5.0], dtype=np.float64),
            **{
                name: np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
                for name in stable_numerics
            },
            retained_feature: np.asarray(["http", "http", "dns", "dns", "dns"], dtype=object),
        }
    )
    report = evaluate_feature_quality(feature_names, frozenset({retained_feature}), train_values)
    assert report.candidate_count_before_filtering == len(feature_names)
    assert report.dropped_feature_count == 1
    dropped = {candidate.name for candidate in report.candidate_features if candidate.dropped}
    assert dropped == {_DURATION}
    dropped_fraction = report.dropped_feature_count / report.candidate_count_before_filtering
    assert dropped_fraction <= preprocessing.client_invalidity_dropped_feature_fraction_threshold
    assert report.client_invalid is False
    assert report.client_invalid_reason is None
    assert preprocessing.feature_missing_or_nonfinite_drop_threshold < 1.0 / 5.0
    fitted = fit_numeric_preprocessor(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
    assert fitted.median == pytest.approx(3.0)
    transformed = transform_numeric(np.asarray([4.0], dtype=np.float64), fitted)
    assert transformed.shape == (1,)


def test_eligibility_and_null_padding_use_per_block_maximum() -> None:
    blocks = build_padded_block_structure(
        (CoarseGroup.DISRUPTION, CoarseGroup.EXPLOITATION),
        {CoarseGroup.DISRUPTION: 2, CoarseGroup.EXPLOITATION: 1},
        {CoarseGroup.DISRUPTION: 1, CoarseGroup.EXPLOITATION: 3},
    )
    assert blocks.padded_size_tuple == (2, 3)
    assert blocks.total_padded_nodes == 5
    assert blocks.block_index_range(0) == range(0, 2)
    assert blocks.block_index_range(1) == range(2, 5)


def test_staged_artifact_promotion_is_atomic_and_torn_records_are_never_reusable(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "outputs")
    payload_bytes = b"promoted-payload"
    draft_directory = store.staging_dir() / "pipeline-draft"
    draft_directory.mkdir(parents=True, exist_ok=True)
    draft = draft_directory / "payload.json"
    draft.write_bytes(payload_bytes)
    active = store.root / "artifacts" / "pipeline.json"
    manifest = artifact_manifest(draft, Sha256Digest("2" * 64), ArtifactStage.RAW).model_copy(
        update={"payload_paths": (str(active),)}
    )
    staged_directory = store.staging_area(manifest.artifact_id)
    staged_directory.mkdir(parents=True, exist_ok=True)
    staged = staged_directory / "payload.json"
    staged.write_bytes(payload_bytes)
    draft.unlink()
    store.promote_artifact(
        manifest,
        artifact_completion(manifest),
        staged_payloads=(StagedPayload(staged, active),),
    )
    assert active.read_bytes() == payload_bytes
    assert store.is_reusable(manifest.artifact_id) is True
    assert store.resolve(manifest.artifact_id).payload_paths == (str(active),)
    torn_active = store.root / "artifacts" / "torn.json"
    torn_draft = draft_directory / "torn.json"
    torn_draft.write_bytes(b"torn")
    torn_manifest = artifact_manifest(
        torn_draft, Sha256Digest("3" * 64), ArtifactStage.RAW
    ).model_copy(update={"payload_paths": (str(torn_active),)})
    torn_active.write_bytes(b"torn")
    store.promote_artifact(torn_manifest, artifact_completion(torn_manifest))
    torn_active.unlink()
    assert store.is_reusable(torn_manifest.artifact_id) is False
    with pytest.raises(ArtifactValidationError):
        store.resolve(torn_manifest.artifact_id)
