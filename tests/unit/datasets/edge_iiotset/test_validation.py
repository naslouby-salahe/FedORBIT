from __future__ import annotations

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.common import ObservedColumnSamples
from fedorbit.datasets.edge_iiotset.schema import edge_iiotset_adapter
from fedorbit.datasets.edge_iiotset.validation import (
    EdgeValidationError,
    LabelObservation,
    validate_binary_multiclass_consistency,
    validate_edge_schema,
)


def _schema():
    config = load_fedorbit_config()
    columns = (
        "frame.time",
        "http.request.method",
        "tcp.ack",
        "Attack_label",
        "Attack_type",
    )
    return edge_iiotset_adapter().resolve_schema(
        columns,
        1.0,
        config.scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum,
        ObservedColumnSamples({"tcp.ack": ("1", "2")}),
    )


def test_edge_schema_validation_accepts_excluded_leakage_fields() -> None:
    validate_edge_schema(_schema())


def test_edge_binary_and_multiclass_labels_must_describe_same_partition() -> None:
    validate_binary_multiclass_consistency(
        (LabelObservation("Normal", 0), LabelObservation("DDoS TCP", 1))
    )
    with pytest.raises(EdgeValidationError):
        validate_binary_multiclass_consistency((LabelObservation("Normal", 1),))
    with pytest.raises(EdgeValidationError):
        validate_binary_multiclass_consistency((LabelObservation("DDoS TCP", 0),))


def test_edge_binary_label_domain_is_closed() -> None:
    with pytest.raises(EdgeValidationError):
        validate_binary_multiclass_consistency((LabelObservation("Normal", 2),))
