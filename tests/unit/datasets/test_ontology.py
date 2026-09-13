from __future__ import annotations

from fedorbit.datasets.ontology import (
    NORMAL_LABEL,
    normalize_label,
)
from fedorbit.types import DatasetLabel, FineLabel


def test_label_row_normalization_is_fixed() -> None:
    assert normalize_label(DatasetLabel(" SQL/Injection ")) == FineLabel("sql_injection")
    assert normalize_label(DatasetLabel("DDoS__TCP")) == FineLabel("ddos_tcp")
    assert normalize_label(DatasetLabel("Normal")) == NORMAL_LABEL
