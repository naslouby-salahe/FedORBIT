from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.datasets.common import (
    ChronologyValidationState,
    DatasetInspectionRequest,
    inspect_dataset,
)
from fedorbit.types import DatasetId

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.skipif(not RAW_ROOT.is_dir(), reason="real raw datasets are unavailable")
def test_selected_release_is_inspected_without_assuming_documented_chronology() -> None:
    edge = inspect_dataset(DatasetInspectionRequest(DatasetId.EDGE_IIOTSET_NETWORK, RAW_ROOT))
    assert edge.row_count == 2_219_201
    assert edge.inconsistent_binary_label_rows == 0
    assert edge.event_time.state == ChronologyValidationState.UNPARSEABLE_EVENT_TIME
    assert edge.event_time.timestamp_pattern_row_count == 2_096_419
    assert edge.event_time.unusable_row_count == 122_782
    assert not edge.valid_for_chronological_preprocessing
    assert tuple((item.label, item.row_count) for item in edge.local_class_counts) == (
        ("Backdoor", 24_862),
        ("DDoS_HTTP", 49_911),
        ("DDoS_ICMP", 116_436),
        ("DDoS_TCP", 50_062),
        ("DDoS_UDP", 121_568),
        ("Fingerprinting", 1_001),
        ("MITM", 1_214),
        ("Normal", 1_615_643),
        ("Password", 50_153),
        ("Port_Scanning", 22_564),
        ("Ransomware", 10_925),
        ("SQL_injection", 51_203),
        ("Uploading", 37_634),
        ("Vulnerability_scanner", 50_110),
        ("XSS", 15_915),
    )
    for dataset in (
        DatasetId.TON_IOT_WINDOWS10_HOST,
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_NETWORK,
    ):
        observed = inspect_dataset(DatasetInspectionRequest(dataset, RAW_ROOT))
        assert observed.event_time.state == ChronologyValidationState.MISSING_FIELD
        assert not observed.valid_for_chronological_preprocessing
