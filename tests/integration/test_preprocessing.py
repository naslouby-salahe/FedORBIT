from __future__ import annotations

from pathlib import Path

import pytest

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.common import (
    ChronologyValidationState,
    DatasetInspectionRequest,
    inspect_dataset,
)
from fedorbit.datasets.ontology import transfer_eligibility
from fedorbit.datasets.splitting import (
    ChronologicalRowCount,
    ChronologicalTimestamp,
    DuplicateGroupChronology,
    DuplicateGroupId,
    assign_duplicate_groups_chronologically,
)
from fedorbit.types import DatasetId, Split

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"


def test_dataset_support_and_chronology_contracts_share_authoritative_config() -> None:
    config = load_fedorbit_config()
    support = config.scientific.transfer_support
    eligibility = transfer_eligibility(
        support.source_train_minimum,
        support.source_meta_minimum,
        support.target_meta_minimum,
        support.target_confirm_minimum,
        support.target_test_minimum,
    )
    assert eligibility.source_eligible and eligibility.target_eligible
    assignment = assign_duplicate_groups_chronologically(
        (
            DuplicateGroupChronology(
                DuplicateGroupId("a"), ChronologicalTimestamp(1.0), ChronologicalRowCount(55)
            ),
            DuplicateGroupChronology(
                DuplicateGroupId("b"), ChronologicalTimestamp(2.0), ChronologicalRowCount(15)
            ),
            DuplicateGroupChronology(
                DuplicateGroupId("c"), ChronologicalTimestamp(3.0), ChronologicalRowCount(10)
            ),
            DuplicateGroupChronology(
                DuplicateGroupId("d"), ChronologicalTimestamp(4.0), ChronologicalRowCount(10)
            ),
            DuplicateGroupChronology(
                DuplicateGroupId("e"), ChronologicalTimestamp(5.0), ChronologicalRowCount(10)
            ),
        ),
    )
    assert assignment.split_of(DuplicateGroupId("a")) == Split.TRAIN
    assert assignment.split_of(DuplicateGroupId("e")) == Split.TEST


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
        assert observed.event_time.state == ChronologyValidationState.VALID
        assert observed.valid_for_chronological_preprocessing
