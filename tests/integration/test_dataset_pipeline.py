from __future__ import annotations

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.ontology import transfer_eligibility
from fedorbit.datasets.splitting import (
    ChronologicalRowCount,
    ChronologicalTimestamp,
    DuplicateGroupChronology,
    DuplicateGroupId,
    assign_duplicate_groups_chronologically,
)
from fedorbit.domain.enums import Split


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
