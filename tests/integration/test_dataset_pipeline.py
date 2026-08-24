from __future__ import annotations

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.datasets.ontology import transfer_eligibility
from fedorbit.datasets.splitting import (
    DuplicateGroupChronology,
    assign_duplicate_groups_chronologically,
)
from fedorbit.domain.enums import Split


def test_dataset_support_and_chronology_contracts_share_authoritative_config() -> None:
    config = load_fedorbit_config()
    support = config.scientific.transfer_support
    eligibility = transfer_eligibility(
        config,
        support.source_train_minimum,
        support.source_meta_minimum,
        support.target_meta_minimum,
        support.target_confirm_minimum,
        support.target_test_minimum,
    )
    assert eligibility.source_eligible and eligibility.target_eligible
    assignment = assign_duplicate_groups_chronologically(
        config,
        (
            DuplicateGroupChronology("a", 1.0, 55),
            DuplicateGroupChronology("b", 2.0, 15),
            DuplicateGroupChronology("c", 3.0, 10),
            DuplicateGroupChronology("d", 4.0, 10),
            DuplicateGroupChronology("e", 5.0, 10),
        ),
    )
    assert assignment.split_of("a") == Split.TRAIN
    assert assignment.split_of("e") == Split.TEST
