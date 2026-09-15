from __future__ import annotations

import json
from collections import OrderedDict
from types import SimpleNamespace
from typing import cast

from fedorbit.config.loading import active_config
from fedorbit.datasets.materialization import MaterializedClient, transfer_concept_groups
from fedorbit.datasets.ontology import (
    NORMAL_LABEL,
    TRANSFER_ONTOLOGY,
    TonNativeLabel,
    normalize_label,
    transfer_concept_for,
    transfer_eligibility,
)
from fedorbit.experiments.synthesis import transfer_ontology_null_padding_rows
from fedorbit.types import (
    DatasetId,
    DatasetLabel,
    DirectedPairName,
    FineLabel,
    OracleTransferConcept,
    Split,
    stable_json,
)


def test_label_row_normalization_is_fixed() -> None:
    assert normalize_label(DatasetLabel(" SQL/Injection ")) == FineLabel("sql_injection")
    assert normalize_label(DatasetLabel("DDoS__TCP")) == FineLabel("ddos_tcp")
    assert normalize_label(DatasetLabel("Normal")) == NORMAL_LABEL


def test_transfer_eligibility_keeps_source_and_target_support_rules_separate() -> None:
    support = active_config().scientific.transfer_support
    eligible = transfer_eligibility(
        support.source_train_minimum,
        support.source_meta_minimum,
        support.target_meta_minimum,
        support.target_confirm_minimum,
        support.target_test_minimum,
    )
    source_ineligible = transfer_eligibility(
        support.source_train_minimum - 1,
        support.source_meta_minimum,
        support.target_meta_minimum,
        support.target_confirm_minimum,
        support.target_test_minimum,
    )
    target_ineligible = transfer_eligibility(
        support.source_train_minimum,
        support.source_meta_minimum,
        support.target_meta_minimum,
        support.target_confirm_minimum - 1,
        support.target_test_minimum,
    )

    assert eligible.source_eligible and eligible.target_eligible
    assert not source_ineligible.source_eligible and source_ineligible.target_eligible
    assert target_ineligible.source_eligible and not target_ineligible.target_eligible


def test_null_padding_records_the_reason_for_present_but_under_supported_concepts() -> None:
    label = FineLabel(TonNativeLabel.DDOS)
    support_by_split = OrderedDict((split, 1) for split in Split)
    materialized = cast(
        MaterializedClient,
        SimpleNamespace(
            dataset=DatasetId.TON_IOT_NETWORK,
            class_manifest=SimpleNamespace(class_names=(label,)),
            class_row_counts={label: support_by_split},
        ),
    )

    rows = transfer_ontology_null_padding_rows(
        DirectedPairName("ton-to-ton"), materialized, materialized
    )
    serialized_rows = json.loads(stable_json(rows))
    ddos = next(row for row in serialized_rows if row["candidate_concept"] == "DDoS")

    assert ddos["source_real"] and ddos["target_real"]
    assert not ddos["action_eligibility"]
    assert ddos["null_reason"] == "DDoS present but below configured support minimum"


def test_transfer_ontology_is_closed_and_sums_support_across_native_classes() -> None:
    ddos_labels = TRANSFER_ONTOLOGY[OracleTransferConcept.DDOS][1][:2]
    assert len(TRANSFER_ONTOLOGY) == len(OracleTransferConcept)
    assert transfer_concept_for(DatasetId.TON_IOT_NETWORK, FineLabel(NORMAL_LABEL)) is None
    assert transfer_concept_for(DatasetId.EDGE_IIOTSET_NETWORK, FineLabel("uploading")) is None
    assert transfer_concept_for(DatasetId.TON_IOT_NETWORK, FineLabel("dos")) is None
    assert transfer_concept_for(DatasetId.TON_IOT_NETWORK, FineLabel("unregistered")) is None
    support_by_label = {
        ddos_labels[0]: OrderedDict(
            (
                (Split.TRAIN, 2),
                (Split.META, 3),
                (Split.VALID, 0),
                (Split.CONFIRM, 4),
                (Split.TEST, 5),
            )
        ),
        ddos_labels[1]: OrderedDict(
            (
                (Split.TRAIN, 7),
                (Split.META, 11),
                (Split.VALID, 0),
                (Split.CONFIRM, 13),
                (Split.TEST, 17),
            )
        ),
    }
    materialized = cast(
        MaterializedClient,
        SimpleNamespace(
            dataset=DatasetId.EDGE_IIOTSET_NETWORK,
            class_manifest=SimpleNamespace(class_names=ddos_labels),
            class_row_counts=support_by_label,
        ),
    )

    group = transfer_concept_groups(materialized.dataset, materialized)[0]

    assert group.concept is OracleTransferConcept.DDOS
    assert group.train_support == 9
    assert group.meta_support == 14
    assert group.confirm_support == 17
    assert group.test_support == 22
