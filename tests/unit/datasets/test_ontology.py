from __future__ import annotations

from fedorbit.config.models import FedorbitConfig
from fedorbit.datasets.ontology import (
    NORMAL_LABEL,
    TRANSFER_CONCEPTS,
    canonicalize_label,
    coarse_group_for,
    native_labels_for,
    transfer_concept_for,
    transfer_eligibility,
)
from fedorbit.domain.enums import CoarseGroup, DatasetId, OracleTransferConcept


def test_label_canonicalization_is_fixed() -> None:
    assert canonicalize_label(" SQL/Injection ") == "sql_injection"
    assert canonicalize_label("DDoS__TCP") == "ddos_tcp"
    assert canonicalize_label("Normal") == NORMAL_LABEL


def test_transfer_ontology_uses_ton_mapping_for_all_ton_clients() -> None:
    assert (
        transfer_concept_for(DatasetId.EDGE_IIOTSET_NETWORK, "ddos_tcp")
        == OracleTransferConcept.DDOS
    )
    for client in (
        DatasetId.TON_IOT_WINDOWS10_HOST,
        DatasetId.TON_IOT_LINUX_PROCESS_HOST,
        DatasetId.TON_IOT_NETWORK,
    ):
        assert transfer_concept_for(client, "ddos") == OracleTransferConcept.DDOS
        assert transfer_concept_for(client, "ddos_tcp") is None
        assert "dos" in native_labels_for(client)
        assert "uploading" not in native_labels_for(client)


def test_transfer_concepts_and_coarse_groups_are_exact() -> None:
    assert TRANSFER_CONCEPTS == tuple(item.value for item in OracleTransferConcept)
    assert coarse_group_for(DatasetId.EDGE_IIOTSET_NETWORK, "ransomware") == CoarseGroup.DISRUPTION
    assert (
        coarse_group_for(DatasetId.TON_IOT_WINDOWS10_HOST, "injection") == CoarseGroup.EXPLOITATION
    )
    assert (
        coarse_group_for(DatasetId.TON_IOT_LINUX_PROCESS_HOST, "scanning")
        == CoarseGroup.ACCESS_AND_DISCOVERY
    )


def test_transfer_eligibility_uses_registered_count_thresholds(
    fedorbit_config: FedorbitConfig,
) -> None:
    support = fedorbit_config.scientific.transfer_support
    eligible = transfer_eligibility(
        fedorbit_config,
        support.source_train_minimum,
        support.source_meta_minimum,
        support.target_meta_minimum,
        support.target_confirm_minimum,
        support.target_test_minimum,
    )
    assert eligible.source_eligible
    assert eligible.target_eligible
    source_null = transfer_eligibility(
        fedorbit_config,
        support.source_train_minimum - 1,
        support.source_meta_minimum,
        support.target_meta_minimum,
        support.target_confirm_minimum,
        support.target_test_minimum,
    )
    assert not source_null.source_eligible
    assert not source_null.present_for_source
    target_null = transfer_eligibility(
        fedorbit_config,
        support.source_train_minimum,
        support.source_meta_minimum,
        support.target_meta_minimum,
        support.target_confirm_minimum - 1,
        support.target_test_minimum,
    )
    assert not target_null.target_eligible
