from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

from fedorbit.config.loading import active_config
from fedorbit.types import CoarseGroup, DatasetId, OracleTransferConcept, SampleCount

NORMAL_LABEL = "normal" #TODO: move to constants or enums
TRANSFER_CONCEPTS = tuple(concept.value for concept in OracleTransferConcept) #TODO: remove this from code and move to tests
TRANSFER_ONTOLOGY: Mapping[
    OracleTransferConcept, tuple[CoarseGroup, tuple[str, ...] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
                                 , tuple[str, ...]] #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this
] = OrderedDict[OracleTransferConcept, tuple[CoarseGroup, tuple[str, ...], tuple[str, ...]]](
    (
        (
            OracleTransferConcept.DDOS,
            (CoarseGroup.DISRUPTION, ("ddos_udp", "ddos_icmp", "ddos_tcp", "ddos_http"), ("ddos",)), #TODO: should be in enums
        ),
        (
            OracleTransferConcept.RANSOMWARE,
            (CoarseGroup.DISRUPTION, ("ransomware",), ("ransomware",)), #TODO: should be in enums
        ),
        (OracleTransferConcept.BACKDOOR, (CoarseGroup.EXPLOITATION, ("backdoor",), ("backdoor",))), #TODO: should be in enums
        (
            OracleTransferConcept.INJECTION,
            (CoarseGroup.EXPLOITATION, ("sql_injection",), ("injection",)), #TODO: should be in enums
        ),
        (OracleTransferConcept.XSS, (CoarseGroup.EXPLOITATION, ("xss",), ("xss",))), #TODO: should be in enums
        (
            OracleTransferConcept.PASSWORD_ATTACK,
            (CoarseGroup.ACCESS_AND_DISCOVERY, ("password",), ("password",)), #TODO: should be in enums
        ),
        (
            OracleTransferConcept.SCANNING,
            (
                CoarseGroup.ACCESS_AND_DISCOVERY,
                ("port_scanning", "fingerprinting", "vulnerability_scanner"), #TODO: should be in enums
                ("scanning",), #TODO: should be in enums
            ),
        ),
        (OracleTransferConcept.MITM, (CoarseGroup.ACCESS_AND_DISCOVERY, ("mitm",), ("mitm",))), #TODO: should be in enums
    )
)
EDGE_ELIGIBLE_LOCAL_CLASSES = frozenset({"uploading"}) #TODO: should be more centralized and in enums
TON_ELIGIBLE_LOCAL_CLASSES = frozenset({"dos"}) #TODO: should be more centralized and in enums


class OntologyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TransferEligibility:
    source_eligible: bool
    target_eligible: bool
    source_train_support_passes: bool
    source_meta_support_passes: bool
    target_meta_support_passes: bool
    target_confirm_support_passes: bool
    target_test_support_passes: bool

    @property
    def present_for_source(self) -> bool:
        return self.source_eligible

    @property
    def present_for_target(self) -> bool:
        return self.target_eligible


def normalize_label(raw: str #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: raw)
                    ) -> str: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (output return)
    normalized = unicodedata.normalize("NFC", raw).strip().casefold() #TODO: should be in enums
    underscored = re.sub(r"[^0-9a-z]+", "_", normalized)
    return re.sub(r"_+", "_", underscored).strip("_")


def _native_mapping(client: DatasetId, concept: OracleTransferConcept) -> tuple[str, ...]: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (output return)
    _, edge_labels, ton_labels = TRANSFER_ONTOLOGY[concept]
    return edge_labels if client == DatasetId.EDGE_IIOTSET_NETWORK else ton_labels


def native_labels_for(client: DatasetId) -> frozenset[str]: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (output return)
    labels = {
        label for concept in OracleTransferConcept for label in _native_mapping(client, concept)
    }
    if client == DatasetId.EDGE_IIOTSET_NETWORK:
        labels.update(EDGE_ELIGIBLE_LOCAL_CLASSES)
    else:
        labels.update(TON_ELIGIBLE_LOCAL_CLASSES)
    return frozenset(labels)


def transfer_concept_for(
    client: DatasetId,
    normalized_label: str, #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: normalized_label)
) -> OracleTransferConcept | None:
    matches = tuple(
        concept
        for concept in OracleTransferConcept
        if normalized_label in _native_mapping(client, concept)
    )
    if len(matches) > 1:
        raise OntologyError(f"label maps to multiple transfer concepts: {normalized_label}")
    return matches[0] if matches else None


def coarse_group_for(client: DatasetId, normalized_label: str) -> CoarseGroup | None: #TODO: do not use primitivies. Use an appropriate alias in Types. And diagnose my tests to identify why the architecture tests didn't catch this (input param: normalized_label)
    concept = transfer_concept_for(client, normalized_label)
    return None if concept is None else TRANSFER_ONTOLOGY[concept][0]


def transfer_eligibility(
    source_train_support: SampleCount,
    source_meta_support: SampleCount,
    target_meta_support: SampleCount,
    target_confirm_support: SampleCount,
    target_test_support: SampleCount,
) -> TransferEligibility:
    if (
        min(
            source_train_support,
            source_meta_support,
            target_meta_support,
            target_confirm_support,
            target_test_support,
        )
        < 0
    ):
        raise OntologyError("transfer support counts must be nonnegative")
    support = active_config().scientific.transfer_support
    source_train_passes = source_train_support >= support.source_train_minimum
    source_meta_passes = source_meta_support >= support.source_meta_minimum
    target_meta_passes = target_meta_support >= support.target_meta_minimum
    target_confirm_passes = target_confirm_support >= support.target_confirm_minimum
    target_test_passes = target_test_support >= support.target_test_minimum
    return TransferEligibility(
        source_eligible=source_train_passes and source_meta_passes,
        target_eligible=target_meta_passes and target_confirm_passes and target_test_passes,
        source_train_support_passes=source_train_passes,
        source_meta_support_passes=source_meta_passes,
        target_meta_support_passes=target_meta_passes,
        target_confirm_support_passes=target_confirm_passes,
        target_test_support_passes=target_test_passes,
    )
