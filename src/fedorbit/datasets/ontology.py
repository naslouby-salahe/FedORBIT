from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from fedorbit.config.loading import active_config
from fedorbit.types import (
    CoarseGroup,
    DatasetId,
    DatasetLabel,
    FineLabel,
    LabelText,
    NativeLabels,
    NativeLabelSet,
    OracleTransferConcept,
    SampleCount,
)

class CanonicalLabel(StrEnum):
    NORMAL = "normal"


class EdgeNativeLabel(StrEnum):
    DDOS_UDP = "ddos_udp"
    DDOS_ICMP = "ddos_icmp"
    DDOS_TCP = "ddos_tcp"
    DDOS_HTTP = "ddos_http"
    RANSOMWARE = "ransomware"
    BACKDOOR = "backdoor"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PASSWORD = "password"
    PORT_SCANNING = "port_scanning"
    FINGERPRINTING = "fingerprinting"
    VULNERABILITY_SCANNER = "vulnerability_scanner"
    MITM = "mitm"
    UPLOADING = "uploading"


class TonNativeLabel(StrEnum):
    DDOS = "ddos"
    RANSOMWARE = "ransomware"
    BACKDOOR = "backdoor"
    INJECTION = "injection"
    XSS = "xss"
    PASSWORD = "password"
    SCANNING = "scanning"
    MITM = "mitm"
    DOS = "dos"


NORMAL_LABEL = CanonicalLabel.NORMAL
TRANSFER_ONTOLOGY: Mapping[
    OracleTransferConcept, tuple[CoarseGroup, NativeLabels, NativeLabels]
] = OrderedDict[OracleTransferConcept, tuple[CoarseGroup, NativeLabels, NativeLabels]](
    (
        (
            OracleTransferConcept.DDOS,
            (
                CoarseGroup.DISRUPTION,
                tuple(FineLabel(label) for label in (EdgeNativeLabel.DDOS_UDP, EdgeNativeLabel.DDOS_ICMP, EdgeNativeLabel.DDOS_TCP, EdgeNativeLabel.DDOS_HTTP)),
                (FineLabel(TonNativeLabel.DDOS),),
            ),
        ),
        (
            OracleTransferConcept.RANSOMWARE,
            (CoarseGroup.DISRUPTION, (FineLabel(EdgeNativeLabel.RANSOMWARE),), (FineLabel(TonNativeLabel.RANSOMWARE),)),
        ),
        (OracleTransferConcept.BACKDOOR, (CoarseGroup.EXPLOITATION, (FineLabel(EdgeNativeLabel.BACKDOOR),), (FineLabel(TonNativeLabel.BACKDOOR),))),
        (
            OracleTransferConcept.INJECTION,
            (CoarseGroup.EXPLOITATION, (FineLabel(EdgeNativeLabel.SQL_INJECTION),), (FineLabel(TonNativeLabel.INJECTION),)),
        ),
        (OracleTransferConcept.XSS, (CoarseGroup.EXPLOITATION, (FineLabel(EdgeNativeLabel.XSS),), (FineLabel(TonNativeLabel.XSS),))),
        (
            OracleTransferConcept.PASSWORD_ATTACK,
            (CoarseGroup.ACCESS_AND_DISCOVERY, (FineLabel(EdgeNativeLabel.PASSWORD),), (FineLabel(TonNativeLabel.PASSWORD),)),
        ),
        (
            OracleTransferConcept.SCANNING,
            (
                CoarseGroup.ACCESS_AND_DISCOVERY,
                tuple(FineLabel(label) for label in (EdgeNativeLabel.PORT_SCANNING, EdgeNativeLabel.FINGERPRINTING, EdgeNativeLabel.VULNERABILITY_SCANNER)),
                (FineLabel(TonNativeLabel.SCANNING),),
            ),
        ),
        (OracleTransferConcept.MITM, (CoarseGroup.ACCESS_AND_DISCOVERY, (FineLabel(EdgeNativeLabel.MITM),), (FineLabel(TonNativeLabel.MITM),))),
    )
)
EDGE_ELIGIBLE_LOCAL_CLASSES = frozenset({FineLabel(EdgeNativeLabel.UPLOADING)})
TON_ELIGIBLE_LOCAL_CLASSES = frozenset({FineLabel(TonNativeLabel.DOS)})


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


def normalize_label(raw: LabelText) -> FineLabel:
    normalized = unicodedata.normalize("NFC", raw).strip().casefold()
    underscored = re.sub(r"[^0-9a-z]+", "_", normalized)
    return FineLabel(re.sub(r"_+", "_", underscored).strip("_"))


def _native_mapping(client: DatasetId, concept: OracleTransferConcept) -> NativeLabels:
    _, edge_labels, ton_labels = TRANSFER_ONTOLOGY[concept]
    return edge_labels if client == DatasetId.EDGE_IIOTSET_NETWORK else ton_labels


def native_labels_for(client: DatasetId) -> NativeLabelSet:
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
    normalized_label: FineLabel,
) -> OracleTransferConcept | None:
    matches = tuple(
        concept
        for concept in OracleTransferConcept
        if normalized_label in _native_mapping(client, concept)
    )
    if len(matches) > 1:
        raise OntologyError(f"label maps to multiple transfer concepts: {normalized_label}")
    return matches[0] if matches else None


def coarse_group_for(client: DatasetId, normalized_label: FineLabel) -> CoarseGroup | None:
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
