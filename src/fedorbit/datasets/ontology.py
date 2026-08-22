from __future__ import annotations

import re
import unicodedata

from fedorbit.domain.enums import CoarseGroup, DatasetId

NORMAL_LABEL = "normal"

TRANSFER_CONCEPTS = (
    "DDoS",
    "Ransomware",
    "Backdoor",
    "Injection",
    "XSS",
    "Password attack",
    "Scanning",
    "MITM",
)

TRANSFER_ONTOLOGY: dict[str, tuple[CoarseGroup, tuple[str, ...], tuple[str, ...]]] = {
    "DDoS": (CoarseGroup.DISRUPTION, ("ddos_udp", "ddos_icmp", "ddos_tcp", "ddos_http"), ("ddos",)),
    "Ransomware": (CoarseGroup.DISRUPTION, ("ransomware",), ("ransomware",)),
    "Backdoor": (CoarseGroup.EXPLOITATION, ("backdoor",), ("backdoor",)),
    "Injection": (CoarseGroup.EXPLOITATION, ("sql_injection",), ("injection",)),
    "XSS": (CoarseGroup.EXPLOITATION, ("xss",), ("xss",)),
    "Password attack": (
        CoarseGroup.ACCESS_AND_DISCOVERY,
        ("password",),
        ("password",),
    ),
    "Scanning": (
        CoarseGroup.ACCESS_AND_DISCOVERY,
        ("port_scanning", "fingerprinting", "vulnerability_scanner"),
        ("scanning",),
    ),
    "MITM": (CoarseGroup.ACCESS_AND_DISCOVERY, ("mitm",), ("mitm",)),
}

EDGE_ELIGIBLE_LOCAL_CLASSES = frozenset({"uploading"})
TON_ELIGIBLE_LOCAL_CLASSES = frozenset({"dos"})


class OntologyError(ValueError):
    pass


def canonicalize_label(raw: str) -> str:
    normalized = unicodedata.normalize("NFC", raw.strip().casefold())
    underscored = re.sub(r"[^0-9a-z]+", "_", normalized)
    return re.sub(r"_+", "_", underscored).strip("_")


def native_labels_for(client: DatasetId) -> frozenset[str]:
    labels: set[str] = set()
    for _, edge_labels, ton_labels in TRANSFER_ONTOLOGY.values():
        labels.update(edge_labels if client != DatasetId.TON_IOT_NETWORK else ton_labels)
    if client != DatasetId.TON_IOT_NETWORK:
        labels.update(EDGE_ELIGIBLE_LOCAL_CLASSES)
    else:
        labels.update(TON_ELIGIBLE_LOCAL_CLASSES)
    return frozenset(labels)


def transfer_concept_for(client: DatasetId, canonical_label: str) -> str | None:
    for concept, (_, edge_labels, ton_labels) in TRANSFER_ONTOLOGY.items():
        native = ton_labels if client == DatasetId.TON_IOT_NETWORK else edge_labels
        if canonical_label in native:
            return concept
    return None


def coarse_group_for(client: DatasetId, canonical_label: str) -> CoarseGroup | None:
    concept = transfer_concept_for(client, canonical_label)
    if concept is None:
        return None
    return TRANSFER_ONTOLOGY[concept][0]
