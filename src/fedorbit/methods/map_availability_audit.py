from __future__ import annotations

import hashlib
from collections import OrderedDict
from datetime import datetime, timedelta
from enum import StrEnum
from typing import cast

from fedorbit.datasets.ontology import TRANSFER_ONTOLOGY
from fedorbit.types import (
    ConceptCount,
    DirectedPair,
    DomainModel,
    DurationMinutes,
    ExposedCoarseGroupId,
    OracleTransferConcept,
    Rfc3339UtcTimestamp,
    Sha256Digest,
    StableJsonPayload,
    stable_json,
)


class MapAvailabilityAuditError(ValueError):
    pass


class ProposedMappingEntry(DomainModel):
    source_public_label: str
    target_public_label: str
    exposed_coarse_group: ExposedCoarseGroupId


class UnresolvedAlternativeEntry(DomainModel):
    affected_public_label: str
    alternatives_considered: tuple[str, ...]


class MapAvailabilityAuditSubmission(DomainModel):
    researcher_id: str
    directed_pair: DirectedPair
    session_start_utc: Rfc3339UtcTimestamp
    session_end_utc: Rfc3339UtcTimestamp
    resources_consulted: tuple[str, ...]
    proposed_mapping: tuple[ProposedMappingEntry, ...]
    unresolved_alternatives: tuple[UnresolvedAlternativeEntry, ...]
    rationale: str


_BLANK_SESSION_TIMESTAMP = Rfc3339UtcTimestamp("1970-01-01T00:00:00Z")


def blank_audit_template(
    researcher_id: str, directed_pair: DirectedPair
) -> MapAvailabilityAuditSubmission:
    return MapAvailabilityAuditSubmission(
        researcher_id=researcher_id,
        directed_pair=directed_pair,
        session_start_utc=_BLANK_SESSION_TIMESTAMP,
        session_end_utc=_BLANK_SESSION_TIMESTAMP,
        resources_consulted=(),
        proposed_mapping=(),
        unresolved_alternatives=(),
        rationale="",
    )


def stable_submission_payload(submission: MapAvailabilityAuditSubmission) -> str:
    ordered_mapping = sorted(
        submission.proposed_mapping,
        key=lambda entry: (
            entry.exposed_coarse_group,
            entry.source_public_label,
            entry.target_public_label,
        ),
    )
    payload = cast(
        StableJsonPayload,
        OrderedDict(
            researcher_id=submission.researcher_id,
            directed_pair=[
                submission.directed_pair.source.value,
                submission.directed_pair.target.value,
            ],
            session_start_utc=submission.session_start_utc,
            session_end_utc=submission.session_end_utc,
            resources_consulted=list(submission.resources_consulted),
            proposed_mapping=[
                cast(
                    StableJsonPayload,
                    OrderedDict(
                        source_public_label=entry.source_public_label,
                        target_public_label=entry.target_public_label,
                        exposed_coarse_group=entry.exposed_coarse_group,
                    ),
                )
                for entry in ordered_mapping
            ],
            unresolved_alternatives=[
                cast(
                    StableJsonPayload,
                    OrderedDict(
                        affected_public_label=entry.affected_public_label,
                        alternatives_considered=list(entry.alternatives_considered),
                    ),
                )
                for entry in submission.unresolved_alternatives
            ],
            rationale=submission.rationale,
        ),
    )
    return stable_json(payload)


def submission_sha256(submission: MapAvailabilityAuditSubmission) -> Sha256Digest:
    return Sha256Digest(
        hashlib.sha256(stable_submission_payload(submission).encode("utf-8")).hexdigest()
    )


def documented_public_labels() -> frozenset[str]:
    labels: set[str] = set()
    for _, (_, edge_labels, ton_labels) in TRANSFER_ONTOLOGY.items():
        labels.update(str(label) for label in edge_labels)
        labels.update(str(label) for label in ton_labels)
    return frozenset(labels)


class SubmissionValidationFailure(StrEnum):
    ELAPSED_TIME_EXCEEDED = "elapsed_time_exceeded"
    UNPARSEABLE_TIMESTAMPS = "unparseable_timestamps"
    UNDOCUMENTED_PUBLIC_LABEL = "undocumented_public_label"
    ORACLE_ARTIFACT_REFERENCED = "oracle_artifact_referenced"
    DUPLICATE_RESEARCHER_ID = "duplicate_researcher_id"


def validate_submission(
    submission: MapAvailabilityAuditSubmission,
    minutes_limit: DurationMinutes,
    public_labels: frozenset[str],
) -> tuple[SubmissionValidationFailure, ...]:
    failures: list[SubmissionValidationFailure] = []
    try:
        started = datetime.fromisoformat(submission.session_start_utc.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(submission.session_end_utc.replace("Z", "+00:00"))
        elapsed_minutes = (ended - started) / timedelta(minutes=1)
        if elapsed_minutes > minutes_limit or elapsed_minutes < 0:
            failures.append(SubmissionValidationFailure.ELAPSED_TIME_EXCEEDED)
    except ValueError:
        failures.append(SubmissionValidationFailure.UNPARSEABLE_TIMESTAMPS)
    for entry in submission.proposed_mapping:
        if (
            entry.source_public_label not in public_labels
            or entry.target_public_label not in public_labels
        ):
            failures.append(SubmissionValidationFailure.UNDOCUMENTED_PUBLIC_LABEL)
            break
    oracle_tokens = {concept.value.casefold() for concept in OracleTransferConcept}
    for resource in submission.resources_consulted:
        if (
            any(token in resource.casefold() for token in oracle_tokens)
            or "oracle" in resource.casefold()
        ):
            failures.append(SubmissionValidationFailure.ORACLE_ARTIFACT_REFERENCED)
            break
    return tuple(failures)


def distinct_researcher_ids(
    submissions: tuple[MapAvailabilityAuditSubmission, ...],
) -> bool:
    return len({submission.researcher_id for submission in submissions}) == len(submissions)


def _public_label_to_oracle_concept(
    edge_or_ton_label: str,
) -> OracleTransferConcept | None:
    for concept, (_, edge_labels, ton_labels) in TRANSFER_ONTOLOGY.items():
        if edge_or_ton_label in {str(label) for label in edge_labels} or edge_or_ton_label in {
            str(label) for label in ton_labels
        }:
            return concept
    return None


def submission_is_complete_one_to_one(
    submission: MapAvailabilityAuditSubmission, required_concept_count: ConceptCount
) -> bool:
    if submission.unresolved_alternatives:
        return False
    if len(submission.proposed_mapping) != required_concept_count:
        return False
    sources = {entry.source_public_label for entry in submission.proposed_mapping}
    targets = {entry.target_public_label for entry in submission.proposed_mapping}
    return len(sources) == len(submission.proposed_mapping) and len(targets) == len(
        submission.proposed_mapping
    )


def oracle_correspondence_accuracy(submission: MapAvailabilityAuditSubmission) -> float:
    if not submission.proposed_mapping:
        return 0.0
    correct = 0
    for entry in submission.proposed_mapping:
        source_concept = _public_label_to_oracle_concept(entry.source_public_label)
        target_concept = _public_label_to_oracle_concept(entry.target_public_label)
        if source_concept is not None and source_concept == target_concept:
            correct += 1
    return correct / len(submission.proposed_mapping)
