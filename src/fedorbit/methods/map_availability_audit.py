from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import cast

from pydantic import JsonValue

from fedorbit.datasets.ontology import TRANSFER_ONTOLOGY
from fedorbit.types import (
    AuditLabel,
    AuditResourceText,
    ConceptCount,
    DatasetId,
    DirectedPair,
    DirectedPairName,
    DomainModel,
    DurationMinutes,
    ExposedCoarseGroupId,
    FieldDescription,
    Fraction,
    Index,
    OracleTransferConcept,
    ResearcherIdentifier,
    Rfc3339UtcTimestamp,
    Sha256Digest,
    StableJsonPayload,
    directed_pair_name,
    stable_json,
)


class ProposedMappingEntry(DomainModel):
    source_public_label: AuditLabel
    target_public_label: AuditLabel
    exposed_coarse_group: ExposedCoarseGroupId


class UnresolvedAlternativeEntry(DomainModel):
    affected_public_label: AuditLabel
    alternatives_considered: tuple[AuditLabel, ...]


class MapAvailabilityAuditSubmission(DomainModel):
    researcher_id: ResearcherIdentifier
    directed_pair: DirectedPair
    session_start_utc: Rfc3339UtcTimestamp
    session_end_utc: Rfc3339UtcTimestamp
    resources_consulted: tuple[AuditResourceText, ...]
    proposed_mapping: tuple[ProposedMappingEntry, ...]
    unresolved_alternatives: tuple[UnresolvedAlternativeEntry, ...]
    rationale: FieldDescription


_BLANK_SESSION_TIMESTAMP = Rfc3339UtcTimestamp("1970-01-01T00:00:00Z")


def blank_audit_template(
    researcher_id: ResearcherIdentifier,
    directed_pair: DirectedPair,
) -> MapAvailabilityAuditSubmission:
    return MapAvailabilityAuditSubmission(
        researcher_id=researcher_id,
        directed_pair=directed_pair,
        session_start_utc=_BLANK_SESSION_TIMESTAMP,
        session_end_utc=_BLANK_SESSION_TIMESTAMP,
        resources_consulted=(),
        proposed_mapping=(),
        unresolved_alternatives=(),
        rationale=FieldDescription(""),
    )


def stable_submission_payload(
    submission: MapAvailabilityAuditSubmission,
) -> str:
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


def documented_public_labels() -> frozenset[AuditLabel]:
    labels: set[AuditLabel] = set()
    for _, (_, edge_labels, ton_labels) in TRANSFER_ONTOLOGY.items():
        labels.update(edge_labels)
        labels.update(ton_labels)
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
    public_labels: frozenset[AuditLabel],
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


def _public_label_to_oracle_concept(
    edge_or_ton_label: AuditLabel,
) -> OracleTransferConcept | None:
    for concept, (_, edge_labels, ton_labels) in TRANSFER_ONTOLOGY.items():
        if edge_or_ton_label in edge_labels or edge_or_ton_label in ton_labels:
            return concept
    return None


def oracle_correspondence_accuracy(
    submission: MapAvailabilityAuditSubmission,
) -> Fraction:
    if not submission.proposed_mapping:
        return 0.0
    correct = 0
    for entry in submission.proposed_mapping:
        source_concept = _public_label_to_oracle_concept(entry.source_public_label)
        target_concept = _public_label_to_oracle_concept(entry.target_public_label)
        if source_concept is not None and source_concept == target_concept:
            correct += 1
    return correct / len(submission.proposed_mapping)


def required_concept_count() -> ConceptCount:
    count: ConceptCount = len(TRANSFER_ONTOLOGY)
    return count


class MapAvailabilityAuditOutcome(StrEnum):
    TRIVIAL_TO_RECONSTRUCT = "Trivial To Reconstruct"
    NOT_DEMONSTRATED_TRIVIAL = "Not Demonstrated Trivial"
    BLOCKED_HUMAN_INPUT_REQUIRED = "Blocked / Human Input Required"


class MapAvailabilityAuditFileName(StrEnum):
    RECORDED_SUBMISSION_SHA256 = "recorded.sha256.json"
    PAIR_OUTCOME = "pair_outcome.json"
    EXPERIMENT_SUMMARY = "human_audit_summary.json"


class MapAvailabilityAuditPairOutcome(DomainModel):
    source: DatasetId
    target: DatasetId
    outcome: MapAvailabilityAuditOutcome
    reason: FieldDescription
    required_researcher_count: Index
    researcher_ids: tuple[ResearcherIdentifier, ...]
    recorded_submission_sha256: tuple[Sha256Digest, ...]
    complete_one_to_one: tuple[bool, ...]
    unresolved_alternative_count: tuple[Index, ...]
    oracle_correspondence_accuracy: tuple[Fraction, ...]


class MapAvailabilityAuditSummary(DomainModel):
    required_researcher_count: Index
    evaluated_pair_count: Index
    trivial_pairs: tuple[DirectedPairName, ...]
    not_demonstrated_pairs: tuple[DirectedPairName, ...]
    blocked_pairs: tuple[DirectedPairName, ...]
    practical_motivation_kill_rule_fires: bool
    wording_restricted_to_pairs: tuple[DirectedPairName, ...]
    practical_motivation_scope: FieldDescription


def summarise_map_availability_outcomes(
    required_researcher_count: Index,
    outcomes: tuple[MapAvailabilityAuditPairOutcome, ...],
) -> MapAvailabilityAuditSummary:
    trivial = tuple(
        _outcome_pair(outcome)
        for outcome in outcomes
        if outcome.outcome is MapAvailabilityAuditOutcome.TRIVIAL_TO_RECONSTRUCT
    )
    demonstrated_not_trivial = tuple(
        _outcome_pair(outcome)
        for outcome in outcomes
        if outcome.outcome is MapAvailabilityAuditOutcome.NOT_DEMONSTRATED_TRIVIAL
    )
    blocked = tuple(
        _outcome_pair(outcome)
        for outcome in outcomes
        if outcome.outcome is MapAvailabilityAuditOutcome.BLOCKED_HUMAN_INPUT_REQUIRED
    )
    kill_rule_fires = bool(outcomes) and not demonstrated_not_trivial and not blocked
    restricted = demonstrated_not_trivial + blocked
    if kill_rule_fires:
        scope = FieldDescription(
            "every evaluated primary pair is trivial to reconstruct; practical unresolved-map "
            "motivation is not supported"
        )
    elif demonstrated_not_trivial and trivial:
        scope = FieldDescription(
            "practical unresolved-map wording is restricted to the pairs that are not "
            "demonstrated trivial"
        )
    elif demonstrated_not_trivial:
        scope = FieldDescription(
            "practical unresolved-map wording covers the registered primary pairs within "
            "measured evidence"
        )
    else:
        scope = FieldDescription(
            "human-audit submissions are incomplete; benchmark-wide natural-unavailability "
            "wording remains forbidden"
        )
    return MapAvailabilityAuditSummary(
        required_researcher_count=required_researcher_count,
        evaluated_pair_count=len(outcomes),
        trivial_pairs=trivial,
        not_demonstrated_pairs=demonstrated_not_trivial,
        blocked_pairs=blocked,
        practical_motivation_kill_rule_fires=kill_rule_fires,
        wording_restricted_to_pairs=restricted,
        practical_motivation_scope=scope,
    )


def _outcome_pair(outcome: MapAvailabilityAuditPairOutcome) -> DirectedPairName:
    return directed_pair_name(outcome.source, outcome.target)


@dataclass(frozen=True, slots=True)
class MapAvailabilityAuditPairDecision:
    directed_pair: DirectedPair
    outcome: MapAvailabilityAuditOutcome
    reason: FieldDescription
    required_researcher_count: Index
    researcher_ids: tuple[ResearcherIdentifier, ...]
    recorded_submission_sha256: tuple[Sha256Digest, ...]
    complete_one_to_one: tuple[bool, ...]
    unresolved_alternative_count: tuple[Index, ...]
    oracle_correspondence_accuracy: tuple[Fraction, ...]

    def record(self) -> MapAvailabilityAuditPairOutcome:
        return MapAvailabilityAuditPairOutcome(
            source=self.directed_pair.source,
            target=self.directed_pair.target,
            outcome=self.outcome,
            reason=self.reason,
            required_researcher_count=self.required_researcher_count,
            researcher_ids=self.researcher_ids,
            recorded_submission_sha256=self.recorded_submission_sha256,
            complete_one_to_one=self.complete_one_to_one,
            unresolved_alternative_count=self.unresolved_alternative_count,
            oracle_correspondence_accuracy=self.oracle_correspondence_accuracy,
        )

    def payload(self) -> Mapping[str, JsonValue]:
        return cast(
            Mapping[str, JsonValue],
            OrderedDict(
                (
                    ("source", self.directed_pair.source.value),
                    ("target", self.directed_pair.target.value),
                    ("outcome", self.outcome.value),
                    ("reason", self.reason),
                    ("required_researcher_count", self.required_researcher_count),
                    ("researcher_ids", list(self.researcher_ids)),
                    (
                        "recorded_submission_sha256",
                        list(self.recorded_submission_sha256),
                    ),
                    ("complete_one_to_one", list(self.complete_one_to_one)),
                    (
                        "unresolved_alternative_count",
                        list(self.unresolved_alternative_count),
                    ),
                    (
                        "oracle_correspondence_accuracy",
                        list(self.oracle_correspondence_accuracy),
                    ),
                )
            ),
        )


def accepted_map_availability_submissions(
    submissions: tuple[MapAvailabilityAuditSubmission, ...],
    minutes_limit: DurationMinutes,
    public_labels: frozenset[AuditLabel],
) -> tuple[MapAvailabilityAuditSubmission, ...]:
    return tuple(
        sorted(
            (
                submission
                for submission in submissions
                if not validate_submission(submission, minutes_limit, public_labels)
            ),
            key=lambda submission: submission.researcher_id,
        )
    )


def decide_map_availability_outcome(
    directed_pair: DirectedPair,
    submissions: tuple[MapAvailabilityAuditSubmission, ...],
    required_researcher_count: Index,
    minutes_limit: DurationMinutes,
    public_labels: frozenset[AuditLabel],
) -> MapAvailabilityAuditPairDecision:
    accepted = accepted_map_availability_submissions(submissions, minutes_limit, public_labels)
    if len(accepted) != required_researcher_count or not distinct_researcher_ids(accepted):
        return MapAvailabilityAuditPairDecision(
            directed_pair=directed_pair,
            outcome=MapAvailabilityAuditOutcome.BLOCKED_HUMAN_INPUT_REQUIRED,
            reason=FieldDescription(
                "required independent researcher submissions are missing, invalid, or duplicated"
            ),
            required_researcher_count=required_researcher_count,
            researcher_ids=(),
            recorded_submission_sha256=(),
            complete_one_to_one=(),
            unresolved_alternative_count=(),
            oracle_correspondence_accuracy=(),
        )
    concept_count = required_concept_count()
    recorded_sha256 = tuple(submission_sha256(submission) for submission in accepted)
    complete_one_to_one = tuple(
        submission_is_complete_one_to_one(submission, concept_count) for submission in accepted
    )
    unresolved_counts = tuple(len(submission.unresolved_alternatives) for submission in accepted)
    accuracies = tuple(oracle_correspondence_accuracy(submission) for submission in accepted)
    trivial = (
        all(complete_one_to_one)
        and all(count == 0 for count in unresolved_counts)
        and all(accuracy == 1.0 for accuracy in accuracies)
    )
    return MapAvailabilityAuditPairDecision(
        directed_pair=directed_pair,
        outcome=(
            MapAvailabilityAuditOutcome.TRIVIAL_TO_RECONSTRUCT
            if trivial
            else MapAvailabilityAuditOutcome.NOT_DEMONSTRATED_TRIVIAL
        ),
        reason=FieldDescription(
            "both researchers reconstructed the complete public map within the timed session"
            if trivial
            else "at least one researcher did not reconstruct the exact complete public map"
        ),
        required_researcher_count=required_researcher_count,
        researcher_ids=tuple(submission.researcher_id for submission in accepted),
        recorded_submission_sha256=recorded_sha256,
        complete_one_to_one=complete_one_to_one,
        unresolved_alternative_count=unresolved_counts,
        oracle_correspondence_accuracy=accuracies,
    )
