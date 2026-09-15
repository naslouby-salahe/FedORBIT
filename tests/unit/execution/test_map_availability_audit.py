from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from fedorbit.config.loading import active_config
from fedorbit.datasets.ontology import TRANSFER_ONTOLOGY
from fedorbit.experiments import audit as experiment_audit
from fedorbit.experiments.audit import (
    RECOVERY_METHOD_ATTEMPTS,
    PacketOnlyRecoveryUnavailability,
    completed_human_audit_outcomes,
    execute_map_availability_applicability_audit,
    registered_recovery_method,
)
from fedorbit.experiments.catalogue import build_catalogue
from fedorbit.experiments.dispatch import ExperimentExecutionRequest
from fedorbit.experiments.synthesis import completed_experiment_metric_records
from fedorbit.infrastructure.artifacts import ArtifactStore, ExecutionError
from fedorbit.infrastructure.workspace import build_layout
from fedorbit.methods.map_availability_audit import (
    MapAvailabilityAuditFileName,
    MapAvailabilityAuditOutcome,
    MapAvailabilityAuditSubmission,
    ProposedMappingEntry,
    UnresolvedAlternativeEntry,
    blank_audit_template,
    decide_map_availability_outcome,
    documented_public_labels,
    required_concept_count,
    submission_sha256,
)
from fedorbit.types import (
    ArtifactState,
    DatasetId,
    DirectedPair,
    ExperimentName,
    ExposedCoarseGroupId,
    FieldDescription,
    HumanAuditFileName,
    MethodName,
    MetricId,
    OracleTransferConcept,
    OverwritePolicy,
    ResearcherIdentifier,
    Rfc3339UtcTimestamp,
    TransferMethod,
)


def _request(experiment: ExperimentName) -> ExperimentExecutionRequest:
    catalogue = build_catalogue()
    return ExperimentExecutionRequest(
        experiment=experiment,
        definition=catalogue.definition(experiment),
        overwrite_policy=OverwritePolicy.REPLACE,
    )


def _submission(
    researcher_id: str,
    pair: DirectedPair,
    mapping: tuple[ProposedMappingEntry, ...],
    unresolved: int = 0,
    elapsed_minutes: int = 60,
) -> MapAvailabilityAuditSubmission:
    started = "2026-01-01T00:00:00Z"
    ended = f"2026-01-01T{elapsed_minutes // 60:02d}:{elapsed_minutes % 60:02d}:00Z"
    return MapAvailabilityAuditSubmission(
        researcher_id=ResearcherIdentifier(researcher_id),
        directed_pair=pair,
        session_start_utc=Rfc3339UtcTimestamp(started),
        session_end_utc=Rfc3339UtcTimestamp(ended),
        resources_consulted=(),
        proposed_mapping=mapping,
        unresolved_alternatives=tuple(
            UnresolvedAlternativeEntry(
                affected_public_label=mapping[0].source_public_label,
                alternatives_considered=(mapping[0].target_public_label,),
            )
            for _ in range(unresolved)
        ),
        rationale=FieldDescription("fixture rationale"),
    )


def _exact_mapping() -> tuple[ProposedMappingEntry, ...]:
    entries: list[ProposedMappingEntry] = []
    for concept, ontology_entry in TRANSFER_ONTOLOGY.items():
        coarse_group, _edge_labels, ton_labels = ontology_entry
        assert concept in set(OracleTransferConcept)
        entries.append(
            ProposedMappingEntry(
                source_public_label=ton_labels[0],
                target_public_label=ton_labels[0],
                exposed_coarse_group=ExposedCoarseGroupId(coarse_group.value),
            )
        )
    return tuple(entries)


def test_every_configured_recovery_method_has_a_certified_implementation() -> None:
    configured = tuple(
        active_config().experiments.map_availability_applicability_audit.packet_only_recovery_methods
    )
    assert configured
    for method in configured:
        assert registered_recovery_method(method) in RECOVERY_METHOD_ATTEMPTS
    assert TransferMethod.GENERIC_EXACT_QAP in RECOVERY_METHOD_ATTEMPTS
    with pytest.raises(ExecutionError):
        registered_recovery_method(cast(MethodName, "not-a-registered-method"))


def test_audit_persists_a_typed_state_for_every_configured_recovery_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT)
    configured = tuple(
        active_config().experiments.map_availability_applicability_audit.packet_only_recovery_methods
    )
    seeds = active_config().scientific.randomness.confirmatory_seeds
    pairs = active_config().scientific.datasets.primary_directed_pairs

    def unavailable(*_arguments: object, **_keywords: object) -> None:
        from fedorbit.datasets.materialization import MaterializationError

        raise MaterializationError("fixture materialization is unavailable")

    monkeypatch.setattr(experiment_audit, "load_or_materialize_client", unavailable)
    execute_map_availability_applicability_audit(store, layout, request)

    records = tuple(
        record
        for record in completed_experiment_metric_records(
            store, ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT
        )
        if record.metric_name is MetricId.PACKET_ONLY_RECOVERY_ACCURACY
    )
    assert len(records) == len(pairs) * len(seeds) * len(configured)
    assert {record.method for record in records} == set(configured)
    for record in records:
        assert not record.valid
        assert record.metric_value is None
        assert (
            record.invalid_reason
            == PacketOnlyRecoveryUnavailability.PAIR_ENDPOINT_UNAVAILABLE.value
        )
    for pair in pairs:
        for seed in seeds:
            for method in configured:
                assert any(
                    record.pair == f"{pair.source.value} -> {pair.target.value}"
                    and record.seed == seed
                    and record.method == method
                    for record in records
                )
    for pair in pairs:
        endpoint = experiment_audit.human_audit_pair_directory(
            layout, request.experiment, pair.source, pair.target
        )
        outcome_path = endpoint / MapAvailabilityAuditFileName.PAIR_OUTCOME
        assert outcome_path.is_file()
        payload = json.loads(outcome_path.read_text(encoding="utf-8"))
        assert payload["outcome"] == MapAvailabilityAuditOutcome.BLOCKED_HUMAN_INPUT_REQUIRED.value
        assert payload["recorded_submission_sha256"] == []
        assert not (endpoint / MapAvailabilityAuditFileName.RECORDED_SUBMISSION_SHA256).is_file()
        for researcher in ("researcher-1", "researcher-2"):
            assert (endpoint / researcher / HumanAuditFileName.TEMPLATE).is_file()
    for manifest in store.all_manifests():
        assert store.resolve(manifest.artifact_id).state == ArtifactState.COMPLETED


def test_human_audit_outcome_requires_exact_complete_independent_submissions() -> None:
    pair = DirectedPair(source=DatasetId.TON_IOT_WINDOWS10_HOST, target=DatasetId.TON_IOT_NETWORK)
    minutes = 60
    labels = documented_public_labels()
    complete = _exact_mapping()
    assert len(complete) == required_concept_count()

    blocked = decide_map_availability_outcome(pair, (), 2, minutes, labels)
    assert blocked.outcome is MapAvailabilityAuditOutcome.BLOCKED_HUMAN_INPUT_REQUIRED
    assert blocked.recorded_submission_sha256 == ()

    trivial = decide_map_availability_outcome(
        pair,
        (
            _submission("researcher-1", pair, complete),
            _submission("researcher-2", pair, complete),
        ),
        2,
        minutes,
        labels,
    )
    assert trivial.outcome is MapAvailabilityAuditOutcome.TRIVIAL_TO_RECONSTRUCT
    assert len(trivial.recorded_submission_sha256) == 2
    assert trivial.oracle_correspondence_accuracy == (1.0, 1.0)

    ambiguous = decide_map_availability_outcome(
        pair,
        (
            _submission("researcher-1", pair, complete, unresolved=1),
            _submission("researcher-2", pair, complete),
        ),
        2,
        minutes,
        labels,
    )
    assert ambiguous.outcome is MapAvailabilityAuditOutcome.NOT_DEMONSTRATED_TRIVIAL

    over_time = decide_map_availability_outcome(
        pair,
        (
            _submission("researcher-1", pair, complete, elapsed_minutes=61),
            _submission("researcher-2", pair, complete),
        ),
        2,
        minutes,
        labels,
    )
    assert over_time.outcome is MapAvailabilityAuditOutcome.BLOCKED_HUMAN_INPUT_REQUIRED

    duplicated = decide_map_availability_outcome(
        pair,
        (
            _submission("researcher-1", pair, complete),
            _submission("researcher-1", pair, complete),
        ),
        2,
        minutes,
        labels,
    )
    assert duplicated.outcome is MapAvailabilityAuditOutcome.BLOCKED_HUMAN_INPUT_REQUIRED


def test_human_audit_payload_records_submission_digests_before_oracle_comparison() -> None:
    pair = DirectedPair(source=DatasetId.TON_IOT_WINDOWS10_HOST, target=DatasetId.TON_IOT_NETWORK)
    submissions = (
        _submission("researcher-1", pair, _exact_mapping()),
        _submission("researcher-2", pair, _exact_mapping()),
    )
    decision = decide_map_availability_outcome(pair, submissions, 2, 60, documented_public_labels())
    payload = decision.record().model_dump(mode="json")
    keys = list(payload)
    assert keys.index("recorded_submission_sha256") < keys.index("oracle_correspondence_accuracy")
    assert cast(list[str], payload["recorded_submission_sha256"]) == [
        submission_sha256(submission) for submission in submissions
    ]
    assert payload["outcome"] == MapAvailabilityAuditOutcome.TRIVIAL_TO_RECONSTRUCT.value
    assert json.loads(json.dumps(payload)) == payload


def test_audit_records_submission_digests_before_the_oracle_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = build_layout(root=tmp_path)
    store = ArtifactStore(layout.execution_root)
    request = _request(ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT)
    pairs = active_config().scientific.datasets.primary_directed_pairs

    def unavailable(*_arguments: object, **_keywords: object) -> None:
        from fedorbit.datasets.materialization import MaterializationError

        raise MaterializationError("fixture materialization is unavailable")

    monkeypatch.setattr(experiment_audit, "load_or_materialize_client", unavailable)
    for pair in pairs:
        for researcher in ("researcher-1", "researcher-2"):
            directory = experiment_audit.human_audit_researcher_directory(
                layout,
                request.experiment,
                pair.source,
                pair.target,
                ResearcherIdentifier(researcher),
            )
            directory.mkdir(parents=True, exist_ok=True)
            template = blank_audit_template(
                ResearcherIdentifier(researcher),
                DirectedPair(source=pair.source, target=pair.target),
            )
            (directory / HumanAuditFileName.SUBMISSION).write_text(
                template.model_dump_json(), encoding="utf-8"
            )
    execute_map_availability_applicability_audit(store, layout, request)
    for pair in pairs:
        endpoint = experiment_audit.human_audit_pair_directory(
            layout, request.experiment, pair.source, pair.target
        )
        recorded_path = endpoint / MapAvailabilityAuditFileName.RECORDED_SUBMISSION_SHA256
        outcome_path = endpoint / MapAvailabilityAuditFileName.PAIR_OUTCOME
        assert recorded_path.is_file()
        recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
        assert set(recorded["recorded_submission_sha256"]) == {"researcher-1", "researcher-2"}
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        assert outcome["outcome"] == MapAvailabilityAuditOutcome.NOT_DEMONSTRATED_TRIVIAL.value
        assert len(outcome["recorded_submission_sha256"]) == 2
        assert outcome["complete_one_to_one"] == [False, False]
        for researcher in ("researcher-1", "researcher-2"):
            assert (endpoint / researcher / HumanAuditFileName.VALIDATED_SHA256).is_file()
    outcomes = completed_human_audit_outcomes(layout, request.experiment)
    assert len(outcomes) == len(pairs)
    assert {outcome.outcome for outcome in outcomes} == {
        MapAvailabilityAuditOutcome.NOT_DEMONSTRATED_TRIVIAL
    }
    for outcome in outcomes:
        assert outcome.required_researcher_count == 2
        assert outcome.researcher_ids == ("researcher-1", "researcher-2")
        assert outcome.complete_one_to_one == (False, False)
        assert len(outcome.recorded_submission_sha256) == 2
