from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from fedorbit.infrastructure.environment import environment_snapshot
from fedorbit.infrastructure.manifests import (
    ArtifactProvenance,
    CompletionManifest,
    ReusableArtifactManifest,
    artifact_id,
    build_artifact_completion,
    recorded_experiment,
)
from fedorbit.infrastructure.provenance import (
    artifact_family_registration,
    artifact_payload_fingerprint,
    configuration_subset_digest,
    implementation_fingerprint,
    subsystem_stage,
    validate_artifact_family_registry,
)
from fedorbit.infrastructure.reuse import (
    ArtifactValidationError,
    normalize_completion_lineage,
    recorded_lineage,
    validate_artifact_lineage,
    validate_completed_artifact,
    validate_completion_manifest,
    validate_payload_checksum,
    validate_reusable_artifact,
    validate_stage_lineage_completeness,
)
from fedorbit.infrastructure.runtime import (
    EfficiencyMeasurement,
    ExecutionEventLog,
    ExecutionLogEvent,
    ExecutionLogger,
    InfrastructureEventName,
    ResourcePhaseObservation,
    current_code_revision,
    execution_logger,
    measure_efficiency,
)
from fedorbit.infrastructure.storage import (
    StorageError,
    atomic_write_json,
    payload_sha256,
    serialized_payload_bytes,
    stage_bytes,
    verify_payload_bytes,
)
from fedorbit.infrastructure.workspace import (
    ExperimentLogDirectory,
    build_layout,
    experiment_event_log_path,
)
from fedorbit.types import (
    ArtifactCacheMissReason,
    ArtifactFingerprint,
    ArtifactIdentifier,
    ArtifactIdentifiers,
    ArtifactPath,
    ArtifactSchemaVersion,
    ArtifactStage,
    ArtifactState,
    ArtifactStoreFileName,
    ArtifactType,
    ConfigurationSection,
    ExecutionEventName,
    FieldDescription,
    OverwritePolicy,
    ScientificSubsystem,
    SemanticCoordinates,
    SemanticCoordinateText,
    SerializedPacket,
    Sha256Digest,
    StableJsonPayload,
    StorageLayoutSegment,
)


class ExecutionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactDependencyEdge:
    parent: ArtifactIdentifier
    descendant: ArtifactIdentifier


@dataclass(frozen=True, slots=True)
class ArtifactStateReport:
    artifact_id: ArtifactIdentifier
    state: ArtifactState
    reason: FieldDescription
    first_changed_dependency: ArtifactIdentifier | None
    nearest_reusable_ancestor: ArtifactIdentifier | None


@dataclass(frozen=True, slots=True)
class StaleDescendantReport:
    superseded: ArtifactIdentifier
    replacement: ArtifactIdentifier
    identity_preserved: bool
    stale_descendants: tuple[ArtifactIdentifier, ...]
    recompute_boundary: ArtifactIdentifier | None
    nearest_reusable_ancestor: ArtifactIdentifier | None


@dataclass(frozen=True, slots=True)
class StagedPayload:
    staged_path: Path
    active_path: Path


@dataclass(frozen=True, slots=True)
class ArtifactAccessContext:
    subsystem: ScientificSubsystem | None = None
    artifacts_read: ArtifactIdentifiers = ()
    provenance: ArtifactProvenance | None = None


@dataclass(frozen=True, slots=True)
class ArtifactPayloadRequest:
    artifact_type: ArtifactType
    coordinates: StableJsonPayload
    semantic_coordinates: SemanticCoordinateText
    stage: ArtifactStage
    payload_path: ArtifactPath
    serialized: SerializedPacket
    upstream_artifact_ids: ArtifactIdentifiers = ()
    configuration_sections: frozenset[ConfigurationSection] = frozenset()
    access: ArtifactAccessContext | None = None


def consumed_artifact_ids(
    manifests: Sequence[ReusableArtifactManifest],
) -> frozenset[ArtifactIdentifier]:
    consumed: set[ArtifactIdentifier] = set()
    for manifest in manifests:
        consumed.update(manifest.upstream_artifact_ids)
    return frozenset(consumed)


def unconsumed_artifact_ids(
    manifests: Sequence[ReusableArtifactManifest],
) -> tuple[ArtifactIdentifier, ...]:
    consumed = consumed_artifact_ids(manifests)
    return tuple(
        manifest.artifact_id
        for manifest in sorted(manifests, key=lambda entry: entry.artifact_id.value)
        if manifest.artifact_id not in consumed
    )


def verify_registered_artifact(
    store: ArtifactStore,
    artifact_id_value: ArtifactIdentifier,
) -> ReusableArtifactManifest:
    return store.resolve(artifact_id_value)


@dataclass(frozen=True, slots=True)
class ArtifactDependencyGraph:
    edges: tuple[ArtifactDependencyEdge, ...]

    @classmethod
    def from_manifests(
        cls,
        manifests: Sequence[ReusableArtifactManifest],
    ) -> ArtifactDependencyGraph:
        known = {manifest.artifact_id for manifest in manifests}
        edges: list[ArtifactDependencyEdge] = []
        for manifest in sorted(manifests, key=lambda entry: entry.artifact_id.value):
            for upstream in manifest.upstream_artifact_ids:
                if upstream in known:
                    edges.append(ArtifactDependencyEdge(upstream, manifest.artifact_id))
        return cls(tuple(edges))

    def ancestors(self, artifact_id: ArtifactIdentifier) -> tuple[ArtifactIdentifier, ...]:
        reached: OrderedDict[ArtifactIdentifier, None] = OrderedDict()
        pending = [artifact_id]
        while pending:
            current = pending.pop()
            for edge in self.edges:
                if edge.descendant != current or edge.parent in reached:
                    continue
                reached[edge.parent] = None
                pending.append(edge.parent)
        return tuple(reached)

    def descendants(self, artifact_id: ArtifactIdentifier) -> tuple[ArtifactIdentifier, ...]:
        reached: OrderedDict[ArtifactIdentifier, None] = OrderedDict()
        pending = [artifact_id]
        while pending:
            current = pending.pop()
            for edge in self.edges:
                if edge.parent != current or edge.descendant in reached:
                    continue
                reached[edge.descendant] = None
                pending.append(edge.descendant)
        return tuple(sorted(reached, key=lambda identifier: identifier.value))

    def distances_from(
        self,
        artifact_id: ArtifactIdentifier,
    ) -> Mapping[ArtifactIdentifier, int]:
        distance: dict[ArtifactIdentifier, int] = {artifact_id: 0}
        pending: list[ArtifactIdentifier] = [artifact_id]
        while pending:
            current = pending.pop(0)
            for edge in self.edges:
                if edge.descendant != current:
                    continue
                candidate = distance[current] + 1
                if candidate < distance.get(edge.parent, candidate + 1):
                    distance[edge.parent] = candidate
                    pending.append(edge.parent)
        return distance

    def nearest_reusable_ancestor(
        self,
        artifact_id: ArtifactIdentifier,
        reusable: Mapping[ArtifactIdentifier, bool],
    ) -> ArtifactIdentifier | None:
        distance = self.distances_from(artifact_id)
        candidates = tuple(
            identifier
            for identifier, reach in distance.items()
            if reach > 0 and reusable.get(identifier, False)
        )
        if not candidates:
            return None
        return min(candidates, key=lambda identifier: (distance[identifier], identifier.value))


class ArtifactStore:
    def __init__(
        self,
        root: Path,
        staging_root: Path | None = None,
        logger: ExecutionLogger | None = None,
    ) -> None:
        self._root = root
        self._manifests = root / StorageLayoutSegment.MANIFESTS
        self._completions = root / StorageLayoutSegment.COMPLETIONS
        self._staging_root = staging_root
        self._logger = logger
        self._index_path = self._root / ArtifactStoreFileName.FINGERPRINT_INDEX
        self._manifest_cache: OrderedDict[ArtifactIdentifier, ReusableArtifactManifest] | None = (
            None
        )

    @property
    def root(self) -> Path:
        return self._root

    def manifest_path(self, artifact_id: ArtifactIdentifier) -> Path:
        return self._manifests / f"{artifact_id.value}.json"

    def completion_path(self, artifact_id: ArtifactIdentifier) -> Path:
        return self._completions / f"{artifact_id.value}.json"

    def staging_dir(self) -> Path:
        if self._staging_root is not None:
            return self._staging_root
        layout = build_layout()
        if layout.execution_root == self._root:
            return layout.staging
        return self._root / StorageLayoutSegment.STAGING

    def staging_area(self, artifact_id: ArtifactIdentifier) -> Path:
        return self.staging_dir() / artifact_id.value

    def discard_staging(self) -> None:
        staging = self.staging_dir()
        if staging.is_dir():
            shutil.rmtree(staging)

    def dependency_graph(self) -> ArtifactDependencyGraph:
        return ArtifactDependencyGraph.from_manifests(self.all_manifests())

    def descendants_of(
        self,
        artifact_id: ArtifactIdentifier,
    ) -> tuple[ArtifactIdentifier, ...]:
        return self.dependency_graph().descendants(artifact_id)

    def nearest_reusable_ancestor(
        self,
        artifact_id: ArtifactIdentifier,
    ) -> ArtifactIdentifier | None:
        graph = self.dependency_graph()
        candidates = graph.ancestors(artifact_id)
        if not candidates:
            return None
        return graph.nearest_reusable_ancestor(
            artifact_id,
            {identifier: self.is_reusable(identifier) for identifier in candidates},
        )

    def artifact_state(self, artifact_id: ArtifactIdentifier) -> ArtifactStateReport:
        report = self._resolve_state(artifact_id, frozenset())
        return replace(
            report,
            nearest_reusable_ancestor=self.nearest_reusable_ancestor(artifact_id),
        )

    def is_reusable(self, artifact_id: ArtifactIdentifier) -> bool:
        try:
            manifest = self.read_reusable(artifact_id)
        except ValueError:
            return False
        return self._is_reusable(manifest)

    def _is_reusable(self, manifest: ReusableArtifactManifest) -> bool:
        if (
            self._resolve_state(manifest.artifact_id, frozenset()).state
            is not ArtifactState.COMPLETED
        ):
            return False
        if not manifest.completion_required:
            return True
        try:
            validate_stage_lineage_completeness(
                manifest, self.read_completion(manifest.artifact_id)
            )
        except (StorageError, ArtifactValidationError):
            return False
        return True

    def _resolve_state(
        self,
        artifact_id: ArtifactIdentifier,
        visiting: frozenset[ArtifactIdentifier],
    ) -> ArtifactStateReport:
        def report(
            state: ArtifactState,
            reason: str,
            changed: ArtifactIdentifier | None = None,
        ) -> ArtifactStateReport:
            return ArtifactStateReport(artifact_id, state, FieldDescription(reason), changed, None)

        if artifact_id in visiting:
            return report(ArtifactState.STALE, "upstream dependency identity forms a cycle")
        if not self.manifest_path(artifact_id).is_file():
            return report(ArtifactState.MISSING, "no reusable artifact manifest is recorded")
        try:
            manifest = self.read_reusable(artifact_id)
        except ValueError as error:
            return report(ArtifactState.INVALID, str(error))
        if manifest.state is not ArtifactState.COMPLETED:
            return report(manifest.state, f"recorded artifact state is {manifest.state.value}")
        if manifest.completion_required:
            try:
                validate_completion_manifest(self.read_completion(artifact_id))
            except (StorageError, ArtifactValidationError) as error:
                return report(ArtifactState.INVALID, str(error))
        marked = visiting | {artifact_id}
        for upstream in manifest.upstream_artifact_ids:
            upstream_state = self._resolve_state(upstream, marked)
            if upstream_state.state is not ArtifactState.COMPLETED:
                return report(
                    ArtifactState.STALE,
                    "a recorded upstream dependency identity is no longer reusable",
                    upstream,
                )
        try:
            validate_payload_checksum(manifest)
        except ArtifactValidationError as error:
            return report(ArtifactState.INVALID, str(error))
        return report(ArtifactState.COMPLETED, "completed and reusable")

    def supersession_report(
        self,
        superseded: ArtifactIdentifier,
        replacement: ArtifactIdentifier,
    ) -> StaleDescendantReport:
        if superseded == replacement:
            return StaleDescendantReport(superseded, replacement, True, (), None, None)
        stale = tuple(
            identifier
            for identifier in self.descendants_of(superseded)
            if identifier != replacement
        )
        if not stale:
            return StaleDescendantReport(superseded, replacement, False, (), None, None)
        graph = self.dependency_graph()
        stale_set = set(stale)
        roots = tuple(
            identifier for identifier in stale if not (set(graph.ancestors(identifier)) & stale_set)
        )
        boundary = min(roots or stale, key=lambda entry: entry.value)
        return StaleDescendantReport(
            superseded,
            replacement,
            False,
            stale,
            boundary,
            self.nearest_reusable_ancestor(boundary),
        )

    def retire_superseded_parent(
        self,
        superseded: ArtifactIdentifier,
        replacement: ArtifactIdentifier,
    ) -> tuple[ArtifactIdentifier, ...]:
        report = self.supersession_report(superseded, replacement)
        if report.identity_preserved:
            return ()
        for stale_artifact_id in report.stale_descendants:
            self._retire(stale_artifact_id)
        if report.stale_descendants:
            self._logger_for().warning(
                FieldDescription(
                    f"superseded parent {superseded.value} retired "
                    f"{len(report.stale_descendants)} stale descendant(s)"
                )
            )
            self._logger_for().event(
                InfrastructureEventName.ARTIFACT_RETIRED,
                superseded=superseded.value,
                replacement=replacement.value,
                stale_descendants=[identifier.value for identifier in report.stale_descendants],
                recompute_boundary=(
                    report.recompute_boundary.value
                    if report.recompute_boundary is not None
                    else None
                ),
            )
        return report.stale_descendants

    def promote_artifact(
        self,
        manifest: ReusableArtifactManifest,
        completion: CompletionManifest,
        staged_payloads: tuple[StagedPayload, ...] = (),
        supersedes: ArtifactIdentifier | None = None,
        access: ArtifactAccessContext | None = None,
    ) -> ReusableArtifactManifest:
        if not manifest.completion_required:
            raise StorageError("completed artifacts must require a completion record")
        validate_artifact_family_registry()
        if access is not None and access.subsystem is not None:
            declared_stage = subsystem_stage(access.subsystem)
            if declared_stage is not manifest.producer_stage:
                raise StorageError(
                    f"scientific subsystem {access.subsystem.value} produces stage "
                    f"{declared_stage.value}, not {manifest.producer_stage.value}"
                )
        provenance = access.provenance if access is not None else None
        try:
            completion = normalize_completion_lineage(manifest, completion, provenance)
        except ValueError as error:
            raise StorageError(str(error)) from error
        if completion.completion_manifest_sha256 != manifest.completion_manifest_sha256:
            manifest = manifest.model_copy(
                update={
                    "completion_manifest_sha256": completion.completion_manifest_sha256,
                }
            )
        try:
            validate_completion_manifest(completion)
            validate_artifact_lineage(manifest, completion)
        except ValueError as error:
            raise StorageError(str(error)) from error
        staging = self.staging_area(manifest.artifact_id)
        with measure_efficiency() as measurement:
            if staged_payloads:
                try:
                    self._promote_staged_payloads(manifest, staged_payloads)
                except ArtifactValidationError as error:
                    raise StorageError(str(error)) from error
            try:
                validate_completed_artifact(manifest, completion)
            except ValueError as error:
                raise StorageError(str(error)) from error
            self.write_reusable(manifest)
            atomic_write_json(
                self.completion_path(manifest.artifact_id),
                completion.model_dump(mode="json"),
            )
        if staging.is_dir():
            shutil.rmtree(staging)
        self._record_promotion(manifest, completion, measurement.result, access)
        if supersedes is not None:
            self.retire_superseded_parent(supersedes, manifest.artifact_id)
        return verify_registered_artifact(self, manifest.artifact_id)

    def register_artifact_payload(
        self,
        request: ArtifactPayloadRequest,
        overwrite_policy: OverwritePolicy,
    ) -> ReusableArtifactManifest:
        registration = artifact_family_registration(request.artifact_type)
        if registration.stage is not None and registration.stage is not request.stage:
            raise StorageError(
                f"artifact family {request.artifact_type.value} is registered at stage "
                f"{registration.stage.value}, not {request.stage.value}"
            )
        sections = request.configuration_sections | registration.cache_key_material
        fingerprint = artifact_payload_fingerprint(
            request.stage,
            request.semantic_coordinates,
            request.upstream_artifact_ids,
            sections,
            registration.implementation_identity,
        )
        if overwrite_policy is OverwritePolicy.REUSE:
            existing = self.find_by_fingerprint(ArtifactFingerprint(fingerprint))
            if existing is not None:
                return existing
        payload = serialized_payload_bytes(request.serialized)
        payload_digest = payload_sha256(payload)
        payload_path = request.payload_path.value
        if payload_path.is_file():
            verify_payload_bytes(payload_digest, payload_path.read_bytes())
        manifest = ReusableArtifactManifest.model_validate(
            OrderedDict(
                artifact_id=artifact_id(request.artifact_type, request.coordinates, fingerprint),
                artifact_type=request.artifact_type,
                semantic_producer_coordinates=request.semantic_coordinates,
                producer_stage=request.stage,
                dependency_fingerprint_sha256=fingerprint,
                upstream_artifact_ids=request.upstream_artifact_ids,
                applicable_configuration_sha256=Sha256Digest(configuration_subset_digest(sections)),
                relevant_code_sha256=Sha256Digest(
                    implementation_fingerprint(registration.implementation_identity)
                ),
                payload_paths=(str(payload_path),),
                payload_sha256=payload_digest,
                schema_version=ArtifactSchemaVersion.V1,
                created_git_commit=current_code_revision().commit,
                created_environment_sha256=environment_snapshot().fingerprint_sha256,
                state=ArtifactState.COMPLETED,
                completion_required=True,
                completion_manifest_sha256=Sha256Digest("0" * 64),
            )
        )
        completion = build_artifact_completion(
            request.semantic_coordinates,
            fingerprint,
            request.payload_path,
            payload_digest,
            manifest.applicable_configuration_sha256,
            manifest.relevant_code_sha256,
            request.stage,
            request.upstream_artifact_ids,
            request.access.provenance if request.access is not None else None,
        )
        manifest = manifest.model_copy(
            update={"completion_manifest_sha256": completion.completion_manifest_sha256}
        )
        staged = stage_bytes(
            payload_path,
            payload,
            self.staging_area(manifest.artifact_id),
        )
        return self.promote_artifact(
            manifest,
            completion,
            staged_payloads=(StagedPayload(staged, payload_path),),
            access=request.access,
        )

    def unconsumed_artifacts(self) -> tuple[ArtifactIdentifier, ...]:
        return unconsumed_artifact_ids(self.all_manifests())

    def write_reusable(self, manifest: ReusableArtifactManifest) -> None:
        atomic_write_json(
            self.manifest_path(manifest.artifact_id),
            manifest.model_dump(mode="json"),
        )
        self._index_fingerprint(
            ArtifactFingerprint(manifest.dependency_fingerprint_sha256), manifest.artifact_id
        )
        if self._manifest_cache is not None:
            self._manifest_cache[manifest.artifact_id] = manifest

    def write_completed(
        self,
        manifest: ReusableArtifactManifest,
        completion: CompletionManifest,
        access: ArtifactAccessContext | None = None,
        request: ArtifactPayloadRequest | None = None,
    ) -> None:
        if request is not None:
            self.register_artifact_payload(request, OverwritePolicy.REPLACE)
            return
        self.promote_artifact(manifest, completion, access=access)

    def read_reusable(self, artifact_id: ArtifactIdentifier) -> ReusableArtifactManifest:
        path = self.manifest_path(artifact_id)
        if not path.is_file():
            raise StorageError(f"no artifact manifest for {artifact_id.value}")
        return ReusableArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def read_completion(self, artifact_id: ArtifactIdentifier) -> CompletionManifest:
        path = self.completion_path(artifact_id)
        if not path.is_file():
            raise StorageError(f"no completion manifest for {artifact_id.value}")
        return CompletionManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def resolve(self, artifact_id: ArtifactIdentifier) -> ReusableArtifactManifest:
        manifest = self.read_reusable(artifact_id)
        validate_reusable_artifact(manifest)
        if manifest.completion_required:
            try:
                validate_completed_artifact(manifest, self.read_completion(artifact_id))
            except ValueError as error:
                raise StorageError(error) from error
        return manifest

    def find_by_fingerprint(
        self, fingerprint_sha256: ArtifactFingerprint
    ) -> ReusableArtifactManifest | None:
        logger = self._logger_for()
        indexed = self._lookup_index(fingerprint_sha256)
        if indexed is not None:
            manifest = self._match_fingerprint(indexed, fingerprint_sha256, logger)
            if manifest is not None:
                logger.event(
                    ExecutionEventName.CACHE_HIT,
                    fingerprint=fingerprint_sha256.value,
                    artifact_id=manifest.artifact_id.value,
                    artifact_path=manifest.payload_paths[0] if manifest.payload_paths else None,
                )
                return manifest
            logger.event(
                ExecutionEventName.CACHE_MISS,
                fingerprint=fingerprint_sha256.value,
                reason=ArtifactCacheMissReason.INDEXED_MANIFEST_INVALID,
            )
            return None
        for manifest in self._load_manifest_cache().values():
            if ArtifactFingerprint(manifest.dependency_fingerprint_sha256) != fingerprint_sha256:
                continue
            resolved = self._match_fingerprint(manifest.artifact_id, fingerprint_sha256, logger)
            if resolved is None:
                continue
            self._index_fingerprint(fingerprint_sha256, resolved.artifact_id)
            logger.event(
                ExecutionEventName.CACHE_HIT,
                fingerprint=fingerprint_sha256.value,
                artifact_id=resolved.artifact_id.value,
                artifact_path=resolved.payload_paths[0] if resolved.payload_paths else None,
            )
            return resolved
        candidates = tuple(
            manifest
            for manifest in self._load_manifest_cache().values()
            if ArtifactFingerprint(manifest.dependency_fingerprint_sha256) == fingerprint_sha256
        )
        logger.event(
            ExecutionEventName.CACHE_MISS,
            fingerprint=fingerprint_sha256.value,
            reason=(
                ArtifactCacheMissReason.MANIFEST_INVALID
                if candidates
                else ArtifactCacheMissReason.ABSENT
            ),
        )
        return None

    def _match_fingerprint(
        self,
        artifact_id: ArtifactIdentifier,
        fingerprint_sha256: ArtifactFingerprint,
        logger: ExecutionLogger,
    ) -> ReusableArtifactManifest | None:
        try:
            manifest = self.read_reusable(artifact_id)
        except ValueError:
            return None
        if ArtifactFingerprint(manifest.dependency_fingerprint_sha256) != fingerprint_sha256:
            return None
        if self.artifact_state(artifact_id).state is not ArtifactState.COMPLETED:
            return None
        if manifest.completion_required:
            try:
                validate_stage_lineage_completeness(manifest, self.read_completion(artifact_id))
            except (StorageError, ArtifactValidationError) as error:
                logger.abstention(
                    SemanticCoordinates(str(manifest.semantic_producer_coordinates)),
                    FieldDescription(f"reuse abstained: {error}"),
                )
                return None
        return manifest

    def _promote_staged_payloads(
        self,
        manifest: ReusableArtifactManifest,
        staged_payloads: tuple[StagedPayload, ...],
    ) -> None:
        active = {str(path) for path in manifest.payload_paths}
        for entry in staged_payloads:
            if str(entry.active_path) not in active:
                raise StorageError(
                    f"staged payload {entry.staged_path} is not a declared payload path of "
                    f"artifact {manifest.artifact_id}"
                )
            if not entry.staged_path.is_file():
                raise StorageError(f"staged payload is absent: {entry.staged_path}")
            if not self._within_root(entry.staged_path):
                raise StorageError(f"staged payload is outside the store root: {entry.staged_path}")
            staged_manifest = manifest.model_copy(
                update={"payload_paths": (str(entry.staged_path),)}
            )
            validate_payload_checksum(staged_manifest)
        for entry in staged_payloads:
            entry.active_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(entry.staged_path, entry.active_path)

    def _retire(self, artifact_id: ArtifactIdentifier) -> None:
        manifest = self.read_reusable(artifact_id)
        self.write_reusable(manifest.model_copy(update={"state": ArtifactState.STALE}))
        for payload_path in manifest.payload_paths:
            path = Path(payload_path)
            if not path.is_file():
                continue
            if not self._within_root(path):
                raise StorageError(f"stale payload is outside the store root: {payload_path}")
            path.unlink()
        completion_path = self.completion_path(artifact_id)
        if completion_path.is_file():
            completion_path.unlink()
        self._drop_index_entry(artifact_id)

    def _within_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._root.resolve())
        except ValueError:
            return False
        return True

    def _logger_for(self) -> ExecutionLogger:
        if self._logger is not None:
            return self._logger
        return execution_logger()

    def _record_promotion(
        self,
        manifest: ReusableArtifactManifest,
        completion: CompletionManifest,
        measurement: EfficiencyMeasurement,
        access: ArtifactAccessContext | None,
    ) -> None:
        logger = self._logger_for()
        lineage = recorded_lineage(completion)
        logger.record(
            ExecutionLogEvent(
                occurred_at=datetime.now(UTC),
                cell_coordinates=SemanticCoordinates(str(manifest.semantic_producer_coordinates)),
                artifact_id=manifest.artifact_id,
                state=ArtifactState.COMPLETED,
                stage=manifest.producer_stage,
                elapsed_seconds=measurement.wall_time_seconds,
                subsystem=access.subsystem if access is not None else None,
                artifacts_read=access.artifacts_read if access is not None else (),
                upstream_artifact_ids=manifest.upstream_artifact_ids,
                unconsumed_artifact_count=len(self.unconsumed_artifacts()),
                provenance_recorded=(
                    lineage.provenance is not None
                    or (access is not None and access.provenance is not None)
                ),
            )
        )
        logger.resource_phase(
            ResourcePhaseObservation(
                phase=FieldDescription(f"artifact-promotion:{manifest.producer_stage.value}"),
                elapsed_seconds=measurement.wall_time_seconds,
                peak_host_rss_mib=measurement.peak_host_rss_mib,
                peak_cuda_allocated_bytes=measurement.peak_cuda_allocated_bytes,
            )
        )
        experiment = recorded_experiment(manifest)
        if experiment is None:
            return
        events_path = experiment_event_log_path(
            self._root,
            experiment,
            ExperimentLogDirectory.EXECUTION,
        )
        ExecutionEventLog(events_path).append(
            InfrastructureEventName.ARTIFACT_PROMOTED,
            OrderedDict(
                artifact_id=manifest.artifact_id.value,
                stage=manifest.producer_stage.value,
                upstream_artifact_ids=[
                    identifier.value for identifier in manifest.upstream_artifact_ids
                ],
                terminal_state=completion.terminal_state.value,
                elapsed_seconds=measurement.wall_time_seconds,
            ),
        )

    def _load_manifest_cache(
        self,
    ) -> OrderedDict[ArtifactIdentifier, ReusableArtifactManifest]:
        if self._manifest_cache is None:
            cache: OrderedDict[ArtifactIdentifier, ReusableArtifactManifest] = OrderedDict()
            if self._manifests.is_dir():
                for path in sorted(self._manifests.glob(StorageLayoutSegment.MANIFEST_GLOB)):
                    if path.name == self._index_path.name:
                        continue
                    manifest = ReusableArtifactManifest.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                    cache[manifest.artifact_id] = manifest
            self._manifest_cache = OrderedDict(
                sorted(cache.items(), key=lambda entry: entry[0].value)
            )
        return self._manifest_cache

    def all_manifests(self) -> tuple[ReusableArtifactManifest, ...]:
        return tuple(self._load_manifest_cache().values())

    def current_completed_manifest(
        self, semantic_producer_marker: str
    ) -> ReusableArtifactManifest | None:
        current: list[ReusableArtifactManifest] = []
        for manifest in self.all_manifests():
            if semantic_producer_marker not in manifest.semantic_producer_coordinates:
                continue
            if self.artifact_state(manifest.artifact_id).state is not ArtifactState.COMPLETED:
                continue
            current.append(manifest)
        if not current:
            return None
        return min(current, key=lambda manifest: manifest.artifact_id.value)

    def _read_index(
        self,
    ) -> OrderedDict[ArtifactFingerprint, ArtifactIdentifier]:
        if not self._index_path.is_file():
            return OrderedDict()
        parsed = json.loads(self._index_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            return OrderedDict()
        entries: OrderedDict[ArtifactFingerprint, ArtifactIdentifier] = OrderedDict()
        for key, value in parsed.items():
            if isinstance(key, str) and isinstance(value, str):
                entries[ArtifactFingerprint(key)] = ArtifactIdentifier(value)
        return entries

    def _lookup_index(
        self,
        fingerprint: ArtifactFingerprint,
    ) -> ArtifactIdentifier | None:
        return self._read_index().get(fingerprint)

    def _write_index(
        self,
        index: OrderedDict[ArtifactFingerprint, ArtifactIdentifier],
    ) -> None:
        atomic_write_json(
            self._index_path,
            cast(
                StableJsonPayload,
                OrderedDict(
                    (indexed_fingerprint.value, indexed_artifact_id.value)
                    for indexed_fingerprint, indexed_artifact_id in index.items()
                ),
            ),
        )

    def _index_fingerprint(
        self,
        fingerprint: ArtifactFingerprint,
        artifact_id: ArtifactIdentifier,
    ) -> None:
        index = self._read_index()
        if index.get(fingerprint) == artifact_id:
            return
        index[fingerprint] = artifact_id
        self._write_index(index)

    def _drop_index_entry(self, artifact_id: ArtifactIdentifier) -> None:
        index = self._read_index()
        retained = OrderedDict(
            (fingerprint, indexed)
            for fingerprint, indexed in index.items()
            if indexed != artifact_id
        )
        if len(retained) != len(index):
            self._write_index(retained)


class RecoveryBoundary:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def discard_interrupted_staging(self) -> None:
        self._store.discard_staging()


def execution_store() -> ArtifactStore:
    layout = build_layout()
    return ArtifactStore(layout.execution_root, layout.staging)
