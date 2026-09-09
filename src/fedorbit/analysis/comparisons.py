from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from fedorbit.config.loading import active_config
from fedorbit.types import (
    ArtifactIdentifier,
    Coefficient,
    DirectedPairName,
    Index,
    NonNegativeFloat,
    RandomSeed,
    RelativeGain,
    SampleCount,
    Score,
    Sha256Digest,
    SignificanceLevel,
    Split,
    StrictResourceValidity,
    TransferMethod,
    is_sha256_digest,
)

type ScoreSeries = tuple[Score, ...]
type RankSeries = tuple[Score, ...]
type PairSeedIdentity = tuple[DirectedPairName, RandomSeed]
type IndexedPairedObservations = tuple[tuple[PairSeedIdentity, PairedObservation], ...]


class PairingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PairingLineage:
    raw_dataset_lineage_sha256: Sha256Digest
    directed_pair: DirectedPairName
    seed: RandomSeed
    split: Split
    target_pre_transfer_checkpoint_artifact_id: ArtifactIdentifier
    target_importance_artifact_id: ArtifactIdentifier
    source_packet_artifact_id: ArtifactIdentifier | None
    action_budget: NonNegativeFloat
    support_budget: Index
    confirmation_budget: Index
    environment_lineage_sha256: Sha256Digest

    def __post_init__(self) -> None:
        if not is_sha256_digest(self.raw_dataset_lineage_sha256):
            raise PairingError("raw dataset lineage must be lowercase SHA-256 hex")
        if not is_sha256_digest(self.environment_lineage_sha256):
            raise PairingError("environment lineage must be lowercase SHA-256 hex")
        if not self.directed_pair:
            raise PairingError("directed pair must be non-empty")
        if not self.target_pre_transfer_checkpoint_artifact_id.value:
            raise PairingError("target checkpoint identity must be non-empty")
        if not self.target_importance_artifact_id.value:
            raise PairingError("target importance identity must be non-empty")
        if not math.isfinite(self.action_budget) or self.action_budget < 0.0:
            raise PairingError("action budget must be finite and nonnegative")
        if self.support_budget < 0 or self.confirmation_budget < 0:
            raise PairingError("support and confirmation budgets must be nonnegative")


@dataclass(frozen=True, slots=True)
class PairedObservation:
    method: TransferMethod
    value: Score
    lineage: PairingLineage

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise PairingError("paired observation value must be finite")


@dataclass(frozen=True, slots=True)
class PairedValues:
    directed_pair: DirectedPairName
    seeds: tuple[RandomSeed, ...]
    method_values: ScoreSeries
    reference_values: ScoreSeries


def validate_paired_observations(
    method_observations: tuple[PairedObservation, ...],
    reference_observations: tuple[PairedObservation, ...],
) -> PairedValues:
    if not method_observations or not reference_observations:
        raise PairingError("paired comparison requires observations from both methods")
    if len(method_observations) != len(reference_observations):
        raise PairingError("paired comparison sample counts differ")
    method_index = _index_observations(method_observations)
    reference_index = _index_observations(reference_observations)
    if tuple(key for key, _ in method_index) != tuple(key for key, _ in reference_index):
        raise PairingError("paired comparison pair/seed identities differ")
    method_values: list[Score] = []
    reference_values: list[Score] = []
    seeds: list[RandomSeed] = []
    directed_pair: DirectedPairName | None = None
    for (method_key, method_observation), (reference_key, reference_observation) in zip(
        method_index, reference_index, strict=True
    ):
        if method_key != reference_key:
            raise PairingError("paired comparison pair/seed identities differ")
        if method_observation.lineage != reference_observation.lineage:
            raise PairingError("paired comparison lineage mismatch")
        if directed_pair is None:
            directed_pair = method_observation.lineage.directed_pair
        elif directed_pair != method_observation.lineage.directed_pair:
            raise PairingError("one contrast cannot pool directed pairs")
        seeds.append(method_observation.lineage.seed)
        method_values.append(method_observation.value)
        reference_values.append(reference_observation.value)
    if directed_pair is None:
        raise PairingError("paired comparison has no directed pair")
    return PairedValues(
        directed_pair,
        tuple(seeds),
        tuple(method_values),
        tuple(reference_values),
    )


def _index_observations(
    observations: tuple[PairedObservation, ...],
) -> IndexedPairedObservations:
    indexed = tuple(
        sorted(
            (
                ((observation.lineage.directed_pair, observation.lineage.seed), observation)
                for observation in observations
            ),
            key=lambda item: item[0],
        )
    )
    keys = tuple(key for key, _ in indexed)
    if len(set(keys)) != len(keys):
        raise PairingError("paired comparison contains duplicate pair/seed cells")
    return indexed


@dataclass(frozen=True, slots=True)
class PairContrastEvidence:
    directed_pair: DirectedPairName
    mean_gain: RelativeGain | None
    holm_p: SignificanceLevel | None
    bca_lower: RelativeGain | None
    strict_resource_valid: StrictResourceValidity
    valid_seed_count: Index


@dataclass(frozen=True, slots=True)
class PairContrastEvidenceSet:
    entries: tuple[PairContrastEvidence, ...]

    def __post_init__(self) -> None:
        names = tuple(entry.directed_pair for entry in self.entries)
        if len(set(names)) != len(names):
            raise ValueError("pair-contrast evidence contains duplicate directed pairs")


class SpearmanError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SpearmanReport:
    rho: Coefficient
    point_count: SampleCount
    pair: DirectedPairName


def descriptive_spearman(
    predicted_values: ScoreSeries,
    realized_values: ScoreSeries,
    directed_pair: DirectedPairName,
) -> SpearmanReport | None:
    minimum = active_config().scientific.statistics.spearman_minimum_valid_points
    if len(predicted_values) != len(realized_values):
        raise SpearmanError("predicted and realized value counts differ")
    if len(predicted_values) < minimum:
        return None
    ranked_predicted = _ranks(predicted_values)
    ranked_realized = _ranks(realized_values)
    rho = _pearson(ranked_predicted, ranked_realized)
    point_count: SampleCount = len(predicted_values)
    return SpearmanReport(
        rho=rho,
        point_count=point_count,
        pair=directed_pair,
    )


def _ranks(values: ScoreSeries) -> RankSeries:
    order = sorted(range(len(values)), key=lambda index: values[index])
    zero_rank: Score = 0.0
    ranks: list[Score] = [zero_rank] * len(values)
    position = 0
    while position < len(order):
        block_end = position
        while (
            block_end + 1 < len(order) and values[order[block_end + 1]] == values[order[position]]
        ):
            block_end += 1
        average_rank = (position + block_end) / 2 + 1
        for offset in range(position, block_end + 1):
            ranks[order[offset]] = average_rank
        position = block_end + 1
    return tuple(ranks)


def _pearson(left: RankSeries, right: RankSeries) -> Coefficient:
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    variance_left = sum((a - mean_left) ** 2 for a in left)
    variance_right = sum((b - mean_right) ** 2 for b in right)
    denominator = math.sqrt(variance_left * variance_right)
    if denominator == 0.0:
        zero: Coefficient = 0.0
        return zero
    correlation: Coefficient = covariance / denominator
    return correlation
