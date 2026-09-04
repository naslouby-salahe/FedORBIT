from __future__ import annotations

import math
import re
import statistics
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from fedorbit.config.loading import active_config
from fedorbit.types import (
    DirectedPair,
    MetricId,
    MultiplicityFamily,
    RandomSeed,
    Split,
    TransferMethod,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContrastRegistryError(ValueError):
    pass


class PairingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PairingLineage:
    raw_dataset_lineage_sha256: str
    directed_pair: str
    seed: RandomSeed
    split: Split
    target_pre_transfer_checkpoint_artifact_id: str
    target_importance_artifact_id: str
    source_packet_artifact_id: str | None
    action_budget: float
    support_budget: int
    confirmation_budget: int
    environment_lineage_sha256: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.raw_dataset_lineage_sha256) is None:
            raise PairingError("raw dataset lineage must be lowercase SHA-256 hex")
        if _SHA256.fullmatch(self.environment_lineage_sha256) is None:
            raise PairingError("environment lineage must be lowercase SHA-256 hex")
        if not self.directed_pair:
            raise PairingError("directed pair must be non-empty")
        if not self.target_pre_transfer_checkpoint_artifact_id:
            raise PairingError("target checkpoint identity must be non-empty")
        if not self.target_importance_artifact_id:
            raise PairingError("target importance identity must be non-empty")
        if not math.isfinite(self.action_budget) or self.action_budget < 0.0:
            raise PairingError("action budget must be finite and nonnegative")
        if self.support_budget < 0 or self.confirmation_budget < 0:
            raise PairingError("support and confirmation budgets must be nonnegative")


@dataclass(frozen=True, slots=True)
class PairedObservation:
    method: TransferMethod
    value: float
    lineage: PairingLineage

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise PairingError("paired observation value must be finite")


@dataclass(frozen=True, slots=True)
class PairedValues:
    directed_pair: str
    seeds: tuple[int, ...]
    method_values: tuple[float, ...]
    reference_values: tuple[float, ...]


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
    method_values: list[float] = []
    reference_values: list[float] = []
    seeds: list[int] = []
    directed_pair: str | None = None
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
) -> tuple[tuple[tuple[str, int], PairedObservation], ...]:
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
    directed_pair: str
    mean_gain: float | None
    holm_p: float | None
    bca_lower: float | None
    strict_resource_valid: bool
    valid_seed_count: int


@dataclass(frozen=True, slots=True)
class PairContrastEvidenceSet:
    entries: tuple[PairContrastEvidence, ...]

    def __post_init__(self) -> None:
        names = tuple(entry.directed_pair for entry in self.entries)
        if len(set(names)) != len(names):
            raise ValueError("pair-contrast evidence contains duplicate directed pairs")


@dataclass(frozen=True, slots=True)
class RegisteredContrast:
    family: MultiplicityFamily
    name: str
    directed_pair: str
    statistic: str

    @property
    def key(self) -> tuple[MultiplicityFamily, str, str]:
        return self.family, self.name, self.directed_pair


@dataclass(frozen=True, slots=True)
class RegisteredFamily:
    family: MultiplicityFamily
    contrasts: tuple[RegisteredContrast, ...]


@dataclass(frozen=True, slots=True)
class RegisteredFamilyInputs:
    entries: tuple[RegisteredFamily, ...]

    def contrasts_for(self, family: MultiplicityFamily) -> tuple[RegisteredContrast, ...]:
        for entry in self.entries:
            if entry.family == family:
                return entry.contrasts
        raise ContrastRegistryError(f"unregistered multiplicity family: {family.value}")

    def registered_keys(self) -> frozenset[tuple[MultiplicityFamily, str, str]]:
        return frozenset(contrast.key for entry in self.entries for contrast in entry.contrasts)


@dataclass(frozen=True, slots=True)
class ContrastPValue:
    family: MultiplicityFamily
    contrast_name: str
    directed_pair: str
    raw_p_value: float
    valid_seed_count: int

    def __post_init__(self) -> None:
        if not self.contrast_name or not self.directed_pair:
            raise ContrastRegistryError("contrast p-value identity must be non-empty")
        if not math.isfinite(self.raw_p_value) or not 0.0 <= self.raw_p_value <= 1.0:
            raise ContrastRegistryError("raw p-value must be finite and lie in [0,1]")
        if self.valid_seed_count < 0:
            raise ContrastRegistryError("valid seed count must be nonnegative")

    @property
    def key(self) -> tuple[MultiplicityFamily, str, str]:
        return self.family, self.contrast_name, self.directed_pair


@dataclass(frozen=True, slots=True)
class ContrastPValueSet:
    entries: tuple[ContrastPValue, ...]

    def __post_init__(self) -> None:
        keys = tuple(entry.key for entry in self.entries)
        if len(set(keys)) != len(keys):
            raise ContrastRegistryError("duplicate pair-specific multiplicity input")

    def value_for(self, contrast: RegisteredContrast) -> ContrastPValue | None:
        for entry in self.entries:
            if entry.key == contrast.key:
                return entry
        return None


@dataclass(frozen=True, slots=True)
class FamilyInputState:
    contrast: RegisteredContrast
    available: bool
    unavailable_reason: str | None = None
    raw_p_value: float | None = None
    holm_p_value: float | None = None
    holm_rank: int | None = None
    family_size: int = 0


@dataclass(frozen=True, slots=True)
class FamilyStateGroup:
    family: MultiplicityFamily
    states: tuple[FamilyInputState, ...]


@dataclass(frozen=True, slots=True)
class FamilyStates:
    entries: tuple[FamilyStateGroup, ...]

    def states_for(self, family: MultiplicityFamily) -> tuple[FamilyInputState, ...]:
        for entry in self.entries:
            if entry.family == family:
                return entry.states
        raise ContrastRegistryError(f"unregistered multiplicity family: {family.value}")


def primary_pair_names() -> tuple[str, ...]:
    return tuple(
        DirectedPair(spec.source, spec.target).direction
        for spec in active_config().scientific.datasets.primary_directed_pairs
    )


def _pair_contrast(
    family: MultiplicityFamily,
    name: str,
    pair: str,
    statistic: str,
) -> RegisteredContrast:
    return RegisteredContrast(family, name, pair, statistic)


def registered_family_inputs() -> RegisteredFamilyInputs:
    families: defaultdict[MultiplicityFamily, list[RegisteredContrast]] = defaultdict(list)
    solver = TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value
    for pair in primary_pair_names():
        families[MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY].append(
            _pair_contrast(
                MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
                f"{solver} vs Local-Only — TEST relative macro-CE gain",
                pair,
                "sign_flip_superiority",
            )
        )
        families[MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR].append(
            _pair_contrast(
                MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
                f"{solver} vs Local-SIR — TEST relative macro-CE gain superiority",
                pair,
                "sign_flip_superiority",
            )
        )
        families[MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR].append(
            _pair_contrast(
                MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
                f"{solver} vs Local-SIR — TEST relative macro-CE gain TOST equivalence",
                pair,
                "tost_equivalence",
            )
        )
        families[MultiplicityFamily.COUPLING_MECHANISM].append(
            _pair_contrast(
                MultiplicityFamily.COUPLING_MECHANISM,
                (
                    "Exact correspondence orbit vs Matched-Resource Rectangular — "
                    "robust coupling value gap"
                ),
                pair,
                "sign_flip_against_zero",
            )
        )
        for suffix in ("difference", "TOST equivalence"):
            statistic = (
                "tost_equivalence" if suffix == "TOST equivalence" else "sign_flip_superiority"
            )
            families[MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY].append(
                _pair_contrast(
                    MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY,
                    (
                        f"{solver} vs Point-Correspondence Commitment — "
                        f"TEST relative macro-CE {suffix}"
                    ),
                    pair,
                    statistic,
                )
            )
            families[MultiplicityFamily.MECHANISM_ABLATIONS].append(
                _pair_contrast(
                    MultiplicityFamily.MECHANISM_ABLATIONS,
                    f"{solver} vs Coupling-Destroyed FedORBIT — TEST relative macro-CE {suffix}",
                    pair,
                    statistic,
                )
            )
        for sparsity_name in (
            "exact sparse s=1 vs exact sparse s=2",
            "exact sparse s=3 vs exact sparse s=2",
            "dense CCP vs exact sparse s=2",
        ):
            families[MultiplicityFamily.SPARSITY_SENSITIVITY].append(
                _pair_contrast(
                    MultiplicityFamily.SPARSITY_SENSITIVITY,
                    sparsity_name,
                    pair,
                    "sign_flip_difference_common_reference",
                )
            )
        families[MultiplicityFamily.CONFIRMATION_SAFETY].append(
            _pair_contrast(
                MultiplicityFamily.CONFIRMATION_SAFETY,
                (
                    "FedORBIT Without Confirmation vs FedORBIT Exact-Sparse Solver with "
                    "confirmation — harmful-transfer rate difference"
                ),
                pair,
                "seed_level_rate_difference_sign_flip",
            )
        )
    return RegisteredFamilyInputs(
        tuple(RegisteredFamily(family, tuple(families[family])) for family in MultiplicityFamily)
    )


def build_family_states(
    available_p_values: ContrastPValueSet,
) -> FamilyStates:
    registry = registered_family_inputs()
    registered_keys = registry.registered_keys()
    unknown = tuple(
        entry.key for entry in available_p_values.entries if entry.key not in registered_keys
    )
    if unknown:
        raise ContrastRegistryError("unregistered confirmatory multiplicity input")
    groups = tuple(
        FamilyStateGroup(
            family_entry.family,
            _family_states(family_entry, available_p_values),
        )
        for family_entry in registry.entries
    )
    if not any(state.available for group in groups for state in group.states):
        raise ContrastRegistryError("no registered family input has enough valid paired seeds")
    return FamilyStates(groups)


def _family_states(
    family_entry: RegisteredFamily,
    available_p_values: ContrastPValueSet,
) -> tuple[FamilyInputState, ...]:
    minimum = active_config().scientific.statistics.minimum_valid_paired_seeds
    present = tuple(
        entry
        for contrast in family_entry.contrasts
        if (entry := available_p_values.value_for(contrast)) is not None
        and entry.valid_seed_count >= minimum
    )
    ordered = sorted(
        present,
        key=lambda entry: (entry.raw_p_value, entry.contrast_name, entry.directed_pair),
    )
    family_size = len(ordered)
    adjusted_by_key: OrderedDict[tuple[MultiplicityFamily, str, str], tuple[float, int]] = (
        OrderedDict()
    )
    running_max = 0.0
    for index, entry in enumerate(ordered):
        scaled = min(1.0, entry.raw_p_value * (family_size - index))
        running_max = max(running_max, scaled)
        adjusted_by_key[entry.key] = (running_max, index + 1)
    states: list[FamilyInputState] = []
    for contrast in family_entry.contrasts:
        entry = available_p_values.value_for(contrast)
        if entry is None:
            states.append(
                FamilyInputState(
                    contrast,
                    False,
                    unavailable_reason="registered input missing",
                    family_size=family_size,
                )
            )
            continue
        if entry.valid_seed_count < minimum:
            states.append(
                FamilyInputState(
                    contrast,
                    False,
                    unavailable_reason="insufficient valid paired seeds",
                    raw_p_value=entry.raw_p_value,
                    family_size=family_size,
                )
            )
            continue
        holm_p, rank = adjusted_by_key[entry.key]
        states.append(
            FamilyInputState(
                contrast,
                True,
                raw_p_value=entry.raw_p_value,
                holm_p_value=holm_p,
                holm_rank=rank,
                family_size=family_size,
            )
        )
    return tuple(states)


class SpearmanError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SpearmanReport:
    rho: float
    point_count: int
    pair: str


def descriptive_spearman(
    predicted_values: tuple[float, ...],
    realized_values: tuple[float, ...],
    directed_pair: str,
) -> SpearmanReport | None:
    minimum = active_config().scientific.statistics.spearman_minimum_valid_points
    if len(predicted_values) != len(realized_values):
        raise SpearmanError("predicted and realized value counts differ")
    if len(predicted_values) < minimum:
        return None
    ranked_predicted = _ranks(predicted_values)
    ranked_realized = _ranks(realized_values)
    rho = _pearson(ranked_predicted, ranked_realized)
    return SpearmanReport(rho=rho, point_count=len(predicted_values), pair=directed_pair)


def _ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks: list[float] = [0.0] * len(values)
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


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    variance_left = sum((a - mean_left) ** 2 for a in left)
    variance_right = sum((b - mean_right) ** 2 for b in right)
    denominator = math.sqrt(variance_left * variance_right)
    if denominator == 0.0:
        return 0.0
    return covariance / denominator


SPEARMAN_METRIC_NAME = MetricId.PREDICTED_REALIZED_SPEARMAN
