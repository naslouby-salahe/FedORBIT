from __future__ import annotations

from dataclasses import dataclass

from fedorbit.analysis.statistics import PValueSet
from fedorbit.config.models import FedorbitConfig
from fedorbit.domain.enums import MultiplicityFamily, TransferMethod


class ContrastRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RegisteredContrast:
    family: MultiplicityFamily
    name: str
    directed_pair: str
    statistic: str


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


@dataclass(frozen=True, slots=True)
class FamilyInputState:
    contrast: RegisteredContrast
    available: bool
    unavailable_reason: str | None = None
    raw_p_value: float | None = None


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


PRIMARY_PAIR_NAMES = ("Edge→Windows", "Windows→Edge", "Edge→Linux", "Linux→Edge")


def _pair_contrast(
    family: MultiplicityFamily,
    name_template: str,
    pair: str,
    statistic: str,
) -> RegisteredContrast:
    return RegisteredContrast(
        family=family,
        name=name_template,
        directed_pair=pair,
        statistic=statistic,
    )


def registered_family_inputs() -> RegisteredFamilyInputs:
    families: dict[MultiplicityFamily, list[RegisteredContrast]] = {
        family: [] for family in MultiplicityFamily
    }
    solver = TransferMethod.FEDORBIT_EXACT_SPARSE_SOLVER.value
    for pair in PRIMARY_PAIR_NAMES:
        families[MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY].append(
            _pair_contrast(
                MultiplicityFamily.PRIMARY_TRANSFER_VS_LOCAL_ONLY,
                f"{solver} vs Local-Only — TEST relative macro-CE gain",
                pair,
                "sign_flip_superiority",
            )
        )
        superiority_name = f"{solver} vs Local-SIR — TEST relative macro-CE gain superiority"
        families[MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR].append(
            _pair_contrast(
                MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
                superiority_name,
                pair,
                "sign_flip_superiority",
            )
        )
        equivalence_name = f"{solver} vs Local-SIR — TEST relative macro-CE gain TOST equivalence"
        families[MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR].append(
            _pair_contrast(
                MultiplicityFamily.EXTERNAL_SOURCE_VS_LOCAL_SIR,
                equivalence_name,
                pair,
                "tost_equivalence",
            )
        )
        coupling_name = (
            "Exact correspondence orbit vs Matched-Resource Rectangular — robust coupling value gap"
        )
        families[MultiplicityFamily.COUPLING_MECHANISM].append(
            _pair_contrast(
                MultiplicityFamily.COUPLING_MECHANISM,
                coupling_name,
                pair,
                "sign_flip_against_zero",
            )
        )
        for suffix in ("difference", "TOST equivalence"):
            statistic = "tost_equivalence" if "TOST" in suffix else "sign_flip_superiority"
            point_name = (
                f"{solver} vs Point-Correspondence Commitment — TEST relative macro-CE {suffix}"
            )
            families[MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY].append(
                _pair_contrast(
                    MultiplicityFamily.POINT_CORRESPONDENCE_SAFETY,
                    point_name,
                    pair,
                    statistic,
                )
            )
            ablation_name = (
                f"{solver} vs Coupling-Destroyed FedORBIT — TEST relative macro-CE {suffix}"
            )
            families[MultiplicityFamily.MECHANISM_ABLATIONS].append(
                _pair_contrast(
                    MultiplicityFamily.MECHANISM_ABLATIONS,
                    ablation_name,
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
        confirmation_name = (
            "FedORBIT Without Confirmation vs FedORBIT Exact-Sparse Solver "
            "with confirmation — harmful-transfer rate difference"
        )
        families[MultiplicityFamily.CONFIRMATION_SAFETY].append(
            _pair_contrast(
                MultiplicityFamily.CONFIRMATION_SAFETY,
                confirmation_name,
                pair,
                "seed_level_rate_difference_sign_flip",
            )
        )
    return RegisteredFamilyInputs(
        tuple(RegisteredFamily(family, tuple(families[family])) for family in MultiplicityFamily)
    )


def build_family_states(
    config: FedorbitConfig,
    available_p_values: PValueSet,
) -> FamilyStates:
    del config
    groups: list[FamilyStateGroup] = []
    for family_entry in registered_family_inputs().entries:
        states: list[FamilyInputState] = []
        for contrast in family_entry.contrasts:
            raw_p_value = available_p_values.value_of(contrast.name)
            if raw_p_value is not None:
                states.append(
                    FamilyInputState(
                        contrast=contrast,
                        available=True,
                        raw_p_value=raw_p_value,
                    )
                )
            else:
                states.append(
                    FamilyInputState(
                        contrast=contrast,
                        available=False,
                        unavailable_reason="insufficient valid paired seeds",
                    )
                )
        groups.append(FamilyStateGroup(family_entry.family, tuple(states)))
    result = FamilyStates(tuple(groups))
    if not any(state.available for group in result.entries for state in group.states):
        raise ContrastRegistryError("no registered family input has a computable p-value")
    return result
