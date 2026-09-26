from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from fedorbit.types import (
    ConditionName,
    ContrastDirection,
    ContrastFamily,
    ContrastName,
    DatasetId,
    DetectorKind,
    DeviceName,
    DeviceStratum,
    ExperimentId,
    FrozenModel,
    PartnerPolicy,
    RandomSeed,
    RelativePath,
    ShareChannel,
    SupportSize,
)


class NBaIoTConfig(FrozenModel):
    relative_directory: RelativePath
    devices: tuple[DeviceName, ...] = Field(min_length=2)
    attack_rows_per_file: int = Field(gt=0)
    benign_support_fraction: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _devices_unique(self) -> Self:
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("nbaiot devices must be unique")
        return self


class GothamConfig(FrozenModel):
    relative_directory: RelativePath
    window_seconds: int = Field(gt=0)
    attack_window_minimum_fraction: float = Field(gt=0.0, lt=1.0)
    benign_support_fraction: float = Field(gt=0.0, lt=1.0)
    benign_label: str = Field(min_length=1)
    excluded_label: str = Field(min_length=1)


class DatasetsConfig(FrozenModel):
    raw_directory: RelativePath
    nbaiot: NBaIoTConfig
    gotham: GothamConfig


class GaussianDetectorConfig(FrozenModel):
    minimum_scale: float = Field(gt=0.0)
    standard_deviation_floor_fraction: float = Field(gt=0.0, lt=1.0)


class AutoencoderDetectorConfig(FrozenModel):
    hidden_width: int = Field(gt=0)
    bottleneck_width: int = Field(gt=0)
    training_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    federated_rounds: int = Field(gt=0)
    federated_local_steps: int = Field(gt=0)
    federated_partner_rows: int = Field(gt=0)

    @model_validator(mode="after")
    def _bottleneck_narrower(self) -> Self:
        if self.bottleneck_width >= self.hidden_width:
            raise ValueError("bottleneck width must be smaller than the hidden width")
        return self


class DetectorsConfig(FrozenModel):
    gaussian: GaussianDetectorConfig
    autoencoder: AutoencoderDetectorConfig


class OperatingPointConfig(FrozenModel):
    nominal_false_positive_rate: float = Field(gt=0.0, lt=1.0)


class PartnerSimilarityConfig(FrozenModel):
    scale_floor: float = Field(gt=0.0)


class StatisticsConfig(FrozenModel):
    bootstrap_resamples: int = Field(gt=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    alpha: float = Field(gt=0.0, lt=1.0)
    saturation_headroom: float = Field(gt=0.0, lt=1.0)
    minimum_recovery_headroom: float = Field(gt=0.0, lt=1.0)
    maximum_auroc_standard_error: float = Field(gt=0.0, lt=0.5)


class ConditionSpec(FrozenModel):
    channel: ShareChannel
    partner_policy: PartnerPolicy | None = None
    partner_count: int | None = Field(default=None, gt=0)
    dose_rows: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        shares = self.channel in {
            ShareChannel.SHARED_MARGINALS,
            ShareChannel.SHARED_COVARIANCE,
            ShareChannel.FEDERATED_AVERAGING,
        }
        if shares != (self.partner_policy is not None):
            raise ValueError("partner policy is required exactly for sharing channels")
        selective = self.partner_policy in {
            PartnerPolicy.DEPLOYABLE_NEAREST,
            PartnerPolicy.ORACLE_NEAREST,
            PartnerPolicy.RANDOM,
        }
        if selective != (self.partner_count is not None):
            raise ValueError("partner count is required exactly for selective partner policies")
        if (self.channel is ShareChannel.SHARED_COVARIANCE) != (self.dose_rows is not None):
            raise ValueError("dose rows are required exactly for the shared covariance channel")
        return self


class ContrastSpec(FrozenModel):
    name: ContrastName
    family: ContrastFamily
    treatment: ConditionName
    control: ConditionName
    support_sizes: tuple[SupportSize, ...] = Field(min_length=1)
    stratum: DeviceStratum
    direction: ContrastDirection


class ExperimentConfig(FrozenModel):
    id: ExperimentId
    dataset: DatasetId
    detector: DetectorKind
    support_sizes: tuple[SupportSize, ...] = Field(min_length=1)
    replicates: int = Field(gt=0)
    conditions: dict[ConditionName, ConditionSpec] = Field(min_length=1)
    contrasts: tuple[ContrastSpec, ...]
    report_operating_point: bool

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        for name, spec in self.conditions.items():
            if spec.channel is ShareChannel.FEDERATED_AVERAGING and (
                self.detector is not DetectorKind.AUTOENCODER
            ):
                raise ValueError(f"condition {name} needs the autoencoder detector")
            if spec.channel is ShareChannel.SHARED_COVARIANCE and (
                self.detector is not DetectorKind.GAUSSIAN
            ):
                raise ValueError(f"condition {name} needs the gaussian detector")
        if self.report_operating_point and self.detector is not DetectorKind.GAUSSIAN:
            raise ValueError("operating point reporting needs the gaussian detector")
        if any(size <= 0 for size in self.support_sizes):
            raise ValueError("support sizes must be positive")
        has_reference = any(
            spec.channel is ShareChannel.LOCAL_FULL_SUPPORT for spec in self.conditions.values()
        )
        contrast_names = [contrast.name for contrast in self.contrasts]
        if len(set(contrast_names)) != len(contrast_names):
            raise ValueError("contrast names must be unique")
        for contrast in self.contrasts:
            if contrast.treatment not in self.conditions or contrast.control not in self.conditions:
                raise ValueError(f"contrast {contrast.name} references an undeclared condition")
            if contrast.stratum is not DeviceStratum.ALL and not has_reference:
                raise ValueError(f"contrast {contrast.name} needs a full-support reference")
            if not set(contrast.support_sizes) <= set(self.support_sizes):
                raise ValueError(f"contrast {contrast.name} uses undeclared support sizes")
        return self


class FedorbitConfig(FrozenModel):
    base_seed: RandomSeed
    workspace_directory: RelativePath
    datasets: DatasetsConfig
    detectors: DetectorsConfig
    operating_point: OperatingPointConfig
    partner_similarity: PartnerSimilarityConfig
    statistics: StatisticsConfig
    experiments: tuple[ExperimentConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_experiments(self) -> Self:
        identifiers = [experiment.id for experiment in self.experiments]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("each experiment may be declared once")
        return self

    def experiment(self, identifier: ExperimentId) -> ExperimentConfig:
        for experiment in self.experiments:
            if experiment.id is identifier:
                return experiment
        raise KeyError(identifier)
