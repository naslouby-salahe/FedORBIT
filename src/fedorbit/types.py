from __future__ import annotations

from enum import StrEnum
from typing import NewType

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

FloatVector = npt.NDArray[np.float64]
FloatMatrix = npt.NDArray[np.float64]

RandomSeed = NewType("RandomSeed", int)
DeviceName = NewType("DeviceName", str)
ConditionName = NewType("ConditionName", str)
ContrastName = NewType("ContrastName", str)
Sha256Digest = NewType("Sha256Digest", str)
SupportSize = NewType("SupportSize", int)
ReplicateIndex = NewType("ReplicateIndex", int)
RelativePath = NewType("RelativePath", str)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetId(StrEnum):
    NBAIOT = "nbaiot"
    GOTHAM = "gotham"


class ExperimentId(StrEnum):
    COLD_START_LADDER = "cold-start-ladder"
    PARTNER_SELECTION = "partner-selection"
    DEEP_DETECTOR = "deep-detector"
    SIMULATED_BOUNDARY = "simulated-boundary"


class DetectorKind(StrEnum):
    GAUSSIAN = "gaussian"
    AUTOENCODER = "autoencoder"


class ShareChannel(StrEnum):
    LOCAL_STANDARDISED = "local-standardised"
    LOCAL_UNSCALED = "local-unscaled"
    LOCAL_STANDARD_DEVIATION_FLOOR = "local-standard-deviation-floor"
    LOCAL_FULL_SUPPORT = "local-full-support"
    SHARED_MARGINALS = "shared-marginals"
    SHARED_COVARIANCE = "shared-covariance"
    FEDERATED_AVERAGING = "federated-averaging"


class PartnerPolicy(StrEnum):
    ALL = "all"
    DEPLOYABLE_NEAREST = "deployable-nearest"
    ORACLE_NEAREST = "oracle-nearest"
    RANDOM = "random"


class ContrastFamily(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    EXPLORATORY = "exploratory"


class ContrastDirection(StrEnum):
    GREATER = "greater"
    LESS = "less"
    TWO_SIDED = "two-sided"


class DeviceStratum(StrEnum):
    ALL = "all"
    SATURATED = "saturated"
    UNSATURATED = "unsaturated"


class EvidenceState(StrEnum):
    MISSING = "missing"
    STALE = "stale"
    MALFORMED = "malformed"
    INCOMPLETE = "incomplete"
    VALID = "valid"


class OverwritePolicy(StrEnum):
    REUSE = "reuse"
    OVERWRITE = "overwrite"


class CliCommand(StrEnum):
    DOCTOR = "doctor"
    PREPROCESS = "preprocess"
    PLAN = "plan"
    SMOKE = "smoke"
    RUN = "run"
    STATUS = "status"
    REPORT = "report"


class RandomPurpose(StrEnum):
    SUPPORT_WINDOW = "support-window"
    PARTNER_SELECTION = "partner-selection"
    PARTNER_SAMPLE = "partner-sample"
    NETWORK_INITIALISATION = "network-initialisation"
    ATTACK_SAMPLE = "attack-sample"
    BOOTSTRAP = "bootstrap"
