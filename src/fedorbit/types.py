from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Protocol, cast

from pydantic import JsonValue


class ClientRole(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SOURCE = "source"
    TARGET = "target"


class DatasetId(StrEnum):
    EDGE_IIOTSET_NETWORK = "edge_iiotset_network"
    TON_IOT_WINDOWS10_HOST = "ton_iot_windows10_host"
    TON_IOT_LINUX_PROCESS_HOST = "ton_iot_linux_process_host"
    TON_IOT_NETWORK = "ton_iot_network"


class RawDatasetDirectory(StrEnum):
    EDGE_IIOTSET = "Edge-IIoTset"
    TON_IOT = "TON-IoT"


class Split(StrEnum):
    TRAIN = "TRAIN"
    META = "META"
    VALID = "VALID"
    CONFIRM = "CONFIRM"
    TEST = "TEST"


class CoarseGroup(StrEnum):
    DISRUPTION = "Disruption"
    EXPLOITATION = "Exploitation"
    ACCESS_AND_DISCOVERY = "Access and Discovery"


class OracleTransferConcept(StrEnum):
    DDOS = "DDoS"
    RANSOMWARE = "Ransomware"
    BACKDOOR = "Backdoor"
    INJECTION = "Injection"
    XSS = "XSS"
    PASSWORD_ATTACK = "Password attack"
    SCANNING = "Scanning"
    MITM = "MITM"


class TransferMethod(StrEnum):
    LOCAL_ONLY = "Local-Only"
    LOCAL_SIR = "Local-SIR"
    COARSE_BLOCK_MEAN = "Coarse Block-Mean"
    COARSE_BLOCK_MIN = "Coarse Block-Min"
    ORBIT_MEAN = "Orbit-Mean"
    MATCHED_RESOURCE_RECTANGULAR = "Matched-Resource Rectangular"
    POINT_CORRESPONDENCE_COMMITMENT = "Point-Correspondence Commitment"
    GENERIC_EXACT_QAP = "Generic Exact QAP"
    FEDORBIT_EXACT_SPARSE_SOLVER = "FedORBIT Exact-Sparse Solver"
    FEDORBIT_DENSE_CCP_FALLBACK = "FedORBIT Dense-CCP Fallback"
    EXACT_MAP_ORACLE = "Exact-Map Oracle"
    FEDORBIT_WITHOUT_CONFIRMATION = "FedORBIT Without Confirmation"
    COUPLING_DESTROYED_FEDORBIT = "Coupling-Destroyed FedORBIT"


class ExperimentName(StrEnum):
    MATHEMATICAL_PRIMITIVE_VALIDATION = "Mathematical Primitive Validation"
    EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION = "Exact Sparse Theorem Exhaustive Validation"
    COUPLING_AND_MAP_BOUND_VALIDATION = "Coupling and Map-Bound Validation"
    DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION = (
        "Dataset, Client, and Strict-Resource Validation"
    )
    BASE_MODEL_HYPERPARAMETER_PILOT = "Base-Model Hyperparameter Pilot"
    SOURCE_RESPONSE_ESTIMATOR_PILOT = "Source-Response Estimator Pilot"
    FINAL_SOURCE_RESPONSE_BAND_VALIDATION = "Final Source-Response Band Validation"
    BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION = "Baseline and Oracle Correctness Validation"
    EXACT_SPARSE_SOLVER_BENCHMARK = "Exact-Sparse Solver Benchmark"
    SYNTHETIC_COUPLING_MECHANISM_VALIDATION = "Synthetic Coupling-Mechanism Validation"
    REAL_PACKET_COUPLING_MECHANISM_VALIDATION = "Real-Packet Coupling-Mechanism Validation"
    COMMON_ACTION_UNDER_UNIDENTIFIED_MAP = "Common Action Under Unidentified Map"
    ROBUST_COMPROMISE_UNDER_UNIDENTIFIED_MAP = "Robust Compromise Under Unidentified Map"
    MAP_DEPENDENT_ACTION_BOUNDARY = "Map-Dependent Action Boundary"
    EXACT_MAP_VALUE_BOUND_VALIDATION = "Exact Map-Value Bound Validation"
    PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER = "Primary Strict Cross-Telemetry Transfer"
    MULTI_SOURCE_SELECTION_VALIDATION = "Multi-Source Selection Validation"
    MECHANISM_ABLATIONS = "Mechanism Ablations"
    SPARSITY_AND_DENSE_FALLBACK = "Sparsity and Dense Fallback"
    TARGET_CONFIRMATION_AND_PORTABILITY = "Target Confirmation and Portability"
    SECONDARY_CROSS_MODALITY_GENERALIZATION = "Secondary Cross-Modality Generalization"
    SEMANTIC_SUFFICIENCY_FRONTIER = "Semantic Sufficiency Frontier"
    WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES = (
        "Weak-Signal, Support, and Heterogeneity Boundaries"
    )
    MAP_AVAILABILITY_APPLICABILITY_AUDIT = "Map-Availability Applicability Audit"
    SCALABILITY_AND_EFFICIENCY = "Scalability and Efficiency"
    STATISTICAL_SYNTHESIS = "Statistical Synthesis"


class ExperimentClassification(StrEnum):
    VALIDATION = "Validation"
    EXPLORATORY = "Exploratory"
    CONFIRMATORY = "Confirmatory"
    CONFIRMATORY_MECHANISM = "Confirmatory mechanism"
    CONFIRMATORY_SAFETY = "Confirmatory safety"
    DIAGNOSTIC = "Diagnostic"
    FAILURE_BOUNDARY = "Failure Boundary"
    ABLATION = "Ablation"
    ROBUSTNESS = "Robustness"
    GENERALIZATION = "Generalization"
    FINAL_EVIDENCE = "FINAL EVIDENCE"
    CONFIRMATORY_ANALYSIS = "Confirmatory ANALYSIS"
    EFFICIENCY = "EFFICIENCY"


class ScalabilityBlockPattern(StrEnum):
    BALANCED = "balanced"
    MAXIMALLY_SKEWED = "maximally_skewed"


class ArtifactState(StrEnum):
    MISSING = "Missing"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    INVALID = "Invalid"
    STALE = "Stale"
    BLOCKED = "Blocked"


class OverwritePolicy(StrEnum):
    REUSE = "reuse"
    REPLACE = "replace"


class TerminalState(StrEnum):
    COMPLETED = "Completed"
    FAILED_INFRASTRUCTURE = "Failed / Infrastructure Failure"
    INVALID = "Invalid"
    FAILED_VALIDATION = "Failed / Validation Failure"
    FAILED_SCIENTIFIC_ALGORITHMIC = "Failed / Scientific Algorithmic Failure"
    TIME_LIMIT = "Time Limit"
    RESOURCE_LIMIT = "Resource Limit"


class RngNamespace(StrEnum):
    SPLIT = "split"
    MODEL_INITIALIZATION = "model_initialization"
    TRAIN_EPOCH_SHUFFLE = "train_epoch_shuffle"
    RESPONSE_SCHEDULE = "response_schedule"
    RESPONSE_BOOTSTRAP = "response_bootstrap"
    ANONYMOUS_NODE_ORDER = "anonymous_node_order"
    CONFIRMATION_SCHEDULE = "confirmation_schedule"
    CONFIRMATION_BOOTSTRAP = "confirmation_bootstrap"
    ASSIMILATION_SCHEDULE = "assimilation_schedule"
    STATISTICAL_BOOTSTRAP = "statistical_bootstrap"
    SYNTHETIC_INSTANCE = "synthetic_instance"
    COUPLING_DESTRUCTION = "coupling_destruction"
    DENSE_START = "dense_start"


class ArtifactType(StrEnum):
    PREPARED_SPLIT = "prepared_split"
    CHECKPOINT = "checkpoint"
    PREDICTION = "prediction"
    RESPONSE_PACKET = "response_packet"
    TARGET_IMPORTANCE = "target_importance"
    SOLVER_RESULT = "solver_result"
    CONFIRMATION_INPUT = "confirmation_input"
    OTHER = "other"


class ArtifactStage(StrEnum):
    RAW = "raw"
    PREPROCESSING = "preprocessing"
    ELIGIBILITY = "eligibility"
    PILOT_SELECTION = "pilot_selection"
    TRAINING = "training"
    SCORING = "scoring"
    RESPONSE = "response"
    TARGET_IMPORTANCE = "target_importance"
    CORRESPONDENCE = "correspondence"
    CONFIRMATION = "confirmation"
    MULTI_SOURCE_SELECTION = "multi_source_selection"
    EVALUATION = "evaluation"
    STATISTICS = "statistics"
    REPORTING = "reporting"


class SemanticCoordinate(StrEnum):
    EXPERIMENT = "experiment"
    DATASET = "dataset"
    SOURCE_CLIENT = "source_client"
    TARGET_CLIENT = "target_client"
    DIRECTED_PAIR = "directed_pair"
    METHOD = "method"
    CONDITION = "condition"
    SUPPORT = "support"
    SEED = "seed"


class FailureCategory(StrEnum):
    INFRASTRUCTURE = "infrastructure"
    VALIDATION = "validation"
    SCIENTIFIC_NULL = "scientific_null"
    SCIENTIFIC_BOUNDARY = "scientific_boundary"
    SOLVER_TIME_LIMIT = "solver_time_limit"
    RESOURCE_LIMIT = "resource_limit"
    SCIENTIFIC_ALGORITHMIC = "scientific_algorithmic"


class MultiplicityFamily(StrEnum):
    PRIMARY_TRANSFER_VS_LOCAL_ONLY = "Primary Transfer vs Local-Only"
    EXTERNAL_SOURCE_VS_LOCAL_SIR = "External Source vs Local-SIR"
    COUPLING_MECHANISM = "Coupling Mechanism"
    POINT_CORRESPONDENCE_SAFETY = "Point-Correspondence Safety"
    MECHANISM_ABLATIONS = "Mechanism Ablations"
    SPARSITY_SENSITIVITY = "Sparsity Sensitivity"
    CONFIRMATION_SAFETY = "Confirmation Safety"


class MetricId(StrEnum):
    CLASS_CONDITIONAL_CROSS_ENTROPY = "Class-Conditional Cross-Entropy"
    MACRO_CROSS_ENTROPY = "Macro Cross-Entropy"
    RELATIVE_MACRO_CE_GAIN = "Relative Macro-CE Gain"
    PRECISION = "Precision"
    RECALL = "Recall"
    F1 = "F1"
    MACRO_F1 = "Macro-F1"
    BALANCED_ACCURACY = "Balanced Accuracy"
    CERTIFIED_ROBUST_PREDICTED_VALUE = "Certified Robust Predicted Value"
    FIXED_ACTION_RECTANGULARIZATION_GAP = "Fixed-Action Rectangularization Gap"
    ROBUST_COUPLING_VALUE_GAP = "Robust Coupling Value Gap"
    COUPLING_UPPER_BOUND_DIAGNOSTIC = "Coupling Upper-Bound Diagnostic"
    EXACT_MAP_ACTION_VALUE = "Exact-Map Action Value"
    ORBIT_RADIUS_MAP_BOUND = "Orbit-Radius Map Bound"
    PREDICTED_REALIZED_SPEARMAN = "Predicted-Realized Spearman Correlation"
    ABSOLUTE_OBJECTIVE_ERROR = "Absolute Objective Error"
    RELATIVE_OBJECTIVE_ERROR = "Relative Objective Error"
    CORRESPONDENCE_CERTIFICATE_VALIDITY = "Correspondence Certificate Validity"
    ACTIVE_IMAGE_CANDIDATES = "Active-Image Candidates"
    LAP_CALLS = "LAP Calls"
    SCENARIO_CUT_COUNT = "Scenario-Cut Count"
    MASTER_ITERATIONS = "Master Iterations"
    DENSE_RELAXATION_BOUND = "Dense Relaxation Bound"
    DENSE_PROJECTED_OBJECTIVE = "Dense Projected Objective"
    DENSE_BOUND_GAP = "Dense Bound Gap"
    DENSE_INTEGRALITY_RESIDUAL = "Dense Integrality Residual"
    PROPOSAL_ACCEPTANCE_RATE = "Proposal Acceptance Rate"
    HARMFUL_ACCEPTED_RATE = "Harmful Accepted Rate"
    USEFUL_ACCEPTED_RATE = "Useful Accepted Rate"
    BENEFICIAL_REJECTED_RATE = "Beneficial Rejected Rate"
    COVERAGE_CONFIRM = "Confirmation Coverage"
    COVERAGE_NO_CONFIRM = "No-Confirmation Coverage"
    COVERAGE_LOSS = "Coverage Loss"
    HARM_RATE_CONFIRM = "Confirm Harmful Rate"
    HARM_RATE_NO_CONFIRM = "No-Confirm Harmful Rate"
    ABSOLUTE_RISK_REDUCTION = "Absolute Risk Reduction"
    RELATIVE_RISK_REDUCTION = "Relative Risk Reduction"
    WALL_TIME = "Wall Time"
    PEAK_HOST_RSS = "Peak Host RSS"
    PEAK_CUDA_ALLOCATED_BYTES = "Peak CUDA Allocated Bytes"
    PACKET_SERIALIZED_BYTE_COUNT = "Packet Serialized Byte Count"
    SOURCE_RESPONSE_OPTIMIZER_STEPS = "Source Response Optimizer Steps"
    TARGET_CONFIRMATION_OPTIMIZER_STEPS = "Target Confirmation Optimizer Steps"
    LIVE_ASSIMILATION_OPTIMIZER_STEPS = "Live Assimilation Optimizer Steps"
    TIMEOUT_INDICATOR = "Timeout Indicator"
    RESOURCE_LIMIT_INDICATOR = "Resource-Limit Indicator"


@dataclass(frozen=True, slots=True)
class ArtifactPath:
    value: Path

    def __post_init__(self) -> None:
        if not self.value.is_absolute():
            raise ValueError("artifact paths must be absolute")


@dataclass(frozen=True, slots=True)
class ArtifactIdentifier:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("artifact identifier must not be empty")


@dataclass(frozen=True, slots=True)
class ArtifactFingerprint:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("artifact fingerprint must not be empty")


@dataclass(frozen=True, slots=True)
class SemanticCoordinates:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("semantic coordinates must not be empty")


@dataclass(frozen=True, slots=True)
class ExperimentCondition:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("experiment condition must not be empty")


@dataclass(frozen=True, slots=True)
class SupportSize:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError("support size must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentSeed:
    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value < 2**32:
            raise ValueError("experiment seed must be in the unsigned 32-bit range")


@dataclass(frozen=True, slots=True)
class ExecutionCell:
    coordinates: SemanticCoordinates
    artifact_identifier: ArtifactIdentifier
    dependency_fingerprint: ArtifactFingerprint


@dataclass(frozen=True, slots=True)
class DirectedPair:
    source: DatasetId
    target: DatasetId

    @property
    def direction(self) -> str:
        return f"{self.source.value} -> {self.target.value}"


@dataclass(frozen=True, slots=True)
class SemanticCell:
    experiment: ExperimentName
    dataset: DatasetId | None = None
    source_client: DatasetId | None = None
    target_client: DatasetId | None = None
    directed_pair: DirectedPair | None = None
    method: TransferMethod | None = None
    condition: ExperimentCondition | None = None
    support: SupportSize | None = None
    seed: ExperimentSeed | None = None

    def identity_json(self, relevance: frozenset[SemanticCoordinate]) -> str:
        present: OrderedDict[str, str | int | float | list[str] | None] = OrderedDict(
            dataset=self.dataset.value if self.dataset is not None else None,
            source_client=self.source_client.value if self.source_client is not None else None,
            target_client=self.target_client.value if self.target_client is not None else None,
            method=self.method.value if self.method is not None else None,
            condition=self.condition.value if self.condition is not None else None,
            support=self.support.value if self.support is not None else None,
            seed=self.seed.value if self.seed is not None else None,
        )
        if self.directed_pair is not None:
            present["directed_pair"] = [
                self.directed_pair.source.value,
                self.directed_pair.target.value,
            ]
        values: OrderedDict[str, str | int | float | list[str] | None] = OrderedDict(
            experiment=self.experiment.value
        )
        for coordinate in relevance:
            value = present.get(coordinate.value)
            if value is not None:
                values[coordinate.value] = value
        return stable_json(values)


class StableSerializationError(ValueError):
    pass


class StableJsonPayload(Protocol):
    __slots__ = ()


def stable_json(value: StableJsonPayload) -> str:
    return json.dumps(_stable_value(value), sort_keys=True, separators=(",", ":"))


def _stable_value(value: StableJsonPayload) -> JsonValue:
    if is_dataclass(value) and not isinstance(value, type):
        return OrderedDict(
            (field.name, _stable_value(getattr(value, field.name))) for field in fields(value)
        )
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, StableJsonPayload], value)
        return OrderedDict(
            (str(key), _stable_value(item))
            for key, item in sorted(mapping.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence = cast(Sequence[StableJsonPayload], value)
        return [_stable_value(item) for item in sequence]
    if isinstance(value, Enum):
        return _stable_value(value.value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StableSerializationError(f"non-finite stable value: {value}")
        return value
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    raise StableSerializationError(f"unsupported stable value: {type(value).__name__}")
