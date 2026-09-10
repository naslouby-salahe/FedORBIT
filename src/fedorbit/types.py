from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, IntEnum, StrEnum
from pathlib import Path
from typing import Annotated, ClassVar, NewType, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def is_sha256_digest(value: str) -> bool:
    return SHA256_HEX.fullmatch(value) is not None


ClientComponentName = NewType("ClientComponentName", str)
DatasetRelativePath = NewType("DatasetRelativePath", str)
RawDatasetPath = NewType("RawDatasetPath", str)
DuplicateGroupIdentifier = NewType("DuplicateGroupIdentifier", str)
Sha256Digest = NewType("Sha256Digest", str)
TabularColumnName = NewType("TabularColumnName", str)
ValidationReason = NewType("ValidationReason", str)
ModelParameterName = NewType("ModelParameterName", str)
DirectedPairName = NewType("DirectedPairName", str)
EvaluationConditionName = NewType("EvaluationConditionName", str)
MetricUnit = NewType("MetricUnit", str)
InvalidReason = NewType("InvalidReason", str)
ContrastName = NewType("ContrastName", str)
StatisticalTestName = NewType("StatisticalTestName", str)
FieldDescription = NewType("FieldDescription", str)
FailureReason = NewType("FailureReason", str)
SupportRecordIdentifier = NewType("SupportRecordIdentifier", str)
CutMasterCounterName = NewType("CutMasterCounterName", str)
AvailabilityReason = NewType("AvailabilityReason", str)
ResourceLimitReason = NewType("ResourceLimitReason", str)
StrictResourceValidity = NewType("StrictResourceValidity", bool)
PValueName = NewType("PValueName", str)
BootstrapPurpose = NewType("BootstrapPurpose", str)
BootstrapDegeneracy = NewType("BootstrapDegeneracy", bool)
ArrayAxis = NewType("ArrayAxis", int)
ClassIndex = NewType("ClassIndex", int)
ContrastCoordinates = NewType("ContrastCoordinates", str)
SourceClientName = NewType("SourceClientName", str)
IneligibilityReason = NewType("IneligibilityReason", str)
ResponseSeedStage = NewType("ResponseSeedStage", str)
Rfc3339UtcTimestamp = NewType("Rfc3339UtcTimestamp", str)
SerializedPacket = NewType("SerializedPacket", str)
ExposedCoarseGroupId = NewType("ExposedCoarseGroupId", str)
GpuName = NewType("GpuName", str)
CudaVersion = NewType("CudaVersion", str)
CpuName = NewType("CpuName", str)
OperatingSystemRelease = NewType("OperatingSystemRelease", str)
PythonVersion = NewType("PythonVersion", str)
FilesystemSlug = NewType("FilesystemSlug", str)
ArtifactFileSuffix = NewType("ArtifactFileSuffix", str)
ArtifactSchemaVersion = NewType("ArtifactSchemaVersion", str)
ArtifactTypeName = NewType("ArtifactTypeName", str)
ArtifactPathText = NewType("ArtifactPathText", str)
ManifestValidationState = NewType("ManifestValidationState", str)
ArtifactLineage = NewType("ArtifactLineage", str)
SemanticCoordinateText = NewType("SemanticCoordinateText", str)
SolverVariablePrefix = NewType("SolverVariablePrefix", str)
MonotonicDeadline = NewType("MonotonicDeadline", float)
SolverStatus = NewType("SolverStatus", str)
TorchPrecision = NewType("TorchPrecision", str)
ExecutionStageName = NewType("ExecutionStageName", str)
ReuseDecision = NewType("ReuseDecision", str)
GitRevision = NewType("GitRevision", str)
ReportSeriesName = NewType("ReportSeriesName", str)
ReportAxisLabel = NewType("ReportAxisLabel", str)
ReportColumnName = NewType("ReportColumnName", str)
ReportArtifactName = NewType("ReportArtifactName", str)
ProducerModuleName = NewType("ProducerModuleName", str)
CorrespondenceBlockId = NewType("CorrespondenceBlockId", str)


class StorageLayoutSegment(StrEnum):
    MANIFESTS = "manifests"
    COMPLETIONS = "completions"
    STAGING = "staging"
    PREPROCESSING = "preprocessing"
    ARTIFACTS = "artifacts"
    DERIVED = "derived"
    MANIFEST_GLOB = "*.json"
    TEMPORARY_FILE_PREFIX = ".tmp-"


class CheckpointDirectorySegment(StrEnum):
    PILOT = "pilot"
    TRAINING = "training"


class InfrastructureLogCoordinate(StrEnum):
    RETRY = "infrastructure-retry"


class DatasetPreprocessingState(StrEnum):
    MATERIALIZED = "materialized"


class RawInventoryArtifact(StrEnum):
    INVENTORIES = "inventories"
    MANIFEST_JSON = "manifest.json"
    CHECKSUMS_JSON = "checksums.json"
    SCHEMA_JSON = "schema.json"


class DuplicateReportColumn(StrEnum):
    RAW_ROW_SHA256 = "raw_row_sha256"
    OCCURRENCE_COUNT = "occurrence_count"
    DUPLICATE_ROW_COUNT = "duplicate_row_count"


class FedorbitConfigSection(StrEnum):
    SCIENTIFIC = "scientific"
    SOLVERS = "solvers"


class RiskReductionColumn(StrEnum):
    ABSOLUTE_RISK_REDUCTION = "arr"
    RELATIVE_RISK_REDUCTION = "rrr"


class ProjectSummaryColumn(StrEnum):
    ARTIFACT_ID = "artifact_id"
    SEMANTIC_PRODUCER_COORDINATES = "semantic_producer_coordinates"
    PRODUCER_STAGE = "producer_stage"
    DEPENDENCY_FINGERPRINT_SHA256 = "dependency_fingerprint_sha256"


class SourceLabel(StrEnum):
    EDGE_IIOTSET = "Edge-IIoTset"
    TON_IOT = "ToN_IoT"


class ReportingPathSegment(StrEnum):
    METRICS = "metrics"
    TABLES = "tables"
    FIGURES = "figures"
    REPRODUCIBILITY = "reproducibility"
    EVIDENCE_SUFFIX = ".evidence.json"
    TABLE_SUFFIX = ".table.json"
    FIGURE_SUFFIX = ".figure.svg"
    SUMMARY_JSON = "summary.json"
    METRIC_RECORDS_CSV = "metric_records.csv"
    METRIC_RECORDS_TEX = "metric_records.tex"
    METRIC_VALUE_SVG = "metric_value.svg"
    METRIC_VALUE_PDF = "metric_value.pdf"
    EXPERIMENTS_CSV = "experiments.csv"
    EVIDENCE_SUMMARY_CSV = "evidence_summary.csv"
    SCIENTIFIC_CONFIGURATION_JSON = "scientific_configuration.json"
    EXECUTION_JSON = "execution.json"


TimestampFieldName = NewType("TimestampFieldName", str)
DatasetLabel = NewType("DatasetLabel", str)
FineLabel = NewType("FineLabel", str)
type DatasetIdentifierText = str
type ExperimentIdentifierText = str


NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
UINT32_LIMIT = 2**32


SupportCount = PositiveInt
ConceptCount = PositiveInt
FeatureCount = PositiveInt
ClassCount = PositiveInt
SampleCount = PositiveInt
ProposalCount = PositiveInt
EpochCount = PositiveInt
BatchSize = PositiveInt
CutCount = PositiveInt
ThreadCount = PositiveInt
ConcurrencyCount = PositiveInt
StepCount = PositiveInt
ReplicateCount = PositiveInt
ResampleCount = PositiveInt
RepetitionCount = PositiveInt
AttemptCount = PositiveInt
DurationMinutes = PositiveInt
TimeBudgetSeconds = PositiveInt
GiBMemory = PositiveInt
ResearcherCount = PositiveInt
PatienceCount = NonNegativeInt
WorkerCount = NonNegativeInt
DecimalPrecision = NonNegativeInt
InvalidPermutationCount = NonNegativeInt
RandomSeed = Annotated[int, Field(ge=0, lt=UINT32_LIMIT)]
DerivedSeed = Annotated[int, Field(ge=0, lt=UINT32_LIMIT)]
Index = NonNegativeInt
type CutMasterCounter = tuple[CutMasterCounterName, Index]
type CutMasterCounters = tuple[CutMasterCounter, ...]
ByteCount = NonNegativeInt
RetryCount = NonNegativeInt
AnonymousNodeIndex = Index
AnonymousNodeDisplayId = NewType("AnonymousNodeDisplayId", str)
type NodeIndices = tuple[Index, ...]
type NodeIndexList = list[Index]
type NodeImageMap = Mapping[Index, Index]


LearningRate = PositiveFloat
InterventionMagnitude = PositiveFloat
Tolerance = PositiveFloat
Floor = PositiveFloat
Budget = PositiveFloat
ConsumedBudget = NonNegativeFloat
ScaleFactor = NonNegativeFloat
Threshold = NonNegativeFloat
AbsoluteMetric = NonNegativeFloat
WeightDecay = NonNegativeFloat
Discrepancy = NonNegativeFloat
StandardError = NonNegativeFloat
ElapsedSeconds = NonNegativeFloat
MemoryMib = NonNegativeFloat
Coefficient = FiniteFloat
RelativeGain = FiniteFloat
Estimate = FiniteFloat
Score = FiniteFloat
Timestamp = FiniteFloat
ConfidenceLevel = FiniteFloat
SignificanceLevel = FiniteFloat
Fraction = FiniteFloat
type ReportCoordinates = tuple[Coefficient, ...]
type ReportColumns = tuple[ReportColumnName, ...]
type RawCellValue = str | int | float | None
RawCellText = NewType("RawCellText", str)
CategoryName = NewType("CategoryName", str)
type RawNumericCellValue = str | int | float
type RawCellSamples = tuple[RawCellValue, ...]
type TabularColumns = tuple[TabularColumnName, ...]
type TabularColumnSet = frozenset[TabularColumnName]
type ComponentColumns = tuple[TabularColumns, ...]
type LabelCounts = tuple[tuple[DatasetLabel, NonNegativeInt], ...]
type LabelText = DatasetLabel | FineLabel
type NativeLabels = tuple[FineLabel, ...]
type NativeLabelSet = frozenset[FineLabel]
type LocalClassNames = tuple[FineLabel, ...]
type ExcludedLocalClasses = tuple[tuple[FineLabel, NonNegativeInt], ...]
FeatureName = NewType("FeatureName", str)
NumericFeatureValue = NewType("NumericFeatureValue", float)
type FeatureNames = tuple[FeatureName, ...]
type RawTabularRow = dict[TabularColumnName, RawCellText]
type RawTabularRows = list[RawTabularRow]
type RawTabularColumns = list[RawCellText]
TimestampSeconds = FiniteFloat
type TimestampRange = tuple[TimestampSeconds, TimestampSeconds]
type CategoryVocabulary = tuple[CategoryName, ...]
type CategorySet = frozenset[CategoryName]
type TextToken = RawCellText | CategoryName
type FeatureValue = str | int | float | None
type FeatureValueMap = Mapping[TabularColumnName, FeatureValue]
type FeatureValuePartitions = tuple[FeatureValueMap, FeatureValueMap]
NormalizedGroupIdentifier = DuplicateGroupIdentifier
type NormalizedGroupIdentifiers = tuple[NormalizedGroupIdentifier, ...]
type ArtifactIdentifiers = tuple[ArtifactIdentifier, ...]
type ArtifactPathTexts = tuple[ArtifactPathText, ...]


class ClientRole(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    EXTERNAL = "external"
    SOURCE = "source"
    TARGET = "target"


class ExitStatus(IntEnum):
    OK = 0
    RUNTIME = 1
    USAGE = 2


class CliCommand(StrEnum):
    DOCTOR = "doctor"
    PREPROCESS = "preprocess"
    PLAN = "plan"
    SMOKE = "smoke"
    RUN = "run"
    STATUS = "status"
    REPORT = "report"


class RuntimeDeviceType(StrEnum):
    CUDA = "cuda"


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


class SemanticPartitionId(StrEnum):
    ORACLE_FINE_SINGLETON_GROUPS = "oracle_fine_singleton_groups"
    PRINCIPAL_THREE_COARSE_GROUPS = "principal_three_coarse_groups"
    ONE_ATTACK_SUPERGROUP = "one_attack_supergroup"


class DatasetModality(StrEnum):
    NETWORK = "network"
    HOST = "host"


class ResearchQuestion(StrEnum):
    EXACT_SPARSE_SEPARATOR_EXACTNESS = "Exact Sparse Separator Exactness"
    JOINT_CORRESPONDENCE_AVOIDS_RECTANGULAR_PESSIMISM = (
        "Joint Correspondence Avoids Rectangular Pessimism"
    )
    ACTION_CERTIFICATION_WITHOUT_FINE_MAP_IDENTIFICATION = (
        "Action Certification Without Fine-Map Identification"
    )
    STRICT_CROSS_TELEMETRY_TRANSFER_UTILITY = "Strict Cross-Telemetry Transfer Utility"
    VALUE_OF_EXTERNAL_PROCEDURAL_EVIDENCE = "Value of External Procedural Evidence"
    OPERATIONAL_RELEVANCE_OF_SPARSE_SUPPORT = "Operational Relevance of Sparse Support"
    TARGET_CONFIRMATION_SAFETY = "Target Confirmation Safety"
    SPARSE_SOLVER_WORK_STRUCTURE_AGREEMENT = "Sparse Solver Work-Structure Agreement"


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


class ExperimentLocalMethod(StrEnum):
    EXACT_ORBIT = "exact_orbit"
    EXACT_SPARSE_SUPPORT_ONE = "exact_sparse_support_one"
    EXACT_SPARSE_SUPPORT_TWO = "exact_sparse_support_two"
    EXACT_SPARSE_SUPPORT_THREE = "exact_sparse_support_three"


type MethodName = TransferMethod | ExperimentLocalMethod


class ComparisonStatistic(StrEnum):
    SIGN_FLIP_SUPERIORITY = "sign_flip_superiority"
    TOST_EQUIVALENCE = "tost_equivalence"
    SIGN_FLIP_AGAINST_ZERO = "sign_flip_against_zero"
    SIGN_FLIP_DIFFERENCE_COMMON_REFERENCE = "sign_flip_difference_common_reference"
    SEED_LEVEL_RATE_DIFFERENCE_SIGN_FLIP = "seed_level_rate_difference_sign_flip"


class ComparisonContrastSuffix(StrEnum):
    DIFFERENCE = "difference"
    TOST_EQUIVALENCE = "TOST equivalence"


class EfficiencyMetricName(StrEnum):
    CUDA_BYTES = "CUDA bytes"
    PACKET_BYTES = "packet bytes"
    SOURCE_RESPONSE_STEPS = "source response steps"
    CONFIRMATION_STEPS = "confirmation steps"
    ASSIMILATION_STEPS = "assimilation steps"


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
    EVIDENCE_CLASSIFICATION = "Evidence Classification"


class EvidenceStatus(StrEnum):
    SUPPORTED = "Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    MECHANISM_ONLY = "Mechanism Only"
    CONDITIONAL = "Conditional"
    NULL_RESULT = "Null Result"
    NOT_SUPPORTED = "Not Supported"
    NOT_TESTED = "Not Tested"


class SimplificationRuleState(StrEnum):
    APPLIED = "Applied"
    NOT_APPLIED = "Not Applied"
    NOT_TESTED = "Not Tested"


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
    ROBUSTNESS_EFFICIENCY = "Robustness / EFFICIENCY"


class ScalabilityBlockPattern(StrEnum):
    BALANCED = "balanced"
    MAXIMALLY_SKEWED = "maximally_skewed"
    MAXIMALLY_SKEWED_TWO_BLOCK = "maximally_skewed_two_block"


class CouplingCompatibility(StrEnum):
    JOINTLY_REALIZABLE = "jointly_realizable"
    INCOMPATIBLE = "incompatible"


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


class ConfigurationSection(StrEnum):
    ACTION = "action"
    GENERATORS = "generators"
    MODELS = "models"
    RESPONSE = "response"
    SOLVERS = "solvers"
    METRICS = "metrics"


class SemanticCoordinate(StrEnum):
    EXPERIMENT = "experiment"
    DATASET = "dataset"
    SOURCE_CLIENT = "source_client"
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
    WORK_STRUCTURE_SPEARMAN = "Work-Structure Spearman"
    PREDICTED_WORK_COORDINATE = "Predicted Work Coordinate"
    RESOURCE_LIMIT_INDICATOR = "Resource-Limit Indicator"
    PACKET_ONLY_RECOVERY_ACCURACY = "Packet-Only Recovery Accuracy"
    STRICT_RESOURCE_VALIDITY = "Strict Resource Validity"
    DETERMINISTIC_REPLAY_CONSISTENCY = "Deterministic Replay Consistency"
    ABSTENTION_INDICATOR = "Abstention Indicator"
    NULL_NODE_COUNT = "Null-Node Count"
    ORBIT_SIZE = "Orbit Size"


class DomainModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, validate_default=True, allow_inf_nan=False
    )


def _nonempty_text(value: str, label: str) -> None:
    if not value:
        raise ValueError(f"{label} must not be empty")


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
        _nonempty_text(self.value, "artifact identifier")


@dataclass(frozen=True, slots=True)
class ArtifactFingerprint:
    value: str

    def __post_init__(self) -> None:
        _nonempty_text(self.value, "artifact fingerprint")


@dataclass(frozen=True, slots=True)
class SemanticCoordinates:
    value: str

    def __post_init__(self) -> None:
        _nonempty_text(self.value, "semantic coordinates")


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
        if not 0 <= self.value < UINT32_LIMIT:
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
    def direction(self) -> DirectedPairName:
        return DirectedPairName(f"{self.source.value} -> {self.target.value}")


@dataclass(frozen=True, slots=True)
class SemanticCell:
    experiment: ExperimentName
    dataset: DatasetId | None = None
    source_client: DatasetId | None = None
    directed_pair: DirectedPair | None = None
    method: TransferMethod | None = None
    condition: ExperimentCondition | None = None
    support: SupportSize | None = None
    seed: ExperimentSeed | None = None

    def identity_json(self, relevance: frozenset[SemanticCoordinate]) -> str:
        present: OrderedDict[str, str | int | float | list[str] | None] = OrderedDict(
            dataset=self.dataset.value if self.dataset is not None else None,
            source_client=self.source_client.value if self.source_client is not None else None,
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
