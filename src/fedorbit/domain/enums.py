from __future__ import annotations

from enum import StrEnum


class ClientRole(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class DatasetId(StrEnum):
    EDGE_IIOTSET_NETWORK = "edge_iiotset_network"
    TON_IOT_WINDOWS10_HOST = "ton_iot_windows10_host"
    TON_IOT_LINUX_PROCESS_HOST = "ton_iot_linux_process_host"
    TON_IOT_NETWORK = "ton_iot_network"


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
    CLAIM_EVIDENCE_ADJUDICATION = "Claim-Evidence Adjudication"


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


class SolverId(StrEnum):
    EXACT_SPARSE = "exact_sparse"
    GENERIC_EXACT_QAP = "generic_exact_qap"
    DENSE_CCP = "dense_ccp"


class ArtifactState(StrEnum):
    MISSING = "Missing"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    INVALID = "Invalid"
    STALE = "Stale"
    BLOCKED = "Blocked"


class TerminalState(StrEnum):
    COMPLETED = "Completed"
    FAILED_INFRASTRUCTURE = "Failed / Infrastructure Failure"
    INVALID = "Invalid"
    FAILED_VALIDATION = "Failed / Validation Failure"
    FAILED_SCIENTIFIC_ALGORITHMIC = "Failed / Scientific Algorithmic Failure"
    TIME_LIMIT = "Time Limit"
    RESOURCE_LIMIT = "Resource Limit"


class ClaimState(StrEnum):
    SUPPORTED = "Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    MECHANISM_ONLY = "Mechanism Only"
    CONDITIONAL = "Conditional"
    NULL_RESULT = "Null Result"
    NOT_SUPPORTED = "Not Supported"
    NOT_TESTED = "Not Tested"


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


class SeedRole(StrEnum):
    PILOT = "pilot"
    CONFIRMATORY = "confirmatory"
    STATISTICAL = "statistical"


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


class ClaimId(StrEnum):
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
