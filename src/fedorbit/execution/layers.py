from __future__ import annotations

from fedorbit.domain.enums import ExperimentName

EXECUTION_LAYERS = (
    "inputs",
    "preprocessing / splits",
    "training / checkpoint selection",
    "scoring and source/target risk derivation",
    "response-packet construction",
    "correspondence / action optimization",
    "target confirmation and live assimilation",
    "TEST evaluation",
    "statistical analysis",
    "reporting",
)


def layer_index(layer: str) -> int:
    for index, candidate in enumerate(EXECUTION_LAYERS):
        if candidate == layer:
            return index
    raise ValueError(f"unknown execution layer: {layer}")


PROGRAMME_PREREQUISITES = (
    ("environment diagnosis", None),
    ("raw-data identity", None),
    ("preprocessing", ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION),
    ("smoke validation", None),
    ("mathematical primitive validation", ExperimentName.MATHEMATICAL_PRIMITIVE_VALIDATION),
    ("exact-sparse theorem validation", ExperimentName.EXACT_SPARSE_THEOREM_EXHAUSTIVE_VALIDATION),
    (
        "dataset/client/resource validation",
        ExperimentName.DATASET_CLIENT_AND_STRICT_RESOURCE_VALIDATION,
    ),
    ("base-model pilot", ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT),
    ("base-model checkpoint selection", ExperimentName.BASE_MODEL_HYPERPARAMETER_PILOT),
    ("source-response pilot", ExperimentName.SOURCE_RESPONSE_ESTIMATOR_PILOT),
    ("final source-response bands", ExperimentName.FINAL_SOURCE_RESPONSE_BAND_VALIDATION),
    ("baseline/oracle validation", ExperimentName.BASELINE_AND_ORACLE_CORRECTNESS_VALIDATION),
    ("exact-sparse solver benchmark", ExperimentName.EXACT_SPARSE_SOLVER_BENCHMARK),
    ("synthetic coupling mechanism", ExperimentName.SYNTHETIC_COUPLING_MECHANISM_VALIDATION),
    ("real-packet coupling mechanism", ExperimentName.COUPLING_AND_MAP_BOUND_VALIDATION),
    ("unresolved-map action diagnostics", ExperimentName.COMMON_ACTION_UNDER_UNIDENTIFIED_MAP),
    ("principal strict transfer", ExperimentName.PRIMARY_STRICT_CROSS_TELEMETRY_TRANSFER),
    ("multi-source diagnostic", ExperimentName.MULTI_SOURCE_SELECTION_VALIDATION),
    ("mechanism ablations", ExperimentName.MECHANISM_ABLATIONS),
    ("sparsity/dense sensitivity", ExperimentName.SPARSITY_AND_DENSE_FALLBACK),
    ("confirmation/portability", ExperimentName.TARGET_CONFIRMATION_AND_PORTABILITY),
    ("secondary generalization", ExperimentName.SECONDARY_CROSS_MODALITY_GENERALIZATION),
    ("semantic sufficiency boundary", ExperimentName.SEMANTIC_SUFFICIENCY_FRONTIER),
    (
        "weak-signal/support/heterogeneity boundaries",
        ExperimentName.WEAK_SIGNAL_SUPPORT_AND_HETEROGENEITY_BOUNDARIES,
    ),
    ("map applicability audit", ExperimentName.MAP_AVAILABILITY_APPLICABILITY_AUDIT),
    ("scalability/efficiency", ExperimentName.SCALABILITY_AND_EFFICIENCY),
    ("statistical synthesis", ExperimentName.STATISTICAL_SYNTHESIS),
    ("claim adjudication", ExperimentName.CLAIM_EVIDENCE_ADJUDICATION),
    ("manuscript evidence export", None),
)
