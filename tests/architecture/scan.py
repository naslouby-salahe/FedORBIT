from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src" / "fedorbit"
TESTS_ROOT = REPOSITORY_ROOT / "tests"

ALLOWED_ROOT_ENTRIES = {
    ".env",
    ".git",
    "CLAUDE.md",
    ".gitignore",
    ".vscode",
    "LICENSE",
    "Makefile",
    "README.md",
    "configs",
    "data",
    "docs",
    "noxfile.py",
    "pyproject.toml",
    "src",
    "stubs",
    "tests",
    "uv.lock",
    "vulture_whitelist.py",
}

PACKAGE_LAYERS: dict[str, int] = {
    "domain": 0,
    "config": 1,
    "runtime": 2,
    "artifacts": 2,
    "datasets": 3,
    "models": 3,
    "strict_interface": 3,
    "oracle": 3,
    "synthetic": 3,
    "training": 4,
    "response": 4,
    "orbit": 4,
    "solvers": 4,
    "baselines": 4,
    "transfer": 4,
    "evaluation": 4,
    "experiments": 5,
    "analysis": 5,
    "execution": 6,
    "reporting": 7,
    "cli": 8,
}

FORBIDDEN_EDGES: dict[str, frozenset[str]] = {
    "reporting": frozenset(
        {
            "datasets",
            "models",
            "strict_interface",
            "oracle",
            "synthetic",
            "training",
            "response",
            "orbit",
            "solvers",
            "baselines",
            "transfer",
            "evaluation",
            "experiments",
            "analysis",
            "execution",
            "cli",
        }
    ),
    "execution": frozenset({"reporting", "cli"}),
    "artifacts": frozenset({"reporting", "cli", "execution", "analysis", "experiments"}),
    "runtime": frozenset({"cli", "reporting", "execution"}),
    "config": frozenset({"cli", "reporting", "execution", "artifacts", "runtime"}),
    "domain": frozenset(PACKAGE_LAYERS.keys()) - {"domain"},
}

VAGUE_MODULE_NAMES = {
    "utils",
    "helpers",
    "common",
    "manager",
    "processor",
    "base",
    "misc",
    "tools",
    "shared",
    "stuff",
    "things",
    "util",
    "helper",
    "misc_utils",
}

BANNED_NAME_FRAGMENTS = (
    "v2",
    "final2",
    "_new",
    "_old",
    "copy2",
    "tmp",
    "temp_",
    "dummy",
    "placeholder",
    "wip",
)

FORBIDDEN_VOCABULARY = (
    "dense exact solver",
    "exact dense solver",
    "dense-exact",
    "privacy guarantee",
    "byzantine-robust",
    "universal transfer",
    "federated learning",
    "v2.0",
    "v3.0",
)

LOCKED_VALUE_CONSTANT_PATTERN = {
    "PRINCIPAL_SPARSE_SUPPORT",
    "SPARSE_SUPPORT_SENSITIVITY",
    "TOTAL_CURRICULUM_BUDGET",
    "COORDINATE_CAP",
    "LINEAR_COST_PER_ACTIONABLE_NODE",
    "MAXIMUM_SOURCE_PROPOSALS_PER_TARGET",
    "COUPLING_OBJECTIVE_UNITS",
    "REALIZED_RELATIVE_MACRO_CE",
    "MACRO_F1_ABSOLUTE",
    "SOURCE_TRAIN_MINIMUM",
    "SOURCE_META_MINIMUM",
    "TARGET_META_MINIMUM",
    "TARGET_CONFIRM_MINIMUM",
    "TARGET_TEST_MINIMUM",
    "MISSING_INDICATOR_TRAIN_RATE_THRESHOLD",
    "RARE_CATEGORY_TRAIN_FREQUENCY_THRESHOLD",
    "FEATURE_MISSING_OR_NONFINITE_DROP_THRESHOLD",
    "CLIENT_INVALIDITY_DROPPED_FEATURE_FRACTION_THRESHOLD",
    "MAXIMUM_EPOCHS",
    "BATCH_SIZE",
    "GRADIENT_CLIP_GLOBAL_L2_NORM",
    "PATIENCE_COMPLETED_EPOCHS",
    "MINIMUM_IMPROVEMENT",
    "LABEL_SMOOTHING",
    "DATALOADER_WORKERS",
    "STATISTICAL_SEED",
    "CONFIDENCE_LEVEL",
    "CI_BOOTSTRAP_REPETITIONS",
    "MINIMUM_VALID_PAIRED_SEEDS",
    "TOST_ALPHA_PER_ONE_SIDED_TEST",
    "SPEARMAN_MINIMUM_VALID_POINTS",
    "MCNEMAR_EXACT_TO_ASYMPTOTIC_DISCORDANT_PAIR_SWITCH",
    "LP_PRIMAL_FEASIBILITY_TOLERANCE",
    "LP_DUAL_FEASIBILITY_TOLERANCE",
    "LP_OPTIMALITY_TOLERANCE",
    "SEPARATOR_CUT_STOPPING_TOLERANCE",
    "EXACT_VALIDATION_ABSOLUTE_TOLERANCE",
    "PERMUTATION_CERTIFICATE_RESIDUAL_TOLERANCE",
    "ACTION_TIE_TOLERANCE",
    "ACTION_TIE_COMPARISON_ROUNDING_PRECISION",
    "LAP_OBJECTIVE_TIE_TOLERANCE",
    "MAXIMUM_CUTS_PER_SUPPORT",
    "LP_THREADS_PER_SOLVE",
    "MAXIMUM_CONCURRENT_SUPPORTS",
    "DETERMINISTIC_RANDOM_SEED",
    "RETRIES_AFTER_INITIAL_INFRASTRUCTURE_FAILURE",
    "SOLVER_CPU_WORKER_CEILING",
    "HOST_RAM_CEILING_GIB_FOR_REGISTERED_EFFICIENCY_RUNS",
    "DETERMINISTIC_KERNEL_WARMUPS",
    "DETERMINISTIC_KERNEL_TIMED_REPETITIONS",
    "SCIENTIFIC_METRIC_DECIMALS",
    "MACRO_F1_DECIMALS",
    "BALANCED_ACCURACY_DECIMALS",
    "P_VALUE_DECIMALS",
    "P_VALUE_LESS_THAN_THRESHOLD",
    "RUNTIME_SECONDS_DECIMALS",
    "MEMORY_DECIMALS",
    "CLASS_RISK_FLOOR",
    "PROBABILITY_LOG_FLOOR",
}

BOUNDARY_PACKAGES = frozenset({"domain", "config", "artifacts", "reporting", "cli"})

CANONICAL_SERIALIZER_BOUNDARY_MODULES = frozenset(
    {
        "domain.canonical",
        "runtime.seeds",
        "artifacts.manifests",
        "artifacts.serialization",
        "artifacts.evidence",
    }
)

TODO_MARKERS = ("TODO", "FIXME", "HACK", "XXX")


@dataclass(frozen=True, slots=True)
class ImportEdge:
    source_module: str
    target_package: str
    lineno: int


def iter_source_files() -> tuple[Path, ...]:
    return tuple(path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def iter_test_files() -> tuple[Path, ...]:
    return tuple(path for path in TESTS_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def relative_module(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    return ".".join(relative.parts)


def package_of(module_name: str) -> str:
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[1] in PACKAGE_LAYERS:
        return parts[1]
    return parts[0]


def parse_module(path: Path) -> ast.Module:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tree


def import_edges(path: Path) -> tuple[ImportEdge, ...]:
    module = relative_module(path)
    tree = parse_module(path)
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name
                if target == "fedorbit" or target.startswith("fedorbit."):
                    parts = target.split(".")
                    target_package = parts[1] if len(parts) > 1 else "fedorbit"
                    edges.append(ImportEdge(module, target_package, node.lineno))
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "fedorbit" or node.module.startswith("fedorbit."))
        ):
            parts = node.module.split(".")
            target_package = parts[1] if len(parts) > 1 else "fedorbit"
            edges.append(ImportEdge(module, target_package, node.lineno))
    return tuple(edges)


def all_import_edges() -> tuple[ImportEdge, ...]:
    edges: list[ImportEdge] = []
    for path in iter_source_files():
        edges.extend(import_edges(path))
    return tuple(edges)


def package_dependency_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {package: set() for package in PACKAGE_LAYERS}
    for edge in all_import_edges():
        source_package = package_of(edge.source_module)
        if edge.target_package in PACKAGE_LAYERS:
            graph[source_package].add(edge.target_package)
    return graph


def public_functions(module: ast.Module) -> tuple[ast.FunctionDef, ...]:
    return tuple(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    )


def public_classes(module: ast.Module) -> tuple[ast.ClassDef, ...]:
    return tuple(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    )


def module_level_constants(module: ast.Module) -> tuple[ast.Assign, ...]:
    return tuple(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.isupper()
    )


def comments_and_docstrings(module: ast.Module) -> tuple[int, ...]:
    lines: list[int] = []
    for node in ast.walk(module):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                lines.append(body[0].lineno)
    return tuple(sorted(set(lines)))


def reexport_only_module(module: ast.Module) -> bool:
    if not module.body:
        return False
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        return False
    return True
