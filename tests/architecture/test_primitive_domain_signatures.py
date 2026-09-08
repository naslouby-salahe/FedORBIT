from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.scan import iter_source_files, parse_module, relative_module

MEASURED_KEYWORDS = (
    "count",
    "seed",
    "threshold",
    "fraction",
    "probability",
    "rate",
    "gain",
    "score",
    "index",
    "size",
    "scale",
    "weight",
    "magnitude",
    "tolerance",
    "floor",
    "epoch",
    "step",
    "budget",
    "support",
    "confidence",
    "alpha",
    "risk",
    "coverage",
    "horizon",
    "replicate",
    "resample",
    "attempt",
    "batch",
    "dropout",
    "precision",
    "objective",
    "truth_value",
    "node_count",
    "orbit_size",
    "client_count",
    "sample_count",
    "total",
    "remaining",
    "category",
)

IDENTITY_KEYWORDS = (
    "dataset",
    "policy",
    "status",
    "method",
    "mode",
    "strategy",
    "direction",
    "outcome",
    "split",
    "stage",
    "class",
    "category",
    "concept",
    "partition",
    "compatibility",
    "alternative",
    "client",
    "sparsity",
    "pair",
    "condition",
    "hash",
    "artifact",
    "fingerprint",
)

EXCLUDED_SUBSTRINGS = ("column", "id", "name", "label", "reason", "message", "field", "unit")
BOUNDARY_PACKAGES = ("reporting", "config")


def _violations(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        for argument in node.args.args:
            if argument.arg in ("self", "cls") or argument.annotation is None:
                continue
            annotation = ast.unparse(argument.annotation)
            name = argument.arg.lower()
            if any(fragment in name for fragment in EXCLUDED_SUBSTRINGS):
                continue
            if _contains_primitive(annotation, {"int", "float", "bool"}) and any(
                keyword in name for keyword in MEASURED_KEYWORDS
            ):
                violations.append(f"{node.name}({argument.arg}: {annotation})")
            if annotation == "str" and any(keyword in name for keyword in IDENTITY_KEYWORDS):
                violations.append(f"{node.name}({argument.arg}: {annotation})")
    return violations


def _contains_primitive(annotation: str, primitives: set[str]) -> bool:
    tree = ast.parse(annotation, mode="eval")
    return any(isinstance(node, ast.Name) and node.id in primitives for node in ast.walk(tree))


def _return_violations(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") or node.returns is None:
            continue
        annotation = ast.unparse(node.returns)
        name = node.name.lower()
        if _contains_primitive(annotation, {"int", "float", "bool"}) and any(
            keyword in name for keyword in MEASURED_KEYWORDS
        ):
            violations.append(f"{node.name}() -> {annotation}")
        if _contains_primitive(annotation, {"str"}) and any(
            keyword in name for keyword in IDENTITY_KEYWORDS
        ):
            violations.append(f"{node.name}() -> {annotation}")
    return violations


def _field_violations(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        annotation = ast.unparse(node.annotation)
        name = node.target.id.lower()
        if any(fragment in name for fragment in EXCLUDED_SUBSTRINGS):
            continue
        if annotation in {"int", "float", "bool"} and any(
            keyword in name for keyword in MEASURED_KEYWORDS
        ):
            violations.append(f"{node.target.id}: {annotation}")
        if annotation == "str" and any(keyword in name for keyword in IDENTITY_KEYWORDS):
            violations.append(f"{node.target.id}: {annotation}")
    return violations


def test_domain_public_signatures_use_canonical_domain_types() -> None:
    findings: list[str] = []
    for path in iter_source_files():
        module = relative_module(path)
        if any(module.startswith(boundary) for boundary in BOUNDARY_PACKAGES):
            continue
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            for argument in node.args.args:
                if argument.arg in ("self", "cls") or argument.annotation is None:
                    continue
                annotation = ast.unparse(argument.annotation)
                name = argument.arg.lower()
                if any(fragment in name for fragment in EXCLUDED_SUBSTRINGS):
                    continue
                if _contains_primitive(annotation, {"int", "float", "bool"}) and any(
                    keyword in name for keyword in MEASURED_KEYWORDS
                ):
                    findings.append(
                        f"{path}:{node.lineno}: {node.name}({argument.arg}: {annotation})"
                    )
                if annotation == "str" and any(keyword in name for keyword in IDENTITY_KEYWORDS):
                    findings.append(
                        f"{path}:{node.lineno}: {node.name}({argument.arg}: {annotation})"
                    )
        for violation in _return_violations(path.read_text(encoding="utf-8")):
            findings.append(f"{path}: {violation}")
    assert not findings, "\n".join(findings)


def test_checker_catches_primitive_seed_param() -> None:
    assert _violations("def run(seed: int) -> None: ...\n") == ["run(seed: int)"]


def test_checker_catches_primitive_threshold_param() -> None:
    assert _violations("def run(threshold: float) -> None: ...\n") == ["run(threshold: float)"]


def test_checker_catches_string_dataset_param() -> None:
    assert _violations("def load(dataset: str) -> None: ...\n") == ["load(dataset: str)"]


def test_checker_catches_primitive_class_collection_param() -> None:
    assert _violations("def run(classes: tuple[int, ...]) -> None: ...\n") == [
        "run(classes: tuple[int, ...])"
    ]


def test_checker_catches_primitive_weight_return() -> None:
    assert _return_violations("def weight_of() -> float: ...\n") == [
        "weight_of() -> float"
    ]


def test_checker_allows_typed_domain_param() -> None:
    source = "from fedorbit.types import SupportCount\ndef run(count: SupportCount) -> None: ...\n"
    assert _violations(source) == []


def test_checker_allows_identifier_string() -> None:
    assert _violations("def look(column_name: str) -> None: ...\n") == []


def test_checker_allows_implementation_scalar() -> None:
    assert _violations("def scale(values: list[float]) -> list[float]: ...\n") == []


def test_evaluation_record_fields_use_domain_types() -> None:
    records_path = Path(__file__).resolve().parents[2] / "src" / "fedorbit" / "analysis" / "records.py"
    assert _field_violations(records_path.read_text(encoding="utf-8")) == []
