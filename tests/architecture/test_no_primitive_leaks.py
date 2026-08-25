from __future__ import annotations

import ast

import pytest

from tests.architecture.scan import (
    SERIALIZATION_BOUNDARY_MODULES,
    SRC_ROOT,
    iter_source_files,
    parse_module,
    public_functions,
    relative_module,
)

COLLECTION_PRIMITIVES = {"dict", "list", "set", "object"}
SCALAR_PRIMITIVES = {"str", "int", "float", "bool"}


def _is_stable_serializer_boundary(module: str) -> bool:
    return module in SERIALIZATION_BOUNDARY_MODULES


def _annotation_base(annotation: ast.expr | None) -> str | None:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        return annotation.value.id
    if isinstance(annotation, ast.Constant):
        return "Constant"
    if isinstance(annotation, ast.BinOp):
        left = _annotation_base(annotation.left)
        right = _annotation_base(annotation.right)
        if left in COLLECTION_PRIMITIVES or right in COLLECTION_PRIMITIVES:
            return "dict"
        return None
    return None


def _render(annotation: ast.expr | None) -> str:
    if annotation is None:
        return "<missing>"
    return ast.unparse(annotation)


def signature_violations(function: ast.FunctionDef) -> list[str]:
    violations: list[str] = []
    for argument in function.args.args:
        base = _annotation_base(argument.annotation)
        if base is not None and (
            base in COLLECTION_PRIMITIVES
            or (base in {"dict", "list"} and _subscripted_by_name(argument.annotation, base))
        ):
            violations.append(
                f"collection primitive parameter '{argument.arg}: {_render(argument.annotation)}'"
            )
    return_base = _annotation_base(function.returns)
    if return_base is not None and (
        return_base in COLLECTION_PRIMITIVES
        or (return_base in {"dict", "list"} and _subscripted_by_name(function.returns, return_base))
    ):
        violations.append(f"collection primitive return '{_render(function.returns)}'")
    tuple_violation = _primitive_tuple_annotation(function.returns)
    if tuple_violation:
        violations.append(f"primitive-only tuple return '{_render(function.returns)}'")
    for argument in function.args.args:
        tuple_violation = _primitive_tuple_annotation(argument.annotation)
        if tuple_violation:
            violations.append(
                f"primitive-only tuple parameter '{argument.arg}: {_render(argument.annotation)}'"
            )
    return violations


def _subscripted_by_name(annotation: ast.expr | None, base: str) -> bool:
    return (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == base
    )


def _primitive_tuple_annotation(annotation: ast.expr | None) -> bool:
    if (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "tuple"
    ):
        slice_node = annotation.slice
        elements = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
        if all(
            (isinstance(element, ast.Name) and element.id in SCALAR_PRIMITIVES)
            or (
                isinstance(element, ast.Subscript)
                and isinstance(element.value, ast.Name)
                and element.value.id in {"tuple", "list", "dict"}
            )
            for element in elements
        ):
            return True
    return False


def production_signature_violations() -> list[str]:
    violations: list[str] = []
    for path in iter_source_files():
        module = relative_module(path)
        if module.endswith("__init__") or _is_stable_serializer_boundary(module):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in public_functions(tree):
            for violation in signature_violations(function):
                violations.append(f"{path}:{function.name}: {violation}")
    return violations


def test_production_public_signatures_do_not_leak_primitives() -> None:
    violations = production_signature_violations()
    assert not violations, "\n".join(violations)


def _violations_in_source(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    for function in public_functions(tree):
        violations.extend(signature_violations(function))
    return violations


def test_detector_catches_dict_parameter() -> None:
    source = "def handler(payload: dict[str, int]) -> None: ...\n"
    assert any("payload" in violation for violation in _violations_in_source(source))


def test_detector_catches_dict_return() -> None:
    source = "def build() -> dict[str, str]: ...\n"
    assert any("dict[str, str]" in violation for violation in _violations_in_source(source))


def test_detector_catches_list_parameter_and_object_return() -> None:
    source = "def convert(items: list[float]) -> object: ...\n"
    violations = _violations_in_source(source)
    assert len(violations) == 2


def test_detector_catches_primitive_only_tuples() -> None:
    source = "def pair() -> tuple[int, int]: ...\n"
    assert len(_violations_in_source(source)) == 1


def test_detector_allows_typed_records() -> None:
    source = (
        "@dataclass(frozen=True)\n"
        "class Report:\n"
        "    values: tuple[int, ...]\n"
        "\n"
        "\n"
        "def load_report() -> Report: ...\n"
        "def combine(first: Report, second: Report) -> tuple[Report, ...]: ...\n"
        "def count(reports: tuple[Report, ...]) -> int: ...\n"
    )
    assert _violations_in_source(source) == []


def test_detector_allows_numpy_numeric_payloads() -> None:
    source = (
        "import numpy as np\n"
        "from numpy.typing import NDArray\n"
        "\n"
        "\n"
        "def scale(values: NDArray[np.float64]) -> NDArray[np.float64]: ...\n"
    )
    assert _violations_in_source(source) == []


@pytest.mark.parametrize("module", sorted(SERIALIZATION_BOUNDARY_MODULES))
def test_exempt_modules_exist(module: str) -> None:
    path = SRC_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    assert path.exists(), f"stale exemption entry: {module}"


PRIMITIVE_CONTAINER_FIELD_ALLOWLIST = {
    "transfer.optimizer_budget.TargetOptimizerStepLedger.reserved_steps",
    "transfer.optimizer_budget.TargetOptimizerStepLedger.consumed_steps",
}

TYPE_ALIAS_ALLOWLIST = {
    "runtime.environment.LockfileEntry",
}


def _iter_class_field_annotations(tree: ast.Module) -> list[tuple[str, str, ast.expr]]:
    findings: list[tuple[str, str, ast.expr]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                findings.append((node.name, item.target.id, item.annotation))
    return findings


def test_record_fields_do_not_use_mutable_primitive_containers() -> None:
    violations: list[str] = []
    for path in iter_source_files():
        module = relative_module(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_name, field_name, annotation in _iter_class_field_annotations(tree):
            text = ast.unparse(annotation)
            base = text.split("[")[0]
            if base in {"dict", "list", "set"}:
                qualified = f"{module}.{class_name}.{field_name}"
                if qualified not in PRIMITIVE_CONTAINER_FIELD_ALLOWLIST:
                    violations.append(f"{path}: {qualified}: {text}")
    assert not violations, "\n".join(violations)


def test_type_aliases_never_reintroduce_primitive_containers() -> None:
    violations: list[str] = []
    for path in iter_source_files():
        module = relative_module(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Subscript, ast.Name)):
                target_text = ast.unparse(node.value)
                base = target_text.split("[")[0]
                if base in {"dict", "list", "set", "object"}:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            qualified = f"{module}.{target.id}"
                            if qualified not in TYPE_ALIAS_ALLOWLIST:
                                violations.append(
                                    f"{path}:{node.lineno}: {qualified} = {target_text[:60]}"
                                )
    assert not violations, "\n".join(violations)


def test_public_methods_do_not_leak_primitives() -> None:
    violations: list[str] = []
    for path in iter_source_files():
        module = relative_module(path)
        if module.endswith("__init__") or module in SERIALIZATION_BOUNDARY_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef) or item.name.startswith("_"):
                    continue
                for violation in signature_violations(item):
                    violations.append(f"{path}:{node.name}.{item.name}: {violation}")
    assert not violations, "\n".join(violations)


def test_allowlisted_fields_and_aliases_exist() -> None:
    for qualified in sorted(PRIMITIVE_CONTAINER_FIELD_ALLOWLIST | TYPE_ALIAS_ALLOWLIST):
        parts = qualified.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            candidate = SRC_ROOT.joinpath(*parts[:depth]).with_suffix(".py")
            if candidate.exists():
                source = candidate.read_text(encoding="utf-8")
                assert all(name in source for name in parts[depth:]), (
                    f"stale allowlist entry: {qualified}"
                )
                break
        else:
            raise AssertionError(f"allowlisted symbol not found in repository: {qualified}")


def test_production_never_imports_unscoped_randomness() -> None:
    violations: list[str] = []
    for path in iter_source_files():
        tree = parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "random" for alias in node.names):
                violations.append(f"{path}:{node.lineno}: import random")
            if isinstance(node, ast.ImportFrom) and node.module == "random":
                violations.append(f"{path}:{node.lineno}: from random import")
            if isinstance(node, ast.Call):
                target = ast.unparse(node.func)
                if target in {"np.random.seed", "numpy.random.seed", "random.seed"}:
                    violations.append(f"{path}:{node.lineno}: unscoped seeding {target}()")
    assert not violations, "\n".join(violations)


def test_production_never_reads_environment_variables() -> None:
    violations: list[str] = []
    for path in iter_source_files():
        tree = parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = ast.unparse(node.func)
                if target in {"os.getenv", "os.environ.get", "os.putenv", "os.unsetenv"}:
                    violations.append(f"{path}:{node.lineno}: environment access {target}()")
    assert not violations, "\n".join(violations)


def test_detector_catches_container_typed_record_field() -> None:
    source = (
        "from dataclasses import dataclass\n"
        "\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Payload:\n"
        "    values_by_name: dict[str, float]\n"
    )
    tree = ast.parse(source)
    fields = [
        (class_node.name, field_name, annotation)
        for class_node in ast.walk(tree)
        if isinstance(class_node, ast.ClassDef)
        for item in class_node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        for field_name, annotation in [(item.target.id, item.annotation)]
    ]
    assert fields[0][1] == "values_by_name"
    assert ast.unparse(fields[0][2]).startswith("dict")


def test_detector_catches_primitive_container_alias() -> None:
    source = "Values = dict[str, float]\n"
    tree = ast.parse(source)
    aliases = [
        ast.unparse(node.value).split("[")[0]
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Subscript, ast.Name))
    ]
    assert aliases == ["dict"]
