from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.scan import (
    BOUNDARY_PACKAGES,
    iter_source_files,
    package_of,
    parse_module,
    public_functions,
    relative_module,
)

COLLECTION_PRIMITIVES = {"dict", "list", "set", "object"}
SCALAR_PRIMITIVES = {"str", "int", "float", "bool"}


def test_boundary_functions_do_not_leak_collection_primitives() -> None:
    for path in iter_source_files():
        module = relative_module(path)
        if module.endswith("__init__"):
            continue
        package = package_of(module)
        if package not in BOUNDARY_PACKAGES:
            continue
        if (
            "fedorbit/domain/canonical.py" in str(path)
            or "fedorbit/runtime/seeds.py" in str(path)
            or "fedorbit/artifacts/manifests.py" in str(path)
            or "fedorbit/artifacts/serialization.py" in str(path)
            or "fedorbit/artifacts/evidence.py" in str(path)
        ):
            continue
        tree = parse_module(path)
        for function in public_functions(tree):
            for argument in function.args.args:
                _check_annotation(argument.annotation, path, function.name, "parameter")
            _check_annotation(function.returns, path, function.name, "return")


def test_boundary_functions_do_not_leak_primitive_only_tuples() -> None:
    for path in iter_source_files():
        module = relative_module(path)
        if module.endswith("__init__"):
            continue
        package = package_of(module)
        if package not in BOUNDARY_PACKAGES:
            continue
        tree = parse_module(path)
        for function in public_functions(tree):
            _check_tuple_shape(function.returns, path, function.name, "return")
            for argument in function.args.args:
                _check_tuple_shape(argument.annotation, path, function.name, argument.arg)


def _check_annotation(annotation: ast.expr | None, path: Path, owner: str, role: str) -> None:
    if annotation is None:
        return
    if isinstance(annotation, ast.Name) and annotation.id in COLLECTION_PRIMITIVES:
        raise AssertionError(f"collection primitive {annotation.id!r} {role} in {path}:{owner}")
    if (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id in COLLECTION_PRIMITIVES
    ):
        raise AssertionError(
            f"collection primitive {annotation.value.id!r} {role} in {path}:{owner}"
        )


def _check_tuple_shape(annotation: ast.expr | None, path: Path, owner: str, role: str) -> None:
    if annotation is None:
        return
    if isinstance(annotation, ast.Name) and annotation.id == "tuple":
        raise AssertionError(f"bare tuple {role} annotation in {path}:{owner}")
    if (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "tuple"
        and _all_elements_primitive(annotation)
    ):
        raise AssertionError(f"primitive-only tuple {role} annotation in {path}:{owner}")


def _all_elements_primitive(annotation: ast.Subscript) -> bool:
    slice_node = annotation.slice
    elements = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
    for element in elements:
        if isinstance(element, ast.Name) and element.id in SCALAR_PRIMITIVES:
            continue
        if (
            isinstance(element, ast.Subscript)
            and isinstance(element.value, ast.Name)
            and element.value.id in {"tuple", "list", "dict"}
        ):
            continue
        return False
    return True
