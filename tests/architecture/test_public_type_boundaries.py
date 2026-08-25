from __future__ import annotations

import ast
import re

from tests.architecture.scan import (
    BOUNDARY_PACKAGES,
    SERIALIZATION_BOUNDARY_MODULES,
    iter_source_files,
    package_of,
    parse_module,
    public_classes,
    public_functions,
    relative_module,
)


def test_public_functions_have_complete_annotations() -> None:
    for path in iter_source_files():
        module_name = relative_module(path)
        if module_name.endswith("__init__"):
            continue
        tree = parse_module(path)
        for function in public_functions(tree):
            assert function.returns is not None, (
                f"public function missing return annotation: {path}:{function.name}"
            )
            for argument in function.args.args:
                assert argument.annotation is not None, (
                    f"public function missing parameter annotation: "
                    f"{path}:{function.name}({argument.arg})"
                )


def test_public_classes_have_annotated_attributes() -> None:
    allowed_unannotated = {"model_config", "__slots__", "__match_args__"}
    for path in iter_source_files():
        tree = parse_module(path)
        for klass in public_classes(tree):
            for node in klass.body:
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and (
                        target.id in allowed_unannotated or target.id.isupper()
                    ):
                        continue
                    raise AssertionError(f"unannotated class attribute in {path}:{klass.name}")


def test_boundary_packages_never_use_any_or_object() -> None:
    for path in iter_source_files():
        module = relative_module(path)
        if package_of(module) not in BOUNDARY_PACKAGES:
            continue
        tree = parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                _assert_annotation_clean(node.annotation, module, path, "attribute")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _assert_annotation_clean(node.returns, module, path, node.name)
                for argument in node.args.args:
                    _assert_annotation_clean(argument.annotation, module, path, argument.arg)


def _assert_annotation_clean(
    annotation: ast.expr | None, module: str, path: object, owner: str
) -> None:
    if annotation is None:
        return
    if isinstance(annotation, ast.Name) and annotation.id in {"Any", "object"}:
        if module in SERIALIZATION_BOUNDARY_MODULES:
            return
        raise AssertionError(f"forbidden {annotation.id!r} annotation in {path}:{owner}")
    if isinstance(annotation, ast.Subscript):
        _assert_annotation_clean(annotation.value, module, path, owner)


def test_boundary_packages_never_return_anonymous_dicts() -> None:
    for path in iter_source_files():
        package = package_of(relative_module(path))
        if package not in BOUNDARY_PACKAGES:
            continue
        if relative_module(path).endswith("__init__"):
            continue
        tree = parse_module(path)
        for function in public_functions(tree):
            _assert_not_dict_annotation(function.returns, path, function.name, "return")


def test_boundary_packages_never_take_anonymous_dict_parameters() -> None:
    for path in iter_source_files():
        package = package_of(relative_module(path))
        if package not in BOUNDARY_PACKAGES:
            continue
        if relative_module(path).endswith("__init__"):
            continue
        tree = parse_module(path)
        for function in public_functions(tree):
            for argument in function.args.args:
                _assert_not_dict_annotation(argument.annotation, path, function.name, argument.arg)


def test_boundary_packages_never_declare_bare_collection_returns() -> None:
    for path in iter_source_files():
        package = package_of(relative_module(path))
        if package not in BOUNDARY_PACKAGES:
            continue
        if relative_module(path).endswith("__init__"):
            continue
        tree = parse_module(path)
        for function in public_functions(tree):
            if function.returns is None:
                continue
            if _is_bare_collection(function.returns):
                raise AssertionError(f"bare collection return annotation in {path}:{function.name}")


def _is_bare_collection(annotation: ast.expr) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id in {"dict", "list", "set", "tuple", "object"}
    return False


def _assert_not_dict_annotation(
    annotation: ast.expr | None, path: object, owner: str, role: str
) -> None:
    if (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "dict"
    ):
        raise AssertionError(f"anonymous dict {role} annotation in {path}:{owner}")


def test_any_never_appears_in_production_annotations() -> None:
    pattern = re.compile(r"\bAny\b")
    for path in iter_source_files():
        tree = parse_module(path)
        source_text = path.read_text(encoding="utf-8")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotation_source = ast.get_source_segment(source_text, node) or ""
                if pattern.search(annotation_source):
                    raise AssertionError(f"Any in annotation at {path}:{node.name}")
