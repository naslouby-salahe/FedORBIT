from __future__ import annotations

import ast

from tests.architecture.scan import iter_source_files, relative_module


def _is_configuration_model(annotation: ast.expr | None) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id.endswith("Config")
    return isinstance(annotation, ast.Attribute) and annotation.attr.endswith("Config")


def test_operational_modules_do_not_accept_application_configuration() -> None:
    violations: list[str] = []
    for path in iter_source_files():
        if relative_module(path).split(".", 1)[0] == "config":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and _is_configuration_model(node.annotation):
                violations.append(f"{path}:{node.lineno}:configuration-field")
            if not isinstance(node, ast.FunctionDef):
                continue
            for parameter in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if _is_configuration_model(parameter.annotation):
                    violations.append(f"{path}:{node.name}:{parameter.arg}")
    assert not violations, "\n".join(violations)


def test_detector_catches_application_configuration_parameter() -> None:
    tree = ast.parse("def execute(config: FedorbitConfig) -> None: ...")
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    assert _is_configuration_model(function.args.args[0].annotation)


def test_detector_catches_application_configuration_field() -> None:
    tree = ast.parse("class Request:\n    configuration: FedorbitConfig\n")
    declaration = tree.body[0]
    assert isinstance(declaration, ast.ClassDef)
    field = declaration.body[0]
    assert isinstance(field, ast.AnnAssign)
    assert _is_configuration_model(field.annotation)


def test_detector_catches_nested_configuration_parameter() -> None:
    tree = ast.parse("def execute(settings: ExactSparseSolverConfig) -> None: ...")
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    assert _is_configuration_model(function.args.args[0].annotation)
