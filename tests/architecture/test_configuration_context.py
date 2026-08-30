from __future__ import annotations

import ast

from tests.architecture.scan import iter_source_files, relative_module


def _is_fedorbit_config(annotation: ast.expr | None) -> bool:
    return isinstance(annotation, ast.Name) and annotation.id == "FedorbitConfig"


def test_operational_modules_do_not_accept_application_configuration() -> None:
    violations: list[str] = []
    for path in iter_source_files():
        if relative_module(path).split(".", 1)[0] == "config":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for parameter in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if _is_fedorbit_config(parameter.annotation):
                    violations.append(f"{path}:{node.name}:{parameter.arg}")
    assert not violations, "\n".join(violations)


def test_detector_catches_application_configuration_parameter() -> None:
    tree = ast.parse("def execute(config: FedorbitConfig) -> None: ...")
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    assert _is_fedorbit_config(function.args.args[0].annotation)
