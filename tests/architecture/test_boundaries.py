from __future__ import annotations

from tests.architecture.scan import (
    FORBIDDEN_EDGES,
    PACKAGE_LAYERS,
    all_import_edges,
    iter_source_files,
    layer_of,
    package_dependency_graph,
    package_of,
)

ALLOWED_TEST_IMPORTS = {"tests", "fedorbit"}
PERMITTED_UPWARD_EDGES = {("datasets.common", "infrastructure.execution")}


def test_no_higher_layer_imports() -> None:
    violations: list[str] = []
    for edge in all_import_edges():
        target_package = edge.target_package
        if target_package.split(".", 1)[0] not in PACKAGE_LAYERS:
            continue
        if (
            layer_of(target_package) > layer_of(edge.source_module)
            and (edge.source_module, target_package) not in PERMITTED_UPWARD_EDGES
        ):
            violations.append(
                f"{edge.source_module}:{edge.lineno} imports higher layer {target_package}"
            )
    assert not violations, "\n".join(violations)


def test_no_forbidden_edges() -> None:
    violations: list[str] = []
    for edge in all_import_edges():
        source_package = package_of(edge.source_module)
        forbidden = FORBIDDEN_EDGES.get(source_package, frozenset())
        if edge.target_package.split(".", 1)[0] in forbidden:
            violations.append(
                f"{edge.source_module}:{edge.lineno} imports forbidden {edge.target_package}"
            )
    assert not violations, "\n".join(violations)


def test_no_import_cycles_between_packages() -> None:
    graph = {
        package: {dependent for dependent in dependents if dependent != package}
        for package, dependents in package_dependency_graph().items()
    }
    visiting: set[str] = set()
    visited: set[str] = set()
    path: list[str] = []

    def visit(package: str) -> None:
        if package in visiting:
            cycle_start = path.index(package)
            cycle = [*path[cycle_start:], package]
            raise AssertionError(f"import cycle between packages: {' -> '.join(cycle)}")
        if package in visited:
            return
        visiting.add(package)
        path.append(package)
        for dependent in sorted(graph[package]):
            visit(dependent)
        path.pop()
        visiting.discard(package)
        visited.add(package)

    for package in sorted(graph):
        visit(package)


def test_no_dynamic_imports_in_production() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        for marker in ("importlib.import_module", "__import__("):
            assert marker not in text, f"dynamic import in {path}: {marker}"


def test_no_production_imports_from_tests() -> None:
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        assert "from tests" not in text, f"production module imports tests: {path}"
        assert "import tests" not in text, f"production module imports tests: {path}"


def test_leaf_packages_do_not_import_execution_or_cli() -> None:
    leaf_packages = {
        "datasets",
        "learning",
        "response",
        "interface",
        "optimization",
        "methods",
        "oracle",
    }
    for edge in all_import_edges():
        source_package = package_of(edge.source_module)
        target_package = edge.target_package
        forbidden_targets = {
            "cli",
            "reporting",
            "experiments",
        }
        if source_package in leaf_packages and (
            target_package.split(".", 1)[0] in forbidden_targets
            or (target_package.startswith("analysis.") and target_package != "analysis.metrics")
            or (
                target_package.startswith("infrastructure.")
                and target_package != "infrastructure.runtime"
                and (source_package, target_package) != ("datasets", "infrastructure.execution")
            )
        ):
            raise AssertionError(
                f"{edge.source_module}:{edge.lineno} leaf package {source_package} "
                f"imports {target_package}"
            )
