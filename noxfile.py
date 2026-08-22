from __future__ import annotations

import nox


def uv_session(session: nox.Session, command: str) -> None:
    session.run("uv", "run", command, external=True)


@nox.session
def format_check(session: nox.Session) -> None:
    uv_session(session, "ruff format --check .")


@nox.session
def lint(session: nox.Session) -> None:
    uv_session(session, "ruff check .")


@nox.session
def typecheck(session: nox.Session) -> None:
    uv_session(session, "pyright")


@nox.session
def contract(session: nox.Session) -> None:
    uv_session(session, "pytest tests/unit/config -k contract -q")


@nox.session
def architecture(session: nox.Session) -> None:
    uv_session(session, "pytest tests/architecture -q")


@nox.session
def unit(session: nox.Session) -> None:
    uv_session(session, "pytest tests/unit -q")


@nox.session
def scientific(session: nox.Session) -> None:
    uv_session(session, "pytest tests/scientific -q")


@nox.session
def integration(session: nox.Session) -> None:
    uv_session(session, "pytest tests/integration -q")


@nox.session
def e2e(session: nox.Session) -> None:
    uv_session(session, "pytest tests/e2e -q")


@nox.session
def smoke(session: nox.Session) -> None:
    uv_session(session, "pytest tests/smoke -q")


@nox.session
def all(session: nox.Session) -> None:
    uv_session(session, "pytest -q")
