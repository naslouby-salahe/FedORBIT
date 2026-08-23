from __future__ import annotations

from typer.testing import CliRunner

from fedorbit.cli.main import app


def test_smoke_command_executes_nonclaim_path() -> None:
    result = CliRunner().invoke(app, ["smoke"])
    assert result.exit_code == 0
