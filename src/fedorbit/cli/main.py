from __future__ import annotations

import typer

from fedorbit.cli.commands.doctor import doctor
from fedorbit.cli.commands.plan import plan
from fedorbit.cli.commands.preprocess import preprocess
from fedorbit.cli.commands.report import report
from fedorbit.cli.commands.run import run
from fedorbit.cli.commands.smoke import smoke
from fedorbit.cli.commands.status import status

app = typer.Typer(name="fedorbit", no_args_is_help=True)
app.command("doctor")(doctor)
app.command("preprocess")(preprocess)
app.command("plan")(plan)
app.command("smoke")(smoke)
app.command("run")(run)
app.command("status")(status)
app.command("report")(report)


def main() -> None:
    raise SystemExit(app())
