from __future__ import annotations

import typer

from fedorbit.config.loading import load_fedorbit_config
from fedorbit.execution.planner import build_plan


def plan() -> None:
    rows = build_plan(load_fedorbit_config())
    typer.echo(f"registered experiments: {len(rows)}")
    for row in rows:
        typer.echo(
            f"{row.experiment.value} | {row.classification.value} | "
            f"planned cells: {row.planned_cells}"
        )
