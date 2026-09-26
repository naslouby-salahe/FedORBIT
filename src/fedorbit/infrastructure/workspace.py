from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedorbit.types import DatasetId, DeviceName, ExperimentId

PREPARED_DIRECTORY = "prepared"
RECORDS_DIRECTORY = "records"
ANALYSIS_FILE = "analysis.json"


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path
    results_directory: Path

    def prepared_path(self, dataset: DatasetId, device: DeviceName) -> Path:
        return self.root / PREPARED_DIRECTORY / dataset.value / f"{device}.npz"

    def records_path(self, experiment: ExperimentId) -> Path:
        return self.root / RECORDS_DIRECTORY / f"{experiment.value}.jsonl"

    def analysis_path(self) -> Path:
        return self.root / ANALYSIS_FILE
