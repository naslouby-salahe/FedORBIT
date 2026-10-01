"""Benign-only serial-dependence and effective-support audit for N-BaIoT.

The repository's N-BaIoT loader preserves each benign CSV's row order and
splits it by prefix. The source is sequential packet-behavior snapshots, but
has no absolute timestamps. This POC estimates featurewise HAC effective n on
the same contiguous support windows used by the cold-start mechanism POCs and
compares each estimate with a row-shuffled negative control. It reads no attack
files or labels.
"""
from __future__ import annotations

import csv
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "support_dependence_audit.csv"
DEVICES = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)
SUPPORT_SIZES = (30, 100, 300, 1000)
REPLICATES = 15
SEED = 202609302


def ess(values: np.ndarray, bandwidth_multiplier: float = 1.0) -> np.ndarray:
    """Bartlett/Newey-West ESS, one estimate per feature column."""
    n = len(values)
    centered = values - values.mean(axis=0, keepdims=True)
    denominator = np.maximum(np.sum(centered**2, axis=0), 1e-12)
    base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lagmax = min(n // 4, max(1, int(round(base * bandwidth_multiplier))))
    long_run_factor = np.ones(values.shape[1])
    for lag in range(1, lagmax + 1):
        autocov_ratio = np.sum(centered[:-lag] * centered[lag:], axis=0) / denominator
        long_run_factor += 2.0 * (1.0 - lag / (lagmax + 1.0)) * autocov_ratio
    return np.clip(n / np.maximum(long_run_factor, 1.0), 1.0, float(n))


def feature_influence(rows: np.ndarray, kind: str) -> np.ndarray:
    centered = rows - rows.mean(axis=0, keepdims=True)
    if kind == "mean":
        return centered
    if kind == "variance":
        squared = centered**2
        return squared - squared.mean(axis=0, keepdims=True)
    raise ValueError(kind)


def summarize(rows: np.ndarray, rng: np.random.Generator, kind: str,
              bandwidth_multiplier: float) -> tuple[float, float, float, float]:
    influence = feature_influence(rows, kind)
    ordered = ess(influence, bandwidth_multiplier)
    shuffled = influence[rng.permutation(len(influence))]
    iid = ess(shuffled, bandwidth_multiplier)
    return tuple(float(x) for x in (
        np.median(ordered), np.quantile(ordered, 0.1),
        np.median(iid), np.median(ordered / len(rows)),
    ))


def main() -> None:
    records: list[dict[str, object]] = []
    for device in DEVICES:
        pool = np.load(DATA / f"{device}.npz")["support_pool"]
        for n in SUPPORT_SIZES:
            metrics = {f"{kind}_{bw}": [] for kind in ("mean", "variance")
                       for bw in (1.0, 2.0)}
            for rep in range(REPLICATES):
                rng = np.random.default_rng(
                    SEED ^ zlib.crc32(f"{device}:{n}:{rep}".encode())
                )
                start = int(rng.integers(0, len(pool) - n + 1))
                rows = pool[start:start + n]
                for kind in ("mean", "variance"):
                    for bw in (1.0, 2.0):
                        metrics[f"{kind}_{bw}"].append(summarize(rows, rng, kind, bw))
            record: dict[str, object] = {"device": device, "support_n": n,
                                        "replicates": REPLICATES}
            for key, values in metrics.items():
                array = np.asarray(values)
                record[f"{key}_ess_median"] = float(np.median(array[:, 0]))
                record[f"{key}_ess_q10"] = float(np.median(array[:, 1]))
                record[f"{key}_shuffled_ess_median"] = float(np.median(array[:, 2]))
                record[f"{key}_fraction_nominal"] = float(np.median(array[:, 3]))
            records.append(record)
            print(device, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("wrote", len(records), OUT)


if __name__ == "__main__":
    main()
