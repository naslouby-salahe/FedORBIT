"""Diagnostic: separate support-calibration error from later benign shift.

Uses only benign rows for adaptation and thresholding. An IID bootstrap holdout
from the support pool is compared with the same fitted model on ordered later
blocks. The released N-BaIoT benign CSV has no absolute timestamps, so this is
an ordered-row diagnostic, not verified wall-clock chronology.
"""
from __future__ import annotations

import csv
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise
from fedorbit.detection.metrics import threshold_at_rate

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "same_period_vs_future_fpr.csv"
DEVICES = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)
SUPPORT_SIZES = (30, 100, 300, 1000)
REPLICATES = 15
SEED = 202609301
RATE = 0.01
EPS = 1e-6
EVAL_ROWS = 10000


def local_builder(rows):
    scale = rows.std(axis=0) + EPS
    return scorer_from_rows(rows, rows.mean(axis=0), scale).score


def shared_builder(rows, shared):
    scale = shared.standard_deviation + EPS
    return scorer_from_rows(rows, shared.mean, scale).score


def crossfit_threshold(rows, builder):
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    scores = np.concatenate((builder(second)(first), builder(first)(second)))
    return threshold_at_rate(scores, RATE)


def sample_rows(rows, rng):
    if len(rows) <= EVAL_ROWS:
        return rows
    return rows[rng.choice(len(rows), size=EVAL_ROWS, replace=False)]


def main():
    data = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    records = []
    for target in DEVICES:
        peers = [summarise(data[name]["support_pool"]) for name in DEVICES if name != target]
        shared = aggregate_equal_weight(peers)
        pool = data[target]["support_pool"]
        midpoint = len(pool) // 2
        early, same_period = pool[:midpoint], pool[midpoint:]
        future = data[target]["test_benign"]
        for n in SUPPORT_SIZES:
            if len(early) < n:
                continue
            for rep in range(REPLICATES):
                seed = SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode())
                rng = np.random.default_rng(seed)
                start = int(rng.integers(0, len(early) - n + 1))
                rows = early[start:start + n]
                same_eval = sample_rows(same_period, rng)
                future_eval = sample_rows(future, rng)
                iid_eval = pool[rng.integers(0, len(pool), size=EVAL_ROWS)]
                row_keys = np.ascontiguousarray(rows).view(
                    np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
                ).ravel()
                same_keys = np.ascontiguousarray(same_eval).view(
                    np.dtype((np.void, same_eval.dtype.itemsize * same_eval.shape[1]))
                ).ravel()
                same_eval = same_eval[~np.isin(same_keys, row_keys)]
                for method, fit in (
                    ("local", local_builder),
                    ("shared_marginals", lambda current: shared_builder(current, shared)),
                ):
                    threshold = crossfit_threshold(rows, fit)
                    score = fit(rows)
                    fpr_same = float(np.mean(score(same_eval) > threshold))
                    fpr_future = float(np.mean(score(future_eval) > threshold))
                    fpr_iid = float(np.mean(score(iid_eval) > threshold))
                    records.append({
                        "device": target, "support_n": n, "replicate": rep,
                        "method": method, "offset": start,
                        "same_period_rows": len(same_eval),
                        "future_rows": len(future_eval),
                        "iid_rows": len(iid_eval),
                        "fpr_same_period": fpr_same,
                        "fpr_future": fpr_future,
                        "fpr_iid_support": fpr_iid,
                        "future_minus_iid": fpr_future - fpr_iid,
                    })
            print(target, n, "done", flush=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    import pandas as pd
    frame = pd.DataFrame(records)
    summary = frame.groupby(["support_n", "method"]).agg(
        same_period_fpr=("fpr_same_period", "mean"),
        iid_support_fpr=("fpr_iid_support", "mean"),
        future_fpr=("fpr_future", "mean"),
        future_minus_iid=("future_minus_iid", "mean"),
        devices_future_gt_iid=("future_minus_iid", lambda x: int((x.groupby(frame.loc[x.index, "device"]).mean() > 0).sum())),
    )
    print(summary.round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
