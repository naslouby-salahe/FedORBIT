"""Robust continuous aggregation of peer predictive marginal scales.

For each peer, form within-device variance plus its squared centre offset from
the peer population centre. Compare arithmetic, trimmed, median, geometric,
and Huber aggregation of this predictive marginal quantity. Target scores use
the target-local centre and OAS geometry; shared-marginal OAS remains the
reference. No attack information enters the scale aggregation.
"""
from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "robust_predictive_peer_scale.csv"
DEVICES = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)
SUPPORT_SIZES = (30, 100, 300, 1000)
REPLICATES = 8
SEED = 202609311
RATE = 0.01
FLOOR = 1e-9
EVAL_ROWS = 5000


def peer_scales(peer_rows):
    means = np.stack([x.mean(axis=0) for x in peer_rows])
    variances = np.stack([x.var(axis=0, ddof=1) for x in peer_rows])
    grand_mean = means.mean(axis=0)
    predictive = np.maximum(variances + (means - grand_mean[None, :])**2, FLOOR)
    ordered = np.sort(predictive, axis=0)
    trimmed = ordered[1:-1].mean(axis=0)
    logv = np.log(predictive)
    med = np.median(logv, axis=0)
    mad = 1.4826 * np.median(np.abs(logv - med), axis=0)
    huber = np.mean(med[None, :] + np.clip(logv - med[None, :],
                                           -1.345 * np.maximum(mad, 1e-3)[None, :],
                                           1.345 * np.maximum(mad, 1e-3)[None, :]), axis=0)
    return {
        "peer_arithmetic": predictive.mean(axis=0),
        "peer_trimmed": trimmed,
        "peer_median": np.median(predictive, axis=0),
        "peer_geometric": np.exp(np.mean(logv, axis=0)),
        "peer_huber_log": np.exp(huber),
    }


def fit_score(rows, scale, center_kind="local"):
    center = rows.mean(axis=0) if isinstance(center_kind, str) and center_kind == "local" else center_kind
    return scorer_from_rows(rows, center, np.sqrt(np.maximum(scale, FLOOR)) + 1e-6).score


def crossfit(rows, scale, center_kind="local"):
    middle = len(rows) // 2
    first, second = rows[:middle], rows[middle:]
    return np.r_[fit_score(second, scale, center_kind)(first),
                 fit_score(first, scale, center_kind)(second)]


def evaluate(benign, attack, threshold):
    y = np.r_[np.zeros(len(benign)), np.ones(len(attack))]
    s = np.r_[benign, attack]
    return (float(roc_auc_score(y, s)), float(roc_auc_score(y, s, max_fpr=RATE)),
            float(np.mean(benign > threshold)), float(np.mean(attack > threshold)))


def main():
    devices = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    records = []
    for target in DEVICES:
        peer_names = [name for name in DEVICES if name != target]
        peer_rows = [devices[name]["support_pool"] for name in peer_names]
        peer_summary = aggregate_equal_weight([summarise(x) for x in peer_rows])
        scales = peer_scales(peer_rows)
        scales["shared_marginals"] = peer_summary.covariance.diagonal()
        pool = devices[target]["support_pool"]
        for n in SUPPORT_SIZES:
            for rep in range(REPLICATES):
                rng = np.random.default_rng(SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode()))
                start = int(rng.integers(0, len(pool) - n + 1))
                rows = pool[start:start+n]
                benign, attack = devices[target]["test_benign"], devices[target]["test_attack"]
                if len(benign) > EVAL_ROWS:
                    benign = benign[rng.choice(len(benign), EVAL_ROWS, replace=False)]
                if len(attack) > EVAL_ROWS:
                    attack = attack[rng.choice(len(attack), EVAL_ROWS, replace=False)]
                for method, variance in scales.items():
                    center = peer_summary.mean if method == "shared_marginals" else "local"
                    model = fit_score(rows, variance, center)
                    threshold = float(np.quantile(crossfit(rows, variance, center), 1-RATE))
                    auc, pauc, fpr, tpr = evaluate(model(benign), model(attack), threshold)
                    records.append({"device": target, "support_n": n, "replicate": rep,
                                    "offset": start, "method": method, "auroc": auc,
                                    "spauc01": pauc, "fpr": fpr, "tpr": tpr})
            print(target, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT, index=False)
    per_device = frame.groupby(["support_n", "method", "device"])[
        ["auroc", "spauc01", "fpr", "tpr"]
    ].mean().reset_index()
    print(per_device.groupby(["support_n", "method"]).mean(numeric_only=True).round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
