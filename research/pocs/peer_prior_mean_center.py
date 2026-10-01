"""Empirical-Bayes peer prior for target centres with shared scale geometry.

This isolates mean adaptation from the empirically useful peer-scale channel.
The target mean is shrunk featurewise toward the peer mean with weight derived
from peer between-device mean dispersion and target mean-estimation variance
using featurewise HAC effective support. The scale remains the pooled peer
predictive marginal; all selection uses benign support only.
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
OUT = ROOT / "research" / "pocs" / "results" / "peer_prior_mean_center.csv"
DEVICES = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)
SUPPORT_SIZES = (30, 100, 300, 1000)
REPLICATES = 8
SEED = 202609312
FLOOR = 1e-9
RATE = 0.01
EVAL_ROWS = 5000


def mean_ess(rows):
    centered = rows - rows.mean(axis=0, keepdims=True)
    denominator = np.maximum(np.sum(centered**2, axis=0), FLOOR)
    lrv = np.ones(rows.shape[1])
    n = len(rows)
    base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lagmax = min(n // 4, 2 * base)
    for lag in range(1, lagmax + 1):
        rho = np.sum(centered[:-lag] * centered[lag:], axis=0) / denominator
        lrv += 2.0 * (1.0 - lag / (lagmax + 1.0)) * rho
    return np.clip(n / np.maximum(lrv, 1.0), 2.0, max(n-1.0, 2.0))


def shrink_center(rows, peer_mean, peer_center_var):
    local_mean = rows.mean(axis=0)
    local_var = np.maximum(rows.var(axis=0, ddof=1), FLOOR)
    ess = mean_ess(rows)
    sampling_var = local_var / ess
    local_weight = peer_center_var / (peer_center_var + sampling_var + FLOOR)
    center = peer_mean + local_weight * (local_mean - peer_mean)
    return center, local_weight


def score(rows, peer, peer_center_var, method):
    if method == "local_center_peer_scale":
        center = rows.mean(axis=0)
    elif method == "shared_marginals":
        center = peer.mean
    elif method == "eb_center_peer_scale":
        center, _ = shrink_center(rows, peer.mean, peer_center_var)
    else:
        raise ValueError(method)
    return scorer_from_rows(rows, center, peer.standard_deviation + 1e-6).score


def crossfit(rows, peer, peer_center_var, method):
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    return np.r_[score(second, peer, peer_center_var, method)(first),
                 score(first, peer, peer_center_var, method)(second)]


def evaluate(benign, attack, threshold):
    y = np.r_[np.zeros(len(benign)), np.ones(len(attack))]
    values = np.r_[benign, attack]
    return (float(roc_auc_score(y, values)), float(roc_auc_score(y, values, max_fpr=RATE)),
            float(np.mean(benign > threshold)), float(np.mean(attack > threshold)))


def main():
    devices = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    records = []
    methods = ("local_center_peer_scale", "shared_marginals", "eb_center_peer_scale")
    for target in DEVICES:
        peer_rows = [devices[k]["support_pool"] for k in DEVICES if k != target]
        summaries = [summarise(x) for x in peer_rows]
        peer = aggregate_equal_weight(summaries)
        means = np.stack([x.mean for x in summaries])
        peer_center_var = np.maximum(np.var(means, axis=0, ddof=1), FLOOR)
        pool = devices[target]["support_pool"]
        for n in SUPPORT_SIZES:
            for rep in range(REPLICATES):
                rng = np.random.default_rng(SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode()))
                start = int(rng.integers(0, len(pool)-n+1))
                rows = pool[start:start+n]
                benign, attack = devices[target]["test_benign"], devices[target]["test_attack"]
                if len(benign) > EVAL_ROWS:
                    benign = benign[rng.choice(len(benign), EVAL_ROWS, replace=False)]
                if len(attack) > EVAL_ROWS:
                    attack = attack[rng.choice(len(attack), EVAL_ROWS, replace=False)]
                _, weights = shrink_center(rows, peer.mean, peer_center_var)
                for method in methods:
                    model = score(rows, peer, peer_center_var, method)
                    threshold = float(np.quantile(crossfit(rows, peer, peer_center_var, method), 1-RATE))
                    auc, pauc, fpr, tpr = evaluate(model(benign), model(attack), threshold)
                    records.append({"device": target, "support_n": n, "replicate": rep,
                                    "offset": start, "method": method, "auroc": auc,
                                    "spauc01": pauc, "fpr": fpr, "tpr": tpr,
                                    "mean_local_center_weight": float(np.mean(weights))})
            print(target, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT, index=False)
    device_means = frame.groupby(["support_n", "method", "device"])[
        ["auroc", "spauc01", "fpr", "tpr", "mean_local_center_weight"]
    ].mean().reset_index()
    print(device_means.groupby(["support_n", "method"]).mean(numeric_only=True).round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
