"""Dependence-adjusted OAS shrinkage for collaborative score geometry.

Estimate effective support for the covariance second moment from HAC
autocorrelation of its matrix-valued influence, then pass that information
count to the existing OAS precision estimator. Peer predictive scales and
target-local centres are kept fixed to isolate the covariance-shrinkage step.
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
from fedorbit.detection.gaussian import GaussianScorer, scorer_from_rows, shrunk_precision
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "dependence_adjusted_oas_bartz_rep30.csv"
DEVICES = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)
SUPPORT_SIZES = (30, 100, 300, 1000)
REPLICATES = 30
SEED = 202609313
FLOOR = 1e-9
RATE = 0.01
EVAL_ROWS = 5000


def mean_ess(rows):
    centered = rows - rows.mean(axis=0, keepdims=True)
    n = len(rows)
    denominator = np.maximum(np.sum(centered**2, axis=0), FLOOR)
    lrv = np.ones(rows.shape[1])
    base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lagmax = min(n // 4, 2 * base)
    for lag in range(1, lagmax + 1):
        rho = np.sum(centered[:-lag] * centered[lag:], axis=0) / denominator
        lrv += 2.0 * (1.0 - lag / (lagmax + 1.0)) * rho
    return np.clip(n / np.maximum(lrv, 1.0), 2.0, max(n - 1.0, 2.0))


def covariance_ess(values):
    centered = values - values.mean(axis=0, keepdims=True)
    n = len(values)
    second = centered.T @ centered / n
    zsz = np.einsum("ni,ij,nj->n", centered, second, centered, optimize=True)
    norms = np.sum(centered**2, axis=1)
    trace_sq = float(np.sum(second * second))
    gamma0 = np.maximum(np.sum(norms**2 - 2.0 * zsz + trace_sq), FLOOR)
    base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lagmax = min(n // 4, 2 * base)
    lrv = 1.0
    for lag in range(1, lagmax + 1):
        dot = np.sum(centered[:-lag] * centered[lag:], axis=1)
        cross = np.sum(dot**2 - zsz[:-lag] - zsz[lag:] + trace_sq)
        gamma = cross / gamma0
        lrv += 2.0 * (1.0 - lag / (lagmax + 1.0)) * gamma
    return float(np.clip(n / max(lrv, 1.0), 2.0, max(n - 1.0, 2.0)))


def bartz_hac_intensity(z):
    """Autocorrelation-aware LW intensity from Bartz & Müller (NIPS 2014)."""
    n, p = z.shape
    second = z.T @ z / n
    trace = float(np.trace(second))
    target = np.eye(p) * (trace / p)
    denominator = max(float(np.sum((second - target) ** 2)), FLOOR)
    quadratic = np.einsum("ni,ij,nj->n", z, second, z, optimize=True)
    norms = np.sum(z**2, axis=1)
    tr_second_sq = float(np.sum(second * second))
    gamma0 = float(np.sum(norms**2 - 2.0 * quadratic + tr_second_sq) / n)
    bandwidth = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    bandwidth = min(bandwidth, n // 4)
    long_run = gamma0
    for lag in range(1, bandwidth + 1):
        products = np.sum(z[:-lag] * z[lag:], axis=1)
        gamma = float(np.sum(products**2 - quadratic[:-lag] - quadratic[lag:]
                              + tr_second_sq) / n)
        long_run += 2.0 * (1.0 - lag / (bandwidth + 1.0)) * gamma
    correction = max(n - 1.0 - 2.0 * bandwidth + bandwidth * (bandwidth + 1.0) / n, 1.0)
    return float(np.clip(long_run / correction / denominator, 0.0, 1.0))


def scorer(rows, center, scale, method):
    if method == "oas_nominal":
        return scorer_from_rows(rows, center, scale)
    z = (rows - center) / scale
    second = z.T @ z / len(rows)
    if method in ("bartz_hac", "collaborative_bartz_mean_oas"):
        intensity = bartz_hac_intensity(z)
        trace = float(np.trace(second))
        covariance = ((1.0 - intensity) * second
                      + intensity * np.eye(second.shape[0]) * trace / second.shape[0])
        return GaussianScorer(center, scale, np.linalg.inv(covariance))
    ess = covariance_ess(z)
    return GaussianScorer(center, scale, shrunk_precision(second, ess))


def fit(rows, peer, method, peer_center_var):
    if method == "shared_marginals":
        return scorer_from_rows(rows, peer.mean, peer.standard_deviation + 1e-6)
    if method in ("collaborative_ess_mean_oas", "collaborative_bartz_mean_oas"):
        local_mean = rows.mean(axis=0)
        local_var = np.maximum(rows.var(axis=0, ddof=1), FLOOR)
        ess = np.maximum(mean_ess(rows), 2.0)
        sampling_var = local_var / ess
        weight = peer_center_var / (peer_center_var + sampling_var + FLOOR)
        center = peer.mean + weight * (local_mean - peer.mean)
    else:
        center = rows.mean(axis=0)
    return scorer(rows, center, peer.standard_deviation + 1e-6, method)


def crossfit(rows, peer, method, peer_center_var):
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    return np.r_[fit(second, peer, method, peer_center_var).score(first),
                 fit(first, peer, method, peer_center_var).score(second)]


def metrics(benign, attack, threshold):
    y = np.r_[np.zeros(len(benign)), np.ones(len(attack))]
    values = np.r_[benign, attack]
    return (float(roc_auc_score(y, values)), float(roc_auc_score(y, values, max_fpr=RATE)),
            float(np.mean(benign > threshold)), float(np.mean(attack > threshold)))


def main():
    devices = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    records = []
    for target in DEVICES:
        peer_names = [k for k in DEVICES if k != target]
        peer_summaries = [summarise(devices[k]["support_pool"]) for k in peer_names]
        peer = aggregate_equal_weight(peer_summaries)
        peer_center_var = np.maximum(np.var(np.stack([s.mean for s in peer_summaries]),
                                            axis=0, ddof=1), FLOOR)
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
                for method in ("oas_nominal", "oas_covariance_ess", "bartz_hac",
                               "shared_marginals", "collaborative_ess_mean_oas",
                               "collaborative_bartz_mean_oas"):
                    model = fit(rows, peer, method, peer_center_var)
                    threshold = float(np.quantile(crossfit(rows, peer, method, peer_center_var), 1-RATE))
                    auc, pauc, fpr, tpr = metrics(model.score(benign), model.score(attack), threshold)
                    records.append({"device": target, "support_n": n, "replicate": rep,
                                    "offset": start, "method": method, "auroc": auc,
                                    "spauc01": pauc, "fpr": fpr, "tpr": tpr,
                                    "mean_ess": float(np.median(mean_ess(rows))),
                                    "covariance_ess": covariance_ess(
                                        (rows - rows.mean(axis=0)) / (peer.standard_deviation + 1e-6)
                                    )})
            print(target, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT, index=False)
    device = frame.groupby(["support_n", "method", "device"])[
        ["auroc", "spauc01", "fpr", "tpr", "covariance_ess"]
    ].mean().reset_index()
    print(device.groupby(["support_n", "method"]).mean(numeric_only=True).round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
