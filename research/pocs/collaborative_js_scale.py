"""Exploratory collaborative James-Stein shrinkage for log-scale geometry.

The peer anchor is the predictive marginal variance (within-device variance
plus between-device centre dispersion). A global positive-part James-Stein
factor shrinks the target's debiased log-variance profile toward that anchor,
using target HAC uncertainty, peer dispersion, and an effective dimension
estimated from variance-influence dependence. All adaptation is benign-only.
"""
from __future__ import annotations

import csv
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "collaborative_js_scale.csv"
DEVICES = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)
SUPPORT_SIZES = (30, 100, 300, 1000)
REPLICATES = 8
SEED = 202609309
FLOOR = 1e-9
RATE = 0.01
EVAL_ROWS = 5000


def peer_prior(peer_rows: list[np.ndarray]):
    means = np.stack([x.mean(axis=0) for x in peer_rows])
    within = np.stack([x.var(axis=0, ddof=1) for x in peer_rows])
    centre = means.mean(axis=0)
    predictive = within + (means - centre[None, :]) ** 2
    peer_var = np.maximum(predictive.mean(axis=0), FLOOR)
    log_peer = np.log(peer_var)
    log_clients = np.log(np.maximum(predictive, FLOOR))
    median = np.median(log_clients, axis=0)
    tau2 = np.maximum((1.4826 * np.median(np.abs(log_clients - median), axis=0)) ** 2,
                      1e-4)
    return peer_var, log_peer, tau2


def local_logscale(rows: np.ndarray):
    n, p = rows.shape
    centered = rows - rows.mean(axis=0, keepdims=True)
    variance = np.maximum(np.mean(centered**2, axis=0), FLOOR)
    influence = centered**2 / variance[None, :] - 1.0
    influence -= influence.mean(axis=0, keepdims=True)
    den = np.maximum(np.sum(influence**2, axis=0), FLOOR)
    base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lagmax = min(n // 4, 2 * base)
    lrv_diag = np.ones(p)
    for lag in range(1, lagmax + 1):
        ac = np.sum(influence[:-lag] * influence[lag:], axis=0) / den
        lrv_diag += 2.0 * (1.0 - lag / (lagmax + 1.0)) * ac
    ess = np.clip(n / np.maximum(lrv_diag, 1.0), 2.0, max(n - 1.0, 2.0))
    df = np.maximum(ess - 1.0, 2.0)
    bias = digamma(df / 2.0) - np.log(df / 2.0)
    y = np.log(variance) - bias
    noise = np.maximum(polygamma(1, df / 2.0), 1e-5)

    # Effective feature dimension for the variance-noise field. Shrinking the
    # empirical correlation halfway toward independence limits small-n rank
    # artifacts while retaining cross-feature dependence signal.
    corr = np.corrcoef(influence, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    corr = 0.5 * corr + 0.5 * np.eye(p)
    effective_dimension = float(np.clip(p * p / np.sum(corr**2), 3.0, p))
    return y, noise, ess, effective_dimension


def adapted_scale(rows: np.ndarray, peer_var: np.ndarray, log_peer: np.ndarray,
                  peer_tau2: np.ndarray):
    y, noise, ess, deff = local_logscale(rows)
    residual = y - log_peer
    total = np.maximum(noise + peer_tau2, 1e-5)
    q = float(np.sum(residual**2 / total) * deff / len(residual))
    local_weight = float(np.clip(1.0 - max(deff - 2.0, 0.0) / max(q, 1e-8), 0.0, 1.0))
    theta = log_peer + local_weight * residual
    return np.exp(0.5 * theta), local_weight, float(np.median(ess)), deff, q


def score(rows, peer_summary, prior, method):
    local_mean = rows.mean(axis=0)
    local_scale = np.sqrt(np.maximum(rows.var(axis=0, ddof=1), FLOOR))
    peer_var, log_peer, tau2 = prior
    if method == "local":
        centre, scale = local_mean, local_scale
    elif method == "peer_scale_local_center":
        centre, scale = local_mean, peer_summary.standard_deviation
    elif method == "shared_marginals":
        centre, scale = peer_summary.mean, peer_summary.standard_deviation
    elif method == "collaborative_js":
        scale, _, _, _, _ = adapted_scale(rows, *prior)
        centre = local_mean
    else:
        raise ValueError(method)
    return scorer_from_rows(rows, centre, scale + 1e-6).score


def crossfit(rows, peer_summary, prior, method):
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    return np.concatenate((score(second, peer_summary, prior, method)(first),
                           score(first, peer_summary, prior, method)(second)))


def metrics(benign, attack, threshold):
    labels = np.r_[np.zeros(len(benign)), np.ones(len(attack))]
    scores = np.r_[benign, attack]
    return (float(roc_auc_score(labels, scores)),
            float(roc_auc_score(labels, scores, max_fpr=RATE)),
            float(np.mean(benign > threshold)), float(np.mean(attack > threshold)))


def main():
    devices = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    records = []
    methods = ("local", "peer_scale_local_center", "shared_marginals", "collaborative_js")
    for target in DEVICES:
        peer_names = [name for name in DEVICES if name != target]
        peer_rows = [devices[name]["support_pool"] for name in peer_names]
        peer_summary = aggregate_equal_weight([summarise(x) for x in peer_rows])
        prior = peer_prior(peer_rows)
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
                scale, alpha, ess, deff, q = adapted_scale(rows, *prior)
                for method in methods:
                    model = score(rows, peer_summary, prior, method)
                    calibration = crossfit(rows, peer_summary, prior, method)
                    threshold = float(np.quantile(calibration, 1.0 - RATE))
                    auc, pauc, fpr, tpr = metrics(model(benign), model(attack), threshold)
                    records.append({"device": target, "support_n": n, "replicate": rep,
                                    "offset": start, "method": method, "auroc": auc,
                                    "spauc01": pauc, "fpr": fpr, "tpr": tpr,
                                    "js_local_weight": alpha, "median_ess": ess,
                                    "effective_dimension": deff, "js_q": q})
            print(target, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUT, index=False)
    frame = pd.DataFrame(records)
    per_device = frame.groupby(["support_n", "method", "device"])[
        ["auroc", "spauc01", "fpr", "tpr", "js_local_weight", "effective_dimension"]
    ].mean().reset_index()
    print(per_device.groupby(["support_n", "method"]).mean(numeric_only=True).round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
