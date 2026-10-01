"""Continuous peer reliability from benign mean/scale predictive likelihoods."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import digamma, logsumexp
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "pocs"))
sys.path.insert(0, str(ROOT / "src"))
from dependence_adjusted_oas import mean_ess, scorer
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "continuous_peer_reliability_oas_nbaiot.csv"
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
RATE = 0.01
EVAL_ROWS = 5000
FLOOR = 1e-8


def log_peer_weights(rows, centers, within, use_scale):
    local_center = rows.mean(axis=0)
    centered = rows - local_center
    local_var = np.maximum(np.mean(centered**2, axis=0), FLOOR)
    mean_eff = np.maximum(mean_ess(rows), 2.0)
    peer_center_var = np.maximum(np.var(centers, axis=0, ddof=1), FLOOR)
    mean_noise = local_var / mean_eff
    mean_var = peer_center_var + mean_noise
    mean_ll = -0.5 * np.mean(np.log(mean_var)[None, :] +
                              (local_center[None, :] - centers)**2 / mean_var[None, :], axis=1)
    if not use_scale:
        logw = mean_ll
    else:
        var_rows = centered**2
        var_eff = np.maximum(mean_ess(var_rows), 2.0)
        df = np.maximum(var_eff - 1.0, 1.0)
        log_local_var = np.log(local_var) - (digamma(df/2) - np.log(df/2))
        scale_noise = 2.0 / df
        grand = centers.mean(axis=0)
        peer_total = np.maximum(within + (centers-grand)**2, FLOOR)
        peer_log = np.log(peer_total)
        peer_log_var = np.maximum(np.var(peer_log, axis=0, ddof=1), 0.025**2)
        combined = peer_log_var + scale_noise
        scale_ll = -0.5 * np.mean(np.log(combined)[None, :] +
                                  (log_local_var[None, :] - peer_log)**2 / combined[None, :], axis=1)
        logw = mean_ll + scale_ll
    logw -= logsumexp(logw)
    return np.exp(logw)


def fit(rows, peer_summaries, peer_center, peer_scale, method):
    centers = np.stack([s.mean for s in peer_summaries])
    within = np.stack([s.covariance.diagonal() for s in peer_summaries])
    if method == "shared_oas":
        return scorer_from_rows(rows, peer_center, peer_scale + 1e-6), np.full(len(centers), 1/len(centers))
    if method == "local_center_peer_scale":
        return scorer_from_rows(rows, rows.mean(axis=0), peer_scale + 1e-6), np.full(len(centers), 1/len(centers))
    weights = log_peer_weights(rows, centers, within, method == "soft_mean_scale")
    soft_center = weights @ centers
    # Each peer contributes within variance plus its squared displacement from
    # the continuously weighted peer center: a predictive marginal scale.
    soft_var = np.sum(weights[:, None] * (within + (centers-soft_center)**2), axis=0)
    model = scorer_from_rows(rows, rows.mean(axis=0), np.sqrt(np.maximum(soft_var, FLOOR)) + 1e-6)
    return model, weights


def crossfit(rows, summaries, center, scale, method):
    mid = len(rows)//2
    first, second = rows[:mid], rows[mid:]
    return np.r_[fit(second, summaries, center, scale, method)[0].score(first),
                 fit(first, summaries, center, scale, method)[0].score(second)]


def main():
    devices = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    records = []
    for target in DEVICES:
        peer_names = [name for name in DEVICES if name != target]
        peer_summaries = [summarise(devices[name]["support_pool"]) for name in peer_names]
        peer = aggregate_equal_weight(peer_summaries)
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
                for method in ("shared_oas", "local_center_peer_scale", "soft_mean",
                               "soft_mean_scale"):
                    model, weights = fit(rows, peer_summaries, peer.mean,
                                         peer.standard_deviation, method)
                    threshold = float(np.quantile(crossfit(rows, peer_summaries, peer.mean,
                                                           peer.standard_deviation, method), 1-RATE))
                    sb, sa = model.score(benign), model.score(attack)
                    y = np.r_[np.zeros(len(sb)), np.ones(len(sa))]
                    score = np.r_[sb, sa]
                    records.append({"device": target, "support_n": n, "replicate": rep,
                                    "offset": start, "method": method,
                                    "auroc": roc_auc_score(y, score),
                                    "spauc01": roc_auc_score(y, score, max_fpr=RATE),
                                    "fpr": np.mean(sb > threshold), "tpr": np.mean(sa > threshold),
                                    "max_peer_weight": float(weights.max()),
                                    "peer_weight_entropy": float(-np.sum(weights*np.log(weights+1e-30)))})
            print(target, n, "done", flush=True)
    frame = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    by_device = frame.groupby(["support_n", "method", "device"])[
        ["auroc", "spauc01", "fpr", "tpr", "max_peer_weight", "peer_weight_entropy"]
    ].mean().reset_index()
    print(by_device.groupby(["support_n", "method"]).mean(numeric_only=True).round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
