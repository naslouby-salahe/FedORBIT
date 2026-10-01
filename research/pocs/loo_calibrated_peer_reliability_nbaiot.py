"""Peer-only LOO calibration of a continuous peer-reliability temperature."""
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
from dependence_adjusted_oas import mean_ess
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "loo_calibrated_peer_reliability_nbaiot.csv"
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
TEMPERATURES = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def peer_loglike(rows, centers, within):
    center = rows.mean(axis=0)
    centered = rows-center
    local_var = np.maximum(np.mean(centered**2, axis=0), FLOOR)
    mean_eff = np.maximum(mean_ess(rows), 2.0)
    mean_var = np.maximum(np.var(centers, axis=0, ddof=1), FLOOR) + local_var/mean_eff
    mean_ll = -0.5*np.mean(np.log(mean_var)[None, :] +
                            (center[None, :]-centers)**2/mean_var[None, :], axis=1)
    var_eff = np.maximum(mean_ess(centered**2), 2.0)
    df = np.maximum(var_eff-1.0, 1.0)
    log_local = np.log(local_var) - (digamma(df/2)-np.log(df/2))
    scale_noise = 2.0/df
    grand = centers.mean(axis=0)
    peer_total = np.maximum(within+(centers-grand)**2, FLOOR)
    peer_log = np.log(peer_total)
    peer_var = np.maximum(np.var(peer_log, axis=0, ddof=1), 0.025**2)
    pred_var = peer_var+scale_noise
    scale_ll = -0.5*np.mean(np.log(pred_var)[None, :] +
                             (log_local[None, :]-peer_log)**2/pred_var[None, :], axis=1)
    return mean_ll+scale_ll


def predictive(peer_summaries, weights):
    centers = np.stack([s.mean for s in peer_summaries])
    within = np.stack([s.covariance.diagonal() for s in peer_summaries])
    center = weights @ centers
    var = np.sum(weights[:, None]*(within+(centers-center)**2), axis=0)
    return center, np.maximum(var, FLOOR)


def loo_temperature(summaries, support_pools, n, seed):
    """Select temperature by peer-only future-benign predictive MSE."""
    n_peers = len(summaries)
    scores = np.zeros(len(TEMPERATURES))
    counts = np.zeros(len(TEMPERATURES))
    rng = np.random.default_rng(seed)
    for held in range(n_peers):
        rows_all = support_pools[held]
        cut = len(rows_all)//2
        past, future = rows_all[:cut], rows_all[cut:]
        size = min(n, max(2, len(past)//2))
        start = int(rng.integers(0, len(past)-size+1))
        support = past[start:start+size]
        truth = summarise(future)
        donors = [summaries[k] for k in range(n_peers) if k != held]
        centers = np.stack([s.mean for s in donors])
        within = np.stack([s.covariance.diagonal() for s in donors])
        ll = peer_loglike(support, centers, within)
        peer_mean_var = np.maximum(np.var(centers, axis=0, ddof=1), FLOOR)
        donor_grand = centers.mean(axis=0)
        donor_total = np.maximum(within+(centers-donor_grand)**2, FLOOR)
        peer_log_var = np.maximum(np.var(np.log(donor_total), axis=0, ddof=1), 0.025**2)
        truth_center = truth.mean
        truth_var = np.maximum(truth.covariance.diagonal(), FLOOR)
        truth_logvar = np.log(truth_var)
        for i, temperature in enumerate(TEMPERATURES):
            lw = temperature*ll
            weights = np.exp(lw-logsumexp(lw))
            pred_center, pred_var = predictive(donors, weights)
            # Standardize the two sufficient-statistic prediction errors by
            # peer-only cross-device dispersion; no attack score is involved.
            mean_loss = np.mean((pred_center-truth_center)**2/peer_mean_var)
            scale_loss = np.mean((np.log(pred_var)-truth_logvar)**2/peer_log_var)
            scores[i] += mean_loss+scale_loss
            counts[i] += 1
    scores /= np.maximum(counts, 1)
    best = int(np.argmin(scores))
    return TEMPERATURES[best], scores


def fit(rows, peer_summaries, peer_center, peer_scale, method, temperature):
    if method == "shared_oas":
        return scorer_from_rows(rows, peer_center, peer_scale+1e-6), np.full(len(peer_summaries), 1/len(peer_summaries))
    if method == "uniform_peer":
        return scorer_from_rows(rows, rows.mean(axis=0), peer_scale+1e-6), np.full(len(peer_summaries), 1/len(peer_summaries))
    centers = np.stack([s.mean for s in peer_summaries])
    within = np.stack([s.covariance.diagonal() for s in peer_summaries])
    ll = peer_loglike(rows, centers, within)
    lw = temperature*ll
    weights = np.exp(lw-logsumexp(lw))
    scale_center, scale = predictive(peer_summaries, weights)
    return scorer_from_rows(rows, rows.mean(axis=0), np.sqrt(scale)+1e-6), weights


def crossfit(rows, summaries, center, scale, temperature):
    mid = len(rows)//2
    first, second = rows[:mid], rows[mid:]
    return np.r_[fit(second, summaries, center, scale, "soft", temperature)[0].score(first),
                 fit(first, summaries, center, scale, "soft", temperature)[0].score(second)]


def main():
    devices = {name: np.load(DATA/f"{name}.npz") for name in DEVICES}
    records = []
    for target in DEVICES:
        peer_names = [name for name in DEVICES if name != target]
        peer_summaries = [summarise(devices[name]["support_pool"]) for name in peer_names]
        peer_pools = [devices[name]["support_pool"] for name in peer_names]
        peer = aggregate_equal_weight(peer_summaries)
        pool = devices[target]["support_pool"]
        for n in SUPPORT_SIZES:
            temp, cv_loss = loo_temperature(peer_summaries, peer_pools, n,
                                            SEED ^ zlib.crc32(f"{target}:{n}:loo".encode()))
            for rep in range(REPLICATES):
                rng = np.random.default_rng(SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode()))
                start = int(rng.integers(0, len(pool)-n+1))
                rows = pool[start:start+n]
                benign, attack = devices[target]["test_benign"], devices[target]["test_attack"]
                if len(benign)>EVAL_ROWS: benign=benign[rng.choice(len(benign), EVAL_ROWS, replace=False)]
                if len(attack)>EVAL_ROWS: attack=attack[rng.choice(len(attack), EVAL_ROWS, replace=False)]
                for method in ("shared_oas", "uniform_peer", "fixed_soft_peer", "loo_soft_peer"):
                    method_temp = 1.0 if method == "fixed_soft_peer" else temp
                    model, weights = fit(rows, peer_summaries, peer.mean, peer.standard_deviation,
                                         "soft" if method in ("fixed_soft_peer", "loo_soft_peer") else method,
                                         method_temp)
                    if method in ("fixed_soft_peer", "loo_soft_peer"):
                        threshold = float(np.quantile(crossfit(rows, peer_summaries, peer.mean,
                            peer.standard_deviation, method_temp), 1-RATE))
                    else:
                        threshold = float(np.quantile(np.r_[fit(rows[:len(rows)//2], peer_summaries,
                            peer.mean, peer.standard_deviation, method, temp)[0].score(rows[len(rows)//2:]),
                            fit(rows[len(rows)//2:], peer_summaries, peer.mean,
                                peer.standard_deviation, method, temp)[0].score(rows[:len(rows)//2])], 1-RATE))
                    sb, sa = model.score(benign), model.score(attack)
                    y=np.r_[np.zeros(len(sb)),np.ones(len(sa))]; ss=np.r_[sb,sa]
                    records.append({"device":target,"support_n":n,"replicate":rep,"offset":start,
                                    "method":method,"temperature":method_temp,
                                    "cv_loss":float(cv_loss[TEMPERATURES.index(temp)]),
                                    "auroc":roc_auc_score(y,ss),
                                    "spauc01":roc_auc_score(y,ss,max_fpr=RATE),
                                    "fpr":np.mean(sb>threshold),"tpr":np.mean(sa>threshold),
                                    "max_peer_weight":weights.max(),
                                    "peer_weight_entropy":-np.sum(weights*np.log(weights+1e-30))})
            print(target,n,"temperature",temp,"done",flush=True)
    frame=pd.DataFrame(records); OUT.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(OUT,index=False)
    by=frame.groupby(["support_n","method","device"])[["auroc","spauc01","fpr","tpr","temperature","max_peer_weight"]].mean().reset_index()
    print(by.groupby(["support_n","method"]).mean(numeric_only=True).round(4).to_string())
    print("wrote",len(frame),OUT)


if __name__ == "__main__": main()
