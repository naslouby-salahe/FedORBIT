"""Exploratory benign-only peer-ESS scale shrinkage on N-BaIoT.

Candidate component: estimate the target variance-influence autocorrelation by
partially pooling its short-support lag-one statistic toward leave-one-device-
out peer autocorrelation summaries. Convert that estimate to effective n,
then combine the target log variance with a robust peer log-variance prior by
normal-normal posterior weighting. Attack rows are opened only after fitting,
for exploratory score evaluation.
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
from scipy.special import digamma, polygamma
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "collaborative_ess_scale_poc.csv"
DEVICES = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)
SUPPORT_SIZES = (30, 100, 300, 1000)
REPLICATES = 15
SEED = 202609307
FLOOR = 1e-10
RATE = 0.01
EVAL_ROWS = 10000


def variance_ac1(rows: np.ndarray) -> np.ndarray:
    centered = rows - rows.mean(axis=0, keepdims=True)
    influence = centered**2
    influence -= influence.mean(axis=0, keepdims=True)
    denominator = np.maximum(np.sum(influence**2, axis=0), FLOOR)
    rho = np.sum(influence[:-1] * influence[1:], axis=0) / denominator
    return np.clip(rho, -0.95, 0.98)


def local_hac_ess(rows: np.ndarray) -> np.ndarray:
    centered = rows - rows.mean(axis=0, keepdims=True)
    influence = centered**2
    influence -= influence.mean(axis=0, keepdims=True)
    denominator = np.maximum(np.sum(influence**2, axis=0), FLOOR)
    n = len(rows)
    base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lagmax = min(n // 4, 2 * base)
    lrv = np.ones(rows.shape[1])
    for lag in range(1, lagmax + 1):
        ac = np.sum(influence[:-lag] * influence[lag:], axis=0) / denominator
        lrv += 2.0 * (1.0 - lag / (lagmax + 1.0)) * ac
    return np.clip(n / np.maximum(lrv, 1.0), 2.0, max(n - 1.0, 2.0))


def collaborative_ess(rows: np.ndarray, peer_rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(rows)
    target_rho = variance_ac1(rows)
    eps = 1e-4
    peer_z = np.arctanh(np.clip(peer_rho, -1.0 + eps, 1.0 - eps))
    target_z = np.arctanh(np.clip(target_rho, -1.0 + eps, 1.0 - eps))
    peer_center = np.median(peer_z, axis=0)
    peer_mad = 1.4826 * np.median(np.abs(peer_z - peer_center), axis=0)
    # Long peer pools make their lag-one sampling variance small relative to
    # between-device spread in this POC; target uncertainty uses Fisher-z.
    tau2 = np.maximum(peer_mad**2 - 1.0 / (200000.0 - 3.0), 1e-6)
    target_var = 1.0 / max(n - 3.0, 1.0)
    local_weight = tau2 / (tau2 + target_var)
    z_post = local_weight * target_z + (1.0 - local_weight) * peer_center
    rho_post = np.tanh(z_post)
    ess = np.clip(n * (1.0 - rho_post) / (1.0 + rho_post), 2.0, float(n))
    return ess, local_weight


def peer_scale_prior(peer_rows: list[np.ndarray]):
    peer_var = np.stack([x.var(axis=0, ddof=1) for x in peer_rows])
    peer_logvar = np.log(np.maximum(peer_var, FLOOR))
    center = np.median(peer_logvar, axis=0)
    mad = 1.4826 * np.median(np.abs(peer_logvar - center), axis=0)
    # For a log sample variance with df nu, asymptotic sampling variance is
    # trigamma(nu/2). Peer pools are large; estimate and remove this noise.
    peer_ess = np.asarray([len(x) for x in peer_rows], dtype=float)[:, None]
    noise = np.median(polygamma(1, np.maximum((peer_ess - 1.0) / 2.0, 1.0)), axis=0)
    tau2 = np.maximum(mad**2 - noise, 1e-8)
    return center, tau2, np.sqrt(np.median(peer_var, axis=0))


def shrink_log_variance(rows: np.ndarray, peer_center: np.ndarray,
                        peer_tau2: np.ndarray, ess: np.ndarray):
    n_eff = np.maximum(ess, 3.0)
    df = np.maximum(n_eff - 1.0, 2.0)
    bias = digamma(df / 2.0) - np.log(df / 2.0)
    observation = np.log(np.maximum(rows.var(axis=0, ddof=1), FLOOR)) - bias
    observation_var = polygamma(1, df / 2.0)
    local_weight = peer_tau2 / (peer_tau2 + observation_var)
    posterior = local_weight * observation + (1.0 - local_weight) * peer_center
    return np.exp(0.5 * posterior), local_weight


def fit(rows: np.ndarray, peer, peer_center, peer_tau2, peer_rho,
        method: str):
    local_mean = rows.mean(axis=0)
    local_scale = np.sqrt(np.maximum(rows.var(axis=0, ddof=1), FLOOR))
    if method == "local":
        return scorer_from_rows(rows, local_mean, local_scale + 1e-6).score
    if method == "peer_scale_local_center":
        return scorer_from_rows(rows, local_mean, peer.standard_deviation + 1e-6).score
    if method == "shared_marginals":
        return scorer_from_rows(rows, peer.mean, peer.standard_deviation + 1e-6).score
    if method == "eb_nominal":
        scale, _ = shrink_log_variance(rows, peer_center, peer_tau2,
                                       np.full(rows.shape[1], len(rows)))
        return scorer_from_rows(rows, local_mean, scale + 1e-6).score
    if method == "eb_local_hac":
        scale, _ = shrink_log_variance(rows, peer_center, peer_tau2,
                                       local_hac_ess(rows))
        return scorer_from_rows(rows, local_mean, scale + 1e-6).score
    if method == "eb_collaborative_ess":
        ess, _ = collaborative_ess(rows, peer_rho)
        scale, _ = shrink_log_variance(rows, peer_center, peer_tau2, ess)
        return scorer_from_rows(rows, local_mean, scale + 1e-6).score
    raise ValueError(method)


def crossfit_scores(rows, peer, peer_center, peer_tau2, peer_rho, method):
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    return np.concatenate((
        fit(second, peer, peer_center, peer_tau2, peer_rho, method)(first),
        fit(first, peer, peer_center, peer_tau2, peer_rho, method)(second),
    ))


def evaluate(benign, attack, threshold):
    labels = np.r_[np.zeros(len(benign)), np.ones(len(attack))]
    scores = np.r_[benign, attack]
    return (float(roc_auc_score(labels, scores)),
            float(roc_auc_score(labels, scores, max_fpr=0.01)),
            float(np.mean(benign > threshold)), float(np.mean(attack > threshold)))


def main():
    arrays = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    summaries = {name: summarise(arrays[name]["support_pool"]) for name in DEVICES}
    records = []
    methods = ("local", "peer_scale_local_center", "shared_marginals",
               "eb_nominal", "eb_local_hac", "eb_collaborative_ess")
    for target in DEVICES:
        peer_names = [k for k in DEVICES if k != target]
        peer_rows = [arrays[k]["support_pool"] for k in peer_names]
        peer_summaries = [summaries[k] for k in peer_names]
        peer = aggregate_equal_weight(peer_summaries)
        peer_center, peer_tau2, _ = peer_scale_prior(peer_rows)
        peer_rho = np.stack([variance_ac1(arrays[k]["support_pool"][:200000])
                             for k in peer_names])
        pool = arrays[target]["support_pool"]
        for n in SUPPORT_SIZES:
            for rep in range(REPLICATES):
                rng = np.random.default_rng(
                    SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode())
                )
                start = int(rng.integers(0, len(pool) - n + 1))
                rows = pool[start:start+n]
                benign = arrays[target]["test_benign"]
                attack = arrays[target]["test_attack"]
                if len(benign) > EVAL_ROWS:
                    benign = benign[rng.choice(len(benign), EVAL_ROWS, replace=False)]
                if len(attack) > EVAL_ROWS:
                    attack = attack[rng.choice(len(attack), EVAL_ROWS, replace=False)]
                for method in methods:
                    model = fit(rows, peer, peer_center, peer_tau2, peer_rho, method)
                    calibration = crossfit_scores(
                        rows, peer, peer_center, peer_tau2, peer_rho, method
                    )
                    threshold = float(np.quantile(calibration, 1.0 - RATE))
                    auc, pauc, fpr, tpr = evaluate(model(benign), model(attack), threshold)
                    record = {"device": target, "support_n": n, "replicate": rep,
                              "offset": start, "method": method,
                              "auroc": auc, "spauc01": pauc, "fpr": fpr, "tpr": tpr}
                    if method == "eb_collaborative_ess":
                        eff, w = collaborative_ess(rows, peer_rho)
                        record["median_ess"] = float(np.median(eff))
                        record["median_local_scale_weight"] = float(
                            np.median(peer_tau2 / (peer_tau2 + polygamma(
                                1, np.maximum((eff - 1.0) / 2.0, 1.0))))
                        )
                        record["median_ess_local_weight"] = float(np.median(w))
                    records.append(record)
            print(target, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
            key for row in records for key in row
        )))
        writer.writeheader()
        writer.writerows(records)
    import pandas as pd
    frame = pd.DataFrame(records)
    device_means = frame.groupby(["support_n", "method", "device"])[
        ["auroc", "spauc01", "fpr", "tpr"]
    ].mean().reset_index()
    print(device_means.groupby(["support_n", "method"]).agg(
        auroc=("auroc", "mean"), spauc01=("spauc01", "mean"),
        fpr=("fpr", "mean"), tpr=("tpr", "mean"),
    ).round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
