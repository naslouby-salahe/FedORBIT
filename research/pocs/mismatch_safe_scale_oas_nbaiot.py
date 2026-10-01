"""Paired N-BaIoT challenge for joint peer/local mean and scale posteriors."""
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "pocs"))
sys.path.insert(0, str(ROOT / "src"))
from mismatch_safe_scale_oas import posterior_scale, peer_prior
from dependence_adjusted_oas import covariance_ess, mean_ess, scorer
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "mismatch_safe_scale_oas_nbaiot.csv"
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


def fit(rows, peer_center, peer_scale, peer_center_var, log_scale_tau2, method):
    if method == "shared_oas":
        return scorer_from_rows(rows, peer_center, peer_scale + 1e-6)
    local_mean = rows.mean(axis=0)
    trust = 1.0
    if method.startswith("scale_power_"):
        trust = float(method.removeprefix("scale_power_"))
    scale, _, _ = posterior_scale(rows, peer_scale, log_scale_tau2, trust)
    local_var = np.maximum(rows.var(axis=0, ddof=1), 1e-9)
    mean_eff = np.maximum(mean_ess(rows), 2.0)
    mean_weight = peer_center_var / (peer_center_var + local_var / mean_eff + 1e-9)
    shrunk_center = peer_center + mean_weight * (local_mean - peer_center)
    if method in ("peer_scale_local_mean", "scale_posterior_local_mean") or method.startswith("scale_power_"):
        center = local_mean
        used_scale = peer_scale if method == "peer_scale_local_mean" else scale
    elif method in ("joint_posterior_nominal", "joint_posterior_cov_ess",
                    "mean_posterior_peer_scale"):
        center = shrunk_center
        used_scale = peer_scale if method == "mean_posterior_peer_scale" else scale
    else:
        raise ValueError(method)
    score_method = "oas_covariance_ess" if method.endswith("cov_ess") else "oas_nominal"
    return scorer(rows, center, used_scale + 1e-6, score_method)


def crossfit(rows, peer_center, peer_scale, center_var, tau2, method):
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    return np.r_[fit(second, peer_center, peer_scale, center_var, tau2, method).score(first),
                 fit(first, peer_center, peer_scale, center_var, tau2, method).score(second)]


def main():
    devices = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    records = []
    for target in DEVICES:
        peer_names = [name for name in DEVICES if name != target]
        peer_summaries = [summarise(devices[name]["support_pool"]) for name in peer_names]
        peer = aggregate_equal_weight(peer_summaries)
        peer_center, prior_scale, tau2, center_var = peer_prior(peer_summaries)
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
                methods = ["shared_oas", "peer_scale_local_mean",
                               "mean_posterior_peer_scale", "scale_posterior_local_mean",
                               "joint_posterior_nominal", "joint_posterior_cov_ess"]
                methods.extend(f"scale_power_{factor}" for factor in (0.1, 0.25, 0.5))
                for method in methods:
                    model = fit(rows, peer_center, prior_scale, center_var, tau2, method)
                    threshold = float(np.quantile(crossfit(rows, peer_center, prior_scale,
                                                           center_var, tau2, method), 1-RATE))
                    benign_score, attack_score = model.score(benign), model.score(attack)
                    y = np.r_[np.zeros(len(benign_score)), np.ones(len(attack_score))]
                    score = np.r_[benign_score, attack_score]
                    records.append({"device": target, "support_n": n, "replicate": rep,
                                    "offset": start, "method": method,
                                    "auroc": roc_auc_score(y, score),
                                    "spauc01": roc_auc_score(y, score, max_fpr=RATE),
                                    "fpr": np.mean(benign_score > threshold),
                                    "tpr": np.mean(attack_score > threshold),
                                    "median_mean_ess": float(np.median(mean_ess(rows))),
                                    "covariance_ess": covariance_ess(
                                        (rows-rows.mean(axis=0))/(prior_scale+1e-6))})
            print(target, n, "done", flush=True)
    frame = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    device = frame.groupby(["support_n", "method", "device"])[
        ["auroc", "spauc01", "fpr", "tpr"]].mean().reset_index()
    print(device.groupby(["support_n", "method"]).mean(numeric_only=True).round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
