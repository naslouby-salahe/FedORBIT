"""Continuous peer/local scale adaptation under support autocorrelation.

Exploratory synthetic challenge for a hierarchical log-scale posterior paired
with covariance-influence OAS.  The prior weight is set from peer log-scale
dispersion and support variance ESS; there is no compatibility gate.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "pocs"))
from dependence_adjusted_oas import covariance_ess, mean_ess, scorer
from fedorbit.detection.moments import aggregate_equal_weight, summarise

OUT = ROOT / "research" / "pocs" / "results" / "mismatch_safe_scale_oas.csv"
SUPPORTS = (30, 100, 300, 1000)
REGIMES = ("iid_match", "ar05_match", "ar08_match", "ar08_shifted_scale",
           "ar08_target_scale_narrow")
REPLICATES = 40
FEATURES = 20
PEERS = 8
SEED = 202609315
FLOOR = 1e-8


def covariance(scale):
    corr = 0.45 ** np.abs(np.subtract.outer(np.arange(FEATURES), np.arange(FEATURES)))
    return corr * np.outer(scale, scale)


def ar_rows(rng, count, mean, cov, rho):
    chol = np.linalg.cholesky(cov + np.eye(len(mean)) * 1e-7)
    eps = rng.normal(size=(count, len(mean))) @ chol.T
    if rho == 0:
        return mean + eps
    out = np.empty_like(eps)
    out[0] = mean + eps[0]
    root = np.sqrt(1.0 - rho**2)
    for t in range(1, count):
        out[t] = mean + rho * (out[t-1] - mean) + root * eps[t]
    return out


def regime_params(regime):
    if regime == "iid_match": return 0.0, np.ones(FEATURES)
    if regime == "ar05_match": return 0.5, np.ones(FEATURES)
    if regime == "ar08_match": return 0.8, np.ones(FEATURES)
    if regime == "ar08_shifted_scale":
        return 0.8, np.tile(np.array([0.5, 2.0]), FEATURES // 2)
    return 0.8, np.tile(np.array([0.5, 1.5]), FEATURES // 2)


def peer_prior(peer_summaries):
    centers = np.stack([s.mean for s in peer_summaries])
    within = np.stack([s.covariance.diagonal() for s in peer_summaries])
    center = centers.mean(axis=0)
    total = within + (centers - center) ** 2
    prior_var = total.mean(axis=0)
    # Between-peer spread in log marginal variance is the EB prior variance.
    log_peer = np.log(np.maximum(total, FLOOR))
    tau2 = np.var(log_peer, axis=0, ddof=1)
    center_var = np.maximum(np.var(centers, axis=0, ddof=1), FLOOR)
    return center, np.sqrt(prior_var), np.maximum(tau2, 0.025**2), center_var


def posterior_scale(rows, prior_scale, tau2, trust=1.0):
    centered = rows - rows.mean(axis=0, keepdims=True)
    local_var = np.maximum(np.mean(centered**2, axis=0), FLOOR)
    # The HAC ESS of squared residuals estimates information for marginal scale.
    ess = np.maximum(mean_ess(centered**2), 2.0)
    df = np.maximum(ess - 1.0, 1.0)
    log_local_unbiased = np.log(local_var) - (digamma(df / 2.0) - np.log(df / 2.0))
    obs_var = 2.0 / df
    weight = trust * tau2 / (tau2 + obs_var)
    post_logvar = np.log(np.maximum(prior_scale**2, FLOOR)) + weight * (
        log_local_unbiased - np.log(np.maximum(prior_scale**2, FLOOR)))
    return np.exp(0.5 * post_logvar), ess, weight


def make_model(rows, center, scale, mode):
    if mode in ("shared_oas", "peer_nominal", "peer_shrunk_nominal",
                "adaptive_joint_nominal"):
        return scorer(rows, center, scale, "oas_nominal")
    if mode == "peer_cov_ess":
        return scorer(rows, rows.mean(axis=0), scale, "oas_covariance_ess")
    if mode == "adaptive_cov_ess":
        return scorer(rows, rows.mean(axis=0), scale, "oas_covariance_ess")
    if mode == "adaptive_nominal":
        return scorer(rows, rows.mean(axis=0), scale, "oas_nominal")
    raise ValueError(mode)


def main():
    rng = np.random.default_rng(SEED)
    records = []
    for regime in REGIMES:
        rho, multiplier = regime_params(regime)
        for n in SUPPORTS:
            for rep in range(REPLICATES):
                peer_means = rng.normal(0, 0.3, size=(PEERS, FEATURES))
                peer_within = np.exp(rng.normal(0, 0.18, size=(PEERS, FEATURES)))
                peer_rows = [ar_rows(rng, 1200, peer_means[k],
                                     covariance(np.sqrt(peer_within[k])), 0.0)
                             for k in range(PEERS)]
                summaries = [summarise(x) for x in peer_rows]
                peer = aggregate_equal_weight(summaries)
                center, peer_scale, tau2, center_var = peer_prior(summaries)
                target_mean = rng.normal(0, 0.3, FEATURES)
                scale = peer_scale * multiplier
                support = ar_rows(rng, n, target_mean,
                                  covariance(scale), rho)
                benign = ar_rows(rng, 500, target_mean, covariance(scale), rho)
                shift = np.zeros(FEATURES)
                active = rng.choice(FEATURES, size=4, replace=False)
                shift[active] = rng.choice((-1.0, 1.0), size=len(active)) * 1.5 * peer_scale[active]
                attack = ar_rows(rng, 500, target_mean + shift, covariance(scale), rho)
                adaptive_scale, scale_ess, peer_weight = posterior_scale(support, peer_scale, tau2)
                local_var = np.maximum(support.var(axis=0, ddof=1), FLOOR)
                mean_eff = np.maximum(mean_ess(support), 2.0)
                mean_weight = center_var / (center_var + local_var / mean_eff + FLOOR)
                adaptive_center = center + mean_weight * (support.mean(axis=0) - center)
                methods = {
                    "shared_oas": (peer_scale, "shared_oas"),
                    "peer_nominal": (peer_scale, "peer_nominal"),
                    "peer_shrunk_nominal": (peer_scale, "peer_shrunk_nominal"),
                    "peer_cov_ess": (peer_scale, "peer_cov_ess"),
                    "adaptive_cov_ess": (adaptive_scale, "adaptive_cov_ess"),
                    "adaptive_nominal": (adaptive_scale, "adaptive_nominal"),
                    "adaptive_joint_nominal": (adaptive_scale, "adaptive_joint_nominal"),
                }
                for method, (used_scale, mode) in methods.items():
                    if mode == "shared_oas":
                        used_center = center
                    elif mode in ("peer_shrunk_nominal", "adaptive_joint_nominal"):
                        used_center = adaptive_center
                    else:
                        used_center = support.mean(axis=0)
                    model = make_model(support, used_center, used_scale, mode)
                    sb, sa = model.score(benign), model.score(attack)
                    y = np.r_[np.zeros(len(sb)), np.ones(len(sa))]
                    values = np.r_[sb, sa]
                    records.append({"regime": regime, "rho": rho, "support_n": n,
                                    "replicate": rep, "method": method,
                                    "auroc": roc_auc_score(y, values),
                                    "spauc01": roc_auc_score(y, values, max_fpr=0.01),
                                    "covariance_ess": covariance_ess(
                                        (support-support.mean(axis=0))/peer_scale),
                                    "median_scale_ess": float(np.median(scale_ess)),
                                    "median_peer_weight": float(np.median(peer_weight)),
                                    "median_mean_peer_weight": float(np.median(mean_weight)),
                                    "median_scale_ratio": float(np.median(adaptive_scale/peer_scale))})
            print(regime, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT, index=False)
    print(frame.groupby(["regime", "support_n", "method"])[
        ["auroc", "spauc01", "median_scale_ess", "median_peer_weight",
         "median_scale_ratio"]].mean().round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
