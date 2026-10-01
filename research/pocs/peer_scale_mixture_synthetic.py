"""Synthetic screen of a multimodal peer prior for collaborative scales.

This is a targeted revision to the single-family EB scale prior: model the
target log-variance as coming from a kernel mixture centered on peer log
variances, then update that prior with the target's ESS-adjusted benign scale
estimate. It uses no attack data. The experiment compares log-scale MSE with
local, uniform-peer, and unimodal normal-normal EB estimates under compatible
bimodal peers and a novel intermediate target regime.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, polygamma, digamma

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "pocs" / "results" / "peer_scale_mixture_synthetic.csv"
SEED = 202609304
REPS = 300
PEERS = 8
FEATURES = 20
PEER_N = 2000


def log_variance_se(n_eff: float) -> tuple[float, float]:
    """Mean bias and variance of log(sample variance) under iid Gaussian data."""
    df = max(float(n_eff) - 1.0, 2.0)
    half = df / 2.0
    bias = float(digamma(half) - np.log(half))
    variance = float(polygamma(1, half))
    return bias, variance


def hac_variance_ess(x: np.ndarray, bandwidth_multiplier: float = 2.0) -> float:
    centered = x - x.mean()
    influence = centered**2
    influence -= influence.mean()
    denominator = max(float(influence @ influence), 1e-12)
    n = len(x)
    base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lagmax = min(n // 4, max(1, int(round(base * bandwidth_multiplier))))
    lrv = 1.0
    for lag in range(1, lagmax + 1):
        rho = float(influence[:-lag] @ influence[lag:]) / denominator
        lrv += 2.0 * (1.0 - lag / (lagmax + 1.0)) * rho
    return float(np.clip(n / max(lrv, 1.0), 2.0, n - 1.0))


def ar1(rng: np.random.Generator, n: int, variance: float,
        rho: float) -> np.ndarray:
    eps = rng.normal(scale=np.sqrt(variance * (1.0 - rho**2)), size=n)
    x = np.empty(n)
    x[0] = rng.normal(scale=np.sqrt(variance))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return x


def robust_sd(x: np.ndarray) -> float:
    median = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - median))
    qsd = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    return float(max(min(mad, qsd), 1e-3))


def estimates(log_local: float, se2: float, peer_logs: np.ndarray):
    # Classical unimodal normal-normal EB baseline, with peer measurement
    # noise negligible because peer support is deliberately large.
    peer_center = float(np.mean(peer_logs))
    tau2 = max(float(np.var(peer_logs, ddof=1)), 1e-8)
    local_weight = tau2 / (tau2 + se2)
    normal_eb = local_weight * log_local + (1.0 - local_weight) * peer_center

    # Nonparametric EB prior: equal-mass Gaussian kernels at peer log scales.
    # Silverman's one-dimensional bandwidth is estimated from peers alone.
    bandwidth = 0.9 * robust_sd(peer_logs) * PEERS ** (-1.0 / 5.0)
    prior_var = max(bandwidth**2, 1e-6)
    component_var = prior_var + se2
    log_weights = -0.5 * (
        np.log(2.0 * np.pi * component_var)
        + (log_local - peer_logs) ** 2 / component_var
    )
    weights = np.exp(log_weights - logsumexp(log_weights))
    component_mean = (se2 * peer_logs + prior_var * log_local) / component_var
    mixture_eb = float(weights @ component_mean)
    effective_peer_count = float(1.0 / np.sum(weights**2))
    return normal_eb, mixture_eb, local_weight, effective_peer_count


def peer_log_scales(regime: str, rng: np.random.Generator) -> np.ndarray:
    if regime == "homogeneous":
        centers = np.zeros(PEERS)
        centers += rng.normal(0.0, 0.08, size=PEERS)
    elif regime in ("bimodal_compatible", "bimodal_novel"):
        centers = np.r_[np.full(PEERS // 2, np.log(0.5)),
                        np.full(PEERS // 2, np.log(2.0))]
        centers += rng.normal(0.0, 0.04, size=PEERS)
    elif regime == "unimodal_heterogeneous":
        centers = rng.normal(0.0, 0.45, size=PEERS)
    else:
        raise ValueError(regime)
    return centers


def target_logs(regime: str, peer_logs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if regime == "homogeneous":
        return rng.normal(0.0, 0.08, size=FEATURES)
    if regime == "bimodal_compatible":
        mode = int(rng.integers(0, 2))
        return np.full(FEATURES, np.log(0.5 if mode == 0 else 2.0)) + rng.normal(0, 0.04, FEATURES)
    if regime == "bimodal_novel":
        return np.full(FEATURES, 0.0) + rng.normal(0, 0.04, FEATURES)
    if regime == "unimodal_heterogeneous":
        return rng.normal(0.0, 0.45, size=FEATURES)
    raise ValueError(regime)


def main() -> None:
    rng = np.random.default_rng(SEED)
    regimes = ("homogeneous", "unimodal_heterogeneous",
               "bimodal_compatible", "bimodal_novel")
    records = []
    for n in (30, 100, 300, 1000):
        for rho in (0.0, 0.5, 0.8):
            for regime in regimes:
                methods = {name: [] for name in
                           ("local", "uniform_peer", "normal_eb", "mixture_eb")}
                weights, peer_counts, ess_values = [], [], []
                for _ in range(REPS):
                    peers = peer_log_scales(regime, rng)
                    truths = target_logs(regime, peers, rng)
                    peer_log_by_feature = np.tile(peers[:, None], (1, FEATURES))
                    # Small per-feature perturbations make target and peer
                    # scales featurewise rather than exactly constant.
                    peer_log_by_feature += rng.normal(0, 0.035, peer_log_by_feature.shape)
                    for j, true_logvar in enumerate(truths):
                        true_var = float(np.exp(true_logvar))
                        x = ar1(rng, n, true_var, rho)
                        local_var = max(float(np.var(x, ddof=1)), 1e-12)
                        n_eff = n if rho == 0.0 else hac_variance_ess(x)
                        bias, se2 = log_variance_se(n_eff)
                        log_local = float(np.log(local_var) - bias)
                        peer_logs = peer_log_by_feature[:, j]
                        normal, mixture, local_weight, peer_count = estimates(
                            log_local, se2, peer_logs
                        )
                        estimates_by_method = {
                            "local": log_local,
                            "uniform_peer": float(peer_logs.mean()),
                            "normal_eb": normal,
                            "mixture_eb": mixture,
                        }
                        for name, estimate in estimates_by_method.items():
                            methods[name].append((estimate - true_logvar) ** 2)
                        weights.append(local_weight)
                        peer_counts.append(peer_count)
                        ess_values.append(n_eff)
                for method, errors in methods.items():
                    records.append({
                        "support_n": n, "target_rho": rho, "regime": regime,
                        "method": method, "log_variance_mse": float(np.mean(errors)),
                        "log_variance_mse_se_rep": float(np.std(errors, ddof=1) / np.sqrt(REPS * FEATURES)),
                        "mean_estimated_ess": float(np.mean(ess_values)),
                        "mean_local_weight_normal_eb": float(np.mean(weights)),
                        "mean_effective_peers_mixture": float(np.mean(peer_counts)),
                        "replications": REPS,
                    })
                print(n, rho, regime, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("wrote", len(records), OUT)


if __name__ == "__main__":
    main()
