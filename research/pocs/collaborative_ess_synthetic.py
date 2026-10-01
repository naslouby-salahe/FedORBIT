"""Synthetic POC for peer-informed autocorrelation shrinkage in scale ESS.

Peers provide feature-wise long-sequence estimates of lag-one correlation in
the variance-influence process. A robust normal-normal posterior on Fisher-z
correlation combines that peer distribution with the target's short support.
The posterior correlation is converted to the AR(1) variance-ESS. This is a
candidate component for uncertainty-weighted scale adaptation, not a detector.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "pocs" / "results" / "collaborative_ess_synthetic.csv"
SEED = 202609306
REPS = 300
PEERS = 8
FEATURES = 20
PEER_N = 2000


def ar1(rng: np.random.Generator, n: int, q: float) -> np.ndarray:
    phi = np.sqrt(np.clip(q, 0.0, 0.98))
    innovation = rng.normal(scale=np.sqrt(1.0 - phi**2), size=n)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + innovation[i]
    return x


def variance_ac1(x: np.ndarray) -> float:
    centered = x - x.mean()
    influence = centered**2
    influence -= influence.mean()
    den = max(float(influence @ influence), 1e-12)
    return float(np.clip((influence[:-1] @ influence[1:]) / den, -0.95, 0.98))


def nw_ess(x: np.ndarray) -> float:
    centered = x - x.mean()
    influence = centered**2
    influence -= influence.mean()
    n = len(x)
    den = max(float(influence @ influence), 1e-12)
    base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lagmax = min(n // 4, 2 * base)
    long_run = 1.0
    for lag in range(1, lagmax + 1):
        ac = float(influence[:-lag] @ influence[lag:]) / den
        long_run += 2.0 * (1.0 - lag / (lagmax + 1.0)) * ac
    return float(np.clip(n / max(long_run, 1.0), 1.0, n))


def collaborative_q(target_qhat: float, n: int,
                    peer_qhat: np.ndarray) -> tuple[float, float]:
    eps = 1e-4
    peer_z = np.arctanh(np.clip(peer_qhat, -1 + eps, 1 - eps))
    target_z = float(np.arctanh(np.clip(target_qhat, -1 + eps, 1 - eps)))
    peer_center = float(np.median(peer_z))
    mad = 1.4826 * np.median(np.abs(peer_z - peer_center))
    # Fisher-z uncertainty is approximately 1/sqrt(n-3). Correct peer
    # heterogeneity for finite peer estimation before the normal-normal update.
    peer_obs_var = float(np.median(1.0 / np.maximum(PEER_N - 3.0, 1.0)))
    tau2 = max(float(mad**2 - peer_obs_var), 1e-6)
    target_var = 1.0 / max(n - 3.0, 1.0)
    local_weight = tau2 / (tau2 + target_var)
    z_post = local_weight * target_z + (1.0 - local_weight) * peer_center
    return float(np.tanh(z_post)), float(local_weight)


def draw_peer_q(regime: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if regime == "homogeneous_match":
        peers = np.clip(rng.normal(0.70, 0.05, PEERS), 0.05, 0.95)
        target = np.clip(rng.normal(0.70, 0.05, FEATURES), 0.05, 0.95)
    elif regime == "heterogeneous_match":
        peers = rng.uniform(0.05, 0.9, PEERS)
        target = rng.uniform(0.05, 0.9, FEATURES)
    elif regime == "target_low_peer_high":
        peers = np.clip(rng.normal(0.70, 0.05, PEERS), 0.05, 0.95)
        target = np.clip(rng.normal(0.10, 0.03, FEATURES), 0.02, 0.2)
    else:
        raise ValueError(regime)
    return peers, target


def main() -> None:
    rng = np.random.default_rng(SEED)
    records = []
    for n in (30, 100, 300, 1000):
        for regime in ("homogeneous_match", "heterogeneous_match",
                       "target_low_peer_high"):
            values = {name: [] for name in
                      ("local_lag1", "collaborative", "NW_2x", "nominal")}
            weights, true_eff = [], []
            for _ in range(REPS):
                q_peers, q_targets = draw_peer_q(regime, rng)
                peer_est = np.asarray([
                    variance_ac1(ar1(rng, PEER_N, q)) for q in q_peers
                ])
                for q in q_targets:
                    x = ar1(rng, n, q)
                    local_q = variance_ac1(x)
                    post_q, local_weight = collaborative_q(local_q, n, peer_est)
                    nw = nw_ess(x)
                    estimates = {
                        "local_lag1": n * (1.0 - local_q) / (1.0 + local_q),
                        "collaborative": n * (1.0 - post_q) / (1.0 + post_q),
                        "NW_2x": nw,
                        "nominal": float(n),
                    }
                    target_effective = n * (1.0 - q) / (1.0 + q)
                    for name, value in estimates.items():
                        values[name].append(value / target_effective)
                    weights.append(local_weight)
                    true_eff.append(target_effective)
            for name, ratios in values.items():
                records.append({
                    "support_n": n, "regime": regime, "method": name,
                    "median_ess_ratio": float(np.median(ratios)),
                    "mean_ess_ratio": float(np.mean(ratios)),
                    "ratio_q10": float(np.quantile(ratios, 0.1)),
                    "ratio_q90": float(np.quantile(ratios, 0.9)),
                    "mean_local_weight": float(np.mean(weights)),
                    "mean_true_ess": float(np.mean(true_eff)),
                    "replications": REPS * FEATURES,
                })
            print(n, regime, "local posterior weight", round(float(np.mean(weights)), 3), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("wrote", len(records), OUT)


if __name__ == "__main__":
    main()
