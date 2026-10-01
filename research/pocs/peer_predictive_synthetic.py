"""Reproduce synthetic hierarchy and AR(1) ESS checks for the peer-scale POC."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "pocs"))
from peer_predictive_student_score import effective_df, peer_prior, predictive_parameters

OUT = ROOT / "research" / "pocs" / "results" / "peer_predictive_synthetic.csv"
SEED = 47013
REPLICATES = 500


def ar1(rng: np.random.Generator, n: int, p: int, rho: float) -> np.ndarray:
    innovations = rng.normal(size=(n, p))
    rows = np.empty_like(innovations)
    rows[0] = innovations[0]
    for t in range(1, n):
        rows[t] = rho * rows[t - 1] + np.sqrt(1.0 - rho * rho) * innovations[t]
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    records: list[dict[str, object]] = []
    for rho in (0.0, 0.5, 0.8):
        rows = ar1(rng, 1000, 64, rho)
        ess = effective_df(rows)
        target = 1000.0 * (1.0 - rho * rho) / (1.0 + rho * rho)
        records.append({
            "check": "ar1_ess", "heterogeneity": "", "rho": rho,
            "method": "hac_ess", "log_mse": "",
            "ess_median": float(np.median(ess)), "ess_target": target,
            "mean_local_weight": "",
        })
    for heterogeneity in (0.1, 0.7):
        losses = {name: [] for name in ("local", "peer", "posterior_mean", "predictive_variance")}
        weights = []
        for _ in range(REPLICATES):
            peer_variances = np.exp(rng.normal(0.0, heterogeneity, size=8))
            peer_rows = [rng.normal(0.0, np.sqrt(v), size=(5000, 1)) for v in peer_variances]
            a0, b0 = peer_prior(peer_rows)
            target_variance = float(np.exp(rng.normal(0.0, heterogeneity)))
            rows = rng.normal(0.0, np.sqrt(target_variance), size=(100, 1))
            local = float(rows.var(axis=0, ddof=1)[0])
            params = predictive_parameters(rows, a0, b0)
            posterior_mean = float(params[4][0] / (1.0 + 1.0 / params[3][0]))
            predictive_variance = float(params[4][0])
            peer_mean = float(b0[0] / (a0[0] - 1.0))
            values = {
                "local": local, "peer": peer_mean,
                "posterior_mean": posterior_mean,
                "predictive_variance": predictive_variance,
            }
            for name, value in values.items():
                losses[name].append(float(np.log(value / target_variance) ** 2))
            d = max(float(params[3][0] - 1.0), 1.0)
            weights.append(d / 2.0 / (float(a0[0]) + d / 2.0 - 1.0))
        for method, values in losses.items():
            records.append({
                "check": "iid_variance_hierarchy", "heterogeneity": heterogeneity,
                "rho": "", "method": method, "log_mse": float(np.mean(values)),
                "ess_median": "", "ess_target": "",
                "mean_local_weight": float(np.mean(weights)) if method == "predictive_variance" else "",
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
