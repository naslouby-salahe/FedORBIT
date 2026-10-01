"""Stress-test support ESS estimators on Gaussian AR(1) sequences.

For variance estimates the influence process is (X_t-mean)^2 centered; under
Gaussian AR(1) its asymptotic lag-k correlation is rho^(2k), giving a known
large-sample benchmark. This evaluates a component needed by collaborative
scale adaptation; it does not use attack data.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "pocs" / "results" / "ess_estimator_stress.csv"
SEED = 202609305
REPS = 200


def ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    innovation = rng.normal(scale=np.sqrt(1.0 - rho**2), size=n)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + innovation[i]
    return x


def influence(x: np.ndarray, kind: str) -> np.ndarray:
    z = x - x.mean()
    if kind == "mean":
        return z
    z2 = z**2
    return z2 - z2.mean()


def acf_sequence(y: np.ndarray, max_lag: int) -> np.ndarray:
    z = y - y.mean()
    denom = max(float(z @ z), 1e-12)
    return np.asarray([float(z[:-lag] @ z[lag:]) / denom
                       for lag in range(1, max_lag + 1)])


def ess_nw(y: np.ndarray, multiplier: float) -> float:
    n = len(y)
    base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lagmax = min(n // 4, max(1, int(round(base * multiplier))))
    acf = acf_sequence(y, lagmax)
    lrv = 1.0 + 2.0 * np.sum((1.0 - np.arange(1, lagmax + 1) /
                              (lagmax + 1.0)) * acf)
    return float(np.clip(n / max(lrv, 1.0), 1.0, n))


def ess_initial_positive(y: np.ndarray) -> float:
    n = len(y)
    acf = acf_sequence(y, min(n // 3, 200))
    # Geyer's initial positive sequence: accumulate adjacent autocorrelation
    # pairs until the first nonpositive pair.
    total = 0.0
    for k in range(0, len(acf) - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        total += pair
    lrv = max(1.0 + 2.0 * total, 1.0)
    return float(np.clip(n / lrv, 1.0, n))


def ess_ar1(y: np.ndarray) -> float:
    acf1 = float(acf_sequence(y, 1)[0])
    rho = float(np.clip(acf1, -0.98, 0.98))
    lrv = (1.0 + rho) / (1.0 - rho)
    return float(np.clip(len(y) / max(lrv, 1.0), 1.0, float(len(y))))


def main() -> None:
    rng = np.random.default_rng(SEED)
    estimators = {"NW_default": lambda y: ess_nw(y, 1.0),
                  "NW_2x": lambda y: ess_nw(y, 2.0),
                  "initial_positive": ess_initial_positive,
                  "AR1_fit": ess_ar1}
    groups: dict[tuple[int, float, str, str], list[float]] = {}
    for n in (100, 300, 1000):
        for rho in (0.0, 0.5, 0.8):
            for kind in ("mean", "variance"):
                target = n * (1.0 - rho) / (1.0 + rho) if kind == "mean" else \
                    n * (1.0 - rho**2) / (1.0 + rho**2)
                for _ in range(REPS):
                    x = ar1(rng, n, rho)
                    y = influence(x, kind)
                    for name, estimator in estimators.items():
                        groups.setdefault((n, rho, kind, name), []).append(
                            estimator(y) / target
                        )
    records = []
    for (n, rho, kind, name), ratios in sorted(groups.items()):
        records.append({
            "n": n, "rho": rho, "quantity": kind, "estimator": name,
            "ess_ratio_to_asymptotic": float(np.median(ratios)),
            "ratio_q10": float(np.quantile(ratios, 0.1)),
            "ratio_q90": float(np.quantile(ratios, 0.9)),
            "replications": REPS,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("wrote", len(records), OUT)
    for row in records:
        if row["quantity"] == "variance":
            print(row["n"], row["rho"], row["estimator"],
                  round(row["ess_ratio_to_asymptotic"], 3),
                  round(row["ratio_q10"], 3), round(row["ratio_q90"], 3))


if __name__ == "__main__":
    main()
