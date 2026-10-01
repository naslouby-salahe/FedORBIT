"""Synthetic support-only sanity check for tail-stability shrinkage."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "pocs"))
from tail_stability_scale_shrinkage import crossfit_tail_pair, select_alpha

OUT = ROOT / "research" / "pocs" / "results" / "tail_stability_synthetic.csv"
SEED = 202609303
REPS = 60
D = 20


def main() -> None:
    rng = np.random.default_rng(SEED)
    correlation = 0.25 * np.ones((D, D)) + 0.75 * np.eye(D)
    scale = np.exp(np.linspace(-0.6, 0.6, D))
    covariance = correlation * np.outer(scale, scale)
    records = []
    for n in (30, 100, 300, 1000):
        regimes = (
            ("compatible", np.ones(D)),
            # Uniform scaling is a negative control: Mahalanobis scores can
            # cancel a common scale factor and the objective should stay flat.
            ("peer_scale_half", np.full(D, 0.5)),
            ("peer_scale_double", np.full(D, 2.0)),
            # Feature-wise disagreement changes the shape of the geometry.
            ("peer_scale_alternating", np.where(np.arange(D) % 2, 2.0, 0.5)),
        )
        chosen_by_regime = {regime: [] for regime, _ in regimes}
        for replicate in range(REPS):
            # Pair each peer-scale condition on the same target support draw.
            rows = rng.multivariate_normal(np.zeros(D), covariance, size=n)
            for regime, peer_multipliers in regimes:
                peer_scale = scale * peer_multipliers
                alpha, objective = select_alpha(rows, peer_scale)
                peer_tails = crossfit_tail_pair(rows, peer_scale, 0.0)
                local_tails = crossfit_tail_pair(rows, peer_scale, 1.0)
                peer_objective = abs(np.log(peer_tails[0]) - np.log(peer_tails[1]))
                local_objective = abs(np.log(local_tails[0]) - np.log(local_tails[1]))
                chosen_by_regime[regime].append(alpha)
                records.append({"support_n": n, "regime": regime, "replicate": replicate,
                                "selected_alpha": alpha, "stability_loss": objective,
                                "peer_loss": peer_objective, "local_loss": local_objective})
        for regime, chosen in chosen_by_regime.items():
            print(n, regime, "mean alpha", round(float(np.mean(chosen)), 3),
                  "median", round(float(np.median(chosen)), 3))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("wrote", len(records), OUT)


if __name__ == "__main__":
    main()
