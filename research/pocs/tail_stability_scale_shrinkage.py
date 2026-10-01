"""Benign-only cross-fit upper-tail stability shrinkage (exploratory POC).

Selects the local-versus-peer marginal-scale blend by minimizing disagreement
between cross-fitted benign upper-tail score means. The objective uses only
the target's benign onboarding window and leave-one-device-out peer scales.
Attack and future-test rows are loaded only for post-selection evaluation.
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
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "tail_stability_scale_shrinkage.csv"
DEVICES = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)
SUPPORT_SIZES = (30, 100, 300, 1000)
REPLICATES = 3
SEED = 77309601
FLOOR = 1e-6
RATE = 0.01
EVAL_ROWS = 10000


def blended_scale(rows: np.ndarray, peer_scale: np.ndarray, alpha: float) -> np.ndarray:
    """Geometric variance blend; alpha=0 is peer, alpha=1 is local."""
    local_var = np.maximum(rows.var(axis=0, ddof=0), FLOOR**2)
    peer_var = np.maximum(peer_scale**2, FLOOR**2)
    return np.exp(0.5 * ((1.0 - alpha) * np.log(peer_var) + alpha * np.log(local_var))) + FLOOR


def fit_score(rows: np.ndarray, peer_scale: np.ndarray, alpha: float):
    return scorer_from_rows(rows, rows.mean(axis=0), blended_scale(rows, peer_scale, alpha)).score


def upper_tail_mean(scores: np.ndarray) -> float:
    count = max(1, int(np.ceil(len(scores) / 4.0)))
    return float(np.mean(np.partition(scores, len(scores) - count)[-count:]))


def crossfit_tail_pair(rows: np.ndarray, peer_scale: np.ndarray, alpha: float):
    middle = len(rows) // 2
    first, second = rows[:middle], rows[middle:]
    first_scores = fit_score(second, peer_scale, alpha)(first)
    second_scores = fit_score(first, peer_scale, alpha)(second)
    return upper_tail_mean(first_scores), upper_tail_mean(second_scores)


def select_alpha(rows: np.ndarray, peer_scale: np.ndarray) -> tuple[float, float]:
    def objective(alpha: float) -> float:
        q_first, q_second = crossfit_tail_pair(rows, peer_scale, alpha)
        return abs(np.log(max(q_first, FLOOR)) - np.log(max(q_second, FLOOR)))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded",
                             options={"xatol": 1e-3, "maxiter": 60})
    return float(result.x), float(result.fun)


def crossfit_scores(rows: np.ndarray, peer_scale: np.ndarray, alpha: float) -> np.ndarray:
    middle = len(rows) // 2
    first, second = rows[:middle], rows[middle:]
    return np.concatenate((
        fit_score(second, peer_scale, alpha)(first),
        fit_score(first, peer_scale, alpha)(second),
    ))


def shared_marginal_score(rows: np.ndarray, peer) -> callable:
    return scorer_from_rows(rows, peer.mean, peer.standard_deviation + FLOOR).score


def crossfit_shared_scores(rows: np.ndarray, peer) -> np.ndarray:
    middle = len(rows) // 2
    first, second = rows[:middle], rows[middle:]
    return np.concatenate((shared_marginal_score(second, peer)(first),
                           shared_marginal_score(first, peer)(second)))


def auc_fpr_tpr(benign, attack, threshold):
    labels = np.r_[np.zeros(len(benign)), np.ones(len(attack))]
    scores = np.r_[benign, attack]
    return (
        float(roc_auc_score(labels, scores)),
        float(roc_auc_score(labels, scores, max_fpr=0.01)),
        float(np.mean(benign > threshold)),
        float(np.mean(attack > threshold)),
    )


def main() -> None:
    datasets = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    records: list[dict[str, object]] = []
    for target in DEVICES:
        peers = [summarise(datasets[k]["support_pool"]) for k in DEVICES if k != target]
        peer = aggregate_equal_weight(peers)
        peer_scale = peer.standard_deviation
        pool = datasets[target]["support_pool"]
        for n in SUPPORT_SIZES:
            for rep in range(REPLICATES):
                rng = np.random.default_rng(SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode()))
                start = int(rng.integers(0, len(pool) - n + 1))
                rows = pool[start:start + n]
                alpha, stability_loss = select_alpha(rows, peer_scale)
                benign = datasets[target]["test_benign"]
                attack = datasets[target]["test_attack"]
                if len(benign) > EVAL_ROWS:
                    benign = benign[rng.choice(len(benign), EVAL_ROWS, replace=False)]
                if len(attack) > EVAL_ROWS:
                    attack = attack[rng.choice(len(attack), EVAL_ROWS, replace=False)]
                methods = {
                    "selected": alpha,
                    "peer_scale_local_center": 0.0,
                    "local_scale_local_center": 1.0,
                    "shared_marginals": None,
                }
                for method, weight in methods.items():
                    if method == "shared_marginals":
                        model = shared_marginal_score(rows, peer)
                        calibration_scores = crossfit_shared_scores(rows, peer)
                    else:
                        model = fit_score(rows, peer_scale, float(weight))
                        calibration_scores = crossfit_scores(rows, peer_scale, float(weight))
                    threshold = float(np.quantile(calibration_scores, 1.0 - RATE))
                    benign_scores, attack_scores = model(benign), model(attack)
                    auc, pauc, fpr, tpr = auc_fpr_tpr(benign_scores, attack_scores, threshold)
                    records.append({
                        "device": target, "support_n": n, "replicate": rep,
                        "offset": start, "method": method,
                        "selected_alpha": alpha, "stability_loss": stability_loss,
                        "auroc": auc, "spauc01": pauc, "fpr": fpr, "tpr": tpr,
                    })
            print(target, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    import pandas as pd
    frame = pd.DataFrame(records)
    device_means = frame.groupby(["support_n", "method", "device"])[
        ["auroc", "spauc01", "fpr", "tpr", "selected_alpha", "stability_loss"]
    ].mean().reset_index()
    print(device_means.groupby(["support_n", "method"]).agg(
        auroc=("auroc", "mean"), median_auroc=("auroc", "median"),
        worst_auroc=("auroc", "min"), spauc01=("spauc01", "mean"),
        fpr=("fpr", "mean"), tpr=("tpr", "mean"), alpha=("selected_alpha", "mean"),
    ).round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
