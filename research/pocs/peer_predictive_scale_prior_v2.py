"""Separate predictive-scale anchor from within-client scale heterogeneity.

The peer prior includes both within-client variance and between-client
centre dispersion, matching the predictive marginal variance used by pooled
peer normalization. Its uncertainty is estimated separately from robust
within-client log-scale heterogeneity and peer variance-estimation noise.
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
from scipy.special import gammaln, polygamma
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
OUT = ROOT / "research" / "pocs" / "results" / "peer_predictive_scale_prior_v2.csv"
DEVICES = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)
SUPPORT_SIZES = (30, 100, 300, 1000)
REPLICATES = 15
SEED = 202609302
EPS = 1e-10
SCALE_FLOOR = 1e-6
N_EVAL = 10000


def effective_df(rows: np.ndarray) -> np.ndarray:
    """Newey-West effective df for squared-residual influence, featurewise."""
    centered = rows - rows.mean(axis=0, keepdims=True)
    influence = centered**2
    influence -= influence.mean(axis=0, keepdims=True)
    denom = np.maximum(np.sum(influence**2, axis=0), EPS)
    n = len(rows)
    lagmax = min(n // 4, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    long_run = np.ones(rows.shape[1])
    for lag in range(1, max(lagmax, 1) + 1):
        ac = np.sum(influence[:-lag] * influence[lag:], axis=0) / denom
        long_run += 2.0 * (1.0 - lag / (lagmax + 1.0)) * ac
    return np.clip(n / np.maximum(long_run, 1.0), 2.0, max(n - 1.0, 2.0))


def peer_prior(peer_rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """IG prior with predictive mean and separate within-scale heterogeneity."""
    means = np.stack([x.mean(axis=0) for x in peer_rows])
    within = np.stack([x.var(axis=0, ddof=1) for x in peer_rows])
    peer_centre = means.mean(axis=0)
    predictive = within + (means - peer_centre[None, :]) ** 2
    dfs = np.asarray([len(x) - 1 for x in peer_rows], dtype=float)[:, None]
    # Match the shared-marginal predictive scale for the prior centre.
    mean = np.maximum(predictive.mean(axis=0), EPS)
    # Estimate prior heterogeneity from within-device scales only. Centre
    # dispersion already enters the predictive mean and is not counted again.
    log_within = np.log(np.maximum(within, EPS))
    log_median = np.median(log_within, axis=0)
    log_mad = 1.4826 * np.median(np.abs(log_within - log_median), axis=0)
    observed_log_heterogeneity = log_mad**2
    log_noise = np.median(polygamma(1.0, np.maximum(dfs, 2.0) / 2.0), axis=0)
    latent_log_heterogeneity = np.maximum(observed_log_heterogeneity - log_noise, 1e-4)
    latent = np.maximum(mean**2 * latent_log_heterogeneity, mean**2 * 1e-8)
    # IG(a,b), with E[V]=b/(a-1)=mean and Var[V]=b^2/((a-1)^2(a-2))=latent.
    a0 = 2.0 + mean**2 / latent
    b0 = mean * (a0 - 1.0)
    return a0, b0


def predictive_parameters(rows: np.ndarray, prior_a: np.ndarray, prior_b: np.ndarray):
    n_eff = effective_df(rows)
    df = np.maximum(n_eff - 1.0, 1.0)
    local_mean = rows.mean(axis=0)
    sample_var = np.maximum(rows.var(axis=0, ddof=1), EPS)
    a = prior_a + df / 2.0
    b = prior_b + df * sample_var / 2.0
    student_df = 2.0 * a
    student_scale = np.sqrt(np.maximum(b * (1.0 + 1.0 / n_eff) / a, EPS))
    predictive_variance = np.maximum(b / np.maximum(a - 1.0, EPS) * (1.0 + 1.0 / n_eff), EPS)
    return local_mean, student_df, student_scale, n_eff, predictive_variance


def predictive_score(rows: np.ndarray, params) -> np.ndarray:
    mean, df, scale, _, _ = params
    z = (rows - mean) / scale
    # Featurewise negative log Student-t predictive density. Constants by
    # feature are retained because they reflect posterior uncertainty/scale.
    log_norm = (
        gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
        - 0.5 * np.log(df * np.pi) - np.log(scale)
    )
    log_density = log_norm - 0.5 * (df + 1.0) * np.log1p(z**2 / df)
    return -2.0 * log_density.sum(axis=1)


def crossfit_scores(rows: np.ndarray, a0: np.ndarray, b0: np.ndarray) -> np.ndarray:
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    return np.concatenate((
        predictive_score(first, predictive_parameters(second, a0, b0)),
        predictive_score(second, predictive_parameters(first, a0, b0)),
    ))


def posterior_gaussian_score(rows: np.ndarray, reference: np.ndarray,
                             prior_a: np.ndarray, prior_b: np.ndarray,
                             held: np.ndarray) -> np.ndarray:
    params = predictive_parameters(reference, prior_a, prior_b)
    return scorer_from_rows(
        reference, params[0], np.sqrt(params[4]) + SCALE_FLOOR
    ).score(held)


def crossfit_posterior_gaussian(rows, prior_a, prior_b):
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    return np.concatenate((
        posterior_gaussian_score(rows, second, prior_a, prior_b, first),
        posterior_gaussian_score(rows, first, prior_a, prior_b, second),
    ))


def crossfit_gaussian_scores(rows: np.ndarray, centre_kind: str, peer_summary=None) -> np.ndarray:
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]

    def score(fit, held):
        if centre_kind == "shared":
            centre, scale = peer_summary.mean, peer_summary.standard_deviation + SCALE_FLOOR
        else:
            centre, scale = fit.mean(axis=0), fit.std(axis=0) + SCALE_FLOOR
        return scorer_from_rows(fit, centre, scale).score(held)

    return np.concatenate((score(second, first), score(first, second)))


def crossfit_local_center_peer_scale(rows: np.ndarray, peer_summary) -> np.ndarray:
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    scale = peer_summary.standard_deviation + SCALE_FLOOR
    return np.concatenate((
        scorer_from_rows(second, second.mean(axis=0), scale).score(first),
        scorer_from_rows(first, first.mean(axis=0), scale).score(second),
    ))


def detector_metrics(benign, attack, threshold):
    labels = np.r_[np.zeros(len(benign)), np.ones(len(attack))]
    scores = np.r_[benign, attack]
    return (
        float(roc_auc_score(labels, scores)),
        float(np.mean(benign > threshold)),
        float(np.mean(attack > threshold)),
        float(roc_auc_score(labels, scores, max_fpr=0.01)),
    )


def draw_eval(rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(rows) <= N_EVAL:
        return rows
    return rows[rng.choice(len(rows), N_EVAL, replace=False)]


def main() -> None:
    devices = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    out: list[dict[str, object]] = []
    for target in DEVICES:
        peer_rows = [devices[k]["support_pool"] for k in DEVICES if k != target]
        prior_a, prior_b = peer_prior(peer_rows)
        peer_summary = aggregate_equal_weight([summarise(x) for x in peer_rows])
        item = devices[target]
        pool, future, attacks = item["support_pool"], item["test_benign"], item["test_attack"]
        for n in SUPPORT_SIZES:
            for rep in range(REPLICATES):
                seed = SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode())
                rng = np.random.default_rng(seed)
                start = int(rng.integers(0, len(pool) - n + 1))
                rows = pool[start:start + n]
                params = predictive_parameters(rows, prior_a, prior_b)
                calibration = crossfit_scores(rows, prior_a, prior_b)
                threshold = float(np.quantile(calibration, 0.99))
                calibration_post_gaussian = crossfit_posterior_gaussian(rows, prior_a, prior_b)
                threshold_post_gaussian = float(np.quantile(calibration_post_gaussian, 0.99))
                threshold_shared = float(np.quantile(
                    crossfit_gaussian_scores(rows, "shared", peer_summary), 0.99
                ))
                threshold_local = float(np.quantile(
                    crossfit_gaussian_scores(rows, "local"), 0.99
                ))
                threshold_local_peer = float(np.quantile(
                    crossfit_local_center_peer_scale(rows, peer_summary), 0.99
                ))
                benign = draw_eval(future, rng)
                attack = draw_eval(attacks, rng)
                pred_b, pred_a = predictive_score(benign, params), predictive_score(attack, params)
                post_gauss = scorer_from_rows(
                    rows, params[0], np.sqrt(params[4]) + SCALE_FLOOR
                )
                postg_b, postg_a = post_gauss.score(benign), post_gauss.score(attack)
                shared = scorer_from_rows(
                    rows, peer_summary.mean, peer_summary.standard_deviation + SCALE_FLOOR
                )
                local_scale = rows.std(axis=0) + SCALE_FLOOR
                local = scorer_from_rows(rows, rows.mean(axis=0), local_scale)
                local_peer = scorer_from_rows(
                    rows, rows.mean(axis=0), peer_summary.standard_deviation + SCALE_FLOOR
                )
                shared_b, shared_a = shared.score(benign), shared.score(attack)
                local_b, local_a = local.score(benign), local.score(attack)
                local_peer_b, local_peer_a = local_peer.score(benign), local_peer.score(attack)
                auc_p, fpr_p, tpr_p, pauc_p = detector_metrics(pred_b, pred_a, threshold)
                auc_pg, fpr_pg, tpr_pg, pauc_pg = detector_metrics(
                    postg_b, postg_a, threshold_post_gaussian
                )
                auc_s, fpr_s, tpr_s, pauc_s = detector_metrics(shared_b, shared_a, threshold_shared)
                auc_l, fpr_l, tpr_l, pauc_l = detector_metrics(local_b, local_a, threshold_local)
                auc_lp, fpr_lp, tpr_lp, pauc_lp = detector_metrics(
                    local_peer_b, local_peer_a, threshold_local_peer
                )
                out.append({
                    "device": target, "support_n": n, "replicate": rep, "offset": start,
                    "mean_ess": float(np.mean(params[3])),
                    "median_posterior_df": float(np.median(params[1])),
                    "auroc_predictive": auc_p, "auroc_shared": auc_s, "auroc_local": auc_l,
                    "auroc_posterior_gaussian": auc_pg,
                    "auroc_local_center_peer_scale": auc_lp,
                    "spauc01_predictive": pauc_p, "spauc01_shared": pauc_s,
                    "spauc01_local": pauc_l,
                    "spauc01_posterior_gaussian": pauc_pg,
                    "spauc01_local_center_peer_scale": pauc_lp,
                    "fpr_predictive": fpr_p, "fpr_shared": fpr_s, "fpr_local": fpr_l,
                    "fpr_posterior_gaussian": fpr_pg,
                    "fpr_local_center_peer_scale": fpr_lp,
                    "tpr_predictive": tpr_p, "tpr_shared": tpr_s, "tpr_local": tpr_l,
                    "tpr_posterior_gaussian": tpr_pg,
                    "tpr_local_center_peer_scale": tpr_lp,
                    "threshold_predictive": threshold,
                })
            print(target, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    import pandas as pd
    frame = pd.DataFrame(out)
    summary = frame.groupby("support_n").agg(
        predictive_auroc=("auroc_predictive", "mean"),
        shared_auroc=("auroc_shared", "mean"),
        local_auroc=("auroc_local", "mean"),
        predictive_fpr=("fpr_predictive", "mean"),
        predictive_tpr=("tpr_predictive", "mean"),
        median_device_auroc_delta_shared=("auroc_predictive", "mean"),
    )
    for n in summary.index:
        g = frame[frame.support_n == n]
        device_delta = (g.groupby("device").auroc_predictive.mean()
                        - g.groupby("device").auroc_shared.mean())
        summary.loc[n, "median_device_auroc_delta_shared"] = device_delta.median()
        summary.loc[n, "positive_devices_vs_shared"] = int((device_delta > 0).sum())
    print(summary.round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
