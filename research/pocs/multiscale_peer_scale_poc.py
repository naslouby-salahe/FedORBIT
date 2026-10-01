"""Exploratory OAS evaluation of multiscale collaborative scale posteriors.

The N-BaIoT schema repeats 23 measurement families over five decay windows.
This POC estimates a peer covariance over each family's five log variances,
combines it with target benign log-variance uncertainty from HAC ESS, and uses
a shared Student-t scale mixture to discount peer structure when the target's
support profile is surprising. Adaptation uses benign support and peer benign
summaries only; attack rows are read only for exploratory post-fit evaluation.
"""
from __future__ import annotations

import csv
import os
import re
import sys
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln, logsumexp, polygamma
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
RAW = ROOT / "data" / "raw" / "N-BaIoT"
OUT = ROOT / "research" / "pocs" / "results" / "multiscale_peer_scale_hacnoise_poc.csv"
DEVICES = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)
WINDOWS = ("L5", "L3", "L1", "L0.1", "L0.01")
SUPPORT_SIZES = (30, 100, 300, 1000)
REPLICATES = 8
SEED = 202609308
FLOOR = 1e-9
RATE = 0.01
EVAL_ROWS = 5000
NU = 4.0
LAMBDA_GRID = np.exp(np.linspace(np.log(0.03), np.log(30.0), 61))


def feature_groups() -> list[np.ndarray]:
    header_path = RAW / DEVICES[0] / "benign_traffic.csv"
    columns = pd.read_csv(header_path, nrows=0).columns.tolist()
    grouped: dict[str, dict[str, int]] = {}
    pattern = re.compile(r"(.+?)_(L5|L3|L1|L0\.1|L0\.01)_(.+)$")
    for index, name in enumerate(columns):
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"Unmapped N-BaIoT feature: {name}")
        prefix, window, statistic = match.groups()
        grouped.setdefault(f"{prefix}::{statistic}", {})[window] = index
    if len(grouped) != 23 or any(set(v) != set(WINDOWS) for v in grouped.values()):
        raise ValueError("Expected 23 complete five-window feature families")
    return [np.asarray([grouped[key][window] for window in WINDOWS], dtype=int)
            for key in sorted(grouped)]


def variance_ac1(rows: np.ndarray) -> np.ndarray:
    centered = rows - rows.mean(axis=0, keepdims=True)
    influence = centered**2
    influence -= influence.mean(axis=0, keepdims=True)
    den = np.maximum(np.sum(influence**2, axis=0), FLOOR)
    rho = np.sum(influence[:-1] * influence[1:], axis=0) / den
    return np.clip(rho, -0.95, 0.98)


def hac_ess(rows: np.ndarray) -> np.ndarray:
    centered = rows - rows.mean(axis=0, keepdims=True)
    influence = centered**2
    influence -= influence.mean(axis=0, keepdims=True)
    den = np.maximum(np.sum(influence**2, axis=0), FLOOR)
    n = len(rows)
    base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lagmax = min(n // 4, 2 * base)
    lrv = np.ones(rows.shape[1])
    for lag in range(1, lagmax + 1):
        ac = np.sum(influence[:-lag] * influence[lag:], axis=0) / den
        lrv += 2.0 * (1.0 - lag / (lagmax + 1.0)) * ac
    return np.clip(n / np.maximum(lrv, 1.0), 2.0, max(n - 1.0, 2.0))


def peer_profile(peer_rows: list[np.ndarray], groups: list[np.ndarray]):
    # Peer profiles are [client, base-feature-group, time-window].
    profiles = np.stack([
        np.stack([np.log(np.maximum(x[:, idx].var(axis=0, ddof=1), FLOOR))
                  for idx in groups]) for x in peer_rows
    ])
    center = np.median(profiles, axis=0)
    residual = profiles - center[None, :, :]
    mad = 1.4826 * np.median(np.abs(residual), axis=0)
    sample_sd = np.std(profiles, axis=0, ddof=1)
    sd = np.where(mad > 1e-5, mad, sample_sd)
    diagonal = np.maximum(sd**2, 1e-7)
    standardized = residual / np.sqrt(diagonal[None, :, :])
    pooled = standardized.transpose(1, 0, 2).reshape(-1, len(WINDOWS))
    corr = np.corrcoef(pooled, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    eigval, eigvec = np.linalg.eigh((corr + corr.T) / 2.0)
    corr = (eigvec * np.maximum(eigval, 0.05)) @ eigvec.T
    norm = np.sqrt(np.diag(corr))
    corr = corr / np.outer(norm, norm)
    return profiles, center, diagonal, corr


def posterior_log_scales(rows: np.ndarray, groups: list[np.ndarray], center: np.ndarray,
                         diagonal: np.ndarray, corr: np.ndarray, robust: bool,
                         correlated_noise: bool = False):
    group_count, windows = center.shape
    local_var = np.stack([
        np.maximum(rows[:, idx].var(axis=0, ddof=1), FLOOR) for idx in groups
    ])
    ess = np.stack([hac_ess(rows[:, idx]) for idx in groups])
    df = np.maximum(ess - 1.0, 2.0)
    bias = digamma(df / 2.0) - np.log(df / 2.0)
    observation = np.log(local_var) - bias
    obs_var = polygamma(1, df / 2.0)
    residual = observation - center
    prior_cov = (np.sqrt(diagonal)[:, :, None] * corr[None, :, :]
                 * np.sqrt(diagonal)[:, None, :])
    if correlated_noise:
        # HAC covariance of the five sample log variances within each family.
        # This captures both overlap between decay windows and row dependence.
        noise_cov = np.empty((group_count, windows, windows))
        for g, idx in enumerate(groups):
            values = rows[:, idx]
            centered = values - values.mean(axis=0, keepdims=True)
            variance = np.maximum(np.mean(centered**2, axis=0), FLOOR)
            influence = centered**2 / variance[None, :] - 1.0
            influence -= influence.mean(axis=0, keepdims=True)
            n = len(values)
            base = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
            lagmax = min(n // 4, 2 * base)
            long_run = influence.T @ influence / n
            for lag in range(1, lagmax + 1):
                cross = influence[lag:].T @ influence[:-lag] / n
                weight = 1.0 - lag / (lagmax + 1.0)
                long_run += weight * (cross + cross.T)
            covariance = (long_run + long_run.T) / (2.0 * n)
            eigval, eigvec = np.linalg.eigh(covariance)
            noise_cov[g] = (eigvec * np.maximum(eigval, 1e-8)) @ eigvec.T
    else:
        noise_cov = np.eye(windows)[None, :, :] * obs_var[:, :, None]
    if not robust:
        total = prior_cov + noise_cov
        solved = np.linalg.solve(total, residual[..., None])[..., 0]
        gain = np.swapaxes(np.linalg.solve(total, prior_cov), -1, -2)
        return center + np.einsum("gij,gj->gi", gain, residual), ess

    # Global Student-t scale mixture, integrated on a fixed log-spaced grid.
    lambdas = LAMBDA_GRID
    shape = rate = NU / 2.0
    log_prior = (shape * np.log(rate) - gammaln(shape)
                 + (shape - 1.0) * np.log(lambdas) - rate * lambdas)
    scaled_prior = prior_cov[None, :, :, :] / lambdas[:, None, None, None]
    total = scaled_prior + noise_cov[None, :, :, :]
    _, logdet = np.linalg.slogdet(total)
    residual_l = residual[None, :, :, None]
    solved_residual = np.linalg.solve(total, residual_l)[..., 0]
    quad = np.sum(residual[None, :, :] * solved_residual, axis=(1, 2))
    log_evidence = log_prior - 0.5 * (
        group_count * windows * np.log(2.0 * np.pi)
        + np.sum(logdet, axis=1) + quad
    )
    weights = np.exp(log_evidence - logsumexp(log_evidence))
    solved_prior = np.linalg.solve(total, scaled_prior)
    gain = np.swapaxes(solved_prior, -1, -2)
    conditional = center[None, :, :] + np.einsum("lgij,gj->lgi", gain, residual)
    return np.einsum("l,lgw->gw", weights, conditional), ess


def model(rows, peer, groups, center, diagonal, corr, method):
    local_mean = rows.mean(axis=0)
    local_scale = np.sqrt(np.maximum(rows.var(axis=0, ddof=1), FLOOR))
    if method == "local":
        scale = local_scale
        score_mean = local_mean
    elif method == "peer_scale_local_center":
        scale = peer.standard_deviation
        score_mean = local_mean
    elif method == "shared_marginals":
        scale = peer.standard_deviation
        score_mean = peer.mean
    elif method in ("multiscale_gaussian", "multiscale_t", "multiscale_hacnoise"):
        theta, ess = posterior_log_scales(rows, groups, center, diagonal, corr,
                                          robust=method == "multiscale_t",
                                          correlated_noise=method == "multiscale_hacnoise")
        scale = np.empty(rows.shape[1])
        for g, idx in enumerate(groups):
            scale[idx] = np.exp(0.5 * theta[g])
        score_mean = local_mean
    elif method == "diagonal_eb":
        theta, ess = posterior_log_scales(rows, groups, center, diagonal,
                                          np.eye(len(WINDOWS)), robust=False)
        scale = np.empty(rows.shape[1])
        for g, idx in enumerate(groups):
            scale[idx] = np.exp(0.5 * theta[g])
        score_mean = local_mean
    else:
        raise ValueError(method)
    return scorer_from_rows(rows, score_mean, scale + 1e-6).score


def crossfit(rows, peer, groups, center, diagonal, corr, method):
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    return np.concatenate((
        model(second, peer, groups, center, diagonal, corr, method)(first),
        model(first, peer, groups, center, diagonal, corr, method)(second),
    ))


def evaluation(benign, attack, threshold):
    labels = np.r_[np.zeros(len(benign)), np.ones(len(attack))]
    scores = np.r_[benign, attack]
    return (float(roc_auc_score(labels, scores)),
            float(roc_auc_score(labels, scores, max_fpr=0.01)),
            float(np.mean(benign > threshold)), float(np.mean(attack > threshold)))


def main():
    groups = feature_groups()
    devices = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    summaries = {name: summarise(devices[name]["support_pool"]) for name in DEVICES}
    methods = ("local", "peer_scale_local_center", "shared_marginals",
               "diagonal_eb", "multiscale_gaussian", "multiscale_t",
               "multiscale_hacnoise")
    records = []
    for target in DEVICES:
        peer_names = [name for name in DEVICES if name != target]
        peer_rows = [devices[name]["support_pool"] for name in peer_names]
        peer_summaries = [summaries[name] for name in peer_names]
        peer = aggregate_equal_weight(peer_summaries)
        _, center, diagonal, corr = peer_profile(peer_rows, groups)
        pool = devices[target]["support_pool"]
        for n in SUPPORT_SIZES:
            for rep in range(REPLICATES):
                rng = np.random.default_rng(
                    SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode())
                )
                start = int(rng.integers(0, len(pool) - n + 1))
                rows = pool[start:start+n]
                benign = devices[target]["test_benign"]
                attack = devices[target]["test_attack"]
                if len(benign) > EVAL_ROWS:
                    benign = benign[rng.choice(len(benign), EVAL_ROWS, replace=False)]
                if len(attack) > EVAL_ROWS:
                    attack = attack[rng.choice(len(attack), EVAL_ROWS, replace=False)]
                for method in methods:
                    scorer = model(rows, peer, groups, center, diagonal, corr, method)
                    calibration = crossfit(rows, peer, groups, center, diagonal, corr, method)
                    threshold = float(np.quantile(calibration, 1.0 - RATE))
                    auc, pauc, fpr, tpr = evaluation(scorer(benign), scorer(attack), threshold)
                    records.append({"device": target, "support_n": n,
                                    "replicate": rep, "offset": start,
                                    "method": method, "auroc": auc, "spauc01": pauc,
                                    "fpr": fpr, "tpr": tpr})
            print(target, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
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
