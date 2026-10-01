"""Collaborative block-correlation geometry for five-window feature groups.

This POC tests whether peer information is useful beyond the marginal scale
preconditioner by transferring the dependence among the five decay windows
within each of the 23 repeated N-BaIoT feature families. The target uses a
local centre and peer predictive marginal scale. Correlation matrices are
Fisher-z pooled across peers and continuously blended with local support
correlations using peer heterogeneity and target effective support.
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight, summarise

DATA = ROOT / "outputs" / "prepared" / "nbaiot"
RAW = ROOT / "data" / "raw" / "N-BaIoT"
OUT = ROOT / "research" / "pocs" / "results" / "peer_block_correlation_scale.csv"
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
SEED = 202609310
FLOOR = 1e-8
RATE = 0.01
EVAL_ROWS = 5000


def feature_groups():
    columns = pd.read_csv(RAW / DEVICES[0] / "benign_traffic.csv", nrows=0).columns.tolist()
    grouped = {}
    pattern = re.compile(r"(.+?)_(L5|L3|L1|L0\.1|L0\.01)_(.+)$")
    for j, name in enumerate(columns):
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"Unmapped feature {name}")
        prefix, window, statistic = match.groups()
        grouped.setdefault(f"{prefix}::{statistic}", {})[window] = j
    return [np.asarray([grouped[key][w] for w in WINDOWS], dtype=int)
            for key in sorted(grouped)]


def corr_matrix(rows):
    centred = rows - rows.mean(axis=0, keepdims=True)
    covariance = centred.T @ centred
    norm = np.sqrt(np.maximum(np.diag(covariance), FLOOR))
    corr = covariance / np.outer(norm, norm)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return nearest_corr(corr)


def nearest_corr(corr):
    corr = (corr + corr.T) / 2.0
    eigval, eigvec = np.linalg.eigh(corr)
    corr = (eigvec * np.maximum(eigval, 0.03)) @ eigvec.T
    norm = np.sqrt(np.maximum(np.diag(corr), FLOOR))
    return corr / np.outer(norm, norm)


def peer_corr_prior(peer_rows, groups):
    # Store Fisher-z pair means and robust between-client variance per group.
    prior = []
    for idx in groups:
        cs = np.stack([corr_matrix(x[:, idx]) for x in peer_rows])
        z = np.arctanh(np.clip(cs, -0.98, 0.98))
        mean = np.mean(z, axis=0)
        med = np.median(z, axis=0)
        tau2 = (1.4826 * np.median(np.abs(z - med[None, :, :]), axis=0)) ** 2
        prior.append((mean, tau2))
    return prior


def target_ess(rows):
    centered = rows - rows.mean(axis=0, keepdims=True)
    influence = centered**2
    influence -= influence.mean(axis=0, keepdims=True)
    den = np.maximum(np.sum(influence**2, axis=0), FLOOR)
    rho = np.sum(influence[:-1] * influence[1:], axis=0) / den
    return np.clip(len(rows) * (1.0 - rho) / (1.0 + rho), 2.0, max(len(rows)-1, 2))


def blended_blocks(rows, groups, prior, rule):
    ess = target_ess(rows)
    blocks, weights = [], []
    for idx, (peer_z, tau2) in zip(groups, prior):
        local_r = corr_matrix(rows[:, idx])
        local_z = np.arctanh(np.clip(local_r, -0.98, 0.98))
        n_eff = max(float(np.median(ess[idx])), 4.0)
        if rule == "local":
            weight = 1.0
        elif rule == "peer":
            weight = 0.0
        else:
            local_var = 1.0 / max(n_eff - 3.0, 1.0)
            # Fisher-z normal-normal posterior weight on the local estimate.
            weight = tau2 / (tau2 + local_var / max(len(DEVICES) - 1, 1) + 1e-8)
            np.fill_diagonal(weight, 1.0)
        z = weight * local_z + (1.0 - weight) * peer_z
        corr = np.tanh(z)
        np.fill_diagonal(corr, 1.0)
        corr = nearest_corr(corr)
        blocks.append(np.linalg.inv(corr))
        if np.isscalar(weight):
            weights.append(float(weight))
        else:
            weights.append(float(np.mean(weight[np.triu_indices(len(idx), 1)])))
    return blocks, float(np.mean(weights)), float(np.median(ess))


def score(rows, peer_summary, groups, prior, method):
    local_mean = rows.mean(axis=0)
    peer_scale = peer_summary.standard_deviation + 1e-6
    if method == "shared_marginals":
        return scorer_from_rows(rows, peer_summary.mean, peer_scale).score
    if method == "peer_scale_local_center":
        return scorer_from_rows(rows, local_mean, peer_scale).score
    if method == "local_oas":
        return scorer_from_rows(rows, local_mean, rows.std(axis=0) + 1e-6).score
    rule = "local" if method == "block_local_corr" else "peer" if method == "block_peer_corr" else "blend"
    blocks, _, _ = blended_blocks(rows, groups, prior, rule)

    def apply(test):
        z = (test - local_mean) / peer_scale
        out = np.zeros(len(test))
        for idx, precision in zip(groups, blocks):
            part = z[:, idx]
            out += np.sum((part @ precision) * part, axis=1)
        return out
    return apply


def crossfit(rows, peer_summary, groups, prior, method):
    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    return np.r_[score(second, peer_summary, groups, prior, method)(first),
                 score(first, peer_summary, groups, prior, method)(second)]


def metrics(benign, attack, threshold):
    labels = np.r_[np.zeros(len(benign)), np.ones(len(attack))]
    values = np.r_[benign, attack]
    return (float(roc_auc_score(labels, values)),
            float(roc_auc_score(labels, values, max_fpr=RATE)),
            float(np.mean(benign > threshold)), float(np.mean(attack > threshold)))


def main():
    groups = feature_groups()
    devices = {name: np.load(DATA / f"{name}.npz") for name in DEVICES}
    records = []
    methods = ("local_oas", "peer_scale_local_center", "shared_marginals",
               "block_local_corr", "block_peer_corr", "block_reliability")
    for target in DEVICES:
        peer_rows = [devices[k]["support_pool"] for k in DEVICES if k != target]
        peer_summary = aggregate_equal_weight([summarise(x) for x in peer_rows])
        prior = peer_corr_prior(peer_rows, groups)
        pool = devices[target]["support_pool"]
        for n in SUPPORT_SIZES:
            for rep in range(REPLICATES):
                rng = np.random.default_rng(SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode()))
                start = int(rng.integers(0, len(pool) - n + 1))
                rows = pool[start:start+n]
                benign, attack = devices[target]["test_benign"], devices[target]["test_attack"]
                if len(benign) > EVAL_ROWS:
                    benign = benign[rng.choice(len(benign), EVAL_ROWS, replace=False)]
                if len(attack) > EVAL_ROWS:
                    attack = attack[rng.choice(len(attack), EVAL_ROWS, replace=False)]
                for method in methods:
                    model = score(rows, peer_summary, groups, prior, method)
                    threshold = float(np.quantile(crossfit(rows, peer_summary, groups, prior, method), 1-RATE))
                    auc, pauc, fpr, tpr = metrics(model(benign), model(attack), threshold)
                    _, local_weight, ess = blended_blocks(rows, groups, prior, "blend")
                    records.append({"device": target, "support_n": n, "replicate": rep,
                                    "offset": start, "method": method, "auroc": auc,
                                    "spauc01": pauc, "fpr": fpr, "tpr": tpr,
                                    "mean_local_corr_weight": local_weight,
                                    "median_ess": ess})
            print(target, n, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT, index=False)
    per_device = frame.groupby(["support_n", "method", "device"])[
        ["auroc", "spauc01", "fpr", "tpr", "mean_local_corr_weight"]
    ].mean().reset_index()
    print(per_device.groupby(["support_n", "method"]).mean(numeric_only=True).round(4).to_string())
    print("wrote", len(frame), OUT)


if __name__ == "__main__":
    main()
