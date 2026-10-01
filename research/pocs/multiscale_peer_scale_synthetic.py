"""Synthetic POC for correlated multiscale peer-scale shrinkage.

N-BaIoT's 115 features are 23 repeated measurement families across five
temporal windows. This screen asks whether a full peer covariance over those
five log-scale coordinates can outperform independent per-feature shrinkage.
The posterior uses known log-variance sampling uncertainty in this synthetic
check; the later real-data POC must estimate it from benign support/peer rows.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.special import polygamma
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "pocs" / "results" / "multiscale_peer_scale_synthetic.csv"
SEED = 202609308
REPS = 120
GROUPS = 23
WINDOWS = 5
PEERS = 8


def profile_covariance(kind: str, magnitude: float) -> np.ndarray:
    if kind == "smooth":
        corr = 0.8 ** np.abs(np.subtract.outer(np.arange(WINDOWS), np.arange(WINDOWS)))
    elif kind == "common_mode":
        corr = 0.65 * np.ones((WINDOWS, WINDOWS)) + 0.35 * np.eye(WINDOWS)
    elif kind == "independent":
        corr = np.eye(WINDOWS)
    else:
        raise ValueError(kind)
    return magnitude**2 * corr


def estimate_profile_prior(peer_profiles: np.ndarray):
    """Estimate group means, per-window variances, and pooled window correlation."""
    # [peer, group, window]
    center = peer_profiles.mean(axis=0)
    residual = peer_profiles - center[None, :, :]
    diagonal = np.var(peer_profiles, axis=0, ddof=1)
    standardized = residual / np.sqrt(np.maximum(diagonal[None, :, :], 1e-10))
    pooled = standardized.transpose(1, 0, 2).reshape(-1, WINDOWS)
    corr = np.corrcoef(pooled, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    eigval, eigvec = np.linalg.eigh((corr + corr.T) / 2.0)
    corr = (eigvec * np.maximum(eigval, 0.05)) @ eigvec.T
    sd = np.sqrt(np.diag(corr))
    corr = corr / np.outer(sd, sd)
    return center, np.maximum(diagonal, 1e-8), corr


def estimators(y: np.ndarray, center: np.ndarray, diagonal: np.ndarray,
               corr: np.ndarray, obs_var: np.ndarray) -> dict[str, np.ndarray]:
    # y, center, diagonal, obs_var all have [group, window] shape.
    local = y
    peer = center
    diag_weight = diagonal / (diagonal + obs_var)
    diagonal_eb = peer + diag_weight * (y - peer)
    multiscale = np.empty_like(y)
    for g in range(y.shape[0]):
        prior_cov = (np.sqrt(diagonal[g])[:, None] * corr
                     * np.sqrt(diagonal[g])[None, :])
        noise_cov = np.diag(obs_var[g])
        # E[theta | y] = mu + Sigma(Sigma+V)^-1(y-mu).
        gain = np.linalg.solve(prior_cov + noise_cov, prior_cov).T
        multiscale[g] = peer[g] + gain @ (y[g] - peer[g])
    # Robust global reliability: a Student-t scale mixture over the entire
    # target's 23 multiscale profiles. A latent precision lambda < 1 inflates
    # the peer prior covariance when target benign evidence is surprising.
    lambdas = np.exp(np.linspace(np.log(0.03), np.log(30.0), 61))
    nu = 4.0
    shape = rate = nu / 2.0
    log_prior_lambda = (shape * np.log(rate) - gammaln(shape)
                        + (shape - 1.0) * np.log(lambdas) - rate * lambdas)
    prior_cov = (np.sqrt(diagonal)[:, :, None] * corr[None, :, :]
                 * np.sqrt(diagonal)[:, None, :])
    noise_cov = np.eye(WINDOWS)[None, :, :] * obs_var[:, :, None]
    residual = y - peer
    lam_column = lambdas[:, None, None, None]
    scaled_prior = prior_cov[None, :, :, :] / lam_column
    total = scaled_prior + noise_cov[None, :, :, :]
    sign, logdet = np.linalg.slogdet(total)
    residual_column = residual[None, :, :, None]
    solved_residual = np.linalg.solve(total, residual_column)[..., 0]
    quadratic = np.sum(residual[None, :, :] * solved_residual, axis=(1, 2))
    log_evidence = log_prior_lambda - 0.5 * (
        GROUPS * WINDOWS * np.log(2.0 * np.pi)
        + np.sum(logdet, axis=1) + quadratic
    )
    solved_prior = np.linalg.solve(total, scaled_prior)
    gain = np.swapaxes(solved_prior, -1, -2)
    conditional = peer[None, :, :] + np.einsum("lgij,gj->lgi", gain,
                                                residual)
    weights = np.exp(log_evidence - logsumexp(log_evidence))
    robust_multiscale = np.einsum("l,lgw->gw", weights, conditional)
    return {"local": local, "uniform_peer": peer,
            "diagonal_eb": diagonal_eb, "multiscale_eb": multiscale,
            "robust_multiscale_eb": robust_multiscale}


def main() -> None:
    rng = np.random.default_rng(SEED)
    regimes = (
        ("smooth_matched", "smooth", 0.45, 0.0),
        ("common_mode_matched", "common_mode", 0.45, 0.0),
        ("independent_matched", "independent", 0.45, 0.0),
        ("smooth_high_heterogeneity", "smooth", 0.8, 0.0),
        ("smooth_target_shift", "smooth", 0.45, 0.75),
    )
    records = []
    for n in (30, 100, 300, 1000):
        # Vary independent information while keeping nominal rows fixed.
        for ess_fraction in (1.0, 0.5, 0.25):
            effective_n = max(3.0, (n - 1.0) * ess_fraction)
            variance = float(polygamma(1, max((effective_n - 1.0) / 2.0, 1.0)))
            obs_var = np.full((GROUPS, WINDOWS), variance)
            for name, structure, magnitude, target_shift in regimes:
                method_errors = {method: [] for method in
                                 ("local", "uniform_peer", "diagonal_eb", "multiscale_eb",
                                  "robust_multiscale_eb")}
                for _ in range(REPS):
                    covariance = profile_covariance(structure, magnitude)
                    centers = rng.normal(0.0, 0.8, size=(GROUPS, 1))
                    peers = centers[None, :, :] + rng.multivariate_normal(
                        np.zeros(WINDOWS), covariance, size=(PEERS, GROUPS)
                    )
                    # The target is drawn from the peer population, with an
                    # optional coherent out-of-prior contrast across windows.
                    truth = centers + rng.multivariate_normal(
                        np.zeros(WINDOWS), covariance, size=GROUPS
                    )
                    if target_shift:
                        contrast = np.array([1, -1, 1, -1, 1], dtype=float)
                        contrast -= contrast.mean()
                        truth += target_shift * contrast[None, :]
                    y = truth + rng.normal(size=(GROUPS, WINDOWS)) * np.sqrt(obs_var)
                    center, diagonal, corr = estimate_profile_prior(peers)
                    estimates = estimators(y, center, diagonal, corr, obs_var)
                    for method, value in estimates.items():
                        method_errors[method].append(float(np.mean((value - truth) ** 2)))
                for method, errors in method_errors.items():
                    records.append({
                        "support_n": n, "ess_fraction": ess_fraction,
                        "effective_n": effective_n, "regime": name,
                        "method": method, "log_variance_mse": float(np.mean(errors)),
                        "replicate_mse_se": float(np.std(errors, ddof=1) / np.sqrt(REPS)),
                        "replications": REPS,
                    })
            print(n, ess_fraction, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("wrote", len(records), OUT)


if __name__ == "__main__":
    main()
