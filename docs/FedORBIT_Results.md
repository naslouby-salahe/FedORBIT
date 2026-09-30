# FedORBIT Empirical Results

Manuscript-facing record of the completed cold-start collaboration study. All values are drawn directly from `outputs/analysis.json` and its `results/*.csv` exports; see `docs/FedORBIT_Results_Frozen.zip` for an immutable copy of the underlying evidence.

## Study status

All four registered experiments completed with exact expected-cell coverage and produced a `VALID`, complete analysis (`outputs/analysis.json.manifest.json`: `complete: true`, 53 records, digests over all four experiment inputs):

- `cold-start-ladder` — 6,489 records, 9 N-BaIoT devices
- `partner-selection` — 4,059 records, 9 N-BaIoT devices
- `deep-detector` — 441 records, 9 N-BaIoT devices
- `simulated-boundary` — 908 records, Gotham2025 simulated devices

## Primary result (n=30)

Shared-marginals collaboration versus three local-only baselines, nine N-BaIoT devices, Wilcoxon signed-rank with Holm adjustment across the three primary contrasts:

| Contrast (shared-marginals vs.) | Mean ΔAUROC | 95% CI | Eligible devices | Pos/Tied/Neg | Signed-rank p | Holm-adjusted p |
|---|---:|---|---:|---|---:|---:|
| local-standardised | +0.1034 | [0.0575, 0.1517] | 9 | 9/0/0 | 0.001953 | 0.005859 |
| local-unscaled | +0.0315 | [0.0182, 0.0463] | 9 | 9/0/0 | 0.001953 | 0.005859 |
| local-standard-deviation-floor | +0.0467 | [0.0191, 0.0797] | 9 | 9/0/0 | 0.001953 | 0.005859 |

All three contrasts are Holm-significant with unanimous positive sign across devices. Mean recovery of the gap to the shared-covariance ceiling: 0.880 (vs. local-standardised, 9/9 devices), 0.752 (vs. local-unscaled, 8/9), 0.808 (vs. local-standard-deviation-floor, 8/9).

## Support-calibrated operating-point diagnostic

The Gaussian thresholds use two-fold cross-fitting on each sampled local support window: its first and second halves score one another, and the threshold is the linear empirical 99th percentile of the resulting support scores. This is a nominal threshold target, not a finite-sample calibration guarantee; at support n=30 it is estimated from only 30 cross-fitted scores. The realized false-positive rate on the later benign test partition is not close to 1%; therefore these values must not be presented as performance at an achieved 1% test FPR. The table reports unweighted means across the nine physical devices (30 paired support windows per device):

| Benign support n | Local-standardised test FPR | Local-standardised test TPR | Shared-marginals test FPR | Shared-marginals test TPR |
|---:|---:|---:|---:|---:|
| 30 | 15.88% | 25.97% | 19.54% | 96.91% |
| 100 | 12.56% | 43.53% | 14.61% | 90.37% |
| 300 | 12.16% | 58.12% | 11.48% | 82.01% |
| 1000 | 9.67% | 73.19% | 8.65% | 86.84% |

This is an important threshold-transfer failure, despite the positive AUROC ranking results. The primary inferential claim is about AUROC; the study does not establish a deployable 1% false-alarm operating point. The persisted records contain empirical FPR/TPR; the descriptive FPR intervals below are derived from those records. The available data do not identify how much of the excess FPR comes from the finite support calibration sample versus chronological benign-distribution change, so that cause remains unresolved and is not tuned against the test partition.

To show the uncertainty across target devices, the secondary FPR means also have descriptive 95% percentile cluster-bootstrap intervals. For each condition and support size, the 30 paired support-draw FPRs are averaged within each of the nine devices, then devices are resampled 10,000 times; the seed is `derive_seed(260926, BOOTSTRAP, "operating-point", condition, str(support_size))` as implemented in `src/fedorbit/infrastructure/runtime.py`. The intervals are not calibrated operating guarantees and do not resolve the source of threshold-transfer error:

| Benign support n | Local-standardised test FPR (95% device-bootstrap CI) | Shared-marginals test FPR (95% device-bootstrap CI) |
|---:|---:|---:|
| 30 | 15.88% (9.94–21.94%) | 19.54% (12.79–27.01%) |
| 100 | 12.56% (7.22–19.23%) | 14.61% (9.62–20.44%) |
| 300 | 12.16% (7.50–18.23%) | 11.48% (7.09–17.01%) |
| 1000 | 9.67% (5.20–16.14%) | 8.65% (4.81–14.03%) |

These intervals resample only the nine observed devices. Because the 30 support-draw FPRs are averaged within each device before resampling, the intervals do not describe support-window variability within a device, uncertainty from a broader device population, or separate calibration-sample noise from temporal drift.

## Dose-response

Shared-marginals vs. local-standardised effect shrinks monotonically as local benign support grows, all contrasts unanimous positive sign (9/9 devices) and significant (signed-rank p = 0.001953 at every n):

| n | Mean ΔAUROC | 95% CI |
|---:|---:|---|
| 30 | +0.1034 | [0.0575, 0.1517] |
| 100 | +0.0451 | [0.0191, 0.0765] |
| 300 | +0.0276 | [0.0102, 0.0517] |
| 1000 | +0.0114 | [0.0038, 0.0237] |

Same monotone decay pattern holds for shared-marginals vs. local-unscaled (+0.0205 → +0.0060) and vs. local-standard-deviation-floor (+0.0193 → +0.0060). Interpretation: the collaborative advantage is a cold-start effect — it is largest when local benign evidence is scarcest and decays as local evidence accumulates.

## Covariance

Shared-covariance (partner-size 300 and 3000) vs. shared-marginals, two-sided, all nine devices:

| Contrast | n | Mean ΔAUROC | 95% CI | Pos/Neg | p |
|---|---:|---:|---|---|---:|
| covariance-300 vs marginals | 30 | +0.0147 | [-0.0007, 0.0347] | 5/4 | 0.301 |
| covariance-300 vs marginals | 1000 | +0.0030 | [-0.0006, 0.0072] | 6/3 | 0.301 |
| covariance-3000 vs marginals | 30 | +0.0045 | [-0.0129, 0.0250] | 5/4 | 0.910 |
| covariance-3000 vs marginals | 1000 | +0.0003 | [-0.0052, 0.0062] | 4/5 | 1.000 |

None of the covariance contrasts reach significance and sign is split across devices at every n. The registered unsaturated-device analysis (devices whose local-standardised AUROC has not already saturated) had **zero eligible devices** at every n, because all nine N-BaIoT devices saturate under full local support. No covariance-harm claim is supported — the honest reading is that covariance did not demonstrate a reliable additional advantage over marginals, positive or negative.

## Partner selection

Two-sided contrasts, nine devices:

| Contrast | n | Mean ΔAUROC | 95% CI | Pos/Neg | p |
|---|---:|---:|---|---|---:|
| deployable-nearest-2 vs all | 30 | -0.0205 | [-0.0404, -0.0039] | 2/7 | 0.039 |
| deployable-nearest-2 vs all | 300 | -0.0041 | [-0.0077, -0.0009] | 3/6 | 0.098 |
| deployable-nearest-2 vs random-2 | 30 | -0.0192 | [-0.0375, -0.0034] | 2/7 | 0.055 |
| oracle-nearest-2 vs all | 30 | +0.0126 | [-0.0005, 0.0306] | 7/2 | 0.301 |
| oracle-nearest-2 vs deployable-nearest-2 | 30 | +0.0331 | [0.0038, 0.0690] | 7/1/1 tie | 0.039 |

Deployable nearest-2 selection (chosen without attack labels) is significantly *worse* than using all partners at n=30 and n=300, and no better than random-2 selection. Oracle nearest-2 (selected using held-out information unavailable at deployment) is directionally better than using all partners and significantly better than deployable nearest-2. Interpretation: naive similarity-based partner selection can actively hurt relative to pooling all partners; the oracle-vs-deployable gap is a diagnostic upper bound on what a real selection rule could gain, not a deployable result.

## Deep detector

Autoencoder replication of the Gaussian marginals effect, nine devices:

| Contrast | n | Mean ΔAUROC | 95% CI | Pos/Neg | p |
|---|---:|---:|---|---|---:|
| marginals vs local-standardised | 30 | +0.1014 | [0.0491, 0.1589] | 9/0 | 0.001953 |
| marginals vs local-standardised | 100 | +0.0443 | [0.0150, 0.0774] | 9/0 | 0.001953 |
| FedAvg vs marginals | 30 | +0.0086 | [0.0002, 0.0183] | 7/2 | 0.203 |
| FedAvg vs marginals | 100 | +0.0044 | [-0.0085, 0.0181] | 6/3 | 0.359 |

The marginals-over-local effect replicates under an autoencoder with unanimous sign and magnitude matching the Gaussian detector (+0.101 vs. +0.103 at n=30). FedAvg's incremental gain *over* marginals is small, not unanimous, and not significant at either n — most of the collaborative benefit is already captured by sharing marginal statistics, not by full model federation.

## Gotham (simulated boundary)

Gotham2025 has 8 attack-observing simulated devices; under the eligibility rule (worst-case AUROC standard error ≤ threshold), only 2 are evaluation targets: `ip-camera-museum-1` and `ip-camera-street-1`. The remaining 6 attack-observing devices are ineligible (2–17 purged benign test rows each): `air-quality-1`, `building-monitor-1`, `city-power-1`, `combined-cycle-1`, `combined-cycle-10`, `domotic-monitor-1`.

Local detection is already near-ceiling on both eligible devices (AUROC ≥ 0.998 across all channels and n). Marginals vs. local-standardised delta is +0.0017 at n=10 and exactly 0 (tied) at n=30/100; covariance vs. local and vs. marginals are likewise ~0 or tied. With only 2 eligible devices no device-level inferential test is meaningful. Conclusion is descriptive and neutral: on this simulated boundary population, no channel produced a material measured effect relative to local detection, and no claim is made about the six ineligible devices.

## Paper-level interpretation

**Strongest central claims.** Lightweight shared marginal statistics (mean/variance) produce a large, unanimous, Holm-significant cold-start AUROC improvement over local-only baselines on nine physical N-BaIoT devices at low local support (n=30), and the effect shrinks monotonically and predictably as local benign support grows. The effect replicates under an autoencoder detector with matching magnitude.

**Secondary/supporting findings.** Federated averaging over marginals adds little beyond marginals alone. Oracle-selected nearest partners beat deployable nearest-2 selection, showing partner-selection quality — not just partner availability — matters.

**Null findings.** Shared covariance shows no reliable advantage over marginals (mixed sign, non-significant at every n and partner size); the registered unsaturated-device covariance analysis had no eligible devices because all nine devices saturate under full local support. Deployable nearest-2 partner selection is no better than random and significantly worse than using all partners.

**Claim boundaries.** All physical-device claims are scoped to Mirai/Gafgyt botnet detection on the nine N-BaIoT devices. Gotham2025 evidence is simulated-testbed evidence describing only 2 evaluable devices and is explicitly not extended to the six ineligible devices or to physical deployments.

**Contribution maturity.** The strongest defensible contribution is the measured low-support information-channel decomposition: what peer statistics add for held-out devices with 30–1,000 benign support windows. The implementation and locked within-cohort evidence are mature, while the journal-level scientific contribution remains developing. The study does not establish broad algorithmic novelty, independent physical generalisation, deployment performance, or an achieved 1% test-FPR operating point.

**Novelty boundary.** This is a measured low-support peer-information channel decomposition, not a new detector or a broad first claim to federated IoT anomaly detection. FedGroup (2024) studies functional-group federated parameter aggregation for supervised attack detection on the same UNSW IoT Analytics cohort, with 253 port/byte/packet features and stratified random splits. That is close prior art for federated, group-based IoT attack detection; it does not compare peer marginal, covariance, and model-weight channels under benign-only target onboarding and future chronological testing. The repository’s Project 2 notebook reads flow and annotation CSVs from an external Google Drive path, derives attack labels from `Timestamp` intervals, retains `Timestamp` among predictors, and uses per-device stratified random splits. This creates a direct timestamp label-proxy risk and provides no chronological test. The repository declares AGPL-3.0 for project/software but no separate processed-data license. FedGroup remains method/code prior art, not a verified reproduction ([paper](https://doi.org/10.1007/s10922-023-09782-9); [repository](https://github.com/BasemSuleiman/2023_Anomaly_Detection_IoT); [Project 2 notebook](https://github.com/BasemSuleiman/2023_Anomaly_Detection_IoT/blob/main/Project%202%20Privacy-Aware%20Anomaly%20Detection%20in%20IoT%20Environments%20using%20FedGroup%20A%20Group-Based%20Federated%20Learning%20Approach/%5Bupdate%5DFL.ipynb)).

**Primary limitations.** Nine physical devices, one botnet family pair, attack rows subsampled per file; Gotham contributes only two evaluable devices; no adaptive attacker; not a privacy analysis; per-feature moments can leak device traffic information; covariance and partner-selection contrasts are descriptive/exploratory without multiplicity-adjusted inferential guarantees.
