# FedORBIT roadmap

Protocol version locked before confirmatory execution on 2026-09-27. The previous cross-schema transfer and sparse-QAP protocol is retired; its evidence and reasons are in `docs/audit/` and in git history at commit `22f41e5d`. The repository's scientific contract is `configs/fedorbit.yaml`; this document explains it and must not disagree with it.

## 1. Question and contribution

**Question.** When a new IoT device has only a few benign windows, what does collaboration with other devices actually buy for benign-only anomaly detection; which minimal shared information delivers it; and when does sharing more hurt?

**Mechanism under test.** Local benign-only detectors fail at onboarding chiefly because the local feature normaliser is unstable at small support. A per-feature mean and standard deviation summary from partner devices (2 × 115 numbers per partner for N-BaIoT) repairs most of that failure. Sharing correlation structure or model weights adds little and can harm devices whose benign behaviour is already well characterised or distinctive. Simple local normalisers recover part of the naive cold-start gap, so federation's own contribution is measured net of them.

**Contribution.** An information-channel decomposition of cold-start collaboration with (i) a dose-response over local support, (ii) a net-of-trivial-baseline estimate of federation's value, (iii) an explicit harm regime for structure sharing, and (iv) the gap between deployable and oracle partner selection. It is not a new detector, and the audit did not find a located 2025–2026 study that measures which shared statistic carries the cold-start benefit against local-normaliser baselines. Nearby work proposes methods rather than decompositions: federated PCA/SVD anomaly detection on Grassmann manifolds, Fed-DTCN, similarity-aware aggregation, per-device personalisation. Novelty is moderate and rests on the decomposition and the boundary results, not on a first-of-kind claim.

## 2. Population and client semantics

* **Primary population (N-BaIoT).** Nine physical commercial IoT devices (doorbells, thermostat, baby monitor, security cameras, webcam), each infected with Mirai and BASHLITE in an isolated network. The experimental unit is the device. A device is a client. Claims are scoped to Mirai/Gafgyt botnet detection on these devices.
* **Boundary population (Gotham2025).** Simulated IoT devices from the Gotham testbed. Attack traffic exists on eight devices; after the eligibility rule two cameras are evaluation targets and the remaining devices are benign-only partners. With two targets the boundary experiment is descriptive: no device-level test is inferentially meaningful and none is claimed. Claims are labelled simulated-testbed evidence and are never made about physical devices.
* **Not claimed.** Other attacks, other deployments, privacy guarantees, robust or secure aggregation, deep federated training beyond the secondary autoencoder, or organisation-level federation.

## 3. Data validity rules (hard gates)

1. Chronology: the first two thirds of each device's benign rows (in file order) form the support pool; the last third is the benign test set.
2. Duplicate purge: benign test rows exactly equal to any support-pool row are removed; the number purged is persisted.
3. No test-set use for scaling, thresholds, partner selection or model selection.
4. Attack rows are sampled once per attack file with a derived seed (`attack_rows_per_file` per file) and are used only for evaluation.
5. Gotham windows: 2-second windows over packets with the excluded label removed; a window is benign if it has no attack packet and attack if more than half its packets are attack packets; mixed windows are dropped.
6. Raw files are hashed at preprocessing and recorded in artifact manifests; a change in raw content, configuration or code invalidates downstream evidence.
7. Non-finite values in prepared arrays abort preparation.
8. Evaluation-target eligibility: a device is evaluated only if the worst-case standard error of its AUROC, `sqrt((m + k + 1) / (12 m k))` for `m` purged benign and `k` attack test rows, is at most `maximum_auroc_standard_error`; other devices are recorded as `ineligible-device` with their counts and serve only as partners. This is a statistic-stability rule, not a performance gate. Preprocessing shows all nine N-BaIoT devices qualify, while only two of Gotham's eight attack-observing devices (`ip-camera-museum-1`, `ip-camera-street-1`) do, because most Gotham benign windows are exact duplicates of support windows and the purge leaves 2–17 benign test windows.

## 4. Design

For each evaluation-target device and each support size `n`, a local support window of `n` consecutive benign rows is drawn at a seed-derived random offset in the support pool. Every condition within a replicate uses the same window and the same test sets, so comparisons are paired. Detectors are scored on the full purged benign test set and the sampled attack rows.

**Detectors.** Primary: Gaussian Mahalanobis detector with an OAS-shrunk second-moment matrix, closed form and exactly federable through moment summaries. Secondary: a small autoencoder (reconstruction error) trained locally or by FedAvg. The autoencoder experiments require CUDA and fail if it is unavailable.

**Information channels (conditions).**

| Channel | What is shared | Notes |
| --- | --- | --- |
| local-standardised | nothing | naive local baseline (local mean and standard deviation) |
| local-unscaled | nothing | trivial baseline: no scaling |
| local-standard-deviation-floor | nothing | trivial baseline: standard deviation floored at half the median |
| local-full-support | nothing | reference: the same detector on the entire support pool; defines attainable headroom and the saturation stratum |
| shared-marginals | per-feature mean and variance of partners (equal partner weights) | the hypothesised carrier of the benefit |
| shared-covariance | partner covariance as pseudo-rows (dose 300 or 3000) | tests whether more structure helps or harms |
| federated-averaging | autoencoder weights (secondary) | client rows: target window plus a partner sample |

**Partner selection.** All partners (default); deployable nearest-k (scale-profile similarity computed from the local window); oracle nearest-k (similarity from the whole support pool; upper-bound diagnostic); random-k.

## 5. Endpoints and diagnostics

* **Primary endpoint:** paired difference in AUROC (attack versus benign test scores), averaged over replicates within a device.
* **Operating point (Gaussian experiments):** benign false-positive rate and attack true-positive rate at a threshold set by two-fold cross-fitting on the local window at the configured nominal false-positive rate. Partner selection and shared statistics are computed once from the whole window.
* **Attainable headroom and recovery:** headroom is the device's `local-full-support` AUROC minus the control AUROC; recovery is the treatment's gain divided by headroom, defined only when headroom reaches `minimum_recovery_headroom`.
* **Saturation stratum:** a device is saturated when its `local-full-support` AUROC is at least one minus `saturation_headroom`. Saturated devices are a reported stratum, not failures.
* **Diagnostics (exploratory):** `scale-mismatch` (median log ratio of window to full-pool standard deviations; not deployable), `split-half-instability` (deployable), and the control AUROC. Spearman association with device-level gains is reported with device-cluster bootstrap intervals and is exploratory.

## 6. Statistical design

* **Independent unit:** the device. Replicates are independent random windows within a device and are averaged before any cross-device inference.
* **Population estimate:** mean of device-level gains with a percentile bootstrap over devices (cluster resampling). With nine devices the intervals are wide and are reported as such.
* **Tests:** exact signed-rank test and exact sign test over devices with non-zero gains; zero gains are ties and never evidence. The signed-rank test is the inferential test; the sign test corroborates.
* **Multiplicity:** Holm adjustment across the pre-declared primary family only. Secondary and exploratory contrasts carry no multiplicity-adjusted claims.
* **Replication:** 30 windows per device and support size for Gaussian experiments; 8 for the autoencoder experiment. Pilot within-device replicate standard deviations of the paired gain were about 0.044 at `n=30`, 0.025 at `n=100` and 0.005 at `n=1000` (median across devices), so replicate noise is small compared with between-device differences; device count, not replicates, limits power. With nine devices the smallest attainable one-sided exact signed-rank p-value is 1/512.
* **Infeasible cells:** a device whose support pool is smaller than `n` gets an explicit `infeasible-support` record for that `n` and is excluded from that contrast only. Effects that are small, conditional or null are reported as results, never as insufficient evidence.

## 7. Experiments

Every experiment below is declared in `configs/fedorbit.yaml`; the CLI runs each by name (`fedorbit run "<name>"`).

| Experiment | Dataset / unit | Detector | Support sizes | Replicates | Conditions | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| cold-start-ladder | N-BaIoT / device | Gaussian | 30, 100, 300, 1000 | 30 | 7 (baselines, marginals, covariance 300 and 3000, reference) | primary claims, dose-response, harm regime, operating point |
| partner-selection | N-BaIoT / device | Gaussian | 30, 100, 300 | 30 | 6 (baseline, reference, all, deployable-nearest-2, oracle-nearest-2, random-2) | deployable versus oracle partner selection |
| deep-detector | N-BaIoT / device | autoencoder | 30, 100 | 8 | 4 (baseline, reference, marginals, federated averaging) | secondary: do the channel conclusions survive a deep detector and real FedAvg? |
| simulated-boundary | Gotham2025 / device | Gaussian | 10, 30, 100 | 30 | 6 | boundary: where local detection already saturates, does sharing help or harm? |

**Seeds.** One `base_seed` in the configuration; every stochastic step derives its seed by hashing (base seed, purpose, dataset, device, support size, replicate, condition where relevant). Window offsets, attack sampling, random partner choice, network initialisation, partner sampling for FedAvg and bootstrap resampling are separate purposes. Network initialisation is shared across conditions within a replicate.

**Primary family (Holm, exact signed-rank, one-sided).** At `n=30` on all nine devices: (P1) shared-marginals over local-standardised; (P2) shared-marginals over local-unscaled; (P3) shared-marginals over local-standard-deviation-floor.

**Secondary contrasts (descriptive, no adjustment).** Dose-response over `n=100, 300, 1000` for P1–P3; recovery on unsaturated devices; covariance-300 and covariance-3000 versus marginals (two-sided on all devices and one-sided on the saturated stratum); deployable and oracle nearest-2 versus all partners and versus random; autoencoder marginals over local and FedAvg versus marginals; Gotham marginals versus local, covariance versus local and versus marginals.

**Artifacts.** `outputs/prepared/<dataset>/<device>.npz` (validated arrays), `outputs/records/<experiment>.jsonl` (typed cell and infeasible-support records), `outputs/analysis.json`, and `results/` tables and figures. Each has a manifest with payload digest, provenance (configuration digest, source digest, input digests) and code revision. Completion is derived from manifests: MISSING, STALE, MALFORMED, INCOMPLETE or VALID; a report is written only from VALID evidence.

**Completion criteria.** An experiment is complete when its records artifact is VALID for the current configuration, code and prepared inputs, covers every eligible device, support size, replicate and condition, and records infeasible cells explicitly. The study is complete when the analysis artifact is VALID over all four experiments.

## 8. Gate classification

| Gate | Class | Treatment |
| --- | --- | --- |
| Chronology, duplicate purge, no test-set use, provenance, seeds, pairing, finite values | validity / invariant | strict, fail closed |
| Support pool at least `n` | feasibility | per cell, recorded as `infeasible-support`, never a study failure |
| Attack and benign test rows present, worst-case AUROC standard error within `maximum_auroc_standard_error` | feasibility (statistic stability) | a device that fails is recorded as `ineligible-device` and serves only as a partner |
| Saturation stratum, minimum recovery headroom | descriptive stratification | pre-declared numbers in configuration; used only to define strata |
| Significance thresholds, materiality thresholds | removed as kill switches | Holm-adjusted p-values are reported for the primary family; effects are reported with intervals |
| Class support per device, minimum device counts | removed | heterogeneity is reported |

## 9. Claims

* **Primary claim.** On the nine physical N-BaIoT devices at 30 benign windows, sharing only per-feature partner mean and variance improves benign-only anomaly detection over the naive local baseline and over each of two trivial local normalisers, evaluated as paired AUROC differences over devices. The size of the improvement over the trivial baselines is reported as measured; the pilot indicates it is much smaller than the improvement over the naive baseline.
* **Secondary claims.** The benefit shrinks with local support; adding covariance structure does not help on average and can harm saturated devices; deployable partner selection at small support is unreliable relative to unselected partners while oracle selection is an upper bound; the autoencoder replicates the channel ordering.
* **Boundary (descriptive).** On the two eligible simulated Gotham cameras, where local detection is expected to be saturated, whether sharing marginals is neutral and sharing covariance harmful is reported per device with replicate intervals and no population inference.
* **Limitations.** Nine devices; one botnet family pair; attack rows subsampled per file; Gotham is simulated and contributes two evaluable devices; no adaptive attacker; not a privacy analysis; per-feature moments can leak information about a device's traffic.

## 10. Reviewer objections and answers

* *The naive local baseline is a strawman.* Answered by local-unscaled and standard-deviation-floor baselines and net-of-baseline contrasts.
* *This is just regularisation.* A shuffled-marginal control was run in the audit pilot (partner rows with columns independently permuted matched real rows); the production design isolates marginals versus covariance directly.
* *The Gaussian detector is simplistic.* A federated-averaging autoencoder is a pre-declared secondary experiment.
* *Only nine devices.* Inference is at the device level with exact tests and cluster bootstrap; heterogeneity is a result; claims are scoped.
* *Simulated boundary data.* Labelled as such and used for boundary evidence only.
* *Cold start is artificial for a data set with large captures.* Support windows model a short onboarding capture; results are reported as a function of support size.

## 11. Evidence pointers

Pilot evidence (single-seed-family, exploratory, not confirmatory) is summarised in `docs/audit/POC Results.md`; scientific decisions and the retired candidates are in `docs/audit/Scientific Decisions.md`; progress in `docs/audit/Progress.md`; the closest-work and objection ledger in `docs/audit/Reviewer Audit.md`. The 250-row implementation ledger for the retired protocol is retained as `docs/Audit Matrix.md`.
