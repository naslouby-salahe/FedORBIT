# Cold-start collaboration algorithm search — checkpoint (phases 4-10)

This is a checkpoint, not a locked protocol. It records real literature search and real, cheap,
seeded exploratory POCs run against `outputs/prepared/nbaiot/*.npz` (read-only), reusing the repository's
own `fedorbit.detection.gaussian` / `fedorbit.detection.moments` scoring code so results are directly
comparable to the confirmatory study in `docs/FedORBIT_Results.md`. No confirmatory experiment was
rerun. Full POC scripts and raw CSVs are in `docs/audit/pocs/` (gitignored scratch); this file is the
minimal permanent summary of what survived.

## Continuation audit — 2026-09-27

The previous labels remain provisional. Recomputed summaries directly from the five existing CSVs
(540 rows per candidate; 15 replicates × 4 supports × 9 devices) agree with the previous means and
device-sign conclusions. The chi-square gate's mean AUROC delta against `shared-marginals` is
−0.00456/−0.00069/−0.00041/−0.00137 for n=30/100/300/1000; it has positive device-mean deltas on
2/9, 3/9, 3/9, and 5/9 devices respectively. Its worst device-mean deltas are −0.02488, −0.00280,
−0.00177, and −0.01343. Its mean feature borrowing fractions are 76.3%, 75.5%, 72.8%, and 69.3%.
These confirm a close average tie with meaningful negative-transfer cells, not safe equivalence.

**A mathematical assumption of the gate fails for the measured data.** The exact pivot
`(n−1) S² / σ² ~ χ²_(n−1)` requires independent Gaussian observations. A controlled 50,000-replicate
calibration using variance-one Student-t(5) samples instead gives lower-tail rejection rates of
14.08%, 16.60%, 17.83%, and 18.51% at n=30/100/300/1000, against the nominal 5%; the Gaussian control
is 4.80%, 5.11%, 5.06%, and 5.04%. Student-t(5) has finite variance but excess kurtosis 6, which
inflates the leading sample-variance risk from `2 σ⁴/n` under Gaussian sampling to
`(κ+2) σ⁴/n = 8 σ⁴/n`. The data are also visibly non-Gaussian: median absolute feature skew / median
feature excess kurtosis in the full support pools are 3.092/18.253 (Danmini), 0.855/−0.583
(Ecobee), 2.107/5.613 (Ennio), and 2.792/22.541 (Philips). The exact chi-square gate therefore has
no calibrated 5% interpretation on this population. This is a real flaw in the prior argument, not
just a caveat about power. Reproducible synthetic calibration is now in the ignored
`pocs/chi_square_gate_tail_calibration.py`.

**A close prior art materially narrows the novelty claim.** Bouzinis et al., *StatAvg: Mitigating
Data Heterogeneity in Federated Learning for Intrusion Detection Systems*, IEEE TNSM 2025, compute
per-client feature means and variances, aggregate them into weighted global population moments,
broadcast these moments, and normalize participating clients before federated IDS training. They
give the pooled-mean/variance equations and evaluate TON-IoT and CIC-IoT-2023. This is much closer
than the earlier matrix acknowledged. StatAvg is supervised model training and global normalization
for participating clients, whereas this study is benign-only anomaly detection and estimates a
held-out device from its own small support plus peer summaries; it does not settle novelty for that
narrower cold-start question. But “federated IDS has not shared feature moments for normalization”
is false and must be removed from any claim. Exact source: DOI
[10.1109/TNSM.2025.3564387](https://doi.org/10.1109/TNSM.2025.3564387); accessible equations and
algorithm were inspected in the full paper.

The existing POC scripts also derive random seeds from Python's process-randomized `hash()`. Their
conditions were paired within each run, so this does not invalidate those paired comparisons, but
re-running does not reproduce the same windows. Future POCs must use stable explicit seed derivation;
the new robust-scale POC does so with CRC32. No prior POC was rerun just to change seeds.

**Robust-local-scale repair tested and failed.** A paired 15-replicate POC compared local sample SD,
normal-consistent local MAD scale, and fixed `shared-marginals` for all 9 devices and n=30/100/300/1000
(540 rows, stable CRC32 seed derivation; production Mahalanobis scorer). MAD's mean AUROC delta vs.
shared-marginals is −0.2493/−0.2288/−0.2384/−0.2287; it is positive on 0/9 device means at every n
(at most 2/135 individual cells positive), and its worst device mean is −0.6043 at n=1000. It also
loses to local sample SD by −0.1575/−0.1743/−0.2122/−0.2153. Median MAD/local-SD scale ratios are
0.426/0.306/0.302/0.329. Diagnosis: the Gaussian-consistency factor 1.4826 makes MAD a central
spread estimate, not the second-moment scale needed by this Gaussian score under these strongly
skewed benign features; shrinking away natural benign tail traffic damages separation. This rejects
MAD as a drop-in repair, not all robust scale methods. The stable POC and raw CSV are
`pocs/robust_local_scale_poc.py` and `pocs/robust_local_scale_results.csv` (gitignored).

The next design direction is risk measured on held-out benign support rows, feature by feature, rather
than a Gaussian null test or central-spread substitution. A concrete candidate to evaluate is a
blocked cross-validated predictive-risk comparison of local vs peer scale, with local centre retained;
its loss would be held-out Gaussian marginal negative log-likelihood
`L(v; x, μ)=0.5[log(v)+(x−μ)^2/v]`. For a fixed true centre, expected loss is minimized at the target
second moment even when the marginal is non-Gaussian, provided it has finite variance. With a fitted
centre, this becomes prediction-risk calibration and also reflects centre-estimation error. A paired
fold comparison could give local evidence a direct abstention role and allow collaboration to fade
when local scale predicts held-out benign support better. Its high variance at n=30 and target
nonstationarity are unresolved; random folds would leak across adjacent support windows, so blocked
folds are required. This is a hypothesis for a cheap POC, not a selected method or novelty claim.

## Live candidate ledger — no winner selected

| Candidate | Novelty strength | Mean ΔAUROC vs shared-marginals (n=30 → 1000) | Negative transfer | Cold-start behavior | Communication | Theory | Deep compatibility | Main reviewer risk |
|---|---|---:|---|---|---|---|---|---|
| C1 precision-weighted EB mean/log-scale | Low; standard random-effects shrinkage; StatAvg narrows the IDS distinction | −0.0931 → −0.0167 | All 9 device means negative at every n | Local weight 0.93–1.00; barely adapts | Uses peer moments | Valid only under its exchangeable-prior model; AUROC poor | Unchecked | Wrong target estimand / device heterogeneity |
| C2 partner-normalized James–Stein | Low; classical estimator | −0.0851 → −0.0093 | 0/9 positive at n=30; 1/9 at n=1000 | Shrinkage weak under real device differences | Uses peer moments | Classical theorem, but does not match detector utility | Unchecked | Imports partner-centre bias |
| C4 dimension-anchored log-scale blend | Low; conventional shrinkage | −0.0667 → −0.0104 | 1/9 positive at n=30; 0/9 thereafter | Smooth n response but still loses | Partner scale only | Heuristic pseudo-count d does not account for target mismatch | Unchecked | Dimension anchor does not establish risk optimality |
| C5 chi-square variance gate | Low; classical test; StatAvg closest IDS prior | −0.0046 → −0.0014 | 2/9 → 5/9 positive device means; worst −0.0249/−0.0134 | Borrows 76.3% → 69.3% of features | Uses peer scale on gated features | Exact pivot is invalid for heavy-tailed/dependent support | Unchecked | Miscalibrated test; no non-inferiority evidence |
| Local MAD scale | Low; established robust statistic | −0.2493 → −0.2287 | No device mean positive at any n | No collaboration; does not recover tails needed here | None | Normal-consistent central spread, not target second moment under skew | Unchecked | Large loss on all devices |
| Blocked CV predictive-risk scale gate | Not assessed; untested | Not run | Not run | Hypothesized borrowing fades when local predictive risk wins | Hypothesized feature subset only | Proper Gaussian marginal score for second-moment prediction with fixed centre | Unchecked | Validation noise, temporal drift, and overlap with method development |

Current strongest candidate: **none**. The blocked predictive-risk comparison is the next exploratory
experiment, not a chosen algorithm. The protocol remains unlocked, and no confirmatory campaign has
started.

## Bar to beat

`shared-marginals` (per-feature partner mean AND partner standard deviation, local covariance shape),
the confirmatory study's strongest channel, not local-only.

## What was tried and rejected

Two mathematically correct shrinkage estimators — a DerSimonian-Laird-style per-feature
precision-weighted empirical-Bayes blend of local vs. partner-aggregate mean/log-std, and a classical
positive-part James-Stein shrinkage of the local mean in partner-normalised coordinates — both lost to
fixed `shared-marginals` at every tested support size (n=30/100/300/1000), by margins of roughly
0.01-0.09 mean AUROC, unanimous or near-unanimous negative sign. Both are internally correct
applications of real theorems; both fail because their implicit assumption (partners are noisy-but-
unbiased estimates of the same underlying quantity as the target) is false here — the nine physical
devices genuinely differ in absolute traffic scale, so a properly calibrated shrinkage law correctly
infers low trust in the partner reference and barely moves away from the (much worse) local estimate.

## Diagnostic result (new to this study; not a novelty claim)

A four-way ablation (local mean+local scale / partner mean+partner scale / local mean+partner scale /
partner mean+local scale) shows the collaboration benefit is carried almost entirely by the SCALE
(standard deviation), not the mean: at n=30, "local mean + partner scale" recovers 87% of the full
local-to-marginals AUROC gap, while "partner mean + local scale" recovers under 9%. This holds at every
support size. This decomposition was not previously isolated in the registered study, which only ever
compared "share nothing" against "share both mean and variance" as one bundled channel. Mechanistically
this points to small-sample instability of the per-feature variance estimate (115 correlated features,
n as low as 30) as the actual failure mode, not miscentring.

## Former provisional candidate: chi-square-gated variance escalation (under active falsification)

**Mechanism.** Keep the local mean always (never replaced by any partner statistic). For each feature j
independently, test H0: local sample variance is consistent with the partner-aggregate variance, using
the standard one-sided lower-tail chi-square test on `(n-1) * local_var_j / partner_var_j ~ chi2_(n-1)`
at alpha=0.05 (a conventional, non-tuned significance level, not fit to this dataset). If the local
variance is significantly *lower* than the partner reference would predict (the specific failure mode
identified above), escalate: replace that feature's scale with the partner's. Otherwise keep the local
variance. This is a per-feature, per-window, evidence-gated decision, not a global switch or a smooth
sample-size-only blend.

**Result (POC, `docs/audit/pocs/chisq_underestimation_gate_poc.py` / `_results.csv`, 15 seeded replicates
x 4 support sizes x 9 devices).** Mean device-level AUROC delta vs. `shared-marginals`: n=30: -0.0046,
n=100: -0.0007, n=300: -0.0004, n=1000: -0.0014 — a near-zero gap at every support size (compare to
-0.07 to -0.09 for the two rejected shrinkage candidates at n=30). Vs. local-standardised it is
unanimous and large: +0.0976 (9/9 devices) at n=30, decaying to +0.0130 (9/9) at n=1000, matching the
registered study's own local-vs-marginals effect size almost exactly (+0.1034 at n=30 in
`docs/FedORBIT_Results.md`). A few individual device/n cells beat `shared-marginals` outright by a small
amount (e.g. Ennio n=1000 +0.0011, Samsung n=30 +0.0001); one cell is clearly worse
(SimpleHome_XCS7_1002 at n=1000, -0.0134). This is honestly reported as a **statistical tie with fixed
shared-marginals**, not a confirmed win — the POC has no significance test and 15 replicates is not the
registered study's 30 — but it is the only candidate of four tested that came close.

**Prior argument (now rejected as a novelty/safety claim).** It shares strictly less
information than `shared-marginals` (never uses partner mean at all, and only uses partner variance for
the subset of features flagged as locally unreliable — a materially smaller external-information
footprint), it is per-feature and per-window adaptive rather than a device-level channel switch, and its
escalation criterion is a textbook, pre-specifiable hypothesis test rather than a hand-tuned threshold or
partner-similarity heuristic.

## Why the gate is not a final candidate at this point

Besides persistent 69–76% feature borrowing, the gate's null calibration relies on Gaussian
independent rows and fails badly under heavy tails. The candidate is therefore demoted pending a
robust risk estimate or another decision principle. See `docs/audit/pocs/checkpoint_poc_results.md`
for prior negative results. No algorithm or confirmatory protocol is locked.
