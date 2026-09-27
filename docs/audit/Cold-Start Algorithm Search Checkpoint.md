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

**Blocked predictive-risk scale selection tested and failed against the real bar.** The candidate kept
the local mean, chose local vs peer scale per feature by comparing the Gaussian marginal negative
log-likelihood `L(v;x,μ)=0.5[log(v)+(x−μ)^2/v]` on the final third of each contiguous support window,
and refit the chosen scale on all n rows. It used a fixed true-support time split, stable CRC32 seeds,
15 paired replicates, 9 devices, and the production Mahalanobis scorer. Mean ΔAUROC vs.
`shared-marginals` was −0.0993/−0.0564/−0.0284/−0.0125 for n=30/100/300/1000; no device-mean delta
was positive at any support size, and the worst device means were −0.2002/−0.2025/−0.1573/−0.0667.
It borrowed only 19.9%/15.2%/18.6%/14.4% of features. Against local-standardised, mean delta was
+0.0017/−0.0011/+0.0004/+0.0002, with no consistent device benefit. Thus this selector nearly
reverts to local-only and is dominated by local-only on the communication axis; it does not explain
the fixed-peer-scale AUROC gain. The normal score is proper for second-moment prediction with a fixed
centre, but that estimation loss is not a surrogate for attack/benign AUROC under OAS regularization.
Failure points to a more basic mechanism question: what does “scale” control in the actual scorer?

**New mathematical mechanism hypothesis: the scale chooses the OAS diagonal shrinkage target.** Let
`S=(X−μ)^T(X−μ)/n` in raw feature units, `D=diag(s_1,…,s_d)` be the chosen normalizer, and
`C=D⁻¹ S D⁻¹` be the standardized second moment. The production scorer shrinks
`C_OAS=(1−λ)C + λ tr(C)/d I`. In raw units this is exactly
`S_OAS(D) = D C_OAS D = (1−λ)S + λ [tr(D⁻¹ S D⁻¹)/d] D²`.
Therefore, when λ>0, choosing local vs peer scales changes an anisotropic diagonal regularization
target (and also changes λ); it is not just replacing a noisy estimate of the target's marginal
variance. If λ=0, the raw Mahalanobis score is invariant to D. This directly links the observed scale
effect to covariance regularization under d=115 and n as low as 30. I independently reconstructed
the raw-space OAS covariance for Danmini n=30 and reproduced production scores to max relative errors
3.5e−15 (local D) and 1.1e−14 (peer D). This is an exact algebraic identity and targeted numerical
check, not yet evidence that the diagonal-target change causes the AUROC gain.

**Factorial OAS target/intensity mechanism POC completed and paired to the exact baseline.** The
factorial used stable CRC32 windows, 15 replicates × 4 supports × 9 devices (540 cells), local means,
raw sample second moments, and the production Mahalanobis scorer. A separate run recomputed
`shared-marginals` on the exact same windows; the peer-production endpoint reproduces its
local-mean/peer-scale score to AUROC error ≤3.34e−16. Holding local OAS intensity and trace factor
fixed while changing only the diagonal target from local to peer marginals gives mean AUROC gains
against local production of +0.0799/+0.0451/+0.0273/+0.0124 at n=30/100/300/1000, positive on all
9 device means at every n. Changing only OAS intensity while holding the local target fixed changes
mean AUROC by +0.00012/−0.00011/−0.00034/−0.00014 against local production. Thus the diagonal target,
not the OAS intensity, drives the measured improvement over local-only.

The comparison to the registered bar is much tighter. Full peer-scale production is −0.00736/−0.00151/
−0.00448/+0.00016 mean AUROC against exact same-window `shared-marginals` for n=30/100/300/1000;
only 4/9 device means are positive at n=30, 2/9 at n=100 and n=300, and 4/9 at n=1000. The peer
diagonal target with local intensity and trace factor is −0.00490/+0.00065/−0.00468/+0.00121 against
shared; positive device means are 5/9, 5/9, 3/9, and 6/9. Worst device-mean deltas for that arm are
−0.03234/−0.00372/−0.02343/−0.00121. It nearly matches the bar on average but is not uniformly safe
and has no material, consistent advantage over the stronger two-summary baseline. Production local
OAS trails shared by −0.08484/−0.04450/−0.03196/−0.01117. Across all 540 cells, reconstructed
raw-space covariance scores match production with max relative error 8.11e−11. This supports a
mechanism claim against local-only; it does not select C8 over shared-marginals or establish a new
estimator. Raw rows are in the gitignored `pocs/oas_scale_target_decomposition.csv` and
`pocs/oas_same_window_shared_baseline.csv`.

**Controlled synthetic partner-mismatch stress test completed.** A separate 600-cell Gaussian simulation
(80 features, 30 paired replicates, n=30/100/300/1000) held the target covariance fixed and varied
feature-wise peer log-scale error `δ_j ~ N(0, q²)`, with q=0, 0.1, 0.25, 0.5, 1.0. Peer-target
OAS uses local support mean and peer diagonal target; local OAS is the comparator. Even with a perfect
peer diagonal (q=0), the average AUROC gain was small (+0.0068 at n=30, +0.0032 at n=100, and around
zero for n≥300), with mixed replicate signs. At q=0.5, mean deltas were −0.0093/−0.0094/−0.0008/−0.0020;
at q=1.0 they were −0.0214/−0.0158/−0.0146/−0.0094. Thus target mismatch can reverse the modest
small-sample benefit, and its harm grows with mismatch. This simulation is intentionally stylized and
cannot establish a real-data threshold or support safe deployment; it reinforces that C8 has no
negative-transfer safeguard. Script and CSV are gitignored at
`pocs/oas_peer_target_mismatch_synthetic.py` and
`pocs/oas_peer_target_mismatch_synthetic.csv`.

**Novelty became more constrained after tracing the estimator family.** The OASD paper (IMF WP
23/257, 2023) already derives Oracle Approximating Shrinkage toward `diag(S)` rather than OAS's
scaled identity and targets high-dimensional inverse-covariance quality for `p>n`. Gray et al.'s
multi-target shrinkage estimator explicitly uses several covariance target matrices, with Bayesian
weights and multiple external-source targets; Oriol (2024) derives a general multi-target linear
shrinkage estimator. FedORBIT's possible difference is replacing `diag(S)` with a held-out peer
population's diagonal target inside a benign-only anomaly detector on a new device. This is a narrow
application/setup distinction, not a new general shrinkage estimator, and needs a direct novelty
comparison. Primary sources: [OASD, IMF WP/23/257](https://www.imf.org/-/media/Files/Publications/WP/2023/English/wpiea2023257-print-pdf.ashx),
[multi-target shrinkage (Gray et al.)](https://arxiv.org/abs/1809.08024), and
[multi-target linear shrinkage (Oriol, 2024)](https://arxiv.org/abs/2405.20086). A just-updated
2026 preprint (Jing et al., arXiv:2606.30615v2, 2026-09-21) is directly relevant to the safe-transfer
agenda: it shrinks a target estimator toward source-specific estimates using estimator covariance,
selects step sizes within a finite-sample risk-improving interval, and adds sources sequentially by
priority rather than pooling them first. It extends the approach to smooth M-estimation. Its exact
setting is not feature-scale OAS for anomaly detection, but it preempts a broad claim to new,
tuning-free, risk-safe, covariance-aware, multi-source collaboration. Any next source-weighting rule
must either instantiate its theory in this detector with a defensible OAS/AUROC loss, or explain a
precise mathematical difference. [Full equations, v2](https://arxiv.org/html/2606.30615).

Next: derive and test a covariance-risk or ranking-risk rule that can detect mismatched peer diagonal
targets before borrowing; evaluate a principled mismatch detector against shared-marginals under real hostile partner subsets,
support dependence, heavy tails, and additional covariance structures; the toy Gaussian stress test
shows a clear reason not to treat a fixed peer target as safe. The candidate remains
unselected; statistical variance-estimation risk and detector ranking loss must not be conflated.

## Live candidate ledger — no winner selected

| Candidate | Novelty strength | Mean ΔAUROC vs shared-marginals (n=30 → 1000) | Negative transfer | Cold-start behavior | Communication | Theory | Deep compatibility | Main reviewer risk |
|---|---|---:|---|---|---|---|---|---|
| C1 precision-weighted EB mean/log-scale | Low; standard random-effects shrinkage; StatAvg narrows the IDS distinction | −0.0931 → −0.0167 | All 9 device means negative at every n | Local weight 0.93–1.00; barely adapts | Uses peer moments | Valid only under its exchangeable-prior model; AUROC poor | Unchecked | Wrong target estimand / device heterogeneity |
| C2 partner-normalized James–Stein | Low; classical estimator | −0.0851 → −0.0093 | 0/9 positive at n=30; 1/9 at n=1000 | Shrinkage weak under real device differences | Uses peer moments | Classical theorem, but does not match detector utility | Unchecked | Imports partner-centre bias |
| C4 dimension-anchored log-scale blend | Low; conventional shrinkage | −0.0667 → −0.0104 | 1/9 positive at n=30; 0/9 thereafter | Smooth n response but still loses | Partner scale only | Heuristic pseudo-count d does not account for target mismatch | Unchecked | Dimension anchor does not establish risk optimality |
| C5 chi-square variance gate | Low; classical test; StatAvg closest IDS prior | −0.0046 → −0.0014 | 2/9 → 5/9 positive device means; worst −0.0249/−0.0134 | Borrows 76.3% → 69.3% of features | Uses peer scale on gated features | Exact pivot is invalid for heavy-tailed/dependent support | Unchecked | Miscalibrated test; no non-inferiority evidence |
| Local MAD scale | Low; established robust statistic | −0.2493 → −0.2287 | No device mean positive at any n | No collaboration; does not recover tails needed here | None | Normal-consistent central spread, not target second moment under skew | Unchecked | Large loss on all devices |
| Blocked CV predictive-risk scale gate | Low; standard validation/transfer-risk principle | −0.0993 → −0.0125 | 0/9 device means positive at every n | Borrows 14–20% of features but nearly reverts to local-only | Low peer-scale use; dominated by zero-communication local-only at similar AUROC | Proper marginal score, but does not target AUROC under OAS | Unchecked | Loss/metric mismatch and negative device cells |
| OAS peer diagonal-target mechanism | Narrow/unassessed; OASD/multi-target covariance shrinkage plus 2026 safe multi-source shrinkage are close prior art | Full peer production −0.0074 → +0.0002; fixed-local-intensity target arm −0.0049 → +0.0012 vs exact shared | + on 9/9 vs local; device-level negative cells remain; Gaussian mismatch SD 0.5 reverses the n=30 gain | Peer target benefit decreases with n; peer intensity contributes little | Requires peer marginal scales | Target effect isolated empirically; no new general shrinkage theory | Model-specific; deep detector untested | Could be existing external-target covariance shrinkage applied to IDS |

Current strongest candidate: **none selected**. The peer diagonal-target mechanism is the leading
mechanistic hypothesis, pending safe-mismatch tests and hostile prior-art audit; it nearly ties but does not beat shared-marginals consistently. The
protocol remains unlocked, and no confirmatory campaign has started.

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

**Prior argument (historical, now rejected as a novelty/safety claim).** The gate used less
information than `shared-marginals` and was per-feature/per-window adaptive, but its classical test is
not a novelty and its Gaussian calibration fails on heavy-tailed support data. These properties do not
rescue it as a safe candidate.

## Why the gate is not a final candidate at this point

Besides persistent 69–76% feature borrowing, the gate's null calibration relies on Gaussian
independent rows and fails badly under heavy tails. The candidate is therefore demoted pending a
robust risk estimate or another decision principle. See `docs/audit/pocs/checkpoint_poc_results.md`
for prior negative results. No algorithm or confirmatory protocol is locked.
