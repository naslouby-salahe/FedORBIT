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

**Scale-estimation risk was measured against held-out benign references.** On the same 540 stable
support windows, each per-feature local/peer SD was compared in log space to the device's complete
`support_pool` SD (an empirical finite-pool reference, not an oracle for future operation). Mean
feature log-SD MSE for local vs pooled-peer scales was 19.82 vs 0.66 at n=30, 12.99 vs 0.66 at n=100,
8.86 vs 0.66 at n=300, and 4.55 vs 0.66 at n=1000. Local scales were below the pool reference in
100%, 97.8%, 93.3%, and 91.1% of device/window cells; local downward log-error RMSE was 3.66/2.68/2.09/
1.40, versus peer downward RMSE 0.15 at every n. Peers also overestimated more (upward log-error RMSE
0.66 vs local 0.08–0.10), so borrowing trades a large local downward tail for peer overestimation.
Within each device, lower peer-vs-local log-scale risk tracked larger peer-vs-local AUROC gain by
Spearman ρ=−0.736/−0.558/−0.406/−0.528 at n=30/100/300/1000. This is descriptive only: overlapping
windows share data, and it does not validate a deployable risk rule. Against a separate held-out
`test_benign` scale reference, mean log-SD MSE was local/peer 14.60/5.97, 8.69/5.97, 5.40/5.97, and
4.12/5.97; local underestimation fractions were 72.8%, 65.6%, 57.8%, 59.4%, versus 26.7% for the fixed
peer scales. The held-out reference also shows local wins at n=1000 on average. This suggests the
mechanism is a substantial low-support downward-scale error in many devices, but it does not justify
blanket peer borrowing, nor explain every device: for Ecobee, peer log-SD MSE (0.45) exceeds local
(0.13) at n=30 while peer AUROC gain is only 0.005. Loss and detector ranking remain only partly
aligned. Exploratory diagnostics: `pocs/oas_scale_risk_diagnostic.py/.csv` (support-pool truth) and a
held-out benign comparison from the same stable windows (calculation recorded in this checkpoint).

**Robust partner-scale aggregation and reduced-statistic sharing revisited.** A paired 15-replicate
POC compared pooled total peer SD (the C8 endpoint), coordinatewise median peer SD, one-low/one-high
trimmed peer SD, square root of median peer variance, and geometric mean peer SD. The latter four use
only each partner's per-feature SD (115 scalars/device) and do not require partner means, unlike pooled
total variance which includes between-peer mean variation. Mean deltas vs exact same-window
`shared-marginals` at n=30/100/300/1000 were: pooled −0.00736/−0.00151/−0.00448/+0.00016; median
−0.00956/−0.00238/−0.00397/+0.00088; trimmed −0.00668/−0.00219/−0.00384/+0.00092; median variance
−0.00941/−0.00236/−0.00395/+0.00088; geometric −0.01044/−0.00462/−0.00400/+0.00103. No robust
summary clearly outperforms the pooled target; trimmed SD's worst device-mean delta was
−0.0301/−0.0105/−0.0218/−0.0007, and it had positive device means on 4/9, 3/9, 3/9, 6/9. This gives
a plausible lower-statistic-communication near-tie, not the requested stronger/safe algorithm. The
POC and CSV are gitignored at `pocs/robust_peer_scale_target_poc.py` and
`pocs/robust_peer_scale_target_results.csv`.

**C9 risk-optimal variance blend derived and falsified in its iid plug-in form.** For a scalar feature,
let local unbiased variance estimate `L` have error variance `V_L`; let peer estimate `P` predict the
new target variance with squared error `B_P=E[(P−θ_target)^2]`, independent of `L`. For
`θhat=(1−w)L+wP`,
`MSE(θhat)=(1−w)^2 V_L+w^2 B_P`, so the conditional minimum is
`w*=V_L/(V_L+B_P)`. Under exchangeable device variances with between-device component `τ²`, K peer
estimates, and per-peer sampling variances `u_k`, a peer mean has predictive error
`B_P=τ²(1+1/K)+K⁻²∑u_k`. The local iid plug-in used the fourth-moment formula
`V_L≈[μ4−((n−3)/(n−1))θ²]/n` with sample moments; its peer heterogeneity term used the nonnegative
method-of-moments estimate of `τ²`. This yields a principled continuous feature-wise weight and, when
`B_P>0`, local-only as n grows. It is not a universal safety guarantee: it assumes independent unbiased
variance estimators and exchangeable target/peers, and says nothing directly about AUROC.

The paired 15-replicate POC gave mean peer weights 5.16%/3.08%/1.58%/0.81% at n=30/100/300/1000;
AUROC was effectively local-only (mean deltas vs `shared-marginals` −0.08494/−0.04448/−0.03156/
−0.01116, 0/9 device means positive). The mean-within-peer variance endpoint alone nearly ties the
baseline (−0.00288/−0.00096/−0.00340/+0.00099) but the risk rule hardly uses it. This is a meaningful
failure of the plug-in, not of the MSE minimizer under its assumptions. Root cause evidence: support
rows are explicitly kept chronological by `load_nbaiot_device`/`chronological_split`; median absolute
feature lag-1 autocorrelation across devices is 0.725 (range 0.072–0.853), remaining 0.365 at lag 10.
Across 30 random chronological windows per device/support, empirical between-window variance of the
local sample variance exceeded the iid fourth-moment estimate by median ratios 4.09/13.93/33.29/16.75
at n=30/100/300/1000; 84.7%/83.9%/90.3%/95.2% of feature-device cells exceeded the iid estimate.
So the closed-form iid risk badly underestimates local instability under temporal dependence/regime
variation, causing the near-zero peer weights. A better continuation is a defensible block-risk
estimate, not inflating `w` with a tuned constant. POC: `pocs/random_effects_variance_risk_poc.py` /
`random_effects_variance_risk_results.csv`; dependence calibration:
`pocs/scale_risk_dependence_calibration.py/.json`.

**C10 replaces the iid local-risk plug-in with peer chronological pseudo-window risk.** For each of the
8 leave-target-out peer devices, the method partitions its chronological benign history into complete,
disjoint windows of the target support size n, computes the per-feature sample variance in each block,
and estimates (i) `P_j`, the mean within-window variance across peers, and (ii) `V_L,j`, the mean
within-peer variance of those block variance estimates. It estimates between-device variance `τ²_j`
from peer-level `P_{k,j}` values after subtracting their finite-block sampling noise, then plugs
`B_P,j=τ²_j(1+1/K)+V_P,j` and `w_j=V_L,j/(V_L,j+B_P,j)` into the MSE-optimal linear blend. No attack
labels or hand-tuned cutoff enter the rule; weights emerge feature-wise and decline from 0.533 to 0.297
as n grows from 30 to 1000. This corrects the iid risk estimator with evidence from serially dependent
peer histories, but assumes the target's temporal variance-estimation behavior resembles the peer
histories and does not guarantee lower detector ranking loss.

On the same stable 15-replicate windows, C10 mean AUROC deltas vs exact same-window `shared-marginals`
were −0.00098/−0.00314/−0.00845/−0.00004 at n=30/100/300/1000. Positive device means: 3/9, 2/9, 3/9,
4/9; worst device means: −0.02196/−0.02108/−0.02644/−0.00211; negative-transfer cells: 41/135,
45/135, 74/135, 70/135. It beats local OAS strongly on average, but does not consistently match
shared-marginals on individual devices, particularly at n=300. The unblended peer block-variance
endpoint scores −0.00060/−0.00100/−0.00506/+0.00062 vs shared, so part of the benefit is still the
peer target, not the adaptive risk weight. This is the strongest next candidate to investigate, not a
selected algorithm; it shares mean/variance-risk information at approximately the same feature-vector
count as the baseline and has no safety guarantee. Its mechanism is adjacent to established dependent-
data block bootstrap methods (Zhang & Cheng, 2014), which estimate distributions for high-dimensional
weakly dependent statistics but do not give a small-n transfer/AUROC guarantee ([paper](https://arxiv.org/abs/1406.1037)). Script/raw data:
`pocs/block_risk_variance_blend_poc.py` and `pocs/block_risk_bias_penalty_results.csv`.

**C11 target-discrepancy penalty tested; it does not improve C10.** The moment identity
`E[(L−P)^2]=V_L+B_P+b²` motivates `b²hat=max((L−P)^2−V_L−B_P,0)` and the derived weight
`V_L/(V_L+B_P+b²hat)`, as a feature-wise penalty for target/peer mismatch. It uses no tuned threshold.
However, its mean peer weights fall to 0.485/0.442/0.352/0.265, and AUROC vs shared is slightly worse
than C10 at every n: −0.00110/−0.00449/−0.01004/−0.00028; positive device means stay 3/9,2/9,3/9,4/9,
while negative cells increase to 44/135,48/135,80/135,75/135. This plug-in discrepancy estimate
shrinks peer use but does not find the harmful features/devices well enough to improve detection; it is
not a safe-transfer guarantee. Result rows share the C10 POC file.

**C12 robust peer-block center tested; robustness changes sensitivity but does not solve safe transfer.** A controlled scalar AR(1) POC (ρ=0.7, 8 peers, 500 replicates, 4,000 benign history rows/peer) compared C10 with its realized-MSE oracle and a C12 probe. C12 replaces the mean peer variance with its median and estimates peer spread using a normal-calibrated MAD, corrected for block-estimator noise. With homogeneous peers, C10 was near the oracle and C12 was similar, slightly better at n=30/100 and slightly worse at n=300/1000. With lognormal device-scale heterogeneity (log-SD 0.45), C12 improved over C10 MSE at all n but remained worse than the oracle, especially n≥300 (C12/oracle MSE 1.04 at n=30, 0.87 at n=100, 1.04 at n=300, 1.12 at n=1000). With one persistent 4× peer, C12’s median center resisted contamination far better than C10 (MSE 0.477/0.782/0.889/0.924 vs C10 1.034/1.053/0.968/0.929 and local 1.000/0.942/0.916/0.911), but its MAD risk estimate missed the lone outlier and retained high peer weights; at n=1000 it still slightly harmed local-only. These are scalar variance-estimation MSE results, not detector metrics. A fourth stress case isolates temporal-risk mismatch: peers switch between variance 0.25 and 2.25 with per-row switch probability 0.01 (and AR(1) ρ=0.7), while the target stays in one randomly selected regime for support and evaluation. For target-conditional variance MSE, C10 weights remain 0.983/0.978/0.962/0.899 at n=30/100/300/1000, far above the oracle 0.597/0.541/0.556/0.468; C10 MSE is 1.60/1.58/1.48/1.55 times the oracle. C12 reduces its weights to 0.973/0.943/0.865/0.692 and its MSE ratio to 1.56/1.50/1.29/1.18, but still borrows too much. The failure occurs because peer block instability is treated as target local-estimator risk even when the target is stable in a distinct operating regime. This directly falsifies the key target-peer risk-comparability assumption; C10/C12 need a target-specific compatibility/risk correction before another detector POC. POC: `pocs/block_risk_synthetic_frontier.py/.csv`.

On the paired real N-BaIoT windows (15 replicates × 9 devices × 4 supports), C12 AUROC deltas vs exact same-window shared-marginals were −0.00149/−0.00385/−0.00802/−0.00043 for n=30/100/300/1000; positive device means 4/9, 2/9, 2/9, 4/9; worst device means −0.02293/−0.02146/−0.02813/−0.00445; negative cells 33/135, 50/135, 77/135, 66/135. Relative to C10, C12 is slightly worse at n=30/100/1000 and +0.00043 better on average at n=300, while positive device means fall 3/9→2/9 and worst-device mean worsens −0.02644→−0.02813. Mean peer weights are 0.656/0.600/0.515/0.428, exceeding C10's at every n, so it does not meet the desired automatic fading or safety behavior. Robust medians/geometric aggregation are established methods (including robust federated aggregation and robust meta-analysis); C12 is a sensitivity variant, not a novelty claim. Do not promote it. Real POC rows share `pocs/block_risk_bias_penalty_results.csv`.

**Bounded-mismatch minimax blend derived; estimable safety envelope remains unresolved.** Let local scale-squared estimator L be unbiased for target θ with variance V_L. Let peer estimator P have mean θ+b, variance B_P, and be independent of L. Then R(w)=E[((1−w)L+wP−θ)^2]=(1−w)^2 V_L+w^2(B_P+b^2). If a defensible bound |b|≤Δ is known, the worst-case-risk minimizer is w*=V_L/(V_L+B_P+Δ²), with worst-case risk V_L(B_P+Δ²)/(V_L+B_P+Δ²)≤V_L. The algebra is standard bounded-bias shrinkage, not by itself a FedORBIT novelty or AUROC guarantee. If no finite mismatch bound is justified, every fixed w>0 can have arbitrarily large risk as |b| grows; uniform safe borrowing is impossible without assumptions or target validation. C11's squared-residual plug-in is too noisy to certify Δ. C13 tests an empirical peer-window envelope as a sensitivity rule; its observed maximum is not a valid finite-sample bound for a new client and must not be called safe. Closest prior art now in the novelty matrix: Abba, Williams & Reich (2024) derive Bayesian shrinkage for normal-means/regression with sparse or bounded-norm task differences; their iid homoscedastic model and target mean loss differ from serially dependent per-feature scale estimation and anomaly AUROC, but they preempt generic bounded-difference shrinkage claims ([full equations](https://arxiv.org/html/2403.17321v2)). TRADER (Lai et al., v2 2026) also uses norm-rescaled multiple source estimates, cosine-informed Dirichlet weights, and a target-only horseshoe component, and proves no-worse-than-target-only behavior under sparse high-dimensional regression assumptions. This substantially narrows any adaptive-source-weighting novelty claim; its supervised coefficient loss and iid regression assumptions still differ from unsupervised scale estimation with serially dependent support and OAS-AUROC loss ([full equations](https://arxiv.org/html/2412.02986v2)).

**C13 empirical-envelope minimax and C14 temporal-mixture posterior screened in synthetic MSE only.** C13 sets `Δhat=max_{k,b}|V_{kb}−P|` from peer window variances and uses `w=V_L/(V_L+B_P+Δhat²)`. It nearly collapses to local-only across homogeneous, heterogeneous, and single-outlier cases; under switching-peer/stable-target it trims MSE versus local from 2.50/2.50/2.75/2.28 to 2.31/2.13/2.21/1.81, but does not beat C10 at n=30/300 or C12 at n=30/100/300. The empirical maximum is not a valid finite-sample bound, so the conditional minimax guarantee does not transfer to C13 as implemented.

C14 fits 1–3 Gaussian mixture components by BIC to raw peer block-variance estimates and uses local variance as the likelihood observation, with within-component dispersion as measurement-noise estimate. At 500 Monte Carlo replicates per condition, scalar variance MSE for C14 vs C10/C12/local was: homogeneous 0.744/0.810/0.958 (n=30), 0.810/0.762/0.941 (100), 0.871/0.871/0.999 (300), 0.900/0.902/0.932 (1000); lognormal heterogeneity 1.300/0.820/1.499, 1.071/0.998/1.180, 1.187/1.266/1.444, 1.269/1.303/1.336; one 4× peer 0.725/1.034/1.000, 0.748/1.053/0.942, 0.868/0.968/0.916, 0.913/0.929/0.911; switching peers/stable target 1.523/1.646/2.495, 1.135/2.090/2.505, 2.242/2.165/2.754, 2.483/2.219/2.276. The posterior mixture helps the mismatch case at n≤100 but fails at n≥300, where target-scale modes disappear from peer windows of target length; it also underperforms C10 under ordinary heterogeneity. This is a useful redesign clue, not evidence for Gaussian-mixture novelty: nonparametric empirical Bayes and Bayesian source-mixture weighting are mature, and TRADER already combines weighted sources with target-local fallback under regression assumptions. Raw exploratory artifact `pocs/block_risk_synthetic_frontier.py/.csv`; neither C13 nor C14 has been evaluated on N-BaIoT AUROC, FPR, or OAS scoring.

**C15 multi-scale mixture failed its mechanism check; calibration exposed two noise-model errors.** The POC `pocs/multiscale_peer_regime_mixture_poc.py/.csv` ran 250 replicates per cell, with eight AR(1) peers (ρ=0.7), 4,000 rows per peer, and support n=30/100/300/1000. It chose short blocks of about 18 rows from a raw-series autocorrelation heuristic, corrected short-block log variances with an effective-n chi-square approximation, fit 1–3 Gaussian components, and conditioned a variance posterior on local support. It selected one component in all homogeneous and stable-target/switching-peer cells and 1.00–1.02 in the outlier cells. In the switching-peer/stable-target case C15 MSE was 1.80/1.58/1.72/1.64× realized-oracle MSE at n=30/100/300/1000; C14 was better at n≤300 and worse at n=1000. In lognormal heterogeneity C15 was 1.20–1.23× oracle, and under one 4× peer it was 1.01–1.08× oracle, while C12 remained materially better under that outlier. A separate 30,000-replicate Gaussian AR(1) calibration per rho/block-size cell (`pocs/log_variance_noise_calibration_poc.py/.csv`; rho=.3/.7/.9; block length 10/20/40) showed the effective-n chi-square approximation miscalibrates log-variance noise: at rho=.7, length 20, it gives variance 0.407 versus empirical 0.241; at rho=.9, length 20, 4.204 versus 0.428. It also misestimates log-bias, with the size and direction depending on rho and block length. Separately, the switching-series squared-autocorrelation estimate adds persistent regime occupancy to sampling noise (floor 1.52 in the focused seeded example), further obscuring component structure. On that example a 0.10 GMM covariance floor recovered component means near -0.98/+1.01, while the chi-square-derived 0.47 floor merged them; the 0.10 result is a sensitivity illustration, not a defensible calibration. Simple n_eff substitution and stationary squared-series autocorrelation are unsuitable without calibration. The dependence-selected block size is only a heuristic and does not implement Politis–White's automatic bootstrap block-length estimator; automatic dependence-adaptive block selection is established prior art ([Politis & White, 2004](https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf)). C15 is neither an estimator nor a novelty result. Next model variance-regime persistence separately from within-regime sampling noise, then test calibration and regime recovery together on homogeneous and switching controls before evaluating heterogeneous or contaminated cells. Do not proceed to real N-BaIoT windows unless regime recovery and competitive risk survive those scenarios. No method is selected and the protocol remains unlocked.

**C16 known-state switching diagnostic: correcting the noise floor is not enough; transition blocks need a temporal model.** A 500-replicate known-state AR(1) POC (`pocs/switching_regime_noise_diagnostic_poc.py/.csv`; eight peers, ρ=.7, variance-state flip probability .01, 4,000 rows/peer, block length 18) found 84.3% of peer blocks stayed in one innovation-variance state. Pure-state blocks had log-variance error variance .263 for both low/high states (mean bias about −.38); blocks crossing a state change had error variance .626 and positive mean log error +.356. C15's squared-series heuristic estimated a mean noise floor of 1.624. In a separate 30-replicate mixture diagnostic, fixing observation noise to the calibrated .263 but treating all blocks as exchangeable selected three components with means about −1.10, −.15, +1.07; the middle component (about 7% weight) represented transition observations. If an oracle discarded all mixed blocks, a two-component model recovered means −1.09/+1.11, but the state labels make that result non-deployable. A known-parameter two-state forward-backward HMM improved block-state accuracy only slightly over thresholding the block log variance (0.949 vs 0.942; Brier 0.044 vs 0.058), and assumes the true state means/noise and transition rate. This diagnostic supports separating regime persistence from sampling noise; it does not establish an adaptive estimator. Hidden Markov switching conditional-variance models are established prior art: Rossi & Gallo model variance across discrete levels governed by a latent Markov chain ([2006 paper](https://doi.org/10.1016/j.jempfin.2005.09.003)). Recent variance-changepoint work studies post-selection uncertainty for piecewise-constant Gaussian variance (Carrington & Fearnhead, 2026, [paper](https://link.springer.com/article/10.1007/s11222-026-10881-1)). Next fit/compare a regime model with estimated parameters against a one-regime model under homogeneous and switching data, including false mode selection and feature/device heterogeneity; then quantify whether a target-support posterior improves variance risk over C10/C12/C14/local/oracle. HMM/changepoint structure itself is not a novelty claim. No real-detector POC yet.

**C17 quadratic-form moment matching improves stationary AR noise calibration, but does not solve regime inference.** For Gaussian AR(1) block `x` with covariance `Σ` and centering matrix `M=I−11ᵀ/n`, the sample variance is `s²=xᵀMx/(n−1)`. Exact quadratic-form moments are `E(Q)=tr(MΣ)` and `Var(Q)=2tr(MΣMΣ)` for `Q=xᵀMx`; Satterthwaite matching gives `ν=tr(MΣ)²/tr(MΣMΣ)`. Using the scaled-χ² log moments yields a better log-bias/noise approximation than C15's scalar effective-n substitution across the 9-cell stationary AR grid (`pocs/ar_quadratic_form_noise_approximation_poc.py/.csv`). At rho=.7/block 20, predicted log-noise variance is .298 vs empirical .241 (old effective-n value .407); at rho=.9/block 20, .671 vs .428 (old .4.204). Bias is also close at rho=.7/block20 (−.365 vs −.351 empirical). This is a real calibration improvement under a known stationary Gaussian AR covariance, derived from standard Gaussian quadratic-form moments; it requires rho and still overstates noise for high rho. It cannot be applied to mixed-regime blocks by treating their volatility persistence as sampling error, and it does not by itself make mixture transfer safe. Next compare its estimated-rho behavior under homogeneous AR controls, then combine only with a temporally explicit switching model if mode selection survives homogeneous false-positive controls. No detector metric, target-risk gain, or novelty claim follows yet.

**Dataset generalization audit: no valid second physical-device population identified.** The existing
Gotham2025 artifacts contain 78 simulated assets, but only 8 attack-observing assets; 6 fail the
registered benign-support stability gate, leaving `ip-camera-museum-1` and `ip-camera-street-1`. Both
are already saturated: the exact archived analysis shows AUROC 1.0 for the local and collaborative
channels at n≥30 and zero delta at n=30/100. This is simulation boundary evidence, not physical-device
replication. The local TON-IoT Windows/Linux/network inventory is each a pooled table; the selected
validation inputs have no usable `ts` column, and the schemas do not identify separate physical IoT
devices. Edge-IIoTset is one scenario/capture CSV with IP endpoint fields and 122,782 rows whose
`frame.time` is not timestamp-shaped; treating IPs or attack scenarios as physical client identities
would change the claim and risks endpoint/capture leakage. Available dataset evidence is in
`outputs/preprocessing/inventories/`, `outputs/preprocessing/validation/`, prepared manifests, and
`outputs/analysis.json`; no eligible second population should be fabricated from these sources.

**Deep-detector compatibility has a positive but very small local-center/peer-scale smoke POC.** The
production autoencoder path normalizes support, benign-test, and attack rows with the same `centre` and
`scale`, then trains a local reconstruction model (`src/fedorbit/study/channels.py` and
`detection/autoencoder.py`). Thus a scale-only normalization arm is model-compatible at the interface,
but its learned reconstruction loss is not the Gaussian OAS quadratic and the covariance-shrinkage
identity does not transfer. The registered secondary experiment's shared-marginals-over-local AE gain
was +0.1014 at n=30 and +0.0443 at n=100 (9/9 devices), but it changes both mean and scale. A new paired
smoke POC kept the local center, substituted peer scale only, trained the same local AE on the same
support rows, and used identical attack/benign tests and initialization seeds: at 300 training steps
and 2 replicates/device, peer-scale minus local-normalization mean AUROC was +0.0990/+0.0551 (9/9
positive device means) at n=30/100; peer-scale/local-center minus full shared-marginals was +0.00097/
+0.00107, positive on 4/9 and 5/9 devices. This suggests the scale-only channel can transfer beyond
Gaussian Mahalanobis normalization, but two seeds are only a compatibility smoke, not evidence of AE
non-inferiority or a candidate win. A larger paired AE POC is required before calling it model-agnostic.
The older ignored AE pilot is confounded because it normalizes even its nominal local baseline with a
pooled partner transform; do not cite it. New POC: `pocs/autoencoder_peer_scale_poc.py` /
`pocs/autoencoder_peer_scale_results.csv`.

**Communication/privacy accounting is still architectural, not a measured wire result.** A peer within-
device SD summary contains d=115 scalars (460 bytes/client as float32, before IDs, encryption, or
transport); a mean-plus-variance summary is 2d=230 scalars (920 bytes). For eight peers this is 3.68KB
vs 7.36KB per target round if each peer sends those vectors directly. Pooled total variance needs
first- and second-moment information because between-peer means contribute; robust within-peer SD
aggregators avoid partner means but change the estimand and show only a near tie. The current runner
retains full `MomentSummary` covariance matrices in memory, so no artifact demonstrates real network
bytes, secure aggregation, or privacy. No privacy guarantee is claimed, and marginal statistics can
leak traffic properties.

**Development/confirmation boundary remains open.** The algorithm search has used the same nine
physical device identities and registered support/test artifacts, including attack AUROC feedback.
A new seed family or more windows from these same fixed arrays does not create an untouched device
population. Nested leave-device-out analysis can estimate robustness but cannot be called independent
confirmation after all nine devices have informed this search. The Gotham boundary is saturated and
simulated; the locally inventoried TON-IoT/Edge-IIoTset tables do not supply an eligible physical
client population. Before any confirmatory claim, the strongest valid route is an untouched release or
newly collected physical-device cohort with predeclared support/attack chronology. If none becomes
available, scope the algorithm comparison as exploratory and state that the external confirmation gate
is unmet; do not use the original frozen study as a confirmatory test of a post-hoc candidate.

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
precise mathematical difference. [Full equations, v2](https://arxiv.org/html/2606.30615). Additional
hostile prior art: Yang et al., *Precise High-Dimensional Asymptotics for Quantifying Heterogeneous
Transfers* (JMLR 2025), derive target-risk phase transitions under covariate/model shift and a
rebalanced hard-sharing estimator with minimax rate; this is not OAS or anomaly ranking, but it
preempts generic claims to deriving transfer-safety from heterogeneity alone
([full paper](https://jmlr.org/papers/v26/24-0454.html)). FedKA (ACML 2023) uses federated feature
distribution matching and reports reduced group-effect negative transfer for supervised unseen-domain
classification ([paper](https://proceedings.mlr.press/v189/sun23a.html)). COMMUTE uses ordered multi-site
source subsets plus target-label validation aggregation with a no-worse-than-best guarantee when
validation is sufficient ([paper](https://doi.org/10.1016/j.jbi.2022.104243)); that validation safeguard
is not available under this benign-only target contract. Together with StatAvg, these make general
claims around moments, adaptive transfer, or safety untenable; only a specific estimator/loss/problem
claim remains to be established.

Next: complete the estimated-parameter HMM-vs-one-state POC currently running on paired homogeneous and switching histories; audit its likelihood, model-selection penalty, and false-regime rate. Separately test C17's quadratic-form calibration when rho is estimated rather than known. If a regime model can recover without false modes, estimate local target variance risk from support and the peer regime model, compare with C10/C12/C14/local/realized oracle over n=30/100/300/1000, and add device heterogeneity/outliers. HMM variance models and changepoint inference are established; novelty must come from a distinct target-risk rule, if any. No N-BaIoT AUROC POC unless recovery and competitive risk first survive these scenarios. No candidate is selected, no risk guarantee applies to detector ranking, and the protocol remains unlocked.

## Live candidate ledger — no winner selected

| Candidate | Novelty strength | Mean ΔAUROC vs shared-marginals (n=30 → 1000) | Negative transfer | Cold-start behavior | Communication | Theory | Deep compatibility | Main reviewer risk |
|---|---|---:|---|---|---|---|---|---|
| C1 precision-weighted EB mean/log-scale | Low; standard random-effects shrinkage; StatAvg narrows the IDS distinction | −0.0931 → −0.0167 | All 9 device means negative at every n | Local weight 0.93–1.00; barely adapts | Uses peer moments | Valid only under its exchangeable-prior model; AUROC poor | Unchecked | Wrong target estimand / device heterogeneity |
| C2 partner-normalized James–Stein | Low; classical estimator | −0.0851 → −0.0093 | 0/9 positive at n=30; 1/9 at n=1000 | Shrinkage weak under real device differences | Uses peer moments | Classical theorem, but does not match detector utility | Unchecked | Imports partner-centre bias |
| C4 dimension-anchored log-scale blend | Low; conventional shrinkage | −0.0667 → −0.0104 | 1/9 positive at n=30; 0/9 thereafter | Smooth n response but still loses | Partner scale only | Heuristic pseudo-count d does not account for target mismatch | Unchecked | Dimension anchor does not establish risk optimality |
| C5 chi-square variance gate | Low; classical test; StatAvg closest IDS prior | −0.0046 → −0.0014 | 2/9 → 5/9 positive device means; worst −0.0249/−0.0134 | Borrows 76.3% → 69.3% of features | Uses peer scale on gated features | Exact pivot is invalid for heavy-tailed/dependent support | Unchecked | Miscalibrated test; no non-inferiority evidence |
| Local MAD scale | Low; established robust statistic | −0.2493 → −0.2287 | No device mean positive at any n | No collaboration; does not recover tails needed here | None | Normal-consistent central spread, not target second moment under skew | Unchecked | Large loss on all devices |
| Blocked CV predictive-risk scale gate | Low; standard validation/transfer-risk principle | −0.0993 → −0.0125 | 0/9 device means positive at every n | Borrows 14–20% of features but nearly reverts to local-only | Low peer-scale use; dominated by zero-communication local-only at similar AUROC | Proper marginal score, but does not target AUROC under OAS | Unchecked | Loss/metric mismatch and negative device cells |
| OAS peer diagonal-target mechanism | Narrow/unassessed; OASD/multi-target covariance shrinkage plus 2026 safe multi-source shrinkage are close prior art | Full peer production −0.0074 → +0.0002; fixed-local-intensity target arm −0.0049 → +0.0012 vs exact shared | + on 9/9 vs local; device-level negative cells remain; Gaussian mismatch SD 0.5 reverses the n=30 gain | Peer target benefit decreases with n; peer intensity contributes little | Pooled total SD uses peer mean+variance summaries; robust peer-SD variants use 115 values/device but only tie | Target effect isolated empirically; no new general shrinkage theory | OAS mechanism is Gaussian-specific; scale-only AE smoke POC nearly ties shared, 2 seeds only | Could be existing external-target covariance shrinkage applied to IDS |
| C9 iid risk-optimal local/peer variance blend | Standard random-effects/MSE shrinkage; no new general estimator claim | −0.0849 → −0.0112 vs shared | No device mean positive; near local-only | Peer weight 5.2% → 0.8%, naturally fades but starts too low | Local fourth moment + peer per-feature variance estimates | MSE-optimal only under independent unbiased estimates; observed window risk is 4–33× iid plug-in due chronology | Scale-only idea could feed AE; not tested yet | Temporal dependence breaks risk estimate; AUROC loss mismatch |
| C10 peer-block variance-risk blend | Random-effects partial pooling with dependent-window risk proxy; bootstrap literature is adjacent | −0.0010 → −0.00004 vs shared; n=300 −0.00845 | Negative cells 30–55% by support; worst device −0.0264 | Peer weight 53% → 30% on real data, but 98% → 90% under regime-risk mismatch | Peer block means/risks, about 2d/device | Conditional linear-MSE derivation; synthetic mismatch MSE 1.48–1.60× oracle | Scale-only interface could feed AE; C10 itself untested | Target-peer temporal comparability fails under distinct stable target regime; no safe bound |
| C11 C10 with moment-estimated target mismatch penalty | Plug-in excess-discrepancy penalty; standard risk estimation | Slightly below C10 at every n | Negative-cell fraction rises vs C10 | Weight 49% → 27%, but does not protect devices | No additional peer payload | Identity-based estimate of squared bias; noisy at feature level | Not evaluated on AE | Penalizes peer use without finding harmful cells |
| C12 median/MAD peer-block risk blend | Standard robust aggregation; close robust meta-analysis and geometric-median FL prior art | −0.0015 → −0.0004 vs shared; slightly better than C10 only at n=300 | Negative cells 24–57% by support; worse worst device than C10 at n=300 | Weight 66% → 43%, above C10 at every n | Same approximate 2d summaries | Resists one 4× peer in scalar MSE, but 1.18–1.56× oracle under regime mismatch | Not evaluated on AE | MAD misses lone-source risk; not safe under target/peer temporal mismatch |
| C13 empirical-envelope minimax blend | Standard bounded-bias/minimax shrinkage; Abba et al. 2024 and TRADER are close | Synthetic only; mostly reverts to local; no AUROC result | Reduces regime-mismatch MSE but does not beat C10/C12 consistently | Peer weight overly conservative under several scenarios | Same 2d plus peer-block envelope summary | Exact minimax rule if bias bound is true; observed maximum is not a coverage-valid bound | Not evaluated on AE | Safety claim unsupported; performance close to local-only |
| C14 peer-window mixture posterior | Empirical Bayes mixture / source-weighted transfer; TRADER and normal-means mixture EB are close | Synthetic only; helps switching regime at n≤100, reverses at n≥300; no AUROC result | No reliable across-scenario advantage | Can target local regime when peer windows preserve it; loses modes as target-length blocks average regimes | Per-feature block distribution / mixture parameters; more than 2d unless components compressed | Posterior optimal only under fitted Gaussian mixture and variance-noise model | Not evaluated on AE | GMM misspecification, component instability, established EB novelty |
| C15 dependence-heuristic short-block mixture posterior | Empirical Bayes plus dependence-adjusted variance noise; automatic block selection is established bootstrap prior art | Synthetic only; one component selected in all 16 cells; C15/oracle MSE 1.01–1.23× across homogeneous/heterogeneous/outlier and 1.58–1.80× under switching-peer mismatch; no AUROC result | Does not recover switching modes; near-local behavior under heterogeneity/outliers | About 18-row blocks across supports; noise floor absorbs regime persistence | Many peer block summaries per feature; larger than 2d unless compressed | Noise-floor approximation failed because volatility persistence was counted as within-regime sampling noise | Not evaluated on AE | Regime recovery absent; empirical-Bayes and block-length novelty claims preempted |

No candidate currently clears both paired detector evidence and the controlled mismatch/contamination suite. C10 remains the strongest real-window lead (near shared on average) but fails target-specific regime-risk comparability; C12 improves some contamination diagnostics but has no stable detector advantage. C14's limited low-support mismatch gain is not a deployable result, and C15 failed to preserve its regime-recovery mechanism after the noise floor accounted for serial dependence. None is selected as the final algorithm. C8 remains the clearest Gaussian mechanism result. Robust peer-SD
sharing is a lower-communication near-tie, and local-center/peer-scale has a two-seed AE smoke result.
No method consistently beats shared-marginals or has a negative-transfer guarantee. The
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
