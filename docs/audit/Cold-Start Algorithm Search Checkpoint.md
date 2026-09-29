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

**C17 quadratic-form moment matching improves stationary AR noise calibration, but does not solve regime inference.** For Gaussian AR(1) block `x` with covariance `Σ` and centering matrix `M=I−11ᵀ/n`, the sample variance is `s²=xᵀMx/(n−1)`. Exact quadratic-form moments are `E(Q)=tr(MΣ)` and `Var(Q)=2tr(MΣMΣ)` for `Q=xᵀMx`; Satterthwaite matching gives `ν=tr(MΣ)²/tr(MΣMΣ)`. Using scaled-χ² log moments yields a better log-bias/noise approximation than C15's scalar effective-n substitution across the nine stationary AR cells (`pocs/ar_quadratic_form_noise_approximation_poc.py/.csv`). At rho=.7/block20, predicted log-noise variance is .298 vs empirical .241 (old effective-n value .407); at rho=.9/block20, .671 vs .428 (old 4.204). Bias at rho=.7/block20 is −.365 vs −.351 empirical. C18 (`pocs/ar_quadratic_form_estimated_rho_poc.py/.csv`) pooled lag-one estimates from eight 4,000-row peers across 500 runs; mean estimated rho differed from truth by <.0005 (SD .0054/.0042/.0024 at rho=.3/.7/.9). Plugging estimated rho barely changed predictions, so rho estimation error is not the main remaining calibration problem: C17 still overstates high-rho log-noise. This is a calibration improvement under a known stationary Gaussian AR covariance, derived from standard Gaussian quadratic-form moments; it still requires an AR model and does not handle switching blocks. It cannot make mixture transfer safe or address transition blocks. Next improve the finite-sample log-quadratic-form calibration beyond Satterthwaite, and combine only with a temporally explicit switching model if mode selection survives homogeneous false-positive controls. No detector metric, target-risk gain, or novelty claim follows yet.

**C19 lognormal moment matching substantially improves the stationary Gaussian calibration.** Given exact `m=E(s²)` and `c²=Var(s²)/m²` from the same quadratic-form traces, approximate `log(s²)` by a lognormal moment match: `Var[log(s²)]≈log(1+c²)`, `E[log(s²)]≈log(m)−0.5log(1+c²)`. Against the 30,000-replicate stationary AR calibration on the same ρ=.3/.7/.9 and block-length 10/20/40 grid (`pocs/ar_lognormal_noise_approximation_poc.py/.csv`), mean absolute log-noise-variance error is .0203, versus 1.046 for C15's effective-n chi-square approximation; maximum error is .0532 over the nine cells. At ρ=.9/block20 the prediction is .416 vs .428 empirical (C17 scaled-χ² prediction .671, C15 .4.204); at ρ=.7/block20 it is .232 vs .241. Mean absolute log-bias error is .0084. This is a useful calibrated likelihood approximation for stationary Gaussian AR blocks, not a new distribution approximation: moment matching for Gaussian quadratic forms is established (e.g. Zhang, Shen & Wu, [arXiv:2005.00905](https://arxiv.org/abs/2005.00905)), and their higher-moment methods seek still better distribution/tail accuracy. The grid is small, assumes known AR(1) covariance, and says nothing about transition blocks, heavy tails, or transfer safety. Test the C19 likelihood inside regime inference and evaluate model selection under homogeneous data before treating it as an estimator component.

**C20 oracle two-regime posterior is a mechanism upper bound and a severe mismatch counterexample.** On the current paired 500-replicate scalar AR POC (`pocs/switching_oracle_posterior_risk_poc.py/.csv`; ρ=.7, 8 peers, 4,000 history rows/peer, n=30/100/300/1000), peer innovations switched between .25 and 2.25 with per-row probability .01. An oracle knew the corresponding marginal-variance prior exactly (θ=.490/4.412, prior mass .5 each) and used the C19 log-variance likelihood for local support. When the target truth was drawn from one of those peer modes, posterior-mean variance MSE was .014 at n=30 and numerically zero at n≥100, versus local sample-variance MSE 1.587/.621/.243/.069 and realized shared-peer marginal MSE 4.137/3.909/3.859/3.845. This is deliberately optimistic: regime levels/probabilities are given, and no detector metric is involved. Under the adjacent stable target truth θ=1.961, absent from the peer prior, posterior MSE rose 4.125/5.305/5.864/6.007, while local MSE was .550/.188/.080/.021 and the finite-history shared-peer estimator gave .031/.158/.221/.247. Thus an exact likelihood and correct within-peer regime model do not protect an unseen target outside source support; forced empirical-Bayes mode selection has severe negative transfer. C21 adds oracle predictive compatibility to this same design, using a separate prior-predictive reference Monte Carlo; that changes the RNG stream, so its samples are a matched resimulation, not the same replicate draws as these C20 numbers. This is not a candidate: it shows both the potential value of a valid target-compatible prior and the need for target-only abstention/model uncertainty. Attack prior art such as TRADER's target-only component; do not claim no-harm from Bayes risk under a correctly specified prior.

**C21 oracle predictive-compatibility gate reduces, but does not remove, negative transfer.** On a matched resimulation of the C20 design, an oracle prior-predictive score `−log p(log s² | two peer modes, C19 likelihood)` was calibrated under the exact matched two-mode model. At nominal α=.05, the fallback-to-local rate on matched targets was 3.6/5.6/6.0/3.8% across n=30/100/300/1000. Under the outside-mode θ=1.961 target, rejection rates were .438/.938/1/1 and gated variance MSE was 3.336/.519/.080/.021 versus local MSE .550/.188/.080/.021; this is much safer by n≥300 but still harmful at n=30 and n=100. Sweeping α exposed the decision tradeoff: at n=30 even α=.5 left outside-target MSE .903, above local .550, while matched-target MSE rose from .014 (no fallback) to 1.302; only full fallback (α=1) restored the local risk. At n=100, α=.3 nearly matched local outside risk (.197 vs .188) while matched MSE was .465 vs .621 local, but this is an oracle-tuned operating point, not a derived rule. The check is a goodness-of-fit test: non-rejection does not certify that borrowed estimation risk is lower. General conformal exchangeability testing for transfer decisions is prior art (Zhou et al., 2017, DOI [10.1016/j.patrec.2016.12.021](https://doi.org/10.1016/j.patrec.2016.12.021)); their validated null asks whether source/target data are exchangeable and uses a conformal p-value to decide transfer. The current gate has even stronger oracle assumptions and still does not ensure no harm. Raw threshold frontier is `pocs/switching_predictive_gate_frontier.csv`; the paired script/result are the C20 artifacts. Next replace the arbitrary predictive-test threshold with an estimated decision-risk/robust-risk comparison, and evaluate by matched, intermediate, and novel target regimes. A feature-level version would also need multiplicity/calibration under serial dependence. Do not promote this gate as a safe algorithm.

**C22 target-only holdout risk selection remains noisy and can still transfer harmfully.** An ignored scalar POC (`pocs/target_only_holdout_risk_poc.py/.csv`) split n contiguous AR(1) support rows in half, computed local sample variance and the oracle C20 posterior on the first half, then selected whichever estimate was closer in squared error to the second-half sample variance. Across 2,000 replicates at ρ=.7, this direct holdout-loss rule often recognized an intermediate target (peer selection .263/.081/.023/0 for n=30/100/300/1000) and cut its posterior harm, but selected-target MSE remained .962/.549/.184/.042 versus first-half local .931/.401/.152/.042. For a novel higher variance θ=7.84, peer selection persisted at .637/.401/.163/.035 and selected MSE was 11.62/7.33/3.52/1.06, worse than first-half local 15.53/6.24/2.23/.71 at n=100–1000. The lower novel variance mostly abstained (.051→0 peer selection) but had MSE .0129/.0057/.0016/.00047 versus first-half local .0090/.0038/.0014/.00047. On matched modes, selection helped over the first-half local estimator but lost much of the oracle posterior gain (e.g., high mode n=30 MSE 2.08 vs posterior .23). Thus target-only holdout loss is more decision-relevant than compatibility, but does not certify no-harm; small held-out support is noisy, costs half the estimation support, and contiguous train/validation blocks remain serially dependent. This is a screening POC with known source modes, not a deployable feature-level rule. Raw artifact is gitignored; no candidate is selected.

**C23 corrects the validation variance's AR bias; the holdout selector still fails the risk gate.** Review of the C22 implementation noted that sample variance with denominator `n−1` is biased for the stationary AR marginal variance after centering. A paired 2,000-replicate follow-up (`pocs/target_only_unbiased_holdout_risk_poc.py/.csv`) compares raw and exact-factor-corrected local variance, scores decisions against a later block whose variance is corrected by the same known-ρ factor, and retains the oracle two-mode posterior. At ρ=.7, correcting the local estimate slightly *increased* its MSE in each tested condition (e.g. intermediate n=100: .436 vs .391 raw; novel-high n=100: 7.15 vs 6.47), so the raw local estimator remains the stronger comparator under squared error despite its downward bias. Correcting the validation proxy lowers but does not remove harmful selections: on novel-high θ=7.84, peer selection was .536/.334/.159/.029 for n=30/100/300/1000, and selected MSE was 11.32/6.80/3.64/.96 versus raw-local 14.93/6.47/2.46/.69. For the intermediate target, corrected-validation selection had MSE 1.32/.66/.195/.045 vs raw-local .971/.391/.149/.043; it avoids most of the catastrophic posterior risk but still loses to local at n≤300. On matched modes it improves over local (e.g. high-mode n=30 selected MSE 1.55 vs raw-local 4.85), but not over the posterior (0.34). This resolves the validation-statistic bias concern, not serial dependence or risk-selection noise: ρ and source modes are oracle-known, and validation blocks are contiguous. No safe rule or candidate results.

**C24 distinguishes Jing et al.'s two shrinkage choices; both are prior art, not FedORBIT candidates.** The ignored POC `pocs/covariance_aware_safe_shrinkage_poc.py/.csv` compares the known-covariance Theorem 2.1 choice `s*=tr(S)−2||S||₂`, which maximizes the theorem's quadratic risk-reduction bound under `tr(S)>2||S||₂`, with Remark 2's unknown-covariance plug-in interval midpoint `s_mid=tr(Ŝ)−||Ŝ||₂`, whose common-risk result assumes the stronger `tr(S)>4||S||₂`. Both use `γ=s/max(||W₂(P−L)||²_Q,s)` in `L+γW₂(P−L)`. This simulation supplies known diagonal covariances to both rules, so it checks the two formulas under an oracle covariance model; it does not simulate covariance estimation for the plug-in rule. In 10,000 paired Gaussian-vector replicates (`p=115`, 8 independent sources, four mismatch patterns, n=30/100/300/1000), `tr(S)/||S||₂=61.9726` at every support size, satisfying both conditions. For all sources shifted by +0.5, theorem-bound-max MSE was .031867/.010447/.003565/.001084 and plug-in-midpoint MSE was .031866/.010447/.003565/.001084, versus local .036320/.010883/.003615/.001088 and fixed precision pooling .249572/.248452/.245353/.234918. Under matched sources, corresponding MSE was .000495/.000165/.000080/.000046 (theorem choice) and .000424/.000144/.000073/.000044 (midpoint), with oracle weight MSE about .000034/.000034/.000034/.000033. Under the fixed mismatch, mean `γ` falls from .1229/.1250 (theorem/midpoint) at n=30 to .0043/.0044 at n=1000; under matched sources it stays about .94. This confirms the prior summary's empirical direction and shows the two rules are practically similar in this high-effective-dimension synthetic setting. It does not establish the theorem's conditions for FedORBIT's serial, non-Gaussian variance summaries or imply detector AUROC/FPR improvement. No sample-variance, real-device, OAS, or detector outcome is included.

**C25 feature-dependence screen challenges direct use of C24's dominance condition.** The ignored diagnostic `pocs/covariance_aware_feature_dependence_poc.py/.csv` estimates effective dimension `tr(C)/λmax(C)` for variance-estimation error. Under an iid-Gaussian proxy `C=Corr(X)∘Corr(X)`, the nine support pools have effective dimensions 2.1–5.2 (median 3.53), far below nominal `d=115` and close to the theorem's strict threshold of 2. This proxy corresponds to *relative* variance error, not raw variance error. A useful qualification follows from the influence function. For a stationary Gaussian vector process with separable covariance `Cov(X_t,X_{t+h})=ρ^{|h|}Σ`, the standardized influence `ψ_{t,j}=(X_{tj}−μ_j)²/σ_j²−1` satisfies `Cov(ψ_{t,j},ψ_{t+h,k})=2ρ^{2|h|}Corr(X)_{jk}²`. Its long-run covariance is `Ω_rel=2(1+ρ²)/(1−ρ²)(Corr(X)∘Corr(X))`: under a common AR coefficient, serial dependence scales this relative-error covariance but leaves its effective-dimension ratio unchanged. For raw variance errors, the covariance is instead `DΩ_relD`, with `D=diag(σ_j²)`, so feature-scale heterogeneity can also change the ratio; the correlation-only proxy cannot establish the raw `S` condition. A new scale-weighted iid-Gaussian proxy `D(Corr(X)∘Corr(X))D` gives median effective dimension 1.46 across devices (range 1.11–2.94), below 2 on 6/9 devices; this is much closer to the raw offset-block covariance than the correlation-only proxy, though Ecobee remains discrepant (2.81 vs offset medians 1.01–1.17) and Provision PT 838 remains high until n=1000. Using empirical covariance across non-overlapping chronological block log-variance vectors gives medians 1.61/1.47/1.38/1.23 for n=30/100/300/1000; the `tr(C)>2λmax(C)` condition holds for only 1/9, 0/9, 0/9, 0/9 devices. A raw sample-variance-vector sensitivity across 16 offset partitions gives medians 1.30/1.32/1.29/1.12; the proxy condition passes 32/144, 23/144, 16/144, and 0/144 partitions. Those offsets overlap, within-partition blocks may remain serially dependent, and large-n covariance ranks are small. These empirical summaries mix estimation noise with operating/regime variation and cannot certify the actual two-summary `S` or its uncertainty. They are descriptive warnings, not theorem tests. Nominal `d=115` therefore cannot justify C24 in FedORBIT. Next estimate the target/source summary-error covariance and its uncertainty under an explicitly checked time-series model, including feature-scale weights; test `S=Q½W₂V₁Q½` only if that model is defensible, then determine whether summary-MSE improvement carries to OAS/AUROC/FPR.

**Prior-art audit of federated AE summary-statistic thresholding is now equation-level.** Laridi, Palmer & Tam (2024, [arXiv HTML](https://arxiv.org/html/2410.09284)) compute each participating client's reconstruction-error mean, variance, skewness, kurtosis, and count separately for normal and anomalous validation examples. The server forms count-weighted global moments (including the between-client mean contribution to variance), creates threshold candidates inside a moment/skewness/kurtosis-adjusted overlap interval between normal and anomaly score distributions, and chooses the threshold maximizing average client F1 after clients score candidates on labeled validation data. The paper also raises whether a client should use local or federated thresholding. This is an adjacent prior for sharing anomaly-score summaries and for federated AE threshold choice; it weakens any broad claim that sharing summary statistics can support anomaly detection or an AE-compatible deployment. It does not implement benign-only input-feature scale estimation, a held-out new physical device, target-specific scale-risk minimization, or AUROC/ranking adaptation. Its labeled anomalous validation and F1 feedback are unavailable under this study's benign-only onboarding protocol, so it does not close the narrower problem gap. See the updated single novelty matrix. The full protocol is materially different from the current scale-only AE smoke: reconstruction-score threshold choice versus normalization statistics used upstream of either Gaussian scoring or AE training.

**New-client deep normalization prior art also narrows the candidate's model-agnostic claim.** Jiang et al.'s UniFed (the earlier TsmoBN preprint; [arXiv HTML](https://arxiv.org/html/2110.09974v3)) explicitly studies external/unseen federated clients. Its algorithm keeps BN parameters local during federated training; at an external client it re-estimates BN activation mean/variance from each incoming test batch and applies EMA updates `μ←τμ+(1−τ)μ_batch`, `σ²←τσ²+(1−τ)σ²_batch`. It gives a lower generalization-error bound for external clients under its image-feature-shift assumptions. Thus target-specific unseen-client normalization and generic deep-detector compatibility are established prior art. It differs from FedORBIT's proposed object (raw input-feature scale used to fit a benign-only detector), has no partner-stat borrowing or risk-optimal local/peer weight, and uses unlabeled target test batches rather than a designated benign support set. Still, any final claim must be narrower than “adaptive normalization for a new client,” and AE compatibility alone cannot distinguish the method. A meaningful comparison should include target-local test-time normalization/BN-stat recomputation where architecturally applicable, while clearly separating its test-batch data contract from benign onboarding.

**Recent soft-statistic partner weighting further preempts generic adaptive-collaboration claims.** The 2025 pFedBBN preprint by Iftee et al. ([full arXiv HTML](https://arxiv.org/html/2511.18066)) performs unlabeled federated test-time adaptation with class-wise pseudo-label-balanced BatchNorm statistics. It defines pair distance `D_ij` as the layer-average of one-half the sum of L2 distances between activation-mean vectors and activation-variance vectors, sets off-diagonal collaboration weights proportional to `exp(−D_ij/τ)`, and gives the self model weight `[1+Σ_{k≠i}exp(−D_ik/τ)]⁻¹`. It exchanges compact statistics and personalized model parameters. This is a close hostile prior for “soft partner weighting from normalization descriptors plus local fallback”; do not present that design pattern as new. Differences remain in the estimand and decision loss: latent BN statistics after pseudo-label adaptation and model aggregation on corrupted image classification versus raw input-feature benign-support variance estimation for held-out IoT anomaly ranking. The preprint describes the approach but provides no finite-sample estimator-risk no-harm guarantee for that similarity weight. Any future method needs a mathematically distinct target-scale risk result and evidence beyond soft similarity weighting itself.

**FedIG's learned zero-shot statistic interpolation is another direct deep-normalization precedent.** The ICLR 2023/OpenReview manuscript “Client-Agnostic Learning and Zero-Shot Adaptation for Federated Domain Generalization” (Yang et al., [OpenReview](https://openreview.net/forum?id=S4PGxCIbznF)) mixes each test instance's BN statistics with the global BN statistics at an unseen client: `μ_t^l=α^lμ_i^l+(1−α^l)μ_G^l`, `σ_t^l=α^lσ_i^l+(1−α^l)σ_G^l`. A source-trained zero-shot adapter generates `α^l` conditioned on channelwise instance/global statistic differences. Thus a learned local/global normalization interpolation at an unseen domain is also established; it is not the proposed contribution. FedIG addresses latent feature normalization for supervised image prediction and obtains no stated finite-sample guarantee for raw benign-scale estimation risk or anomaly AUROC. Together with UniFed and pFedBBN, this leaves only a narrow possible contribution around a distinct estimand/loss plus defensible partner-risk control, not adaptation, interpolation, statistics-based similarity, or deep compatibility in isolation.

**Unseen-client transferability scoring and negative-transfer-aware source selection are also established.** The 2023 MICCAI TGMA paper (Yang, Liu & Yuan, [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_66)) transfers multiple source models to an unlabeled target without source data. It introduces a label-free transferability metric and uses an instance-level transferability matrix for target pseudo-label correction plus a domain-level matrix for source-model selection/initialization. The publisher page confirms only an abstract preview; its full equations are not available there, and the linked [authors' code repository](https://github.com/CityU-AIM-Group/TGMA) currently returns 404. TGFed (Niu et al., [Expert Systems with Applications, 2026](https://www.sciencedirect.com/science/article/pii/S095741742503564X)) makes a closely related unseen-client FL move: publisher-accessible text confirms source-model transferability is based on prediction consistency for original versus masked target samples, followed by selected-model aggregation and mask-consistency mean-teacher pseudo-label adaptation. [OpenAlex currently lists the article as closed access with no repository full text](https://openalex.org/works/W4415228012). These works preempt broad claims that unseen-client transferability, dynamic source choice, pseudo-label adaptation, or negative-transfer-aware source-model selection are new. Their task/loss differs from FedORBIT: labeled-source segmentation models are selected/adapted on unlabeled target imagery, with no benign-support variance estimation or anomaly-ranking risk. Exact TGMA/TGFed transfer-score and aggregation equations remain an explicit audit gap; do not claim a formula-level novelty distinction until primary full text or code is available. The single novelty matrix records this access limit.

**C26 estimated-HMM regime recovery separates pure-block detection from transitions, but does not establish a target-transfer rule.** The ignored POC `pocs/estimated_hmm_model_selection_poc.py/.csv` finished 200 replicates per scenario (8 independent AR(1) peer sequences, 4,000 rows each, rho=.7, block length 18, four starts, up to 200 EM iterations). It compares a shared symmetric two-state Gaussian-emission HMM (5 BIC parameters) with a one-state Gaussian model (2 parameters). On homogeneous data, the HMM won BIC in 1/200 runs and median `BIC_1−BIC_2` was −17.1; six fits reached the 200-iteration cap. On switching data (innovation variance .25/2.25, per-row flip probability .01), the HMM won 200/200, median BIC improvement was 1429.1, median fitted block-flip probability .135 (theoretical row-Markov block-boundary flip probability .144), and median fitted emission means/variances were −1.033/+1.028 and .294/.288. A replay audit `pocs/replay_hmm_transition_accuracy.py/.csv` regenerates the same sequences and applies the saved parameters without refitting; it exactly reproduces the saved pure-block accuracy. Median pure-block accuracy is .995 (minimum .989), versus .984 for a simple midpoint threshold using the same fitted emission means. Mixed-transition blocks averaged 278.6 of 1776 blocks per replicate (15.7%); HMM majority-state accuracy there is .700 median (minimum .621), slightly below the midpoint threshold's .703. Thus most gain is on pure blocks, and the model does not resolve transition ambiguity. The earlier 20-replicate inline output has no saved result to compare numerically. This remains an exploratory regime-recovery diagnostic: emissions are uncalibrated Gaussian block log-variances rather than C19's likelihood, and the BIC calculation treats correlated block emissions as independent, so large BIC deltas are not calibrated evidence. The parent script now reports undefined homogeneous accuracy as “n/a” without a warning and records iteration-cap hits. Next compare C15/C17/C19 emission calibrations in one likelihood and test state recovery under varied autocorrelation, switch rates, and within-peer heterogeneity before fitting any target prior. HMM/changepoint structure is established prior art, not a novelty claim.

**C27 calibrated-noise HMM comparison is complete; noise calibration changes false selection and transition recovery modestly.** The ignored POC `pocs/calibrated_hmm_emission_poc.py/.csv` ran 200 paired replicates per scenario (homogeneous or switching) and calibration (C15/C17/C19), using one shared symmetric HMM likelihood. Each calibration supplies only a fixed Gaussian log-emission noise variance; the two state means and block flip probability are estimated. At rho=.7/block18, the fixed variances are C15 .4720, C17 .3293, C19 .2507. On homogeneous data the two-state HMM won within-calibration BIC in 0/200 C15, 0/200 C17, and 5/200 C19 runs; the five C19 improvements were small (median among selected runs about 1.09). On switching data, the HMM won 200/200 under all three calibrations. Median pure-block accuracy was .993/.995/.995 (C15/C17/C19), with paired median gains over the midpoint threshold of .0094/.0108/.0107; transition-block majority accuracy was .696/.703/.707, still below the corresponding midpoint thresholds (.713/.713/.713), with paired median differences −.0174/−.0103/−.0066. Median fitted block-flip probabilities were .1255/.1335/.1384 versus the theoretical .144 block-boundary flip probability; fitted state means remained near −1/+1. No fit hit the 200-iteration cap. Homogeneous pure-label accuracy is not meaningful because the data contain only one state. BIC selection is interpretable within each fixed calibration; absolute BIC improvements should not be ranked across calibrations because they assume different fixed noise models. Since means are free, the shared log-bias offsets cancel; this isolates noise scale rather than every part of each approximation. The comparison still assumes known rho and conditionally independent block emissions, so its BIC is not calibrated for serial dependence. The small transition-accuracy changes do not resolve transition ambiguity or establish target-transfer benefit. HMM/changepoint models remain established prior art. Next vary autocorrelation, switch rate, and within-peer heterogeneity under paired controls before fitting any target prior.

**C28 stationary HAC covariance diagnostic remains below C24's raw-scale dominance threshold.** The ignored POC `pocs/hac_variance_summary_covariance_poc.py/.csv` computes Bartlett/Newey-West long-run covariance of raw and relative sample-variance influence series from each N-BaIoT support pool, at bandwidths 5/18/50 and on full/first-half/second-half chronological segments (81 device/segment/bandwidth rows). For raw-scale covariance, median `tr(C)/λmax(C)` across nine devices is 1.388/1.378/1.360 for full series at bandwidth 5/18/50; no device/segment passes `tr(C)>2λmax(C)` in any bandwidth/segment combination. First-half medians are 1.369/1.366/1.299; second-half 1.452/1.384/1.327. Relative-scale median effective dimensions are larger and bandwidth-sensitive: full 1.799/1.674/1.497, with 2/9, 1/9, 0/9 passing; first-half 1.835/1.649/1.495 with 2/9, 1/9, 0/9; second-half 1.754/1.709/1.570 with 2/9, 2/9, 0/9. The raw-scale HAC proxy therefore agrees that the `W₂=Q=I` sufficient condition is not met, though it gives lower effective dimension than some offset-partition diagnostics. This is not a test of C24's actual `S=Q½W₂V₁Q½`: the same device series supplies influence and centering estimates, stationarity/mixing is unverified, chronological halves can contain regime changes, and target/source cross-covariance and peer weighting are absent. It is a sensitivity diagnostic, not a theorem test. Next establish a defensible stationary segment/model and uncertainty for the actual target/source summary-error covariance before using the theorem condition.

**C29 paired HMM sensitivity controls show that calibration matters most at high autocorrelation.** The ignored POC `pocs/hmm_switching_sensitivity_poc.py/.csv` spans rho=.3/.7/.9, mean per-row switch probability .005/.01/.03, and peer transition-rate heterogeneity 0/.5, with 20 replicates per cell and eight peers × 4,000 rows (1,080 paired calibration rows). At heterogeneity .5, individual peer rates span 0.5–1.5× the cell mean; the fitted symmetric HMM still shares one transition probability and two emission means across peers. C15/C17/C19 fixed emission-noise variances are recalculated for each rho. Across the 120 runs per rho/calibration, the two-state model won within-calibration BIC 120/120 for all methods at rho=.3/.7. At rho=.9, it won 47/120 for C15 versus 120/120 for C17 and C19. The rho=.9 C15 emission variance is 4.54, versus .710/.431 for C17/C19; median pure-block accuracy is .891/.975/.976, and median transition-block accuracy .556/.669/.684. The respective midpoint transition accuracies are .698/.694/.692, so HMM transition scoring remains lower under every calibration. Across rho=.3/.7/.9, paired ±50% peer-rate heterogeneity has smaller and less consistent effects than calibration; at rho=.7 its median transition-accuracy change versus common peer rates is about −.01 to −.015. No fit hit the iteration cap. This is a coarse sensitivity screen (20 replicates/cell), not a target-transfer estimator. BIC selection is interpreted only within each fixed noise calibration, and even those BICs treat conditionally dependent block emissions as independent. C15's loss of switching selection/recovery at rho=.9 is consistent with its severe high-rho emission-noise overestimate, while C17/C19 improve pure-state recovery; neither resolves transition blocks or establishes transfer benefit. Next focus on a defensible stationary segment/model and uncertainty for actual target/source summary-error covariance before returning to C24's `S` condition.

**C30 chronological CUSUM rejects second-moment stability in the full N-BaIoT support pools.** The ignored POC `pocs/variance_influence_cusum_poc.py/.csv` aggregates each device's raw variance influence vector into non-overlapping 18-row means, then scans a simultaneous max CUSUM over 115 features and break fractions 0.15–0.85. A circular moving-block bootstrap with 199 replicates uses 18/54/108 raw-row block lengths. At lengths 18 and 54, all 9 devices have bootstrap p≤.05; at length 108, 8/9 do (Ecobee p=.115; Danmini .035). The CUSUM statistics range 6.79–33.12, versus 18-row-block 95% critical values 3.92–4.38 and 54-row-block critical values 5.64–6.56. Selected-feature variance ratios after/before the scanned break range .056–2.67, with selected fractions from .15 to .85; these are post-selection descriptions, not unbiased effect estimates. Thus the full chronological support pools do not support a single stationary second-moment approximation at the tested scale, even though the bootstrap p-values are block-length-sensitive for some devices. Covariance and volatility break tests for dependent multivariate data are established prior art ([Aue et al. 2009](https://arxiv.org/abs/0911.3796); [Bours et al. 2021](https://doi.org/10.1111/sjos.12508)); moving-block bootstrap for stationary observations dates to Künsch ([1989](https://doi.org/10.1214/aos/1176347265)). This max statistic accounts for scanning features and cut points within each device, but not multiplicity across devices/block lengths; p-values have minimum resolution .005. The MBB null assumes stationarity/mixing and has no synthetic size calibration here, and this tests second-moment stability only—not strict stationarity, mixing, or the actual target/source `S`. Do not use full-pool HAC or C24 theorem conditions as if stationarity were established. Next apply the same predeclared stability check separately to chronological halves and only consider a local covariance model if a segment survives block-length sensitivity; otherwise record that this benchmark cannot validate C24's assumptions.

**C31 fixed chronological halves also fail the second-moment stability screen.** The ignored POC `pocs/variance_influence_half_cusum_poc.py/.csv` applies C30 unchanged inside the first and second halves, with the 50/50 split set before inspecting C30's break locations. In each half, 9/9 devices have p≤.05 at MBB lengths 18 and 54; 9/9 first halves and 8/9 second halves do at length 108 (Ecobee is the only second-half non-rejection, p=.105). Median max-CUSUM is 13.08 in the first halves and 15.06 in the second. Selected variance ratios after/before the within-half scan range .011–.315 in first halves and .781–16.74 in second halves; many maxima occur at the 15%/85% trim boundary, consistent with continued drift rather than a clean interior break. The same caveats as C30 apply: bootstrap p resolution is .005, multiplicity across devices/lengths is unadjusted, the null presumes stationarity/mixing, and failure to reject would not prove stationarity. The broad predeclared halves are not plausible stable windows, so no target/source covariance or C24 theorem condition should be estimated from these pools as if it had a stationary sampling law. The repository loader preserves row order but retains only the 115 statistical feature columns; UCI distributes per-device CSV feature files, while the original paper describes these as behavior snapshots from packet streams, with no per-row timestamp or session identifier in the released feature matrix ([UCI dataset record](https://archive.ics.uci.edu/dataset/442/detection_of_IoT_botnet_attacks_N_BaIoT); [Meidan et al.](https://arxiv.org/abs/1805.03409)). A narrower interval would therefore need external capture/session boundaries or an independently specified support protocol; choosing a segment from these scans and testing it on the same rows would be circular.

**C32 exact finite-window covariance avoids stationarity but rarely activates the C24 rule.** The ignored POC `pocs/finite_window_covariance_risk_poc.py/.csv` enumerates every contiguous n-row start in each finite target support pool at n=30/100/300/1000, exactly matching the production uniform-start design. It computes the exact covariance `V1` of the local `ddof=0` variance vector, without a stationary-process assumption, and tests the special case `W2=Q=I`. The peer vector is the fixed production-style equal-weight aggregate of the other seven devices' full-pool moment summaries, including between-device mean variation. The condition `tr(V1)>2||V1||₂` passes in only 5/36 device/support cells (2/9 at n=30, 2/9 at 100, 1/9 at 300, 0/9 at 1000); the stronger `tr(V1)>4||V1||₂` passes in 0/36. When the condition fails, the theorem-bound rule has γ=0 and equals local-only. In the five passing cells it uses small mean γ=.027–.060; its exact finite-design MSE for the random-window mean improves over local in all 5 cells by about 3–6%, but this is an oracle finite-pool calculation and not a Gaussian-theorem validation. Against the full-pool variance target, the local random-window estimator is biased (median bias RMSE across devices 8.34/6.82/5.67/4.51 for n=30/100/300/1000); the theorem rule improves MSE only in the same 5 cells and is otherwise exactly local. The fixed peer vector has lower MSE than local against the full-pool target in 30/36 cells, but can be substantially mismatched to the random-window mean, so raw pooling is not a safe comparator. This finite-design view is distinct from C28's HAC covariance: it conditions on the observed ordered pool and the explicitly randomized start, but it does not estimate peer uncertainty, actual C24 `S`, future benign risk, detector AUROC/FPR, or prove the published risk bound applies to these non-Gaussian window summaries. Next test whether the limited finite-design risk gain carries to OAS/AUROC only as a benchmark-specific diagnostic; no general safety or novelty claim follows.

**C33 the C32 finite-design rule does not beat shared marginals or control held-out FPR.** The ignored POC `pocs/c24_finite_window_oas_poc.py/.csv` pairs the update with existing OAS window draws and 540 detector cases (15 replicates × 9 devices × 4 supports). Only the 5 cells with positive exact-oracle `V1` condition have nonzero γ (75 windows); the other 465 outputs equal local-only. Within those 75 active windows, C24 AUROC exceeds local in 58 and is lower in 17; median paired gain is .0048 and mean .0504, with the mean driven by large gains in some n=30 Provision runs. By support size, median/mean ΔAUROC vs local is .043/.097 at n=30, .011/.026 at n=100, and .0001/.004 at n=300. C24 loses to shared marginals in 53/75 active windows (median ΔAUROC −.0029, mean −.0268); mean AUROC is .923 for C24 versus .950 for shared marginals on those same cases. At a per-model threshold set to the training-window 95th score percentile, held-out benign FPR averages .899 local and .893 C24, while attack TPR is 1.000 for both. This is poor operating-point behavior; the threshold uses no separate benign calibration set. The `V1` condition is an oracle: it uses every target-pool window and is unavailable to a new client that sends only one short support window. This does not validate the Gaussian theorem assumptions, provide a deployable weight, establish future-device safety, or support novelty. Retire this C24 route for the current N-BaIoT data contract; only revisit on a less saturated benchmark if its support protocol makes `V1` estimable and gives detector-level value over shared marginals.

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
transport); a mean-plus-variance summary is 2d=230 scalars (920 bytes). For eight peers those direct
uploads would be 3.68KB vs 7.36KB per target round. However, this is not the C48/C75 estimator: those
POCs pool total peer covariance, whose diagonal includes between-peer mean offsets, so reproducing
their peer scale requires per-peer means and variances (at least 2d summary scalars), not only a d-value
within-peer SD. Once a coordinator has computed the pooled scale, sending the target only d scale values
instead of 2d mean-plus-scale values halves that target-facing normalization state; this does not reduce
the peer-to-coordinator upload established by C48/C75. The current runners retain full `MomentSummary`
covariance matrices in memory and do not measure wire bytes, secure aggregation, or privacy. No privacy
guarantee is claimed, and marginal statistics can leak traffic properties.

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
precise mathematical difference. The exact two-sample result is specific: for independent Gaussian
summary estimators `X1~N(theta1,V1)`, `X2~N(theta2,V2)` and fixed quadratic loss `Q`, define the
common-mean precision pool `Xbar`, direction `W2(X2-X1)`, and `S=Q^(1/2)W2 V1 Q^(1/2)`. Its SURE
weight is `t_hat=min(1,tr(S)/||W2(X2-X1)||_Q^2)`, but the paper explicitly notes that data-dependent
SURE minimization is not itself guaranteed to reduce risk. Theorem 2.1 gives a risk-improving interval
`0<s<2tr(S)−4||S||₂` under `tr(S)>2||S||₂`; its bound-maximizing choice is `s=tr(S)−2||S||₂`.
Separately, Remark 2's plug-in midpoint is `tr(Ŝ)−||Ŝ||₂`, selected from the interval
`[tr(Ŝ)−2||Ŝ||₂,tr(Ŝ)]`; the common-risk theorem assumes `tr(S)>4||S||₂`. C24 now compares
both choices, and its synthetic ratio `tr(S)/||S||₂≈61.97` satisfies both conditions. These are
distinct formulas and assumptions; neither guarantee has been established for FedORBIT's serial
variance summaries. The guarantee is for squared estimator risk, not OAS detector ranking. The multi-source extension
updates source by source using a priority score, and the asymptotic oracle result assumes homogeneous
and heterogeneous sources are separated above/below the `sqrt(p/n)` noise scale. Therefore it is a
powerful theoretical template but not directly a no-harm FedORBIT solution: target benign windows are
serially dependent, the 115 variance summaries are not established as jointly Gaussian, covariance is
unknown at cold start, and estimation-risk dominance need not imply AUROC/FPR dominance.
[Full equations, v2](https://arxiv.org/html/2606.30615). Li & Ignatiadis' PAS (ICML 2025) is a related
compound-mean method: it first builds per-problem prediction-powered estimators, then uses a
correlation-aware unbiased risk estimate (CURE) to choose across-problem shrinkage. Its problems are
iid/exchangeable and it has a separate unlabeled-prediction channel, so it is not a device-summary
transfer method, but it preempts generic claims to adaptive unbiased-risk shrinkage over many features
([paper](https://proceedings.mlr.press/v267/li25ak.html)). Additional
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

Continuation after C33: C24 remains retired on this N-BaIoT data contract, while its theorem remains prior art. C34–C36 tested full-covariance likelihood selection, joint mean/scale selection, and target-conditioned peer weights; none establishes a consistent device-level gain over shared marginals, and C36's tiny n≥300 mean gains have device-clustered intervals crossing zero. No alternative physical-device data population has emerged: prepared outputs contain nine N-BaIoT devices and the simulated Gotham assets; the existing TON-IoT and Edge-IIoTset audits do not support physical-client identity/chronology. Continue with a loss/identifiability audit rather than promote a proxy selected on the same attacks. C20–C23 and C26–C31 remain diagnostics only; the protocol is unlocked and no confirmation run has started.

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
| C61 shared-center/median-peer-SD OAS | Low/unclear; coordinatewise robust scale aggregation inside an established external-target OAS family | ΔAUROC −.0050/−.0015/+.0003/+.0006; at n=1000 pAUC(.05) +.00364 vs shared | 2/9→8/9 positive device means; worst mean −.0265→−.00003 from n=30 to 1000 | Fixed peer scale at every n; AUROC loses at n≤100 and low-FPR benefit is mainly n=1000 | No reduction vs 2d outgoing shared moments; each peer scale summary is d values | C60/C64 synthetic OAS comparisons show a shared/median near-tie and no no-harm property; C63 locates the change in OAS's effective diagonal covariance target, not its shrinkage intensity | C62 AE at n=1000: AUROC +.00007, pAUC(.01) +.00436 vs shared; all device intervals cross zero | Objective/post-hoc selection risk, 9 devices only, no second physical cohort, and robust target aggregation prior art |
| C84–C86 source-attack-assisted OAS action selection | Low; directly adjacent to cross-domain anomaly transfer (ACT, TLNP); relaxes benign-only | Selected AUROC Δ (n=30/100/300/1000): C85 −.00023/−.00064/−.00001/−.00307; C86 −.00023/−.00070/+.00018/−.00305 | C85 n=1000 harms 10% of draws; worst device mean −.0296 (Philips), worst Mirai device mean −.0575; pAUC(.01) −.00959 | Five-fold selector changes action more often at larger n but source ranking does not transfer reliably; target oracle action headroom only +.00028→+.00140 AUROC | ~412–442 KB source attack rows/target vs 7.36 KB eight-peer mean/scale summaries (~56–60×); raw labeled examples and privacy exposure | Source-labeled transfer has no target-ranking no-harm result; standard domain-transfer objective | Not tested; OAS only | Source attack/normal domain separation saturates selector AUROC; target attack-family reversal, small n, one nine-device cohort, and established labeled-source anomaly-transfer prior art |
| C87 source-attack + target-benign regularized linear boundary | Low; one-class-data transfer is established (Chen & Liu 2014), with LOCIT/ACT also close | ΔAUROC vs shared OAS (n=30/100/300/1000): +.00589/+.01261/+.00764/+.00405; standardized pAUC(.01): +.01179/+.02069/+.02640/+.04867 | At n=1000 AUROC improves in 68.9% of paired cells and 8/9 device means; Danmini is worst (−.00115). pAUC lower tails stay negative (q10 at .01: −.0668 to −.0808) | Target support labels the benign class; gains broadly increase to n=300/1000, but n=30/100 AUROC is mostly two Provision-device gain | ~910 labeled source attack rows/target, 412–442 KB float32 vs 7.36 KB peer moment summaries; ~56–60× | Standard class-balanced L2 logistic ERM; no transfer or ranking safety bound | Not tested; standalone linear detector | Gains are attack-family/device dependent, same nine-device cohort, label/privacy and communication burden, direct one-class transfer prior art; no clean confirmation split |
| C88–C90 shared-OAS/logistic score blends | Low; score/rank fusion is established, including reciprocal-rank ensembles (Marques et al. 2023); z-score normalization is standard | C88 cross-fit rank blend is below full-support OAS at n=30/100 and nearly ties at n=1000; C89 support-ECDF saturation hurts pAUC(.01) by −.2846 at n=1000; C90 standard z blend α=.50 gives ΔAUROC +.00020/+ .00045/+.00223/+.00128 and ΔpAUC(.01) +.00115/+.00094/+.00404/+.01068 at n=30/100/300/1000 | C90 α=.50 reaches positive device means on 9/9 at n≥300, but n=1000 pAUC(.01) paired q05 is −.00064 and worst draw −.01153; small lower-tail failures persist | Benign support calibrates/fits both endpoints; C88 loses information by training on only 2/3 support; C89's bounded ECDF saturates outside support; C90's unbounded affine z scores preserve ordering | Same high raw-attack training payload as C87; transforms add only benign support-score summaries | Convex score fusion after mean/SD or median/MAD calibration; no no-harm guarantee | Standalone OAS and linear score paths; AE untested | Blend gains are tiny relative to C87 and do not remove low-FPR tail harms; one reused cohort, rank-fusion prior art, exploratory only |
| C91 leave-one-source-device-out maximin blend selection | Low; leave-one-domain-out model selection and worst-group regret/utility criteria are established | Select α in {0,.25,.5,.75,1} by maximizing the worst pseudo-device mean ΔpAUC(.01) vs α=0; held-out results give ΔpAUC(.01) +0/+.00193/+.00550/+.02276 and ΔAUROC +0/+.00039/+.00247/+.00211 at n=30/100/300/1000 | Target device means positive on 0/9, 6/9, 7/9, 9/9; at n=100 q10 pAUC(.01) −.00015 and worst-device mean −.00127; at n=1000 worst paired delta −.01244 | Selects α=0 for all targets at n=30 and α=.75 for all at n=1000; at the largest support it collapses to the fixed C90 α=.75 blend | Same ~419 KB raw source attack payload/target as C87; LODO requires eight pseudo-target fits per target and support size (10 windows/fold) | Empirical rule `argmax_α min_j mean_r ΔpAUC01(j,r,α)`; source-client exchangeability is assumed but not tested; no generalization or no-harm guarantee | Same OAS and linear endpoints; AE untested | Gives up C87's n=30 mean lift, retains lower-tail harm at n=100/1000, adds no gain beyond fixed α=.75 at n=1000; one fixed cohort and direct LODO/maximin prior art |
| C92 benign-distance source-attack relevance diagnostic | Low; benign similarity weighting and anomaly-source compatibility are established (LOCIT 2020); no new rule | Single-source mean ΔAUROC +.00644/+.01164/+.00559/+.00348 and pAUC(.01) +.01776/+.01789/+.02221/+.04342 vs shared OAS | Only 36/72–57/72 source-target mean pAUC pairs positive; q10 −.06560 to −.08096; worst pair mean −.078 to −.090 at n≤300 | RMS benign mean/log-SD distance correlations with gain are mostly positive, contrary to “closer helps”; mixed across windows, no usable monotone selector | Per-source raw attack rows needed; not a validated communication reduction | No estimator of ranking risk; a benign marginal distance need not predict attack-family transfer | Standalone logistic; AE untested | Family reversal, large low-tail harm, same nine-device cohort; per-source deltas are not additive to pooled C87; diagnostic only |
| C76/C80 local-center/median-peer-SD AE | Low/unclear; robust per-feature peer aggregation plus standard raw-input normalization | At 30 paired draws/device, vs shared AUROC −.01294/−.00477 and pAUC(.01) −.01767/−.00812 at n=30/100 | 2/9 AUROC-positive device means; Provision devices and SimpleHome XCS7-1002 show harms | Transfers benefit vs local, but median peer scales remain fixed at both supports | 115 float32 SD values/peer, half the 2d full-marginal upload; no wire benchmark | Median aggregation handles outliers but gives no target mismatch/no-harm result | Local AE tested across 30 paired support draws/device; median scale trails pooled-total scale | Genuine statistic-payload reduction but no quality-preserving gain; nine fixed test populations, no independent confirmation |
| C9 iid risk-optimal local/peer variance blend | Standard random-effects/MSE shrinkage; no new general estimator claim | −0.0849 → −0.0112 vs shared | No device mean positive; near local-only | Peer weight 5.2% → 0.8%, naturally fades but starts too low | Local fourth moment + peer per-feature variance estimates | MSE-optimal only under independent unbiased estimates; observed window risk is 4–33× iid plug-in due chronology | Scale-only idea could feed AE; not tested yet | Temporal dependence breaks risk estimate; AUROC loss mismatch |
| C10 peer-block variance-risk blend | Random-effects partial pooling with dependent-window risk proxy; bootstrap literature is adjacent | −0.0010 → −0.00004 vs shared; n=300 −0.00845 | Negative cells 30–55% by support; worst device −0.0264 | Peer weight 53% → 30% on real data, but 98% → 90% under regime-risk mismatch | Peer block means/risks, about 2d/device | Conditional linear-MSE derivation; synthetic mismatch MSE 1.48–1.60× oracle | Scale-only interface could feed AE; C10 itself untested | Target-peer temporal comparability fails under distinct stable target regime; no safe bound |
| C11 C10 with moment-estimated target mismatch penalty | Plug-in excess-discrepancy penalty; standard risk estimation | Slightly below C10 at every n | Negative-cell fraction rises vs C10 | Weight 49% → 27%, but does not protect devices | No additional peer payload | Identity-based estimate of squared bias; noisy at feature level | Not evaluated on AE | Penalizes peer use without finding harmful cells |
| C12 median/MAD peer-block risk blend | Standard robust aggregation; close robust meta-analysis and geometric-median FL prior art | −0.0015 → −0.0004 vs shared; slightly better than C10 only at n=300 | Negative cells 24–57% by support; worse worst device than C10 at n=300 | Weight 66% → 43%, above C10 at every n | Same approximate 2d summaries | Resists one 4× peer in scalar MSE, but 1.18–1.56× oracle under regime mismatch | Not evaluated on AE | MAD misses lone-source risk; not safe under target/peer temporal mismatch |
| C13 empirical-envelope minimax blend | Standard bounded-bias/minimax shrinkage; Abba et al. 2024 and TRADER are close | Synthetic only; mostly reverts to local; no AUROC result | Reduces regime-mismatch MSE but does not beat C10/C12 consistently | Peer weight overly conservative under several scenarios | Same 2d plus peer-block envelope summary | Exact minimax rule if bias bound is true; observed maximum is not a coverage-valid bound | Not evaluated on AE | Safety claim unsupported; performance close to local-only |
| C14 peer-window mixture posterior | Empirical Bayes mixture / source-weighted transfer; TRADER and normal-means mixture EB are close | Synthetic only; helps switching regime at n≤100, reverses at n≥300; no AUROC result | No reliable across-scenario advantage | Can target local regime when peer windows preserve it; loses modes as target-length blocks average regimes | Per-feature block distribution / mixture parameters; more than 2d unless components compressed | Posterior optimal only under fitted Gaussian mixture and variance-noise model | Not evaluated on AE | GMM misspecification, component instability, established EB novelty |
| C15 dependence-heuristic short-block mixture posterior | Empirical Bayes plus dependence-adjusted variance noise; automatic block selection is established bootstrap prior art | Synthetic only; one component selected in all 16 cells; C15/oracle MSE 1.01–1.23× across homogeneous/heterogeneous/outlier and 1.58–1.80× under switching-peer mismatch; no AUROC result | Does not recover switching modes; near-local behavior under heterogeneity/outliers | About 18-row blocks across supports; noise floor absorbs regime persistence | Many peer block summaries per feature; larger than 2d unless compressed | Noise-floor approximation failed because volatility persistence was counted as within-regime sampling noise | Not evaluated on AE | Regime recovery absent; empirical-Bayes and block-length novelty claims preempted |
| C16 known-parameter switching diagnostic | Markov-switching variance HMM; Rossi & Gallo 2006 is direct prior art | Known-parameter state accuracy improves 0.942→0.949 vs thresholding; calibrated iid GMM treats transitions as a third mode | No estimated-parameter estimator, transfer-risk result, or AUROC result | Requires oracle state means/noise and transition rate | Peer temporal sequence/block log variances | Known-parameter forward-backward smoother | Not evaluated on AE | Markov-switching variance model is established; advantage is small and oracle-dependent |
| C26 estimated-parameter HMM diagnostic | Shared two-state Gaussian HMM on peer block log-variance sequences | Selects 1/200 homogeneous vs 200/200 switching runs; pure-block accuracy .995 vs .984 midpoint threshold, but transition accuracy .700 vs .703 | Not a transfer estimator; target mismatch/safety untested | Strong on pure blocks, ambiguous on transition blocks; nominal BIC ignores emission serial dependence | Peer time-series/block log variances; per-replicate parameter fit | Estimated shared means, variances, and block flip probability; no calibrated BIC guarantee | Not evaluated on AE | HMM regime detection is established prior art; highly controlled AR simulation only |
| C17 Satterthwaite noise-calibration diagnostic | Gaussian quadratic-form moment matching for AR(1) sample variance | Improves stationary log-noise calibration vs C15; remains inaccurate at high rho | No collaboration rule or detector result | Requires AR(1) covariance parameter | No extra communication in principle if dependence summaries are shared | Standard quadratic-form moments; approximation still overstates noise | Not evaluated on AE | Calibration only; no transfer or novelty claim |
| C18 estimated-rho calibration diagnostic | Plug pooled peer rho estimates into C17 | 8×4,000 peer rows estimate rho precisely, but approximation bias persists | No collaboration rule or detector result | Does not address volatility switching | Small autocorrelation summary | High-rho log-noise approximation remains materially off | Not evaluated on AE | Accurate rho alone does not fix misspecified log-quadratic-form distribution |
| C19 lognormal AR log-variance calibration | Exact Gaussian AR quadratic-form first two moments matched to lognormal moments; Gaussian quadratic-form approximation literature is established | Stationary Gaussian simulation only; no AUROC | Not a borrowing rule | Calibrates n/rho dependent log-likelihood; high accuracy on tested ρ/n grid | Peer/target autocorrelation estimate; no added moments | Mean abs log-noise error .0203 vs 1.046 for C15 on 9 cells; this is calibration only | Not evaluated on AE | Known AR covariance required; not valid for regime-crossing blocks; approximation prior art is mature |
| C20 oracle two-regime posterior diagnostic | Two-point prior over peer volatility modes plus C19 target likelihood | MSE .014 vs local 1.587 at n=30 when target matches modes; catastrophic 4.13–6.01 MSE for an unseen intermediate target | Illustrates severe negative transfer under support mismatch; prior is oracle-known | Excellent only under correct source-target support; Bayes posterior cannot abstain under this model | Peer mode probabilities/levels plus local support | Posterior mean Bayes-optimal only under specified two-point prior and squared variance loss | Not evaluated on AE | Oracle prior is non-deployable; requires target-only fallback and mismatch analysis |
| C21 oracle predictive-compatibility gate | Local fallback when peer-mixture prior predictive p-value < α; conformal exchangeability transfer tests are direct prior art | α=.05 only avoids 4–6% matched-target use and still leaves MSE 3.34/.52 vs local .55/.19 at n=30/100 under unseen target; α sweep has no derived safe operating point | Reduces harm at larger n; low-support compatibility evidence is inadequate | Can abstain as n grows; low-n false acceptance remains costly | Requires peer regime prior and target score; calibration assumes known model | Posterior predictive goodness-of-fit test, not risk guarantee | Not evaluated on AE | Zhou et al. (2017) already test source-target exchangeability for transfer; oracle assumptions and significance-threshold dependence |
| C22 target-only holdout risk selector | Choose local variance vs oracle two-mode posterior using later target block | Matched/intermediate/novel scalar AR screen; half-support validation often lowers posterior mismatch harm but still raises MSE vs local for intermediate n≤300 and novel-high n=100–1000 | No guarantee; contiguous validation is serially dependent | Discards half the support for estimation | Target-only validation block; peer prior is still oracle | Held-out risk selection is established; no finite-sample no-harm result here | Not evaluated on AE | Uses wrong raw validation variance bias in original form; C23 corrects it |
| C23 AR-bias-corrected target-only holdout selector | Correct local/validation variance by known AR centering factor, then choose by held-out squared loss | Correction raises local MSE vs raw in all tested cells; novel-high selected MSE 6.80/3.64/.96 vs raw local 6.47/2.46/.69 at n=100/300/1000 | Harmful peer selection remains 33.4/15.9/2.9% at n=100/300/1000 | Half support used; not reliably safe | Target-only validation plus oracle source prior | Corrects validation bias only; selection noise and temporal dependence remain | Not evaluated on AE | Oracle rho; validation blocks contiguous; no risk bound |
| C24 covariance-aware safe shrinkage baseline (Jing et al. 2026) | Two-summary estimator `X1 + γ W2(X2−X1)`; compares Theorem 2.1 `s=tr−2||S||₂` and plug-in midpoint `s=tr−||S||₂` | Paired 10k known-covariance Gaussian-summary reps, p=115: both choices below local in all 16 cells; fixed +.5 shift theorem-choice MSE .031867/.010447/.003565/.001084 vs local .036320/.010883/.003615/.001088; fixed pooling badly harms | Theorem condition `tr(S)>2||S||₂`; midpoint common-risk result requires `tr(S)>4||S||₂`; both hold at synthetic ratio 61.9726, but FedORBIT's serial scale summaries have not been shown to qualify | Fixed mismatch mean γ declines .1229→.0043 (theorem) and .1250→.0044 (midpoint) from n=30 to 1000; near .94 matched | Pooled source summary; 8 peers | Direct prior art; squared summary-estimation risk only, no detector outcome | Not evaluated on AE; AUROC/FPR alignment unknown | Oracle covariance, independent Gaussian summaries, and no transfer-risk guarantee for feature-scale/OAS detector loss |
| C34 target-side OAS blocked-CV variance-target blend | Standard held-out Gaussian likelihood selects a local/peer OAS diagonal target; no novelty claim | ΔAUROC vs shared −.0136/−.0095/−.0074/−.0016 for n=30/100/300/1000; below on all 9 device means; improves over local | No device-level advantage over shared; featurewise safety is not tested | Selected peer weight is unstable across n and CV direction; no automatic fading | Peer feature scales only | Gaussian NLL is a proper model score, but serial/non-Gaussian support and AUROC mismatch remain | Same covariance can score AE inputs; not tested | Standard CV shrinkage selection; split sensitivity and lack of gain over shared |
| C35 joint mean/variance OAS validation blend | CV chooses local/peer means and diagonal variances jointly | No novelty; standard Gaussian predictive model selection | ΔAUROC vs shared −.0119/−.0087/−.0065/−.0014; 0/9 positive device means at n=100; improves local-only but fails shared | Weight choices remain non-monotone in n | Summary vectors only | NLL gains negatively track AUROC gains | AE not evaluated | Proper density loss does not match detector-ranking objective |
| C36 target-benign CV simplex over peers | CV optimizes 8 peer weights and pools full peer mean/variance summaries | Low; pFedBBN soft normalization-stat weighting and generic predictive model averaging are close | ΔAUROC vs shared −.0072/−.0023/+.0023/+.0010; all 9-device 95% t intervals include zero; held-out FPR remains .245 at n=1000 | Weight mass concentrates on ~1.6–1.9 peers; total external weight stays 1 | Needs all peer summaries to select; output is still 2d | No negative-transfer rule; CV likelihood weakly predicts AUROC | AE not evaluated | Small apparent gain over shared at n≥300, donor-specific harm and misspecified chronology |

**C35 joint mean/variance validation also fails to beat shared marginals.** The ignored `pocs/joint_oas_validation_blend_poc.py/.csv` extends C34 to a 5×5 grid: target mean is blended between the training-window mean and peer mean, while the OAS diagonal target is blended between local and peer variances. It uses the same two-direction blocked Gaussian NLL and paired windows. Mean AUROC deltas versus shared are −.01185/−.00874/−.00648/−.00142 at n=30/100/300/1000; positive device means occur on 1/9, 0/9, 1/9, and 2/9 devices. It improves over local by +.06023/+ .03131/+ .02262/+ .00882, but selected mean-weight β averages .40/.36/.39/.46 and variance-weight α .54/.46/.60/.52, with no decline as n increases. The detector-loss gap does not track the validation score: across same-window rows, Spearman correlations between validation-NLL advantage over shared and AUROC advantage over shared are −.227/−.381/−.459/−.519 by n. These are descriptive paired-window associations, not independent inference. The extension shows the C34 issue is not only a fixed local center: selecting both channels under Gaussian likelihood still does not recover the fixed shared baseline's detector ranking. Both selectors are standard model-selection procedures, not novelty candidates. Next revisit partner selection with target-specific weights over all peers rather than hard nearest-2; compare against equal pooling and all target-only endpoints, while treating learned normalization-stat weights as established prior art and guarding against severe small-n weight overfit.

**C36 target-benign CV peer weighting revisits source selection; it nears the bar only at larger n and remains unsafe.** The ignored `pocs/cv_peer_weight_selection_poc.py/.csv` uses SLSQP to choose a simplex over eight full-device moment summaries by two-direction held-out multivariate Gaussian NLL (the weights are shared across folds), then uses the resulting weighted mean and total variance in OAS. On 15 paired starts per device/support (540 rows; stable CRC32 seeds), AUROC deltas versus equal-weight `shared-marginals` are −.00721/−.00231/+.00226/+.00099 at n=30/100/300/1000; positive device means are 0/9, 3/9, 5/9, 7/9. Mean delta versus local-only is +.06487/+ .03774/+ .03136/+ .01123. Across nine device means, unadjusted t-based 95% intervals for the paired ΔAUROC are [−.0162,+.0018], [−.0079,+.0033], [−.0033,+.0078], and [−.0006,+.0026], all including zero; repeated window starts are not treated as independent devices. At n=1000 the aggregate result is only +.00099 despite 91/135 positive window rows, and the Provision PT 737E target is −.0117 averaged across all its rows; therefore there is no device-level no-harm property. The learned weights are concentrated: mean effective source count is 1.88/1.77/1.82/1.57 of 8 and mean maximum weight .71/.75/.73/.81. This does not show collaboration fading in total weight (the peer weights still sum to one); it concentrates borrowing onto a few donors. The optimizer succeeds on 97–100% of cells. Held-out likelihood improvement is a poor safety proxy: its rowwise association with AUROC gain over uniform pooling is Spearman .06/.01/.08/.15 by n, and sign agreement is only 31–67%.

A same-window operating-point replay is in `pocs/cv_peer_weight_operating_point_poc.py/.csv`. Each model's score threshold is the 95th percentile of that support window's in-sample benign scores; held-out benign FPR for local/selected/equal-share is .848/.718/.708 at n=30, .723/.617/.629 at 100, .516/.449/.461 at 300, and .275/.245/.252 at 1000. Attack TPR is 1.000/1.000/1.000, 1.000/1.000/1.000, 1.000/1.000/.999, and .999/.997/.996 respectively. Nearly every model/window exceeds 5% held-out benign FPR (local/selected/shared: 135/135/135 at n=30; 134/133/133 at 100; 133/131/132 at 300; 122/122/121 at 1000). Selection modestly improves this empirical operating point over equal pooling at n=100–1000, but leaves 24.5% FPR at n=1000 and essentially perfect attack recall, consistent with a saturated ranking benchmark and chronology mismatch. This is not a threshold-calibration guarantee because the train scores are reused to set the threshold and the support streams fail the earlier stability checks. Source-weighting from normalization descriptors is already prior art (pFedBBN 2025); the present NLL-optimized linear pooling is generic predictive model averaging. No selection rule is promoted. C38 below applies a proper support split and finite-sample conformal rank before any adaptive calibrator is considered.

**C38 proper split-conformal calibration respects the support data contract but fails prospectively under the observed chronology.** The ignored `pocs/split_conformal_oas_fpr_poc.py/.csv` uses the first two-thirds of each support window to fit local/shared OAS scores and the last third as benign calibration, with the exact split-conformal 95% rank `ceil((m+1)·.95)` and infinity when that rank exceeds the m calibration scores. On 15 paired starts ×9 devices ×four n, n=30 has m=10, so the cutoff is infinite for both models (0 FPR and 0 attack TPR on test); this is a finite-sample resolution limit, not useful detection. At n=100, 300, 1000 the median m is 34/100/334 and finite cutoffs exist. Calibration FPR is 0/.04/.0449 for both endpoints, as expected from the conservative order statistic, but future chronological test-benign FPR is local/shared .4609/.2901, .3758/.2894, and .2236/.2004; held-out attack TPR is .832/.9723, .9225/.9612, and .9313/.9624. For shared at n≥100, 102–118 of 135 cells exceed 5% test FPR. This is an exploratory transfer in time: the finite-sample split-conformal guarantee requires exchangeability between calibration and future benign scores, which is contradicted by C30/C31's second-moment instability and by these results. Standard conformal therefore cannot certify the operating point here.

**Adaptive score calibration is now a close prior, not an unexplored general direction.** W1-ACAS (Martinez Gil et al., ICLR 2026) accepts model-agnostic anomaly scores, maps them to weighted-conformal p-values using a calibration-score weight vector, and chooses weights by minimizing empirical 1-Wasserstein distance from p-value CDF to Uniform subject to an effective sample size constraint. Its weighted-quantile bound has a deviation term in weighted total-variation distances between the score sequence and sequences formed by swapping test/history scores; the bound is not numerically useful without controlling those terms. It directly preempts generic claims to post-hoc adaptive FPR-calibrated anomaly scores and can in principle wrap a FedORBIT Mahalanobis or AE score. It does not learn feature-level collaboration, and a monotone p-value transform does not by itself improve AUROC. CADES (Zhang et al., ICML 2025) instead uses time-rescaled continuous-time event sequences, two nonconformity scores, and Bonferroni correction for finite-sample conditional FPR control; its point-process likelihood assumptions/data modality differ from N-BaIoT's 115-dimensional packet-feature rows. Primary sources: [W1-ACAS ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/f54c9fc57aa6e1c72400cc127917fcf8-Paper-Conference.pdf); [CADES ICML 2025](https://proceedings.mlr.press/v267/zhang25dn.html); [time-series adaptive conformal analysis (Zaffran et al., 2022)](https://proceedings.mlr.press/v162/zaffran22a/zaffran22a.pdf). C39 has now probed benign-only score adaptation; next reproduce the official W1-ACAS update on a smaller score stream to isolate implementation effects, keeping ranking and FPR claims separate.

**C39 benign-only W1 score-weight compatibility probe does not fix the observed operating point.** The ignored POC `pocs/w1acas_benign_score_compatibility_poc.py/.csv` is an independent implementation of the paper's weighted upper-tail p-value and empirical W1-to-Uniform objective, not an exact copy of the Granite wrapper. It uses 9 devices × two detector-fit sizes (`n=100,1000`) × three contiguous starts × local/shared OAS models (108 rows). For each start, it fits the detector on a contiguous benign prefix, learns weights from the following 500 chronological benign support scores at nominal α=.05 (trailing score history ≤200; weight mass constrained to at least 19; batch=10; three projected Adam steps at lr=.001), freezes the terminal weight/history vector, and reports future test benign FPR and attack TPR. Adaptation uses only benign support rows; test attack labels are evaluation-only. The online benign-support alert fraction is .091/.078 for local/shared at n=100 and .062/.060 at n=1000, already above the target .05. Future test-benign FPR for equal-weight/W1 mappings is .1760/.1745 local and .1292/.1222 shared at n=100; at n=1000 it is .2538/.2511 local and .2412/.2389 shared. Average attack TPR changes by at most .0001. W1 moves shared FPR by only 0.2–0.7 percentage points and leaves it far above 5%; each of the nine device means remains above 5% for local, and seven/eight of nine for shared at n=100/1000. This indicates the same chronological transfer problem seen in C38, not evidence against the full W1-ACAS paper algorithm.

The frozen finite-history p-value map is intentionally scored separately from raw ranking. Equal-weight and learned-weight mappings have identical empirical AUROC here (both coarsen scores to finite calibration ranks); versus raw AUROC, the average equal/W1 p-value AUROC falls from .915/.970 to .884/.947 (local/shared) at n=100, and from .982/.994 to .912/.925 at n=1000. On average, 79% of attack scores exceed the benign calibration-history maximum, so many high scores collapse to the same minimum p-value; this is a resolution/chronology limitation of this frozen mapping, not proof that an online adaptive sequence must have the same AUROC. The POC is a small compatibility diagnostic, its batched projected optimizer is not an exact author-code replication, and no alarm-rate guarantee is claimed. The official method explicitly prefers a past score history that is ideally non-anomalous; a known contamination upper bound changes the p-value interpretation by an offset, while an unknown bound and the paper's total-variation term leave current drift uncontrolled. Next reproduce the official update on a smaller sequence to separate implementation effects from the benign-stream result, or move to another candidate family; do not promote adaptive conformal as a collaboration algorithm.

**C37 benign-only, distribution-free AUROC no-harm is not identifiable when attacks are unrestricted.** Let `P0` be the target benign distribution and `s` a score with randomized tie handling. Define its benign percentile `h_s(z)=Pr_{X~P0}[s(X)<s(z)]+0.5 Pr[s(X)=s(z)]`; then `AUROC(s;Q,P0)=E_{Z~Q} h_s(Z)` for attack distribution `Q`. For two candidate scores, if their benign-percentile functions differ at any attack-reachable region, choose `Q` concentrated where the proposed score's percentile is lower; its AUROC is then lower. Conversely, if one score were no worse for every unrestricted `Q`, its percentile function would have to dominate pointwise; since both percentile transforms are Uniform(0,1) under continuous `P0`, that dominance implies equality almost surely, so the scores induce the same benign ranking. This is a direct derivation for ranking loss, not a claim about the fixed N-BaIoT attack mixture. It explains why target-benign CV cannot provide a distribution-free AUROC no-harm guarantee when local/collaborative scores reorder examples. It aligns with Nalisnick et al.'s finding that anomaly identity is ill-posed without prior knowledge and perfect density estimation alone cannot guarantee correct anomaly detection ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8700034/)); that paper studies density/representation non-identifiability, while the percentile argument here concerns candidate ranking under unknown `Q`. Future safety claims must state an attack/threat distribution or move to a benign false-alarm objective with its own stable temporal calibration assumptions. C36's held-out FPR confirms the latter is not solved by its weight optimizer. C38 and C39 test static and adaptive score calibration respectively; both show that nominal calibration on support streams does not transfer to later N-BaIoT benign windows under the current chronology. Keep the nine-device attack AUROC exploratory and do not lock a protocol.

**C34 target-side multivariate OAS validation blend tested; it helps local-only but fails the registered bar.** The ignored POC `pocs/target_oas_validation_blend_poc.py/.csv` uses two-direction blocked cross-validation (train/validation size ratios 2:1 and 1:2; held-out NLL weighted by validation-row count) to select a variance-target interpolation `α∈{0,.1,…,1}` between local window variance and peer variance. It keeps the target support mean, refits the selected OAS model on all support rows, and compares paired test AUROC against exact-window local and shared-marginals endpoints (15 starts × 9 devices × four n; 540 rows, stable CRC32 seeds). The selected model beats local OAS by mean AUROC +.05852/+ .03053/+ .02166/+ .00864 at n=30/100/300/1000, but trails shared marginals by −.01356/−.00952/−.00744/−.00160; it is below the shared baseline on all 9 device means at every n. It exceeds shared on 50/41/36/37 of 135 rows. Mean selected α is .623/.496/.645/.580 and the peer endpoint is selected in 45.9%/40.0%/54.8%/51.1% of cases, so borrowing is not monotonically reduced as n grows. Direction matters: the earlier one-way holdout pass (superseded after identifying split sensitivity) had mean deltas vs shared −.0490/−.0252/−.0197/−.0041 and selected α median 0 through n=300; the final two-direction block CV gave the paired results above. The decision is unstable to a scientifically reasonable CV redesign. Held-out multivariate Gaussian NLL is more directly matched to the OAS covariance than C7's featurewise NLL, but remains misspecified for highly skewed serial streams and is not anomaly-ranking risk. This is known covariance-estimator selection, not a new method: Gaussian held-out likelihood is a standard shrinkage/CV criterion ([Zhou et al., 2011](https://jmlr.csail.mit.edu/papers/volume12/zhou11a/zhou11a.pdf)); recent structured shrinkage also selects targets by held-out NLL ([symmetry-aware shrinkage preprint](https://arxiv.org/abs/2605.17111)). Next test whether jointly selecting mean and variance interpolation with the same target benign likelihood changes the comparison; do not promote C34.

**Conformal mismatch-envelope redesign has an exact small-client resolution barrier.** Exchangeability testing for transfer is direct prior art: Zhou et al. define a conformal p-value for the null that source/target sequences are exchangeable and transfer only when the p-value clears a chosen significance level ([2017 paper](https://www.sciencedirect.com/science/article/pii/S0167865516303762); expanded equations in [their 2019 PMLR paper](https://proceedings.mlr.press/v105/zhou19a/zhou19a.pdf)). With eight peer devices plus one target profile, an exact client-level rank test has only 9 attainable ranks, so its smallest p-value is 1/9≈.111; a conventional α=.05 or .10 gate cannot reject peer incompatibility. At α=.20 it can reject at most the most extreme rank, with low resolution, and the validity would still require target/peer exchangeability that is not established for these nine heterogeneous, temporally unstable physical devices. Weighted conformal under covariate shift requires known or accurately estimated density ratios and does not remove arbitrary joint shift ([Tibshirani et al., 2019](https://arxiv.org/abs/1904.06019)). Federated conformal methods now explicitly handle heterogeneous client calibration/distribution shift, which preempts generic “conformal federated safety” novelty ([Shi et al., UAI 2026](https://proceedings.mlr.press/v337/shi26a.html); [Wen et al., UAI 2026](https://proceedings.mlr.press/v337/wen26a.html)). Thus conformal ranks cannot supply the missing C13 featurewise finite-sample mismatch envelope here; keep this avenue closed for the current nine-device contract, while continuing algorithm search.

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


## Latest continuation — C40 source-matched W1-ACAS score-wrapper audit

**C40 narrows the C39 result and still does not establish a reliable 5% operating point.** The ignored POC `pocs/w1acas_reference_update_poc.py/.csv` ports the public Granite `AdaptiveWeightedConformalScoreWrapper` recurrence for scalar OAS anomaly scores: 20 initial benign score values, per-score causal p-value calculation, rolling context of at most 200 scores, persistent AdamW state at lr=.001, batch context 10, one optimizer step per incoming score, the authors' L1-box projection, proximity initialization, and the code's finite-sample correction. It omits forecast generation and uses a bounded window of 200, so it is a source-matched core update rather than a full execution of the Granite forecasting wrapper. It runs 9 devices × n=100/1000 × 3 starts × local/shared models (108 rows). The detector is fit on a contiguous support prefix; the next 500 benign support rows seed/adapt W1; then 500 chronological rows from the dataset's known-benign test split drive online updates, followed by a separate 500-row benign holdout and the randomly sampled attack set evaluated against the same frozen post-adaptation state. No attack row/label enters weight adaptation. Since the experiment selects the known-benign test stream by the dataset's class split, it is a clean-stream diagnostic, not an operational demonstration that benign rows can be identified online.

Against equal-weight conformal scoring, online false-alarm fractions on the 500-row future adaptation stream move from .1152 to .1041 (local) and .0693 to .0616 (shared) at n=100; at n=1000 they move from .0619 to .0590 (local) and .0627 to .0590 (shared). But on the following frozen 500-row chronological holdout, shared FPR is .1044/.1053 (equal/W1) at n=100 and .1127/.1143 at n=1000; local is .1195/.1210 and .1233/.1267. Across shared rows, 15/27 exceed 5% for each mapping at both support sizes, and the Ennio shared-score device mean is about .53. The learned weights remain diffuse (effective count about 193–195 of 200). Attack TPR from this same frozen post-adaptation state is .9298/.9300 for shared at n=100 and .9847/.9847 at n=1000; ranking AUROC of the frozen conformal map is exactly the same for equal and learned weights in these paired cells, while finite-rank mapping is below raw-score AUROC. Thus the reference online update modestly reduces alarms while adapting on a clean stream, but does not transfer that reduction to the next benign block; the shift evidence persists. Attack TPR is only a separate frozen-state diagnostic because the provided attack order is randomized, so no temporal attack-stream claim is supported.

This supersedes C39's broader static-map interpretation: source-matched online updates materially reduce FPR during the benign adaptation segment, but the benefit does not persist on a later held-out block. W1-ACAS remains close prior art for generic adaptive p-value/FPR calibration, not a FedORBIT collaboration mechanism. Its paper's nonexchangeable bound contains unknown score-sequence total-variation terms and it expects past scores to be ideally non-anomalous; neither issue is repaired by the current experiment. Primary references are the [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/f54c9fc57aa6e1c72400cc127917fcf8-Paper-Conference.pdf) and [public scalar update implementation](https://github.com/ibm-granite/granite-tsfm/blob/main/tsfm_public/toolkit/conformal.py).

**Historical next-task note, completed by C41–C45 below.** The peer-risk calibration audit and target-local HAC/bootstrap follow-ups are complete as exploratory diagnostics. The current continuation task is recorded at the end of C45.


## C41 — target-specific risk calibration and oracle-weight diagnostic for C10

**C41 directly falsifies C10's peer-to-target local-risk transport assumption, but its oracle correction does not establish an AUROC win.** First, `pocs/c10_peer_risk_target_calibration_poc.py/.csv` compares C10's leave-target-out mean peer block-variance risk (`Vpeer`) with the target device's own variance of disjoint chronological block variances, measured separately on its early and late benign support halves. Across target/device halves, the median `Vpeer / Vtarget` ratio is 1.32/2.49 at n=30, 1.20/1.73 at n=100, 1.55/5.59 at n=300, and 3.99/7.55 at n=1000 (early/late). Peer risk exceeds target risk on roughly 64–80% of feature coordinates, depending on n/half. Consequently, mean C10 feature weight is .533/.495/.398/.297, versus target-risk-only weights .402/.366/.304/.192 using early and .375/.342/.264/.162 using late target block risk. This is direct evidence that the peer risk proxy often inflates the local-risk numerator and therefore borrows more than a target-specific risk estimate would. At n=1000 the target-half estimate is based on only about 16 complete blocks, so its large ratios are noisy and descriptive; at n=30 the many blocks give a more stable mismatch signal.

Second, `pocs/c10_target_risk_oracle_poc.py/.csv` uses target risk from the early half (much more target data than the deployed n-row cold start) and evaluates randomly selected n-row windows from the later half, with 15 paired starts per device. Against exact shared-marginals, mean C10 AUROC deltas in this different chronological design are +.00347/+.00278/−.00144/−.00162 at n=30/100/300/1000. Replacing only its local-risk numerator with the target-history oracle gives +.00628/+.00466/−.00018/+.00004; adding a squared-discrepancy penalty gives +.00498/+.00266/−.00046/−.00040. Across the nine device means, the unadjusted t-based 95% intervals include zero for every method and n. Target-risk-only beats C10 on 7/9, 5/9, 7/9, and 5/9 device means by n, but it does not consistently beat shared-marginals; the bias-penalized version is weaker at n=100/300/1000. These results differ from the earlier C10 benchmark because support training windows are now restricted to the later chronology while risk is estimated on the earlier half, so do not pool the AUROC deltas across designs.

C41 therefore identifies a meaningful C10 failure mode (cross-client local-risk calibration), yet the nondeployable target-history oracle only nudges mean AUROC and leaves device uncertainty large. It is a mechanism diagnosis, not a candidate result.

## C42 — target-only Andrews–Bartlett HAC risk calibration

The ignored POC `pocs/c10_target_hac_risk_poc.py`, with raw outputs `c10_target_hac_risk_results.csv` and `c10_target_hac_risk_calibration.csv`, tests a deployable target-local risk estimate using only n benign rows per risk window. For each feature it forms the sample-variance influence series `q_t=(x_t−x̄)^2−s²`, estimates its long-run variance with a Bartlett HAC kernel and Andrews (1991) AR(1)-plug-in bandwidth `b=1.1447(α(1)n)^(1/3)`, and divides by n. Risk is the median over 15 seeded windows from the first chronological half; peer predictive risk remains C10's `Bpred`. Calibration compares this estimate, an iid Gaussian plug-in, and C10's peer risk to variance across disjoint n-row target blocks in the later half. The OAS comparison uses 15 seeded later-half windows ×9 devices ×4 support sizes, with identical 5,000-row benign and attack evaluation subsets across methods for tractability. These are exploratory, not confirmatory; use of known dataset test classes is limited to offline evaluation.

The target HAC estimator fails risk calibration decisively. Its median featurewise predicted/future risk ratios across target devices are .045/.037/.129/.049 at n=30/100/300/1000, with 84–94% of compared features underpredicted. The iid Gaussian plug-in is also low (.067/.043/.036/.024), so this is not just a bandwidth misspecification. The C10 peer proxy is much closer to one in the aggregate (median ratios 2.49/1.73/5.59/7.55, with large device variation), which explains why replacing its numerator with the HAC estimate sharply reduces peer weight: mean HAC weight .080/.043/.053/.036 versus C10 .533/.495/.397/.297.

Against exact same-window `shared-marginals` in this later-half split, HAC-risk blend mean AUROC deltas are −.05847/−.04117/−.02001/−.01290, positive device means 0/1/1/0 of 9, and worst device deltas −.1862/−.1572/−.1099/−.0779. The unadjusted device-level 95% interval excludes zero only at n=30 (−.1111,−.0058); the other intervals include zero. C10 in the same design gives +.00349/+.00280/−.00144/−.00141; paired results differ from C41's 15-replicate full-test-set benchmark because this sweep evaluates a fixed 5k+5k test subset. HAC behaves close to local-only and is not a viable C10 correction.

Failure analysis: Andrews–Bartlett HAC estimates variance under a locally stationary influence process, but the actual risk target here is variance across later chronological support blocks. The N-BaIoT windows contain slow distribution changes and heavy tails that are absent from most n-row influence windows; HAC and iid risk both miss that between-window component. This rejects this HAC instantiation, not all target-local finite-window diagnostics. C43 and C44 below test whether automatic dependence-aware block selection fixes the miss and isolate the stationary versus switching mechanism. No production method is selected and no confirmation protocol is locked.

## C43 — corrected Politis–White circular block risk estimate

The ignored POC `pocs/c10_target_pw_block_risk_poc.py` tests featurewise circular block lengths selected by the Politis–White (2004) automatic rule, including the Patton–Politis–White (2009) correction, on the same variance-influence series. The circular overlapping-block variance estimate is computed with its equivalent Bartlett weighted autocovariance form. As in C42, each estimate uses n target benign rows, risk is aggregated over 15 early-half windows, and calibration/evaluation use later chronology; the AUROC screen uses paired 5k benign/5k attack evaluation subsets.

It also underpredicts later target-block variance risk: median featurewise predicted/future ratios .033/.073/.104/.029 at n=30/100/300/1000, with 82–95% of features underpredicted. Median selected block length grows 1/3/10/15 rows. The estimated HAC/bootstrap weight averages .069/.052/.057/.040, far below C10's .533/.495/.397/.297. Mean AUROC deltas versus exact shared marginals are −.07087/−.03648/−.01325/−.00931, positive device means 0/2/1/0, and worst-device deltas −.2032/−.1564/−.0579/−.0460. The n=30 device-level 95% t interval excludes zero on the negative side; the other three include zero. This is not a deployable correction.

The block selector is not simply picking tiny lengths everywhere: it grows with n and reaches 44 rows in the switching synthetic case at n=1000. Its failure on N-BaIoT is consistent with nonstationarity beyond the single observed target window, which cannot be inferred from local autocovariances alone. The equations and block-length selector are established bootstrap statistics, not novelty. See the direct [Politis–White paper](https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf) and their [2009 correction](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf).

## C44 — synthetic calibration separates stationary dependence from switching variance

The ignored `pocs/target_variance_risk_synthetic_calibration.py/.csv` runs 1,500 Monte Carlo n-row windows for each n and compares iid fourth-moment risk, Andrews–Bartlett HAC, and Politis–White circular risk against the realized Monte Carlo variance of the unbiased sample variance. In stationary Gaussian AR(1) data (ρ=.7), HAC's mean predicted/true risk ratio rises .478/.702/.766/.976 for n=30/100/300/1000; circular block risk gives .432/.672/.759/.975. The iid plug-in stays at .300–.373. Thus both dependence-aware methods approach calibration as stationary support grows.

In a two-state AR(1) process with innovation variances .25 and 2.25 and per-row switch probability .01, the same ratios fall to .176/.167/.198/.235 for HAC and .158/.162/.223/.361 for circular risk. A single n-window often shows one regime and cannot estimate the unconditional future-window risk that includes movement between regimes. This controlled result supports chronology/regime heterogeneity as a material C42/C43 failure mechanism; it does not prove every real device follows a Markov-switching law. It explains why tuning a stationary HAC bandwidth cannot repair the real-data calibration gap.

## C45 — target-window CUSUM as a predictor of later risk miss

The ignored `pocs/target_cusum_future_risk_probe.py/.csv` compares each feature's normalized variance-influence bridge CUSUM from one early n-row target window against the later chronological target-block variance risk and its ratio to the Politis–White estimate. It uses 15 paired early windows per target/support size, 9 devices, all 115 features, and no attack data.

The signal is not stable enough to define a collaboration rule. Median within-window Spearman correlation between CUSUM and log(future target risk / PW risk) is −.121/−.038/−.038/−.049 at n=30/100/300/1000; only 2/9 device medians are positive at n=30 and 4/9 at each larger n. The top-CUSUM feature quintile's median log risk ratio is lower than the bottom quintile's by −1.479/−.157/−.544/−.865, but both groups remain strongly underpredicted: the top-quintile median log ratios are still 3.749/3.679/3.253/2.864. Direction and size vary by device. This is only a feature-level prognostic screen; no significance threshold, rejection rate, or CUSUM-based OAS rule was evaluated. A within-window shift signal cannot reveal a future regime change that has not yet appeared.

The C42–C45 path now establishes that local stationary risk estimation can work on controlled stationary AR data, while it cannot anticipate future regime occupancy from one short support window. Do not turn CUSUM into a gate based on this association. Resume the broader candidate audit from first principles: revisit featurewise scale-only collaboration under explicit target/peer discrepancy and contaminated-peer conditions, and search for a meaningful Pareto improvement over fixed `shared-marginals`; update the hostile literature comparison as the mechanism changes. Keep all results exploratory and the confirmatory protocol unlocked.

## C46 — featurewise robust-scale summaries under discrepancy and one-peer contamination

The ignored POC `pocs/robust_target_peer_contamination_oas_synthetic.py/.csv` provides a paired mechanism screen for 64 correlated Gaussian features, eight peers with 2,000-row scale summaries, target support `n=30/100/300/1000`, five scenarios (matched clean, peer log-SD heterogeneity .35, one peer at 4× or .25× variance, and a target at 2× peer scale), and 120 seeded replications per cell. Each candidate sees the same OAS test benign/attack samples and production scorer. This is synthetic diagnostic evidence, not a result on the nine physical devices.

Robust C12's scale-risk blend lowers variance RMSE relative to fixed shared marginals under ordinary peer heterogeneity (for n=30/100/300/1000: .826/.757/.752/.700 vs 1.38/1.23/1.34/1.30) and a 4×-variance outlier (.842/.738/.701/.744 vs 1.95/1.98/1.97/2.04). That does not yield an AUROC gain: paired mean ΔAUROC vs shared is −.0061/−.0023/−.0005/−.0005 under heterogeneity and −.0058/−.0037/−.0008/+.0001 with the high outlier; the candidate beats shared in only 39–48% and 35–51% of replicates, respectively. With a .25× peer, shared marginal variance is already preferable by RMSE at most n, and C12 is worse. On the 2× novel target, C12+discrepancy brings RMSE closer to local (4.53/3.66/3.13/2.91 vs shared 9.41/9.29/8.75/9.00) but its AUROC remains below shared by .0065/.0029/.0003/.0007. In matched-clean cells its discrepancy estimate is unstable (RMSE .67–.78 against .03–.04 for shared), because featurewise squared-discrepancy minus noisy estimated risks is itself noisy. These results verify the existing C10–C12 conclusion: robust peer aggregation can protect a variance estimator from an outlier, but lower scale MSE does not establish detector-ranking benefit or a no-harm rule.

## C47 — sequential covariance-aware multi-source shrinkage on variance summaries

The ignored POC `pocs/sequential_source_scale_shrinkage_synthetic.py/.csv` adapts Jing et al.'s 2026 Algorithm 1 to vectors of featurewise sample variances. It has 64 correlated Gaussian features, eight source summaries, `n=30/100/300/1000`, 60 paired replications per cell, and the same five scenarios and OAS attack construction as C46. It uses the exact Gaussian Wishart covariance of each variance-summary vector and supplies those covariance matrices to the method as an oracle; peer mean summaries are also simulated and included in the fixed shared-marginals baseline. The pooled single-step rule, source-by-source priority updates, robust peer median/trimmed summaries, local-only, and shared marginals all use paired target/test draws.

The sequential estimator's mean paired ΔAUROC vs shared-marginals at n=30/100/300/1000 is: matched clean −.00315/−.00278/+.00079/−.00013; heterogeneous −.00729/−.00003/+.00042/−.00081; one 4×-variance peer −.00849/−.00156/+.00013/+.00051; one .25×-variance peer −.00354/+.00148/−.00039/−.00060; and 2×-scale novel target −.00706/+.00004/+.00079/−.00049. It wins 28–55% of paired replications across these cells; its paired ΔAUROC 10th percentile ranges from −.0306 to −.0045. Thus source sequencing does not produce a meaningful or stable OAS ranking gain over the fixed baseline. Yet it reduces variance RMSE in several heterogeneous cases: for a 4× outlier, .693/.733/.757/.657 vs shared 2.16/2.40/2.06/1.99; for the novel target, 5.22/3.17/3.12/2.70 vs 9.24/8.44/9.15/8.61. Under matched-clean peers, sequential RMSE .106/.214/.480/.626 is above the shared summary's .033/.035/.035/.033, showing the selected adaptive updates do not recover the stable peer-pooling efficiency even when all sources match. It assigns very small mean gamma to the flagged 4× peer (.0049/.0035/.0032/.0028) while retaining larger aggregate homogeneous-peer gamma (.665/.341/.154/.071), so the source-ordering mechanism is visible, but not sufficient for detector utility.

This is a controlled adaptation of established prior art, not a novelty claim. Jing et al., “Tuning-Free Covariance-Aware Sequential Shrinkage for Multi-Source Estimation,” arXiv:2606.30615v2 (2026-09-21), already provides covariance-aware update directions, a risk-improving shrinkage interval, source-priority ordering, and an updated covariance proxy ([full paper](https://arxiv.org/html/2606.30615v2)). Its theory is for independent Gaussian mean estimators and smooth M-estimation, with a mixed-source separation/effective-dimension regime; it is not a theorem for serial sample-variance vectors or anomaly AUROC. C47 is unusually favorable to the adaptation because it supplies exact covariance matrices. Real use would need stable target/source covariance estimates, matrix operations over 115 features, and a valid loss connection; these are not currently available under the cold-start benign-only contract. C24 already screened the paper's pooled two-summary update, and C47 now screens the sequential extension. Neither transfers its estimation-risk guarantee to the OAS detector objective.

C46–C47 therefore close this specific scale-robustness/sequential-SURE branch for now, but do not close the algorithm search. They reinforce the loss mismatch: variance-estimation improvements under synthetic contamination and target discrepancy have not improved the paired OAS ranking over fixed shared marginals. Earlier empirical-Bayes mixture C14 and serial-regime C15–C27 already cover adjacent random-effects and switching-variance routes, so avoid rerunning those exact models. Continue with the remaining live candidate question: whether a benign-only statistic can predict *detector-ranking* harm or benefit under a specified attack family without reusing attack labels, and separately whether lower communication at near-tie quality is a defensible Pareto result. Any next experiment should state its estimand and attack-family limits before tuning; do not claim arbitrary-AUROC safety (see C37). No protocol is locked and no confirmatory run has started.

## C48 — six-seed local-center/peer-scale AE compatibility and payload audit

The ignored pocs/autoencoder_peer_scale_poc.py/.csv extends the earlier two-seed smoke to six paired support draws per device at n=30/100, training the same 300-step local autoencoder three ways: local center/local scale, local center/peer scale, and full shared mean+scale. It covers all nine N-BaIoT devices and uses the same fixed benign/attack test rows per device for every condition; the test labels are used only for offline AUROC. This is a model-compatibility screen, not confirmatory inference: there are six random support draws per device, but only nine fixed physical-device/test populations.

Local-center/peer-scale versus full shared marginals has mean ΔAUROC −.00510 at n=30 and −.00138 at n=100; paired window wins are 38.9% and 51.9%, positive device means 2/9 and 6/9, and worst device means −.03811 and −.03106. Unadjusted t intervals across the nine fixed device means are [−.01492,+.00472] and [−.01083,+.00806]. Relative to local normalization it gains +.07876/+ .05847, while full sharing gains +.08386/+ .05985. Thus scale-only transfer retains most average AE gain at these n, but has visible device-specific negative deltas and the six-seed screen does not establish non-inferiority to full sharing. Compared with the two-seed smoke, the larger seed set removes the apparent tiny positive mean delta versus shared.

The POC forms its scale with `aggregate_equal_weight` over per-peer `MomentSummary` objects. That pooled total scale includes between-peer mean offsets, so reproducing it requires each peer to contribute mean and variance information (at least 2d scalars); the experiment establishes no peer-to-coordinator upload reduction. After the pooled scale is computed, the target-facing normalization state is d=115 scale values (460 bytes as float32) instead of 2d=230 mean-plus-scale values (920 bytes) for full shared marginals, a 50% reduction in that downstream vector only. These are arithmetic counts, not measured wire bytes: the POC keeps full covariance matrices in memory and does not measure framing, cryptography, transport, or protocol overhead. Sending only d per-peer within-device scales could reduce uploads, but would define a different estimator and is not the method evaluated here. UniFed and client-agnostic FL normalization already preempt broad unseen-client normalization novelty, while the scale-only raw-input transform is an interface-level adaptation only. No update was made to the AE architecture or FedAvg path.

This strengthens the claim that scale-only collaboration can be consumed by both OAS and a local AE, but not the stronger claim that it is model-agnostic, safe, or as accurate as shared marginals. The closest attack-specific outlier-detection prior found for the next threat-model audit is Kalan, Neugut & Kpotufe, “Transfer Neyman-Pearson Algorithm for Outlier Detection” (AISTATS 2025; [PMLR paper](https://proceedings.mlr.press/v258/kalan25a.html)). It studies transfer with rare target abnormal data and guarantees over changes in abnormal distributions, so it is adjacent to attack-distribution-aware selection but does not satisfy this benign-only cold-start information contract. Continue by stating a threat family explicitly and determine whether a benign-only plug-in expected-ranking criterion adds information beyond an assumed attack prior; compare against this prior art and test reversal under a neighboring attack family. Do not make a distribution-free AUROC safety claim. Protocol remains unlocked.

## C49 — benign-only expected-AUROC scale choice depends on the threat prior and does not beat shared

The ignored POC pocs/benign_only_threat_model_selection_poc.py chooses among local-center scale blends
v(λ)=(1−λ)s²_target+λ median_k(s²_peer,k), λ∈{0,.25,.5,.75,1}, and full shared-marginals.
The threat family is fixed before fitting: either (A) sparse additive mean shifts on six of 64 features,
independent random signs, magnitude .8 × robust peer SD, or (B) sparse scale inflation on six
features by multiplying residuals by two (fourfold variance). There are four synthetic populations:
matched clean, heterogeneous peers, one 4×-variance peer, and a target with 2× peer scale. There are
40 paired draws per cell, eight peers with 2,000-row scale summaries, and production OAS. Selection uses
only the n target-benign support rows, peer summaries, and synthetic perturbations; a three-fold
out-of-fold protocol keeps each benign validation row out of its scoring-model fit. Held-out simulated
Gaussian benign/attack draws are shared across candidate actions and selection rules and are evaluation
only.

The first in-sample-bootstrap implementation was invalid: it scored bootstrap copies of support rows
used to fit the detector, and overestimated selected-versus-shared AUROC by +.00668 for sparse shifts
and +.00516 for variance inflation. Its ignored raw file is preserved as
pocs/benign_only_threat_model_selection_resubstitution_diagnostic.csv; the corrected script and final
CSV are pocs/benign_only_threat_model_selection_poc.py and
pocs/benign_only_threat_model_selection_results.csv.

Cross-fitting reduces but does not eliminate selector optimism. The mean out-of-fold predicted gain
over shared is +.00116 for sparse shifts and +.00187 for variance inflation; actual held-out gains are
−.00018 and −.00145 (prediction-minus-evaluation gaps +.00134/+ .00332). Against the matching held-out
threat, the selected action beats shared in only 24.7%/7.5% of paired draws, with lower-tail
ΔAUROC quantiles −.00182/−.00535; the selected method's cell-average is below shared in 10/16 and
16/16 scenario/support cells. For the sparse mean-shift prior, matching-family cell means range
−.00158 to +.00128 across settings; for the variance-inflation prior they are negative in every cell
(n=30 losses range −.00340 to −.00487 across scenarios, approaching zero by n=1000). A neighboring
family check further weakens transport: sparse-shift-selected actions lose to shared on all 16
variance-inflation cells (mean −.00170), while variance-inflation-selected actions lose on 12/16
sparse-shift cells (mean −.00059). The selector chooses different conditional weights (mean λ=.223
vs .699 among non-shared choices) but does not convert that adaptation into reliable AUROC benefit.

This is not a general benign-only ranking predictor: its positive simulated score advantage fails on
fresh draws, especially under scale inflation. C37 still rules out distribution-free AUROC no-harm.
Closest prior art now includes Kalan et al.'s TLNP, which tunes a target/source abnormal-data loss
under a Type-I error range (it has scarce target anomalies, unlike this contract), and normal-only
synthetic anomaly/perturbation methods. Schlüter et al.'s NSA creates labeled image anomalies from
normal images via Poisson-blended patches and reports that perturbation parameters encode assumptions
about unknown anomalies ([paper](https://arxiv.org/abs/2109.15222)); 2026 PCU trains a tabular
representation on controlled corruption magnitudes to measure epistemic uncertainty
([paper](https://proceedings.mlr.press/v337/allaoui26a.html)). These preempt generic novelty for
perturbation-defined threat families. C49's specific issue is whether such a family can select a
peer/local scale action for a new IoT device, but it has no observed gain and no novelty claim. C50
now tests a no-mixture-weight redesign: maximize the minimum out-of-fold AUROC across the two declared
threats, then evaluate both separately. Do not treat this two-family set as exhaustive.

**Attack-label oracle diagnostic.** C49 also records, strictly as an evaluation upper bound, the best
λ from the five scale blends under each actual held-out attack family. Even this unavailable
attack-label oracle has mean AUROC change vs full shared marginals −.00015 for sparse shifts and
−.00197 for variance inflation; eight of 16 and 15 of 16 scenario/support cell means are negative,
respectively. Under the variance-inflation family, no choice from the local-center scale-blend grid
matches the fixed shared baseline in 15/16 cells. This identifies an action-space limitation as well
as selector noise; it is not an available training signal.

## C50 — maximin out-of-fold threat-set selection does not improve on shared

The ignored pocs/benign_only_threat_set_robust_selection_poc.py chooses the action maximizing the
minimum of the two C49 out-of-fold pseudo-AUROCs across sparse mean-shift and sparse variance-
inflation attacks. Its candidates and four populations are unchanged, and it adds no hand-chosen
threat-mixture weight. It uses 40 paired synthetic replications per scenario/support cell, then
evaluates each chosen action on both held-out threat families.

Against shared marginals, maximin selection has mean ΔAUROC −.00018 on sparse mean-shift and −.00170
on variance-inflation draws, wins 24.7%/9.2% of paired draws, and has 10/16/16 of 16 cell means below
shared. Its 10th-percentile paired differences are −.00182/−.00650. The procedure chooses full shared
in 49.7% of draws; its mean selected non-shared λ is .223, so it lands close to the sparse-shift-only
selector and does not improve the variance-inflation failure. This controlled maximin rule therefore
does not establish Pareto improvement or negative-transfer protection, even for the declared two-
family set.

C49–C50 provide a sharper statement than the initial benign-only rank-prediction question: this OAS
scale action family has little headroom over shared marginals for the declared synthetic attacks,
and the empirical out-of-fold maximizer fails to identify the small conditional differences. The
simple five-point scale blend is also not rescued by an oracle λ on variance-inflation attacks. Do not
refine its Monte Carlo selector as if it were the missing algorithm. Continue the broader search on a
different estimand or object—for example, featurewise score-preserving calibration or threat-
structured detector robustness—with a concrete mechanism and a hostile prior-art comparison before
claiming improvement. No confirmation protocol is locked.

## C51 — diagonal top-k residual scores lose to shared OAS in sparse and dense screens

The ignored `pocs/sparse_score_geometry_poc.py/.csv` asks whether the negative C49–C50 scale-selection
result is only a restriction of the OAS ranking loss. It uses the production `scorer_from_rows` for
local and shared OAS. A predeclared comparison family replaces the quadratic score with the mean of
the largest k coordinatewise squared standardized residuals, k∈{1,4,8,16,64}; it crosses local or
shared centers with local, median-peer, or pooled-shared scales. No attack labels choose among scores.
There are 64 correlated features, eight peer summaries from 2,000 rows, 40 paired repetitions in
each of four support sizes and four peer/target populations, and shared benign/attack evaluation draws
for four declared perturbations: six-feature ±0.8-SD mean shifts, six-feature 4× variance, dense
±0.2-SD mean shifts, and 1.1× dense residual scaling. This is a synthetic mechanism screen, not an
all-nine-device or confirmatory evaluation.

Shared OAS mean AUROC is .6121/.8101/.5758/.7489 for sparse mean, sparse variance, dense mean, and
dense variance respectively. OAS using target-local center/scale stays close under mean shifts
(ΔAUROC −.0010/−.0007 for sparse/dense) but trails by −.0037 for sparse variance and −.0013 for dense
variance. Every fixed top-k diagonal score has lower mean AUROC than shared OAS in all 16
scenario/support cell means for sparse mean and both dense families. The strongest predeclared
featurewise version for sparse mean (top-4, shared center/local scale) still averages −.0519 AUROC,
wins 1.9% of paired draws, and has 10th-percentile Δ −.0877. For sparse variance the strongest is
top-1 with shared center/local scale: mean Δ −.0168, wins 17.5%, 10th percentile −.0364. In that
family, local OAS itself wins only 15.8% of paired draws; no top-k version closes its gap to shared
OAS. Dense-family top-k results are also substantially below baseline (best mean deltas about
−.0387/−.0671).

The failure points to loss geometry rather than an obvious benefit from diagonal sparsity: in these
moderately correlated 64-feature populations, OAS's covariance regularization improves ranking, while
max/top-k coordinate scores pay a multiple-comparison/noise cost. Borrowing peer or pooled scale does
not fix the gap; the strongest top-k variants use local scales. No payload/accuracy Pareto benefit is
shown because the accuracy loss is large. The score family is only a first sparse-aware screen; it
does not rule out rank-HC, covariance-aware sparse scans, or other dependency-aware detectors. Hostile
prior art includes Donoho–Jin higher criticism and Stoepker et al.'s 2025 rank-based higher criticism
across referentials. The latter uses coordinate/referential ranks and the statistic
`T(R)=max_q (N_q(R)-np_q)/sqrt(np_q(1-p_q))`, with finite-sample null calibration by Monte Carlo;
it handles a different unit/independence model but preempts generic sparse-threshold aggregation.
Accordingly C51 is a negative diagnostic, not a FedORBIT candidate or novelty claim. Continue to a
dependency-aware threat score or a different target estimand; do not tune k on these attack labels or
promote this diagonal score family.

## C52–C53 — precision-adjusted sparse score helps one synthetic threat, then loses on real pooled attacks

C51's diagonal score failure suggested separating correlation handling from sparsity. The ignored
`pocs/covariance_aware_sparse_score_poc.py/.csv` tests the one-coordinate Gaussian score/GLR statistic
`S₁(x)=max_j ((Ωr)_j²/Ω_jj)`, where `r` is the support-centered, feature-scaled observation and `Ω`
is the OAS precision fitted to the same benign support under local, peer-scale/local-center, or full
shared normalization. For fixed known Gaussian covariance this is the maximum one-coordinate mean
shift likelihood-ratio score. C52 uses 64 correlated features, eight peer summaries of 2,000 rows,
four support sizes, four mismatch/contamination populations, 40 paired draws/cell, and 800 benign and
attack evaluation observations per threat. It compares this score with production OAS on the same
test draws and includes one-feature ±1-SD and six-feature ±0.8-SD mean shifts, six-feature 4×
variance, and two dense perturbations.

The result is threat-conditional. On six-feature variance inflation, shared-normalized `S₁` exceeds
shared OAS by mean AUROC +.0128 (80.3% paired wins, 10th percentile −.0045), and local-center/
peer-scale `S₁` gains +.0110 (78.6% wins, 10th percentile −.0055); both have positive mean deltas
in all 16 scenario/support cells. Full shared OAS mean AUROC is .8108 for this threat. Under a
one-feature mean shift, local-center/peer-scale `S₁` loses .0052 on average (35.6% wins); for six-
feature mean shifts it loses .0322 (5.2% wins). Dense mean and variance losses are .0306 and .0951.
The local OAS score is near shared for sparse mean but loses .0035 on sparse variance. Thus the
precision-adjusted statistic repairs C51's synthetic sparse-variance weakness, but it is not a
general sparse threat score and the gain comes from a specified score/attack match.

The direct hostile prior is Zhang, “Testing High Dimensional Mean Under Sparsity” (arXiv:1509.08444,
version dated 2026-03-22). For a Gaussian sample with precision `Γ`, it derives the maximum
subset likelihood-ratio statistic `LR_n(k)=n max_|S|=k Z_Sᵀ Γ_SS⁻¹ Z_S`, with a feasible
dependence-aware diagonal approximation `T_n(k)=n max_|S|=k Z_Sᵀ diag(Γ_SS)⁻¹ Z_S`; at k=1 this
reduces to `n max_j Z_j²/Γ_jj`, where `Z=Γ X̄`. It also proposes choosing k with a standardized
maximum across k and simulation-based null calibration. C52's one-observation detector use is a
different application, but the score mechanism itself is direct prior art; no novelty claim remains.
The full equations were inspected, including the assumptions for feasible precision estimation.

Because the synthetic gain was paired, consistent, and used a featurewise transfer interface, C53
screened the rule across all nine physical N-BaIoT devices before dismissing it. The ignored
`pocs/covariance_aware_sparse_nbaiot_poc.py/.csv` uses 15 paired contiguous support windows at each
n=30/100/300/1000, fixed shared held-out subsets of up to 5,000 benign and 5,000 attack rows per
device, and production `scorer_from_rows`. It compares full shared OAS, local OAS, local-center/
median-peer-scale OAS, and each corresponding precision-max score. Across the pooled held-out attack
mixture, the median-peer-scale OAS ΔAUROC vs shared is −.0168/−.0030/−.0002/−.0012; its paired
window win shares are 50.4%/51.1%/57.0%/68.9%, but only 3/9, 3/9, 2/9, and 5/9 device means are
positive, with worst device means −.0527/−.0131/−.0023/−.0173. The precision-max peer-scale score
loses by −.0114/−.0112/−.0121/−.0190, wins only 17.8%/15.6%/11.1%/14.1% of paired windows, and
has no positive device-mean at any support size (worst device −.0401/−.0293/−.0262/−.0571). Its
full-shared counterpart gains a negligible +.0018 at n=30, then loses −.0073/−.0122/−.0201 as n
increases. Mean full-shared OAS AUROC is already .9783/.9826/.9916/.9942, so ceiling limits some
headroom, but the peer-scale max-score regressions at n≥100 remain material.

C52–C53 therefore identify a valid but established score test with one narrow synthetic operating
regime, followed by negative real-device evidence. This is not the missing general FedORBIT algorithm.
The formula offers O(d) scoring after a precision fit, but C53 shows no detection or device-safety
advantage; the peer-scale message payload was not separately benchmarked here. Leave the score as a
threat-specific diagnostic. Continue on another estimand or collaboration-risk object, with a
different prior-art search before another detector-score variant; protocol remains unlocked.

## C54 — attack-family stratification does not recover the sparse-score gain

Because C53 pooled many attack classes, C54 re-runs the same paired methods separately on the two
N-BaIoT attack families. The ignored `pocs/sparse_score_nbaiot_family_poc.py/.csv` rebuilds each
family's attack rows from the raw files using the repository's configured attack sampling seed and
then verifies exact array equality against the prepared `test_attack`; the labels are therefore
grounded in the actual artifact order rather than guessed from row values. It retains C53's fixed
per-device held-out benign/attack subsamples, 15 paired support windows at n=30/100/300/1000, and
shared OAS, local-center/peer-scale OAS, and their precision-max score variants. Evaluation labels
are used only for family-specific AUROC.

The precision-max peer-scale score remains below shared OAS for both families at every support size:
Gafgyt mean ΔAUROC −.0117/−.0113/−.0126/−.0217 and Mirai −.0085/−.0115/−.0114/−.0155. Its paired
window win shares are 16.3%/15.6%/8.9%/12.6% for Gafgyt and 40.0%/30.5%/23.8%/21.9% for Mirai.
The full-shared precision-max score produces only tiny gains at n=30 (+.0015 Gafgyt, +.0033 Mirai),
with positive device means on just 1/9 and 2/9 devices, and loses for both families at n≥100. Thus
the C53 regression is not just cancellation between Gafgyt and Mirai. C54 provides no real attack
family in which peer-scale precision-max beats the fixed shared OAS baseline on average.

C52's synthetic six-feature variance result therefore does not transport to either real attack
family under this chronology. Retain C52–C54 as a scoped mechanism diagnostic plus a failed transfer
attempt. The next search should change the objective or collaboration-risk object, not add another
unvalidated sparse score. Existing temporal FPR calibration and C37 AUROC-identifiability results
remain constraints; do not treat the move as permission to make an arbitrary-attack safety claim.

## C55 — deep-normalization literature closes broad statistic-sharing compatibility claims

A fresh primary-source pass revisited whether a scale-only collaboration layer could claim general
deep-detector compatibility or federated normalization as a contribution. It cannot. Population
Normalization (Wang et al., CVPR 2025) replaces batch-estimated activation moments with trainable
population mean/scale parameters. Its normalization is
`(x−μ)/sqrt(γ²+ε)` followed by learned affine parameters; the population constraints set the
normalized second moment to one and the population mean to `μ`. The `μ,γ` parameters are trained
alongside model weights locally and aggregated by FedAvg. Its Noisy Population Normalization variant
perturbs `μ` and `γ` multiplicatively during training to restore stochastic regularization. This is
close to federated statistics in deep models, but it is learned hidden-layer normalization inside a
supervised training objective, not a new physical client's raw-input benign estimator or anomaly
ranking rule. Primary source: [CVPR 2025 paper and official open-access PDF](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Population_Normalization_for_Federated_Learning_CVPR_2025_paper.html).

FedFD-A (Yang et al., ECCV 2024) is closer to transfer at inference: for every layer it randomly
interpolates local and global normalization statistics as `μΔ=uμk+(1−u)μG` and
`σΔ=uσk+(1−u)σG`, with each channel's `u` sampled from `Uniform(0,1)` during feature
diversification. Its unseen-domain adapter also interpolates instance and global moments,
`μ*=αμinstance+(1−α)μG` and `σ*=ασinstance+(1−α)σG`, where a learned adapter predicts `α` from
the instance/global statistic differences. It trains supervised image classifiers and learns
representations across source domains; it does not select an estimator from target benign-only
support or control anomaly-ranking transfer risk. Primary source: [ECCV 2024 paper, full text and
equations](https://arxiv.org/html/2407.08245v1).

These results narrow the novelty boundary further: raw or hidden-layer moment sharing, local/global
interpolation, stochastic statistic perturbation, and compatibility with neural models are all
established design patterns. The remaining possible distinction is a specific target-side
estimand—such as anomaly-ranking or operating-point risk under benign-only onboarding—and a valid
feature-level decision rule for that estimand. That distinction is still unsupported: C37 rules out
distribution-free AUROC no-harm, C38–C40 fail prospectively for FPR under chronology shift, and
C49–C50's attack-prior rules do not beat fixed shared marginals. No candidate or novelty claim is
promoted by C55, and the confirmation protocol remains unlocked. The next POC must start from a new,
explicitly bounded estimand and derive its observable information requirements before scoring test
attacks; repeating moment interpolation, similarity weights, or a benign likelihood selector would
revisit established or already-failed designs.

**Current single novelty matrix (updated through C65).** “FedORBIT explored family” denotes the
tested local/shared/peer scale estimators and their benign-only selectors, not a surviving algorithm.

| Dimension | FedORBIT explored family | Closest prior methods | Remaining distinction / status |
|---|---|---|---|
| Target problem | Cold-start one-class anomaly ranking on a new IoT device | FedFD-A: unseen-domain supervised image generalization; pFedBBN: federated test-time adaptation | Narrow application/protocol difference only; no demonstrated general method novelty |
| New/unseen client | Yes, evaluation target excluded from peer summaries | FedFD-A zero-shot domain; pFedBBN unseen/shifted clients; PN global deployable model | New-client setup is established |
| Benign-only | Target adaptation uses benign support and no target attack labels | Closest normalization papers use supervised source task labels or pseudo-label-balanced adaptation | Contract differs, but has not yielded a winning rule |
| Shared information | Per-feature mean/scale summaries | PN aggregates learned hidden-layer moments; FedFD-A uses global BN moments; pFedBBN shares BN descriptors/models | Raw-feature sufficient summaries are a narrower representation, not a new sharing principle |
| Adaptation unit | Featurewise input mean/scale or OAS covariance target | FedFD-A interpolates each channel/layer; pFedBBN weights peer models from layerwise BN distances | C63 derives the exact effective raw-space OAS diagonal target; C64 finds no systematic ranking or covariance-risk gain for the median-peer target over full shared. This remains a parameterization of established OAS shrinkage, not a new adaptation principle |
| Uncertainty model | Sampling, block, robust, mixture, validation, and threat-prior variants explored | Jing et al. 2026 covariance-aware source shrinkage; PN learns moments as parameters | C65 finds covariance-estimation error is nearly uncorrelated with AUROC/pAUC gain within the controlled cells; a generic variance/covariance-risk weight is not detector-risk adaptation |
| Negative-transfer handling | Local fallback, discrepancy penalties, gates, robust/maximin selection | pFedBBN similarity weights; TRADER target-only component and source shrinkage | No no-harm guarantee for FedORBIT AUROC/FPR; C37 impossibility remains |
| Partner handling | Equal pool, peer scale, soft weights, selected action | pFedBBN softmax over BN-statistic distances; TRADER source-weighted regression | Generic adaptive partner selection is preempted; target loss remains distinct |
| Closed form / optimization | Mostly closed-form shrinkage; CV and threat selectors tested | PN constrained optimization; FedFD-A learned adapter; Jing SURE step size | Any new rule must show an exact estimand and a genuinely different risk derivation |
| Theoretical basis | Conditional MSE algebra for scale estimates; no AUROC guarantee | Classical empirical Bayes/shrinkage; Jing safe interval under Gaussian summary loss | Existing theory does not transfer automatically to serial benign support or AUROC |
| IoT anomaly setting | N-BaIoT physical devices, with chronology/stability limits | Closest methods use vision or supervised regression | Setting-specific empirical evidence only; no eligible second physical population |
| Deep-detector compatibility | C62 applies the same shared-center/median-peer-scale input normalization to a local AE | FedFD-A, PN, and other normalization-transfer methods | Interface compatibility holds; C62 shows no reliable AE gain and no new model-agnostic claim |
| Main mathematical difference | Target raw-feature benign estimator plus OAS ranking objective; C61 varies the peer-scale target and C62 reuses it for AE input normalization | Learned/interpolated hidden-layer BN moments, robust target aggregation, or supervised coefficient transfer | C61's n=1000 low-FPR crossover is small; C62 adds compatibility evidence only; novelty and independent confirmation remain absent |

## C56–C57 — low-FPR partial AUC reveals a narrow n=1000 crossover, not a safe winner

C56 changes the evaluation objective rather than the estimator. The ignored POC
`pocs/low_fpr_sparse_nbaiot_poc.py/.csv` reuses C53's exact nine physical devices, 15 paired
chronological support starts at n=30/100/300/1000, deterministic per-device benign/attack test
subsets of up to 5,000 rows, and all six C53 production OAS/precision-max scores. It adds
standardized partial AUROC capped at 1% and 5% FPR plus interpolated TPR at those FPR points. The
peer-scale candidate is still local-center/median-peer-scale OAS; the fixed comparator is full
shared OAS. The test-benign ROC operating points are descriptive ranking diagnostics, not a
deployment threshold or calibration guarantee.

The objective changes the n=1000 comparison: shared OAS has mean full AUROC .9942, standardized
pAUC(.01) .8905, pAUC(.05) .9532, interpolated TPR@1% .8870, and TPR@5% .9616. Local-center/
median-peer-scale OAS has .9931, .8999, .9566, .8986, and .9742 respectively. Its paired mean
deltas are −.00117, +.00939, +.00337, +.01161, and +.01263; standardized-pAUC window win shares are
70.4%/69.6%, while interpolated TPR window win shares are only 21.5%/6.7% because the empirical
benign quantiles yield many ties. Across the nine device means, pAUC(.01) is positive on 6/9 and
pAUC(.05) on 4/9; TPR@1% is positive on 2/9. Paired 95% t intervals across the nine device means
are [−.00364,.02242] for pAUC(.01), [−.00352,.01026] for pAUC(.05), and [−.01880,.04202] for
TPR@1%. This is a plausible low-FPR crossover against a saturated full-AUROC baseline, but not
device-level evidence of a reliable advantage. At n≤300, peer-scale OAS pAUC(.01) deltas are
−.00602/−.00455/−.00233, so the effect also does not appear as a general cold-start gain.

C57 (`pocs/low_fpr_nbaiot_family_poc.py/.csv`) checks whether the n=1000 partial-AUC crossover is
specific to Gafgyt or Mirai. It reconstructs labels from raw attack files using C54's configured
sampling seeds and verifies exact equality with prepared `test_attack`, then evaluates the same
paired support/test indices. At n=1000, local-center/peer-scale OAS has mean pAUC(.01) deltas
+.01080 for Gafgyt (6/9 positive device means; 95% device-level t interval [−.00100,.02260]) and
+.00352 for Mirai (3/7; [−.01303,.02007]). At the 5% cap, Gafgyt is +.00446 (5/9) but Mirai is
−.00234 (1/7); Mirai's seven-device t interval is [−.00439,−.00029]. Gafgyt/Mirai full-AUROC
deltas remain +.00077/−.00499, with only 5/9 and 1/7 positive device means respectively. Two devices
have no eligible Mirai rows, so that family result covers seven devices. This points to a threat-
family and operating-region interaction rather than a generally better score. Neither family has
positive 1%-cap device-level evidence with an interval excluding zero.

The sparse precision-max scores remain far below shared OAS on these partial-AUC metrics, so the
C52 score mechanism does not re-enter consideration. C56–C57 suggest that local-center/peer-scale
OAS may trade a tiny full-AUROC loss for a low-FPR ranking gain at n=1000, particularly for Gafgyt;
the evidence is exploratory, attack-family-specific, uses the same attack-exposed devices that
informed method development, and has no untouched confirmation population. Keep the existing
candidate disposition unchanged. A follow-up is justified only if a predeclared low-FPR operating
region becomes the actual study estimand and an independent physical-device confirmation cohort is
available; do not tune support sizes or operating regions to these observed crossovers. The
chronological test FPR instability from C38–C40 also means partial-AUC improvement cannot be
translated into a prospective 1% or 5% false-alarm guarantee.

## C58–C60 — preserving the shared center isolates the low-FPR effect to peer scale choice

C58 (`pocs/low_fpr_channel_ablation_poc.py/.csv`) ablates center and scale separately at n=1000,
using C53/C56's same paired windows and test subsets, five production OAS variants, and C54's exact
Gafgyt/Mirai label reconstruction. The full shared-marginals model is the common reference. Replacing
only its pooled scale by coordinatewise median peer SD while retaining the shared center gives mean
ΔAUROC +.00062, pAUC(.01) +.00814, and pAUC(.05) +.00364. Replacing only the center with the local
support mean while keeping shared scale gives −.00193/−.00271/−.00195. Replacing both center and
scale gives −.00117/+.00939/+.00337. Thus the low-FPR shift is attributable to the scale target.
The local center worsens AUROC and pAUC(.05), while adding only .00125 to pAUC(.01); that small
difference is uncertain. Across nine device means, the
shared-center/peer-scale AUROC delta is 8/9 positive, mean +.00062 with unadjusted 95% t interval
[−.00008,.00131], worst −.00003. Its pAUC(.05) is positive on 8/9, mean +.00364, interval
[+.00003,+.00725], worst −.00368; pAUC(.01) is positive on 7/9, mean +.00814, interval
[−.00269,+.01898], worst −.02214. Under Gafgyt, AUROC and pAUC(.05) deltas are +.00076 (9/9;
95% device interval [+.00009,+.00143]) and +.00465 (8/9; [+.00092,+.00837]); under Mirai they
are +.00016 (5/7) and +.00167 (5/7), with both intervals crossing zero. Intervals are unadjusted
exploratory summaries across the fixed nine-device set, not confirmatory inference.

C59 (`pocs/low_fpr_shared_center_peer_scale_poc.py/.csv`) carries this exact new combination across
all registered n=30/100/300/1000 supports (15 paired starts/device) and both attack families. The
pooled AUROC deltas versus shared marginals are −.00498/−.00152/+.00027/+.00062; positive device
means are 2/9, 2/9, 4/9, and 8/9; worst device means are −.02647/−.00784/−.00147/−.00003. Its
standardized pAUC(.01) deltas are +.00338/−.00001/+.00205/+.00814, positive device means 4/9,
4/9, 4/9, 7/9. pAUC(.05) deltas are −.00228/−.00118/+.00027/+.00364, positive device means
3/9, 3/9, 4/9, 8/9. Paired window win shares for AUROC grow from 54.8% to 84.4%; pAUC(.05) win
shares are 63.0%/68.9%/68.9%/83.7%. The apparent benefit therefore concentrates at larger support,
and is not a general cold-start improvement. At n=1000 the small positive AUROC and pAUC(.05)
device means do not establish a broad Pareto gain; repeated windows are nested within only nine
devices, test populations were reused during development, and pAUC(.05)'s interval is barely above
zero before accounting for the many support/metric/family comparisons.

C60 reuses the existing paired 64-feature correlated-Gaussian mechanism suite, 120 replications per
cell, n=30/100/300/1000, and its five declared settings (matched clean, heterogeneous peers, one
4× peer, one 0.25× peer, and a 2× novel target). It adds shared-center/median-peer-SD to the existing
production-OAS comparison. Relative to the same synthetic full-shared baseline, candidate mean
ΔAUROC across support sizes is approximately zero throughout: matched clean −.00000/−.00000/+.00000/
−.00000; heterogeneous peers +.00007/−.00076/−.00029/−.00002; 4× peer +.00006/−.00096/+.00023/
+.00010; 0.25× peer −.00064/+.00015/−.00021/+.00004; and novel target −.00021/−.00020/+.00007/
+.00004. Across these 20 cells it does not produce a material positive synthetic AUROC gain; its
10th-percentile effects remain negative (worst cell −.00876). This fails to explain the real n=1000
partial-AUC shift as a general mechanism and supplies no safety guarantee under source-target
mismatch.

C61 is a **provisional low-FPR lead**, not the selected FedORBIT algorithm: keep the shared center,
replace the pooled scale with coordinatewise median peer SD, and use the existing OAS score. It is
the strongest new result this turn because the scale-only ablation improves the n=1000 pAUC(.05)
mean on 8/9 devices and across both attack families, with a tiny positive AUROC shift on 8/9. Its
closest methods are robust federated aggregation, multi-target covariance shrinkage, and OASD's
alternative diagonal covariance target, all already in the novelty ledger. The mathematical
distinction is only the combination of shared center plus robust peer-scale target for this specific
new-device OAS ranking objective; novelty is weak and unsupported. Communication remains 2d values
for a shared center/scale message, the rule borrows on every feature at every support size, no
automatic fade or safe-transfer property is established. C62 now tests the exact combination with
the AE and finds only a small, uncertain n=1000 low-FPR near-tie. The full shared-marginals
comparator still wins on AUROC at n≤100. Keep the protocol unlocked; do not tune a support cutoff or
call the observed operating region confirmatory.

## C62 — exact C61 normalization is compatible with the local AE, without a demonstrated gain

The ignored `pocs/autoencoder_shared_center_median_scale_poc.py/.csv` evaluates the exact C61
transform in the secondary reconstruction detector, addressing the prior C48 gap. It uses all nine
N-BaIoT devices, six paired contiguous support draws at n=30/100/300/1000, the same held-out benign
and attack rows for every condition, and identical model initialization and minibatch RNG seeds
within each paired draw. Every AE is trained locally for 300 steps with hidden/bottleneck widths
64/16, batch size 64, and learning rate .001, using only normalized benign support. Five conditions
are compared: local center/local scale; full shared-marginals center/scale; exact C61 shared center/
median peer SD; local center/median peer SD; and C48 local center/pooled peer scale. AUROC and
standardized pAUC capped at 1%/5% FPR are evaluation metrics only; no threshold or online calibration
claim is made.

For exact C61 versus full shared AE at n=30/100/300/1000, mean AUROC deltas are
−.00259/−.00386/−.00163/+.00007; positive device means are 4/9, 1/9, 3/9, and 6/9. At n=1000,
the device-level mean is +.00007 with 95% t interval [−.00015,+.00029], worst device −.00031, and
63.0% paired-window wins. The pAUC(.01) deltas are −.00028/−.00242/−.00068/+.00436, with 6/9,
5/9, 4/9, and 7/9 positive device means. At n=1000 pAUC(.01) has a +.00436 device-mean delta,
95% interval [−.00451,+.01323], worst device −.01617, and 66.7% paired-window wins. The pAUC(.05)
deltas are −.00379/−.00648/−.00195/+.00072; at n=1000 the device interval is
[−.00150,+.00294], 6/9 device means are positive, worst device −.00322, and paired-window wins are
63.0%. Thus the raw-input transform can feed a deep detector, and its low-FPR direction at n=1000
is similar to OAS, but the effect is small and uncertain. At n≤300 it does not outperform full shared
normalization on mean AUROC or either partial-AUC cap.

C62 therefore supports interface-level AE compatibility, not a model-agnostic performance claim or
non-inferiority to fixed shared marginals. It does not alter C61's provisional status: AUROC is
essentially tied only at n=1000, all paired device-level intervals for C61's AE gains cross zero,
and the same nine attack-exposed device/test populations informed the broader search. The AE result
does not repair C61's synthetic near-tie, weak novelty, absence of automatic collaboration fading,
or lack of independent physical-device confirmation. No protocol is locked.

## C63 — C61 changes OAS's effective diagonal target; no reliable benign-only gain predictor emerges

The ignored `pocs/oas_peer_scale_diagonal_target_audit.py/.csv` reuses the paired n=1000 C59 OAS
windows and scores; it does not retrain or rescore attacks. It calculates peer-only scale and
between/within descriptors from benign `support_pool` arrays, then joins those descriptors to the
already-computed device-level AUROC and standardized pAUC deltas strictly for post-hoc diagnosis.
No attack labels enter any descriptor or candidate rule.

For fixed shared center and support second moment `S`, let `D=diag(s)` be the external scale,
`C=D^-1 S D^-1`, and let OAS shrink `C` to `(1-alpha)C + alpha*tau*I`, where
`tau=trace(C)/d`. The raw-unit covariance represented by the resulting quadratic score is exactly
`(1-alpha)S + alpha*tau*D^2`. Thus a change from full shared scale to coordinatewise median peer SD
changes the *shape of the OAS diagonal target* as well as the standardized sample covariance; it is
not merely a different way of expressing the same pooled estimate. This also explains why the
scale choice is not generally score-invariant under OAS, despite exact invariance to a uniform
rescaling when the corresponding covariance transform is applied.

Across the nine targets, median peer SD is below the full pooled peer SD on 96.5%–100% of features.
The median across-feature SD of log scale ratios is 0.193–0.249. Across the 15 n=1000 windows/device,
the RMS log ratio between effective candidate and shared OAS diagonal targets is 0.405–0.505, while
the mean change in OAS `alpha` is only −0.00040 to +0.00015. The candidate therefore materially
changes which diagonal target is used, while barely changing the scalar shrinkage intensity.

The post-hoc rank association between the 10th percentile of effective-target log ratios and C61's
device-level deltas is negative for AUROC (Spearman −0.80), pAUC(.01) (−0.85), and pAUC(.05)
(−0.77); the 90th-percentile associations are also negative (−0.77/−0.73/−0.73). These are only
nine device-level observations, with feature summaries and outcomes examined after the C61 screen;
they are neither valid selector evidence nor confirmation. The single low-FPR-harmed Provision
camera is consistent with a target-shape mismatch explanation, but cannot establish it. Peer
between-center/within-variance ratios show no stable direction that would enable a safe gate. The
mechanism audit therefore explains the implementation mathematically but does not explain the
performance crossover or yield a reliable benign-only prediction rule.

C63 strengthens the hostile novelty reading: C61 is an externally chosen coordinatewise diagonal
target inside conventional OAS shrinkage. Robust diagonal targets and scale aggregation are
established statistical patterns, and no distinct estimand or safety principle has appeared. Keep
C61 provisional and do not tune a selector against these nine post-hoc associations. A useful next
mechanism check is a controlled factorial experiment that varies target-vs-peer diagonal mismatch
and off-diagonal correlation separately, scores the raw-space OAS target directly, and checks when
local, full-shared, and median-peer targets win on covariance error and anomaly ranking. No protocol
lock or candidate promotion follows from C63.

## C64 — controlled mismatch favors target-local scales, while the median-peer target remains a near-tie

The ignored `pocs/oas_diagonal_target_factorial_poc.py` evaluates the C63 mechanism in a controlled
population using the production `scorer_from_rows` OAS implementation. It crosses featurewise
target/peer log-scale mismatch SD `{0, .35, .70}`, peer log-SD heterogeneity `{0, .40}`, target
factor-correlation strength `{0, .65}`, and support `n={30,100,300,1000}`. There are 100 paired
replications per cell (48 cells total). Each draw has 32 features, eight peer sufficient-statistic
draws calibrated to 2,000 benign samples per peer, common target covariance for paired methods,
1,200 held-out benign rows, and 800 shifted-mean attacks. It compares local center/local scale,
full shared center/scale, C61
shared center/median peer SD, shared center/local scale, and a shared-center oracle target SD. The
oracle is a mechanism bound only. Outcomes are exploratory; no test or physical-client claim is
made.

Across the 48 cells, the C61 median-peer arm is essentially tied to full shared: mean deltas are
−.00004 AUROC, −.00005 standardized pAUC(.01), and −.00006 standardized pAUC(.05). Only 22/48,
25/48, and 23/48 cell means, respectively, are positive. The worst cell-mean deltas are −.00174,
−.00063, and −.00111; worst within-cell 10th percentiles are −.01915, −.00565, and −.01072. The
median-peer arm's grand mean covariance-Frobenius error is also effectively tied (candidate minus
shared +.00026 relative error), with lower error in 48.3% of paired draws; diagonal-variance RMSE
gain averages only +.00007, with lower error in 48.7%. Changing peer heterogeneity or correlation
does not produce a stable median-peer advantage. This reproduces C60's near-tie with an explicit
factorial design and error metrics.

The local scale ablation moves in the expected direction as target/peer feature scales diverge.
At zero mismatch and n=30, local OAS trails full shared by .01492 AUROC and .00717 pAUC(.05), and
its relative covariance error is .07941 worse on average. At mismatch SD .35, local's mean pAUC(.05)
delta is −.00013/.00462/.00163/.00081 for n=30/100/300/1000; at mismatch SD .70 it is
+0.00494/+0.01131/+0.00775/+0.00322. For mismatch .35/.70 the corresponding covariance-Frobenius gains
over full shared are +0.10661/+0.12601 at n=30 and remain +0.03688/+0.02484 at n=1000. However,
low-FPR lower tails remain negative: for mismatch .70, local pAUC(.05) 10th-percentile deltas are
−.03114/−.01542/−.01175/−.00818. Correlation changes the crossover; for example, at mismatch .35,
n=30, and correlation .65, local loses .00692 pAUC(.05) on average, while the uncorrelated arm
gains .00665. Lower covariance error therefore does not guarantee better anomaly ranking in every
mechanism cell.

The center-fixed local-scale arm shows that this pattern is scale-specific rather than a benefit
from replacing the peer center: it has similar covariance-risk and ranking trends to the full local
arm. The oracle-target-scale arm improves mean covariance error in 82.6% of draws and gains .00445
pAUC(.05) overall versus full shared, but it uses the true target diagonal and is not deployable.
This leaves a real estimation problem: detect featurewise target/peer scale mismatch from benign
support with enough uncertainty accounting to avoid reverting to noisy local scales when the
populations match. C64 does not solve that problem; it gives no support cutoff, selector, or safe
transfer guarantee.

C64 therefore sharpens the failure analysis. Median peer SD does not improve over pooled peer scale
in this controlled Gaussian design, even when heterogeneity and correlation are varied. The useful
signal is whether target variance differs from peer variance: fixed sharing helps when they match
and hurts covariance estimation under mismatch, while local scaling can improve ranking in some
mismatch regimes but has harmful lower-tail cells. The next candidate should be a target-scale
estimator with explicit uncertainty and random-effects mismatch, compared against local and shared
baselines; it must be evaluated as a statistical-risk rule, not selected from these oracle scenarios.
This candidate family is adjacent to standard random-effects/empirical-Bayes variance shrinkage, so
novelty remains weak until a distinct estimand or justified safety property is established. No
protocol is locked.

## C65 — covariance-estimation error is not a useful stand-alone proxy for ranking gain

The ignored `pocs/oas_covariance_error_ranking_alignment_audit.py` reuses C64's paired 24,000-row
synthetic output; it does not fit models, change any method, or rescore attacks. For each of the 48
factorial cells and 100 paired replications, it compares an estimator's reduction in relative
covariance Frobenius error or diagonal-variance RMSE against its AUROC and standardized pAUC gains
over full shared. This isolates the relationship within fixed target-mismatch, heterogeneity,
correlation, and support-size conditions, avoiding pooled correlations driven by the scenario grid.

Across the 48 cells, local-scale covariance-risk gain has essentially zero median Spearman
association with ranking gain: for Frobenius error, rho is +.011 for AUROC, −.008 for pAUC(.01),
and +.011 for pAUC(.05); for diagonal RMSE the corresponding medians are +.019, +.003, and +.002.
Only 48%–56% of cellwise correlations are positive. The shared-center/local-scale arm is similar.
Even the oracle-target-scale arm has small median correlations (+.039 at most), despite its much
larger average covariance-error reduction. Across paired replications, covariance error improves
while ranking gets worse in about 18%–30% of cases, depending on estimator/loss; the reverse
discordance also occurs in about 9%–26%. Among nonzero changes, sign discordance is roughly 39%–51%.
These are descriptive summaries over a controlled Gaussian mean-shift attack family, not tests of
independence or universal claims.

This explains why earlier variance-risk improvements did not reliably translate into detector
ranking improvements. For quadratic score matrix `A`, common-covariance Gaussian benign and
mean-shift attack distributions have expected score gap
`E[Z^T A Z]−E[X^T A X]=δ^T A δ`; the attack direction `δ` therefore matters, while a global
Frobenius covariance error weights all directions and does not encode that attack objective. Score
variance and the attack distribution also affect AUROC/pAUC, so this identity is explanatory, not
an AUROC formula or a selector. C37 already establishes that arbitrary-attack AUROC no-harm is
unidentifiable from benign data alone. C65 reinforces that a Gaussian covariance-MSE result cannot
be presented as anomaly-ranking safety or used to select collaboration without a bounded threat
model.

C65 does not retire the target-scale problem: C64 shows settings where local scale reduces
covariance error and sometimes improves low-FPR ranking, while its harmful lower tail remains. It
does retire covariance MSE as a sufficient surrogate for choosing that scale in this detector.
Any next risk-based candidate needs an explicit decision estimand that is observable under benign-
only onboarding and a mathematically stated attack/threat scope; otherwise evaluate scale estimation
as a statistical subproblem without claiming it selects a safer anomaly score. No new candidate,
novelty claim, or protocol lock follows from C65.

## C66 — peer-correlation OAS targets show a low-support signal, but no safe or efficient candidate

The ignored `pocs/peer_correlation_oas_target_poc.py` replaces only the OAS covariance target's
shape. It keeps C59's shared center/scale, support windows, attack and benign test subsets, and
standard OAS shrink intensity fixed. For each target device, it averages the other eight devices'
benign correlation matrices, then interpolates that matrix with identity at weights .25/.50/.75.
The result CSV contains 4,500 paired candidate rows across nine devices, four support sizes
(30/100/300/1000), 15 windows per device/size, three weights, and pooled plus family-specific
metrics. Summaries are paired to the exact C59 shared-center/shared-scale rows. The weight sweep is
exploratory and uses the same N-BaIoT attack sets; it is not validation of a tuning rule.

Pooled results have a small-support low-FPR signal, strongest at n=30, that fades or reverses as
support grows. At weight .50, mean device deltas over shared are +.03593 standardized pAUC(.01) at
n=30 and +.01436 at n=100, but the worst device means are −.06428 and −.05192; only 7/9 and 5/9
device means are positive. At n=300/1000 the same arm falls to +.00651/−.00140, and pooled AUROC
is −.00002/−.00048. Weight .75 gives larger n=30 pAUC(.01) mean (+.03710), with a worse-device
loss of −.07753; its pAUC(.01) mean turns negative at n=300/1000 (−.00729/−.01343). Weight .25 is
less volatile but has the smallest n=30 gain (+.03062) and nearly zero n=1000 AUROC/pAUC(.05).
No setting avoids material device harm.

The apparent pooled gain is attack-family dependent. At weight .50 and n=30, mean pAUC(.01) gains
are +.08003 for Mirai versus +.00787 for Gafgyt; Gafgyt's worst device mean is −.13122. At n=1000,
Mirai's gain is only +.00472 on average (2/7 positive device means), while Gafgyt averages −.00618
(3/9 positive). This is consistent with the C37 limit: benign-only evidence cannot certify ranking
safety for arbitrary attacks, and the observed same-dataset gains cannot justify a family-aware
selection rule.

The raw peer mean correlation target is rank deficient in this dataset, so using it directly caused
singular covariance inversion. The evaluated identity interpolation makes each target invertible,
but adds another weight with no benign-only selection rule. A dense 115-feature correlation matrix
has 6,555 unique off-diagonal values per client, versus 230 values for the two shared mean/scale
vectors: 28.5x the outgoing scalar payload under triangular encoding (52,440 scalars from eight
peers to the aggregator, versus 1,840 baseline scalars). This ignores headers, quantization, and
secure-aggregation overhead. Multiple-target covariance shrinkage is established prior art, as
recorded in the novelty matrix; C66 supplies no distinct novelty claim. It is a provisional
low-support mechanism observation only, not a candidate promotion, independent confirmation, or
protocol recommendation. No weight is selected and no protocol is locked.

## C67 — featurewise empirical Bayes adapts to some mismatch, but the peer prior cannot detect an isolated target shift

The ignored `pocs/featurewise_random_effects_oas_target_poc.py` adds one arm to C64's exact paired
48-cell Gaussian factorial, recreating the same seeded support, peer summaries, and held-out benign
and mean-shift attack rows. For feature j, it bias-corrects peer and target log sample variances under
the Gaussian chi-square model. The peer mean log variance is the prior mean; its between-peer
variance minus known peer sampling variance, truncated at zero, estimates the random-effect variance
`tau_j^2`. Given target log sample variance `z_tj` with known sampling variance `v_n`, it uses normal-
normal shrinkage: `v_post=(1/tau_j^2+1/v_n)^−1`, `m_post=v_post*(mu_j/tau_j^2+z_tj/v_n)`, and
sets the target scale to `sqrt(E[sigma_j^2|data]) = exp((m_post+v_post/2)/2)`. OAS intensity, shared
center, correlation, attack generation, and evaluation remain unchanged. This is a standard
random-effects empirical-Bayes construction adjacent to C1/C9/C12, not a novelty claim.

Across all 48 cells and 100 paired replications per cell, the EB arm's mean deltas versus full
shared are +.00350 AUROC, +.00077 standardized pAUC(.01), and +.00162 pAUC(.05). The corresponding
10th percentiles over paired replications are −.00583/−.00267/−.00463, and only 34/48, 32/48, and
34/48 cell means are positive. Its worst cell is the matched-target condition with peer heterogeneity
.40, correlation .65, and n=30: AUROC −.00758, pAUC(.01) −.00223, and pAUC(.05) −.00491. In that
condition the target is at the peer-population center, but empirical heterogeneity makes the prior
expect a new target effect and the short support cannot reliably identify that it is the central
case. Conversely, when peer heterogeneity is zero but the target has a featurewise scale shift, the
estimated prior variance is zero and the estimator remains near shared instead of adapting to the
novel target. It therefore fails in both directions of the cold-start problem: it can over-adapt to
a matched target and under-adapt to a shifted one.

The EB arm reduces mean relative covariance Frobenius error by .02380 and diagonal variance RMSE by
.01014 versus full shared, but ranking changes do not follow these error improvements. Covariance
error improves while AUROC worsens in 23.3% of paired cases and while pAUC(.01) worsens in 23.0%;
this is consistent with C65's loss-mismatch finding. The candidate needs each peer's per-feature
variance vector to estimate `tau_j^2`, preserving between-peer detail (8d values at the aggregator
in this simulation, versus 2d in one final shared mean/scale vector). This may be smaller than
collecting every peer's full 2d moment pair, but it prevents simple secure aggregation and exposes
between-peer heterogeneity. It remains synthetic-only and does not establish target/device safety.
Do not promote it or tune its
prior from these attack outcomes. The strongest current algorithmic direction remains unresolved;
next investigate a target-specific mismatch model that can distinguish an isolated newcomer shift
from ordinary between-peer heterogeneity, while preserving low-FPR ranking under a stated threat
family. Prior work on empirical-Bayes mixture and discrepancy estimators, plus C37/C49/C50/C65, makes
any benign-only ranking-safety claim especially vulnerable. No protocol is locked.

## C68 — matched peer correlation helps sparse shifts and hurts a different mean-shift direction

The ignored `pocs/peer_correlation_transfer_factorial_poc.py` isolates C66's off-diagonal mechanism.
It uses 32 features, eight peers with 2,000 benign rows each, target support n=30/100/300/1000, and
60 paired replications per cell. Target and peer populations have identical unit marginal variances
and zero centers. Peer correlation is a convex mixture of the target correlation and an unrelated
factor correlation at mismatch levels 0/.25/.50/.75/1. The same target support, test benign rows,
and test attack rows compare the OAS identity target with peer-correlation targets at weights
.25/.50/.75; OAS intensity is held fixed. Two predeclared mean-shift directions are evaluated:
five sparse coordinate shifts and shifts along the target correlation's leading eigenvector. This
is a controlled Gaussian mechanism screen, not a general threat set or a weight-selection experiment.

When peer and target correlations match and n=30, weight .75 gains +.05195 AUROC, +.03544
standardized pAUC(.01), and +.05998 pAUC(.05) over the identity-target OAS for sparse-coordinate
shifts; all 60 paired draws improve. Under the same target and peer covariance, that candidate loses
−.01925 AUROC, −.00186 pAUC(.01), and −.00510 pAUC(.05) for the leading-eigenvector shifts; only
0/60, 12/60, and 1/60 draws improve. At mismatch .50, n=30, and weight .50, sparse-shift gains
remain (+.01999/+0.01068/+0.02106), while leading-eigenvector shifts lose
(−.00766/−.00069/−.00212). Across support sizes, sparse-shift AUROC gains shrink from +.0157 at
zero mismatch to approximately zero or slightly negative at full mismatch; low-FPR gains also
approach zero. Increasing peer-target weight scales up both the sparse-shift benefit and the
leading-eigenvector harm. Even exact correlation transfer therefore does not provide attack-family
no-harm within these two mean-shift families.

C68 supplies a controlled explanation for why peer-correlation borrowing can show strong low-support
benefit on a particular attack mix, as in C66, while remaining unsafe as a generic anomaly-ranking
improvement. The mechanism depends on both target-peer correlation alignment and attack direction;
benign covariance alone does not reveal which direction will dominate. This is a direct instance of
the C37 ranking-identifiability constraint, not a tuning problem that can be solved by selecting the
best weight from these two attack families. Multi-target covariance shrinkage remains established
prior art, and its dense message cost and target correlation rank issues remain. No candidate is
promoted, no weight selected, and no protocol locked. Continue the search on an algorithm object whose
benefit can be justified against an explicit operational loss without inspecting attack labels.

## C69–C70 — classical variance-ratio empirical Bayes is prior art and does not beat fixed shared scale

The ignored `pocs/f_modeling_variance_ratio_poc.py` and `pocs/inverse_gamma_variance_ratio_poc.py` reuse C59's exact 9-device support windows, paired test subsets, production scorer, and full shared/local comparators (15 windows per device and support size). For each feature, the target-to-peer variance ratio is modeled as `r_j | θ_j ~ θ_j χ²_(n−1)/(n−1)`. C69 applies the F-EBV order-statistic estimator from Kwon & Zhao, “On F-modelling-based empirical Bayes estimation of variances,” *Biometrika* 110(1), 2023 ([DOI](https://doi.org/10.1093/biomet/asac019)). Its theorem studies the number of parallel variance problems `d→∞` at fixed degrees of freedom. Here there are only 115 dependent feature problems, while degrees of freedom rise from 29 to 999, so the asymptotic regime is reversed and the independent/exchangeable parallel-problem model is implausible.

C69 fails severely even in its four synthetic chi-square screens (400 replicates, 115 features): with matched unit ratios, mean scale relative-squared loss for F-EBV grows from about 1,053 at n=30 to `8.2e21` at n=1000, while raw local loss falls from .1006 to .0020. The same high-n collapse appears under lognormal feature heterogeneity and sparse/global target shifts. On N-BaIoT, F-EBV versus exact shared has mean AUROC deltas −.09040/−.05601/−.01339/−.00261 and standardized pAUC(.01) deltas −.22645/−.17439/−.10929/−.05357 for n=30/100/300/1000; it is positive on 0/9, 0/9, 0/9, and 1/9 device means for AUROC. Only 25.0%/16.9%/9.5%/3.9% of features move closer to shared scale as n grows, consistent with estimator collapse rather than useful shrinkage.

C70 tests the modified Smyth inverse-gamma empirical-Bayes action, using moment estimates that subtract the known chi-square sampling variance before estimating cross-feature prior spread. This conventional parametric EB version repairs the synthetic risk pathology: under matched ratios, relative-squared loss is about .0010/.0004/.0001/0 versus raw-local .0967/.0226/.0069/.0020; under lognormal heterogeneity it is .0469/.0181/.0064/.0020, and it also substantially beats fixed peer scale under sparse/global mismatch. This is estimator-risk evidence under iid Gaussian chi-square feature models, not evidence that the resulting score ranks attacks better.

On the same 540 real paired windows, modified Smyth scale beats C69 F-EBV at n=30/100 on AUROC by +.07760/+.04390 and pAUC(.01) by +.18055/+.11473, but at n=1000 its deltas versus F-EBV are −.00137 AUROC and +.01176 pAUC(.01). It remains below exact shared at every n: mean AUROC deltas −.01280/−.01211/−.00470/−.00399 and pAUC(.01) deltas −.04591/−.05966/−.04744/−.04181. Only 1–3 of 9 device means are positive by metric/support, and the worst-device AUROC mean reaches −.05446 at n=30. It improves over local scale at n≤300, but the n=1000 low-FPR outcomes are mixed. Thus the moment correction is a useful synthetic estimator repair and a detector-level failure against the registered shared baseline, not a candidate.

C10's persistently high peer weight has a concrete explanation, but no validation: its weight is `V_L/(V_L+B_P)`, and the peer block variance-risk proxy `V_L` is itself inflated relative to the held-out target risk. C41's median peer/target risk ratios range from 1.20 to 7.55 across support sizes/halves, while target-local HAC/bootstrap estimates in C42–C43 underpredict future target risk. These results indicate cross-device/regime mismatch and within-target nonstationarity, not proof that an alternative target-risk estimate can safely lower the weight. Neither C69 nor C70 resolves this transport problem.

The synthetic and N-BaIoT raw rows remain gitignored in `docs/audit/pocs/`; C70's real summary is `pocs/inverse_gamma_variance_ratio_summary.csv`. No Gotham evaluation was repeated: the existing 78-device Gotham release has 70 benign-only emulations, 8 attack emulations, 7,670 unknown-label rows, and is not a second physical cohort or the 115-feature target population. The protocol remains unlocked.

## C71–C72 — dependence-aware variance-ratio EB repairs some synthetic risk, not real detector transfer

C71 (`pocs/variance_ratio_dependence_stress_poc.py`) tests C70 while varying temporal AR(1) correlation (`rho=.0/.5/.8`) and cross-feature equicorrelation (`0/.4/.8`) independently, across the same four target-scale scenarios, 115 features, n=30/100/300/1000, and 60 paired draws per cell. In the matched case, the exact peer scale is the truth and has zero loss; at strongest dependence (rho-time=.8, rho-feature=.8), C70 log-MSE is .2948 versus .4250 for raw local at n=30, declining to .0063 versus .0092 at n=1000, but both lose to exact peer. Under lognormal heterogeneity at that dependence, C70 is .2812 versus .3873 local and .1217 peer at n=30; by n=1000 it approaches local (.0105 versus .0107) and beats peer (.1241). Under the six-feature 4x sparse shift, C70 is .3037 versus .4081 local and .1003 peer at n=30, but reaches .0091 versus .0092 local and .1003 peer at n=1000. Thus the corrected EB rule can beat noisy local ratios while still selecting the wrong compromise relative to peer scale; cross-feature dependence and strong temporal correlation reduce or reverse its heterogeneous-target advantage. These are Gaussian mechanism tests, not ranking results.

C71 also evaluates future benign scale on 15 paired contiguous support/future blocks per device and support size. It reads only each device's prepared benign `support_pool`, uses other devices' pooled variances as the shared reference, and compares local ratios, peer scale, and C70 ratios by mean feature log-squared error against the next contiguous block. Across devices, log-MSE for local/C70/peer is 12.15/32.49/50.88 at n=30, 9.23/19.00/35.84 at n=100, 6.40/12.33/26.85 at n=300, and 5.42/5.88/14.09 at n=1000. C70 beats peer scale on average but loses to local scale at every n; the effect is device-unstable. At n=30 it improves over local on only 2/9 device means, and its severe losses on Danmini, Samsung, and both SimpleHome devices dominate the average. This diagnoses a target/time transport problem; it does not imply local scale gives better AUROC, since C65 shows covariance error need not predict ranking.

C72 (`pocs/acf_adjusted_variance_ratio_eb_poc.py`) replaces C70's common `n−1` degrees of freedom with a feature-specific Newey–West effective df estimated from the squared-residual influence series. Mean effective df is 24.2/73.5/204.9/607.2 at nominal n=30/100/300/1000. This is a standard HAC/effective-sample-size adjustment, not a new EB principle. In paired N-BaIoT detector scoring it remains below exact shared scale: mean AUROC deltas are −.01447/−.01160/−.00473/−.00392 and pAUC(.01) deltas are −.05695/−.07042/−.04981/−.04293. It is essentially unchanged from C70 (AUROC differences from C70 −.00167/+ .00051/−.00003/+ .00007; pAUC(.01) differences −.01104/−.01076/−.00237/−.00111). It is substantially better than local on low support, but still has negative device means against shared and does not improve C70's ranking outcome.

A separate 120-replicate synthetic check (`pocs/acf_df_empirical_bayes_synthetic_poc.py`) compares C70's iid df, estimated HAC df, and an AR(1)-formula oracle df. The estimated-df version lowers mean log-MSE than C70 in 4/6 temporal/feature-dependence cells for matched/global shifts and 4–6/6 cells for sparse shifts, but it cannot beat exact shared scale in any matched cell. Under high dependence and heterogeneous feature scales, peer scale can outperform both C70 and C72 at small n. The approximate oracle df is not uniformly better than the HAC estimate. Conclusion: temporal-dependence correction is statistically relevant to this estimator, but it does not solve the unknown target-heterogeneity/scale-transport problem or produce detector gain.

No new novelty claim follows: C71 is a model-diagnostic factorial; C72 combines classical HAC effective-df estimation with prior empirical-Bayes variance shrinkage. The real future-block evidence strengthens the case that the remaining error is not just the iid chi-square noise term. The next useful variant must model target-specific temporal/regime mismatch and feature-prior uncertainty without inspecting attack labels, then demonstrate a paired detector or operational-loss benefit. No candidate is promoted and the protocol remains unlocked.

## C73 — flexible VASH-style inverse-gamma mixture is prior art; first port over-shrinks detector scales

A hostile literature search found a closer variance-shrinkage method than the single inverse-gamma C70 model: Lu & Stephens, “Variance adaptive shrinkage (vash),” *Bioinformatics* 32(22), 2016 ([paper](https://doi.org/10.1093/bioinformatics/btw483), [official vashr source](https://github.com/mengyin/vashr)). The paper assumes independent sample variances with known chi-square degrees of freedom and a unimodal prior; it represents the prior as a mixture of inverse-gamma distributions sharing a common mode, estimates component weights and mode by marginal likelihood, and returns posterior variance estimates. This directly preempts flexible empirical-Bayes variance shrinkage. Its genomics setting has no peer transfer, unseen device, dependent benign chronology, OAS detector, or attack-ranking objective.

C73 (`pocs/vash_mixture_variance_ratio_poc.py`) makes a first constrained Python port on the target-to-peer ratios from the same 540 paired N-BaIoT windows. It fits a 12-component geometric grid of inverse-gamma shapes with a common mode by alternating weight EM and scalar marginal-likelihood mode optimization; the detector action is the mixture posterior `E[theta^2|r]/E[theta|r]` under relative-squared ratio loss. This is deliberately not described as an exact vashr reproduction: the local environment has no R, the grid is fixed rather than centered on a separately estimated one-component fit, and the official variance-versus-precision prior comparison is omitted.

Against exact shared scale, mean AUROC deltas are −.05967/−.03278/−.01684/−.00519 for n=30/100/300/1000, positive on 0/9, 0/9, 1/9, and 3/9 device means. Standardized pAUC(.01) deltas are −.20847/−.15194/−.10937/−.05274, with only 0/9, 1/9, 1/9, and 2/9 positive device means. It also loses to C70 by −.04688/−.02067/−.01214/−.00120 AUROC and −.16256/−.09227/−.06193/−.01092 pAUC(.01). Against local scale it helps mean AUROC at n≤300, but pAUC(.01) only gains at n=30 (+.02377) and is negative thereafter. The fitted mixture uses only 1.14–2.24 components above 1% mean mass, suggesting this fixed grid/marginal-likelihood fit collapses close to a single component rather than adapting flexibly. This is a negative parameterization screen, not a reason to dismiss VASH itself; the method is established prior art in any case. Do not promote C73.

C73 strengthens the reviewer objection that the C69–C70 estimator family is standard empirical Bayes, while C71–C72 show that dependence adjustment does not solve the detector loss mismatch. The live conclusion remains no algorithm superior to fixed shared marginals. A faithful VASH reproduction is optional only if it answers a real unresolved mechanism question; priority should remain on target-specific temporal/regime compatibility and detector-relevant operational loss. No protocol is locked.

## C74 — paper-faithful VASH core confirms the detector-loss failure

C74 (`pocs/vash_faithful_variance_ratio_poc.py`) follows the paper's main fitting equations more closely than the constrained C73 screen. Starting with the log-moment single-component estimate, it builds Lu–Stephens' shape grid around that estimate, fits a shared prior mode and mixture weights by alternating marginal-likelihood optimization and EM, and fits/compares both variance-unimodal and precision-unimodal prior families. The component priors are inverse-gamma with `beta_k=(alpha_k+1)c` for common variance mode or `beta_k=(alpha_k−1)/c` for common precision mode. The fit selects the higher marginal likelihood. It evaluates the paper's published variance estimate `1 / E[1/theta | r]` and, separately, C70's loss-matched action `E[theta²|r]/E[theta|r]`. This is a paper-equation port, not bit-for-bit package parity: R is unavailable, and the package's SQUAREM acceleration is replaced by ordinary fixed-point alternation.

On the same 540 paired N-BaIoT windows, the published action's mean AUROC deltas vs exact shared are −.08223/−.03931/−.01893/−.00447 at n=30/100/300/1000, with 0/9, 0/9, 1/9, and 2/9 positive device means. Its pAUC(.01) deltas are −.22632/−.14981/−.10714/−.03839, positive on 0/9, 1/9, 2/9, and 2/9 device means. The action stays close to local scale: AUROC gains vs local are only +.00232/+ .00084/+ .00095/+ .00018, and pAUC(.01) gains +.00592/+ .00133/+ .00062/−.00051. The relative-risk action moves 74–81% of feature ratios closer to peer scale, but it remains below shared by AUROC −.0824/−.0393/−.0189/−.0045 and pAUC(.01) −.2265/−.1498/−.1071/−.0384. The distinction between published and loss-matched actions therefore does not rescue detector performance; more peer movement is not the missing ingredient.

The fitted prior selects the variance-unimodal family for 61.5% of n=30 windows, 46.7% at n=100, 45.9% at n=300, and 43.0% at n=1000, with about 3.1–3.4 mixture components above 1% mass. The published action moves only 41–47% of feature ratios closer to peer scale, while the relative-risk action moves 74–81%. C73's fixed-grid approximation was therefore not responsible for the principal detector failure: the paper-centered grid, both prior orientations, and both point actions all miss the stronger fixed shared-marginal score on this attack mixture. This does not reject VASH for its variance-estimation objective or every detector/use case. VASH remains established prior art, and no FedORBIT candidate or novelty claim follows.

## C75 — expanded AE scale-only transfer still does not establish a quality-preserving Pareto gain

C75 (`pocs/autoencoder_peer_scale_followup_poc.py`) extends C48's local-center/peer-scale AE screen from 6 to 30 paired support draws per device at n=30/100. The 24 new draws (replicates 6–29) use the same deterministic support-window derivation, 300-step local autoencoder, fixed per-device benign/attack test rows, and paired model initialization for local/local, local/peer-scale, and full shared-marginal normalization. AUROC uses all 30 draws per device; low-FPR standardized pAUC uses the 24 new draws because the first six C48 rows did not store those metrics. This remains exploratory, reuses the same nine devices and test populations, and is not a non-inferiority or confirmation study.

Local-center/peer-scale minus full shared has mean device AUROC deltas −.00646 at n=30 and −.00119 at n=100. Across nine fixed device means, unadjusted t 95% intervals are [−.01456,+.00164] and [−.00595,+.00357]; positive means occur on 3/9 and 4/9 devices, worst-device means are −.03060/−.01603, and paired support draws improve in 43.0%/50.0% of cases. For pAUC(.01), mean deltas are −.01064/−.00658, intervals [−.02313,+.00186]/[−.01394,+.00078], and only 3/9 device means are positive at either n. For pAUC(.05), mean deltas are −.00463/−.00214, intervals [−.01181,+.00256]/[−.00734,+.00307], with 3/9 and 2/9 positive device means. Thus the near-average AUROC gap remains small but all metrics point below fixed sharing; no prespecified non-inferiority margin exists, so these intervals do not certify equivalence.

Against local/local normalization, peer-scale/local-center gains +.07999/+ .05422 mean AUROC, positive on 9/9 and 8/9 device means. Its mean pAUC(.01) gains are +.23767/+ .20923 (9/9 and 8/9 positive), and pAUC(.05) gains +.22558/+ .17824 (9/9 positive). Scale-only transfer therefore retains substantial AE benefit over local normalization, but the stronger full shared endpoint remains ahead on average and especially on the low-FPR point estimates.

The POC computes peer scale using `aggregate_equal_weight` on peer `MomentSummary` objects; the resulting pooled total variance includes between-peer mean offsets. Reproducing that value therefore requires per-peer means and variances (at least 2d scalars), and C75 establishes no peer-to-coordinator upload reduction. Once computed, the target-facing normalization state is d=115 scale values (460 bytes as float32) versus 2d=230 mean-plus-scale values (920 bytes) for full sharing: half the downstream vector size only. The POC retains full covariance matrices in memory and measures neither wire bytes nor framing, secure aggregation, or transport overhead. Exchanging only d within-peer scale values would be a different estimator and was not evaluated here. This is not a demonstrated end-to-end efficiency gain or safe accuracy Pareto improvement. The result strengthens AE interface compatibility evidence but does not justify “model-agnostic,” no-harm, or candidate-promotion claims. Continue the main search on detector-relevant risk/temporal compatibility and keep the protocol unlocked.

## C76 — within-peer median SD cuts the statistic upload, but its AE gap is concentrated on Provision cameras

C76 (`pocs/autoencoder_median_peer_scale_followup_poc.py`) expands C62's exact local-center/median-peer-SD AE arm from six to 18 paired support draws per device at n=30/100. It adds replicates 6–17 using the same deterministic support-window and model seeds, fixed test rows, and 300-step local AE. Conditions are local/local, local-center/median peer SD, local-center/pooled total peer SD, and full shared marginals. The 18 draws comprise the six existing C62 draws and 12 new ones; metrics are AUROC and standardized pAUC at .01/.05 FPR. This is still exploratory on the same nine physical devices/test populations, not independent confirmation or a non-inferiority study.

Against full shared marginals, local-center/median-peer-SD mean device deltas at n=30/100 are −.00797/−.00446 AUROC, −.01414/−.01106 pAUC(.01), and −.00956/−.00658 pAUC(.05). Positive device means are 2/9 at both supports for AUROC, 1/9 and 2/9 for pAUC(.01), and 2/9 at both supports for pAUC(.05); paired-draw win rates range from 35.8% to 46.9%. Unadjusted t intervals across nine device means contain zero for all six outcomes. The worst AUROC device means are −.04984/−.01833; Provision PT-737 contributes −.04984/−.01617 and Provision PT-838 −.02020/−.01833. Against local normalization, the median-SD arm gains +.07596/+ .05323 AUROC and +.23951/+ .21621 pAUC(.01); gains are positive on all nine devices for AUROC at both supports and on eight/nine for pAUC(.01).

The pooled-total-peer-scale arm is closer to full sharing: its mean AUROC deltas are −.00282/−.00185, pAUC(.01) −.00708/−.00631, and pAUC(.05) −.00258/−.00140. Median-SD minus pooled-scale device-mean deltas average −.00515/−.00261 AUROC, −.00706/−.00474 pAUC(.01), and −.00698/−.00518 pAUC(.05); most of the AUROC shortfall is on the two Provision cameras. This suggests the choice of total versus within-peer scale matters for these populations, but does not establish why or yield a benign-only gate.

The median estimator does have a distinct message accounting: each of eight peers can send only its d=115 within-device SD values (460 bytes float32), or 3.68 KB raw statistic payload per target before framing; full mean-plus-scale uploads are 2d per peer (7.36 KB total). This is a 50% reduction in peer-to-coordinator statistic payload by arithmetic, unlike C48/C75's pooled-total-scale estimator, but wire protocol and overhead remain unmeasured. The accuracy comparison does not establish quality preservation: fixed full sharing leads on all mean metrics, confidence intervals are not equivalence evidence, and the Provision harms remain visible. Keep this as a real communication/accuracy tradeoff and a candidate diagnostic, not a Pareto survivor. Next determine whether the two-device gap is caused by scale magnitude versus peer heterogeneity using benign-only descriptors declared before a separate threat-family test; do not tune a selector on the attack outcomes already seen.

## C77 — peer center offsets inflate pooled scale error without a commensurate AE ranking change

C77 (`pocs/synthetic_peer_center_scale_ae_factorial_poc.py`) directly crosses four mechanisms in a controlled 32-feature, eight-peer Gaussian AE screen: peer-center offset SD `{0,.75}` in base-scale units, peer log-SD heterogeneity `{0,.4}`, featurewise target/peer log-SD mismatch `{0,.4}`, and target AR(1) dependence `{0,.7}`. At each of n=30/100 and 10 paired replications/cell, it compares local/local, local-center/median-peer-SD, local-center/pooled-total-peer-SD, and full shared marginals. The same benign support and evaluation rows, threat realizations, and AE initialization are used across methods. Evaluation uses sparse four-feature signed mean shifts and four-feature 2× variance inflation; it records AUROC, standardized pAUC(.01/.05), and target log-scale RMSE. Peer summaries are exact population quantities in this screen, intentionally removing source estimation noise; this is a mechanism experiment, not a deployment simulation or candidate selection.

The mathematical separation is the law of total variance: for feature j, pooled total variance is `K⁻¹Σ_k σ²_kj + K⁻¹Σ_k(μ_kj−μ̄_j)²`, while median peer SD summarizes only the within-peer `σ_kj`. Local-center normalization gives the peer-center dispersion no direct target-center role. Averaged over the balanced remaining factors, increasing peer-center offset SD from 0 to .75 raises pooled-scale log-RMSE from .2449 to .3618, while median-peer-SD log-RMSE is .2218 and .2149; median minus pooled RMSE is −.0231/−.1469. Thus median scale better estimates target marginal scale when source means vary.

That estimator-risk advantage barely moves mean detector ranking. Median minus pooled AUROC averages +.00011 with zero peer-center offsets and +.00171 at offset SD .75; corresponding pAUC(.01) differences are +.00019/+ .00019 and pAUC(.05) −.00029/+.00059. The paired-draw 10th percentiles remain negative (AUROC −.01021/−.01227; pAUC(.01) −.00480/−.00769; pAUC(.05) −.00722/−.00934), so the mean does not imply no-harm. Against full shared, median peer SD's average AUROC deltas at n=30 are −.00446/−.00373 for the two threats; at n=100 they are +.00202/+ .00324. All corresponding 10th-percentile deltas are negative across the balanced factorial. Gains at n=100 are small, synthetic, threat-family-specific, and do not establish a general ranking advantage.

C77 does not explain why C76's median-scale AE arm harmed two Provision cameras: even a large controlled separation between scale RMSE does not produce a similarly large AUROC/pAUC separation, and lower scale RMSE is not a reliable proxy for detector ranking. This aligns with C65's loss-mismatch result. C78 finds a post-hoc association between peer center dispersion and C76/C80 device deltas, but the nine-point descriptor is not a validated gate; C79's fresh-seed primary-OAS factorial likewise finds estimator-risk differences without stable ranking separation. C77 adds no novelty claim beyond the standard total-variance decomposition and robust within-peer aggregation. No candidate is promoted; protocol remains unlocked.

## C78 — peer center dispersion is a post-hoc clue for C76, not a selector

C78 (`pocs/autoencoder_median_peer_scale_benign_descriptor_diagnostic.py`) recomputes benign-only descriptors from each target's 30 C62/C76/C80 support windows and the other devices' `support_pool` summaries. For each feature it estimates the variance of peer means divided by mean peer within-device variance; the reported device descriptor is its featurewise median. Provision PT-737/PT-838 have the largest ratios among the nine devices (.259/.255); the next largest is Philips at .212. Those two Provision devices also have some of the largest median-minus-pooled AE AUROC harms. With all 30 paired windows, the across-device Spearman associations between this descriptor and the already-seen median-minus-pooled AUROC are −.683 at n=30 and −.467 at n=100.

This is post hoc: the nine-device outcome was already inspected, there is no held-out device-level validation, and the correlation misses other harms such as SimpleHome XCS7-1002 at n=30. A target-local/peer-median log-scale mismatch descriptor is not a complete explanation either; SimpleHome XCS7-1003 shows a large scale mismatch without a comparable detector loss. The ratio could motivate a preregistered synthetic hypothesis, but cannot select between median and pooled scale or support a safety claim. This re-audit narrows the earlier 18-window correlation and keeps the whole descriptor family diagnostic-only.

## C79 — fresh-seed primary-OAS factorial finds scale-risk gains without ranking separation

C79 (`pocs/synthetic_peer_center_scale_oas_followup_poc.py`) repeats the C77 mechanism factors with 100 new paired seeds per cell and the production OAS scorer: peer-center offset SD `{0,.75}`, peer log-SD heterogeneity `{0,.4}`, target log-SD mismatch `{0,.4}`, target AR(1) dependence `{0,.7}`, and support n=30/100. It uses 32 features, eight peers, and four declared neighboring threats: 2- or 8-feature 1.25-SD mean shifts and 4-feature 1.5× or 8-feature 2× variance inflation. The same support, benign rows, attacks, and OAS fit inputs are paired across local, local-center/median-peer-SD, local-center/pooled-total-SD, and full shared-marginal methods. The source summaries are exact population values; the 51,200 result rows isolate the effect of estimator choice without finite peer-summary noise.

Median-minus-pooled AUROC means range from −.00032 to +.00011 over the n/threat groups, while q10 ranges from −.01042 to −.00377 and the median-scale arm wins only 34–39% of paired draws. For pAUC(.01), mean differences range −.00050 to +.00004, with q10 as low as −.01288; pAUC(.05) ranges −.00051 to +.00002, with q10 as low as −.01260. Across center-offset factors, median-minus-pooled AUROC averages −.00011 with no source-center heterogeneity and +.00009 at offset SD .75; corresponding pAUC differences remain near zero. Nevertheless, source-center offsets widen the scale-error gap: log-scale RMSE for median versus pooled is .2181/.2407 without offsets and .2168/.3616 at offset SD .75. Thus the law-of-total-variance decomposition strongly changes scale risk while detector rankings remain nearly tied on average and retain harmful lower tails.

Against full sharing, median-scale mean AUROC deltas at n=30 are −.00808/−.01864/−.00881/−.01415 for the four threat variants in the order above; at n=100 they are −.00260/−.00735/−.00328/−.00441. All cell-balanced q10 results are negative. This fresh primary-detector run reproduces the loss/ranking mismatch under distinct attacks and rejects a broad claim that avoiding between-peer means improves detector ranking. It does not refute robust within-peer scales for their own estimation objective. No scale selector or candidate gain is established.

## C80 — 30 paired AE windows strengthen the pooled-scale advantage over median peer SD

C80 (`pocs/autoencoder_median_peer_scale_30rep_followup_poc.py`) extends C76 from 18 to 30 paired support windows per device at n=30/100. Replicates 18–29 use the same support seed derivation, fixed device test rows, four normalization conditions, 300-step production local AE, and paired initialization; the combined analysis includes C62 replicates 0–5 and C76 replicates 6–17. Each action therefore has 270 paired device-window draws per support. This increases support-seed precision but does not add physical devices or independent test populations.

Local-center/median-peer-SD versus full shared has mean device AUROC deltas −.01294/−.00477 at n=30/100; pAUC(.01) −.01767/−.00812 and pAUC(.05) −.01294/−.00539. Only 2/9 device means are positive for AUROC at either support; worst device means are −.05041/−.01867, with the strongest n=30 AUROC loss on SimpleHome XCS7-1002 and further harms on both Provision cameras. Unadjusted t intervals across nine device means include zero, which is not non-inferiority evidence. Against local/local normalization, median scale still gains +.07350/+ .05063 AUROC and +.23597/+ .21352 pAUC(.01), positive for AUROC on 9/9 and 8/9 devices.

The pooled-total-scale/local-center arm is closer to full sharing at n=30/100: AUROC −.00646/−.00119, pAUC(.01) −.01019/−.00564, and pAUC(.05) −.00472/−.00115. Median minus pooled device-mean deltas are −.00648/−.00358 AUROC, −.00748/−.00248 pAUC(.01), and −.00823/−.00425 pAUC(.05). The two Provision cameras remain harmed by median scale; SimpleHome XCS7-1002 is also a substantial n=30 loss. Thus the C76 direction persists with 30 windows, but its magnitude is support-seed sensitive, especially at n=30. The actual per-peer payload remains 50% lower for median within-peer SD; no quality-preserving gain is established. Keep this as an explicit communication/accuracy tradeoff and do not promote it.

## C81 — finite peer sampling adds scale error but barely changes the median-versus-pooled OAS result

C81 (`pocs/finite_peer_summary_dependence_oas_poc.py`) replaces C79's exact source summaries with eight actual peer AR(1) sequences per replication. It crosses peer sample size m={100,1000,5000}, peer and target autocorrelation ρ={0,.7}, center-offset SD={0,.75}, peer log-SD heterogeneity={0,.4}, target log-SD mismatch={0,.4}, and target support n={30,100,300,1000}: 384 cells × 30 paired draws. Four neighboring mean-shift/variance-inflation threats and seven OAS actions yield 322,560 result rows. Sampled median SD, sampled pooled total scale, and shared sample mean are compared with exact-population median/pooled references and local-only. The same target support and attack/evaluation rows are paired across actions. Source features use a 32-dimensional correlated Gaussian AR process; these are controlled mechanism results, not N-BaIoT confirmation.

For stationary Gaussian AR(1), the expected centered sample variance is `q_m(ρ)σ²`, where `q_m(ρ)=1−2Σ_{h=1}^{m−1}(m−h)ρ^h/[m(m−1)]`; the peer sample-mean variance is `σ²[m+2Σ_{h=1}^{m−1}(m−h)ρ^h]/m²`. At ρ=.7 and m=100, these are .9544σ² and .0551σ², compared with σ² and .0100σ² under independence. This explains why finite peer sampling and temporal dependence affect within-SD and between-sample-mean terms differently; C81 evaluates the uncorrected production-style summaries, not a proposed correction.

Across target/peer mismatch factors, mean feature log-SD RMSE for median sample versus exact median at m=100/1000/5000 is .2270/.2199/.2191 under independent peers and .2408/.2226/.2206 at peer ρ=.7. For pooled sample versus exact pooled it is .3072/.3025/.3026 and .3101/.3034/.3025, respectively. Thus finite-source error shrinks with m and is larger under serial dependence, especially for the median-SD summary. Yet the OAS metrics barely move when replacing sampled with exact source scales: at m=100, ρ=.7, median-sample minus exact-median AUROC averages −.00004 (paired q10 −.00187), and pooled-sample minus exact-pooled averages −.00005 (q10 −.00164). Sample shared center/scale versus exact shared has a larger AUROC effect, −.00102 (q10 −.00865) in that condition, showing peer-center noise matters more than source-scale noise here, though the magnitude remains small.

Against sampled full shared marginals, median/local-center mean AUROC deltas at n=30/100/300/1000 are −.01199/−.00432/−.00135/−.00031; pooled/local-center gives −.01223/−.00433/−.00135/−.00030. For pAUC(.01), the corresponding median values are −.00620/−.00284/−.00099/−.00030 and pooled values −.00638/−.00290/−.00101/−.00032. For pAUC(.05), median gives −.00919/−.00380/−.00132/−.00031 and pooled −.00944/−.00382/−.00132/−.00031. These averages balance all source sample sizes, dependence levels, heterogeneity/mismatch factors, and threat variants; lower-tail paired deltas remain negative. Median versus pooled mean detector differences stay close to zero as peer m and dependence vary. C81 therefore confirms that realistic peer-summary noise does not rescue the upload-saving median scale or explain C80's N-BaIoT device harms; scale-estimation error and OAS ranking loss remain weakly aligned. The result does not establish non-inferiority, transfer safety, or a new candidate. Next test the derived AR finite-sample bias correction only as a risk-matched sensitivity analysis, then retain it only if it changes detector utility as well as scale RMSE.

## C82 — AR(1) moment correction improves some scale-risk cells but leaves detector rankings unchanged

C82 (`pocs/ar_bias_corrected_peer_scale_oas_poc.py`) tests the C81 follow-up with eight finite AR(1) peers, peer m={100,1000}, peer ρ={0,.7}, four target supports, two center-offset levels, two peer log-SD heterogeneity levels, and two target-scale mismatch levels: 128 cells × 30 paired replicates. Four attack variants and eleven actions yield 168,960 complete rows; all method/threat/replicate combinations are balanced and contain no missing values. The earlier progress line incorrectly used a 256-cell denominator; this bookkeeping typo is fixed in the ignored script and did not affect the run or its output. Target support, benign test rows, attacks, and seeds are paired across methods.

The exact stationary-Gaussian AR(1) moment factors are `q_m(ρ)=1−2Σ_{h=1}^{m−1}(m−h)ρ^h/[m(m−1)]` for the expectation of the centered sample variance divided by σ², and `c_m(ρ)=[m+2Σ_{h=1}^{m−1}(m−h)ρ^h]/m²` for sample-mean variance divided by σ². The oracle correction divides each peer sample variance by `q_m`; pooled total variance additionally subtracts the expected finite-peer sample-mean noise `(K−1)/K × mean_k[c_m s²_k/q_m]` from the observed between-peer mean variance, clipped at zero. The estimated-ρ arm uses a pooled lag-one estimate from peer sequences. This is an AR moment correction, not a newly proposed estimator. The correction removes only this stationary sampling component; it does not model temporal regime change, peer-specific ρ, or non-Gaussianity. Correcting sample variances by `q_m` also does not make their median square roots exactly unbiased.

The pooled lag-one estimate behaved well under this deliberately common-ρ design: mean estimated ρ was .683 at true .7 with m=100 and .698 with m=1000; at true zero it was .002 and .001. The median-scale correction reduced mean feature log-SD RMSE from .23878 to .23426 (oracle ρ) / .23454 (estimated ρ) for m=100, ρ=.7, while the exact-source median reference was .21709. At m=100, ρ=0 it changed .22649 not at all; at m=1000 effects were at most about .001. For pooled total scale, oracle correction changed RMSE from .30650 to .30489 at m=100, ρ=0, but slightly worsened it from .30912 to .30938 at m=100, ρ=.7; the corresponding exact pooled references were .30100 and .30160. At m=1000 the changes were below .0002. Thus the scale-risk effect is modest and not uniformly favorable.

Against sampled full shared marginals, corrected median/local-center AUROC deltas at n=30/100/300/1000 were −.01917/−.00689/−.00165/−.00034; oracle and estimated corrections have the same values to reported precision as raw median scale. Corrected pooled/local-center deltas were −.01930/−.00690/−.00163/−.00033. The paired pAUC(.01) deltas were −.01014/−.00392/−.00118/−.00031 for median and −.01018/−.00390/−.00120/−.00029 for pooled; pAUC(.05) was −.01509/−.00562/−.00164/−.00050 for median and −.01513/−.00564/−.00163/−.00050 for pooled. Correction-versus-raw mean detector changes were at most 5×10⁻⁷ for AUROC and pAUC(.01), and 4×10⁻⁶ for pAUC(.05). The median correction rescales all feature scales by a common factor in this design, which leaves Mahalanobis ranking essentially invariant; pooled correction changes features slightly through its between-mean term but does not improve ranking.

C82 therefore answers the specific C81 follow-up negatively: finite AR source variance bias can be reduced modestly, but the correction does not improve the ranking objective or close the shared-marginals gap. The ranking/scale-risk mismatch persists, and estimated common ρ is not a safe correction under heterogeneous or regime-changing peer dependence. No candidate or novelty claim follows. Return to the broader unresolved problem of predicting target-relevant ranking benefit, with local-scale and peer-scale errors kept distinct from benign-only anomaly-ranking utility.

## C83 — a bounded-likelihood-ratio AUROC guarantee still collapses to the unchanged score

To test whether C37's arbitrary-attack impossibility could be softened into a useful benign-only safety gate, C83 (`pocs/bounded_lr_auroc_rank_robustness_poc.py`) derives the exact distributionally robust AUROC change for a fixed candidate score `s1` versus baseline `s0` when the attack law is restricted to reweighting the target benign law `P0`: `Q≪P0`, `0≤w=dQ/dP0≤M`, and `E_P0[w]=1`. Write `H_s(z)=P0(s(X)<s(z))+0.5P0(s(X)=s(z))` and `Δ(z)=H_s1(z)−H_s0(z)`. Then the worst-case AUROC change is `inf_w E_P0[wΔ]=M∫_0^(1/M) F_Δ^{-1}(u)du`, the lower-tail average of the benign rank-percentile change. Since each continuous-score probability integral transform has mean 1/2, `E_P0[Δ]=0`; for any `M>1`, this infimum is strictly negative whenever the candidate changes ranks on a set of positive benign probability. Thus even this restricted likelihood-ratio ball cannot certify a strict AUROC improvement over a rank-changing baseline. A no-harm gate that includes the baseline must abstain unless the candidate preserves its ranking almost surely. This is a standard lower-tail/DRO variational result, an extension of C37's interpretation rather than a new algorithm or theorem claim.

The ignored synthetic diagnostic uses a 32-feature correlated Gaussian benign law, n=100 target support, eight finite peers, and production OAS for local, median-peer-scale/local-center, and shared-marginals scores. On 50,000 independent benign reference points, local and median-peer-scale percentile transforms differ from shared on >99.9% of reference points in this realization (Spearman ρ=.9761/.9765). The empirical exact worst-case AUROC changes against shared are −.00323/−.00320 at `M=1.01`, −.01552/−.01528 at `M=1.1`, and −.03972/−.03950 at `M=2`, respectively. These values are lower bounds attained by attacks that only reweight benign support; they are not estimates for N-BaIoT attacks. Such bounded-density attacks cannot place mass outside the benign law's support, so the guarantee's practical threat scope is narrow for genuine anomalies. In particular, neither this calculation nor C37 licenses a distribution-free negative-transfer claim, and the result supplies no way to select a better score from benign data alone.

The closest direct optimization prior located is Ma & Lejeune's DR-AUC, which optimizes a hinge surrogate of pairwise classification AUC over a Kantorovich/Wasserstein ambiguity set using labeled positive-negative pairs; it does not solve benign-only score selection for a cold-start target ([paper](https://doi.org/10.1016/j.orl.2020.05.012), [preprint](https://arxiv.org/abs/2002.07345)). More broadly, the `L∞` likelihood-ratio uncertainty set and lower-tail extremization are established DRO machinery. Novelty is not claimed. C83 confirms that adding a mild bounded-reweighting assumption still yields only baseline-preserving abstention; the live search must therefore pursue a different operational estimand with observable validation evidence or explicitly use a defensible attack prior. No confirmatory protocol is locked.

## C84–C86 — labeled source attacks do not select a transferable OAS action

These three ignored POCs revisit whether external attack examples can provide the threat information that benign-only validation lacks. For each of the nine physical N-BaIoT devices as held-out target, the other eight devices supply labeled attack exemplars, up to 64 rows per source-device × family cell for Gafgyt/Mirai. Source attack family labels were reconstructed from raw files and checked against the prepared attack arrays. Target attack labels are excluded from action selection. The selector chooses among shared full marginals, target-local center/local scale, and target-local center/median-peer-SD; it then refits the chosen action on the target's full benign support. There are ten paired support windows at each n=30/100/300/1000 and 3,000 target benign and attack evaluation rows per cell. Source messages average about 910 labeled attack rows (896–960), 115 features, or 412–442 KB at float32 per target, versus 7,360 bytes for eight peers' two-vector mean/scale summaries, roughly 56–60 times as much before labels, protocol, or secure-transport overhead.

C84 used one quarter of target benign support for validation and source AUROC for action selection. It selected shared in 90/90, 87/90, 87/90, and 80/90 cells as n increased. The selected action was effectively the shared baseline: mean AUROC changes were 0, +.000052, −.000036, and +.000292. The small n=1000 gain did not replicate under cross-fitting.

C85 replaced the single holdout with five-fold cross-fitting. Shared was selected in 88/90, 81/90, 77/90, and 69/90 cells. Mean selected-action AUROC changes versus shared were −.00023/−.00064/−.00001/−.00307 at n=30/100/300/1000; standardized pAUC(.01) changes were −.00430/−.00437/−.00017/−.00959. At n=1000 only 13.3% of paired cells improved and 10% worsened. The oracle that knows which of the three actions wins on target attacks has only +.00028 to +.00140 mean AUROC headroom over shared, indicating that this action set itself offers little average upside. Negative transfer is concentrated in Mirai: among seven eligible targets the selected action averages −.00855 Mirai AUROC, with 5.7% paired wins; Philips averages −.0575. Gafgyt averages +.00034, demonstrating a source-family/target-family reversal.

C86 retains the five-fold split but selects by standardized source pAUC(.05), targeting low-FPR ranking. Shared is selected in 88/90, 80/90, 79/90, and 67/90 cells; local/peer-scale selections increase with n. Mean AUROC changes are −.00023/−.00070/+.000183/−.003052 and pAUC(.01) changes −.004296/−.004391/+.002009/−.007462. The n=300 point gain has a zero 10th percentile and only 11% paired wins; harms remain at n=1000. Source low-FPR performance also predicts target benefit poorly. The output CSV was fully written before a stale summary-print expression caused the process to exit nonzero; the ignored script was corrected afterward, and the expensive run was not repeated because its complete output is intact.

The direct prior-art threat is ACT: Wang et al. (AAAI 2023) use labeled source normal/anomalous graph nodes and unlabeled target nodes, optimizing `L_joint=L_dom+L_con`; `L_dom` aligns source/target normal representations through Sinkhorn-approximated Wasserstein distance while penalizing source anomalies' deviation from the normal reference, followed by target pseudo-label/deviation learning ([official paper](https://ojs.aaai.org/index.php/AAAI/article/download/25591/25363)). TLNP (Kalan et al., AISTATS 2025) is also close but relies on scarce target anomalies and a Neyman–Pearson constraint. C84–C86 differ in adapting raw-feature OAS normalization actions from other devices' labeled attacks, without target anomaly labels, but this narrow difference does not overcome the selector's failed transfer, high raw-data payload, attack-family reversal, and established labeled-source anomaly-transfer prior art. No candidate is promoted. The result diagnoses that source anomaly labels do not by themselves identify a target-optimal normalization action; next work should test another observable operational estimand or redesign the adaptation object, then search its closest prior art before another large POC. No protocol is locked.

## C87 — source attacks plus target benign support improve mean OAS AUROC, with an unresolved lower-tail/device tradeoff

C87 changes the adaptation object from choosing among three OAS normalization actions to training a target-specific class boundary. For each of the nine physical N-BaIoT targets, the other eight devices contribute up to 64 prepared labeled attack rows per available source-device × Gafgyt/Mirai family. The target contributes only benign support rows at n=30/100/300/1000. Source attacks are positive examples and target support benigns are negative examples. Using the same shared-marginal center/scale as the OAS reference to standardize features, the class-balanced L2 logistic detector solves the standard penalized empirical log-loss objective; C∈{.01,.1,1} was a first 10-window sensitivity screen, followed by 30 paired support windows/device with C=.01/.1. Target attack labels enter only evaluation. Each cell uses the same support window and target test subset for both logistic and `shared-marginals`/local OAS. The expanded CSV has 4,320 complete rows: 9 targets × 4 supports × 30 paired windows × four methods; the 960 missing Mirai-family rows are expected because two devices have no Mirai test attacks.

At 30 windows and C=.01, mean AUROC changes versus shared OAS are +.00589/+.01261/+.00764/+.00405 for n=30/100/300/1000. Standardized pAUC(.01) changes are +.01179/+.02069/+.02640/+.04867 and pAUC(.05) changes +.01412/+.02696/+.02381/+.03242. AUROC paired-win rates are 57.4%/60.7%/55.9%/68.9%, with positive device means on 6/9, 6/9, 8/9, and 8/9 devices. The worst device mean is Danmini at −.00169/−.00178/−.00164/−.00115; C=.1 is similar. At n=1000, C=.01 pAUC(.01) is positive on 7/9 device means but has paired q10 −.07891; pAUC(.05) is positive on 8/9 but q10 −.01706. Thus positive averages and sign counts do not provide low-tail protection.

The AUROC advantage at n=30/100 is concentrated on Provision PT-737/PT-838 (+.0128/+ .0282 mean device deltas at n=30; +.0415/+ .0660 at n=100). Excluding those two devices, mean AUROC deltas are +.00171/+ .00086 at n=30/100, +.00346 at n=300, and +.00417 at n=1000, so the larger-support gain is not only a Provision effect. Family averages at n=1000 also differ: Gafgyt AUROC improves +.00501 while Mirai improves only +.00056; both standardized pAUC(.01) means improve (+.05671/+ .01916), but family-wise paired lower tails remain negative. The initial 10-window sensitivity run found mean AUROC deltas within .0007 across C=.01/.1/1, so the direction is not visibly dependent on one tested C; this does not substitute for broader support or device replication.

The adaptation is standard supervised transfer under a one-class target contract, not a new estimator. Chen & Liu (2014) directly formulate two-class target learning when only one target class is observed: source AdaBoost weak features use `h_f(x)=sign log(p_f⁺/p_f⁻)`, then per-feature Gaussian-process regression maps the observed target-class distribution to the missing class distribution, `p̂_T⁺=R(p_T⁻)` ([paper](https://doi.org/10.1016/j.patrec.2013.07.017); [equations/preprint](https://cvlab.cse.msu.edu/pdfs/Chen_Liu_PRL2014.pdf)). LOCIT and ACT provide further labeled-source anomaly-transfer prior art recorded in the novelty matrix. C87 instead trains a regularized linear log-odds boundary directly from source attack examples and target benign support, with shared OAS scaling; this specific N-BaIoT application does not create a broad algorithmic novelty claim.

Communication remains a large weakness: targets receive 896–960 raw float32 attack rows of 115 features (mean 910 rows, 419 KB) versus 7.36 KB for eight peers' two-vector mean/scale summaries, about 57× before framing/security overhead. A trained linear model is only about 464 bytes at float32, but fitting it centrally requires shipping the raw source attack examples in this POC; privacy-preserving distributed training was not implemented. The method is a standalone linear detector and has not been tested with the AE. The fixed N-BaIoT physical devices/test sets are reused, so no independent confirmation or second-cohort claim follows.

C87 is the strongest positive mean detector-transfer signal in this source-attack branch, but not the final algorithm: its pAUC lower tails still contain sizable losses, Danmini remains slightly harmed, low-support gains depend heavily on two devices, Mirai AUROC transfer nearly vanishes at n=1000, and privacy/communication cost is high. C88–C90 below test benign-calibrated score fusion and do not remove those concerns. No protocol is locked.

## C88–C90 — rank transforms lose score information; affine blends preserve OAS but leave a smaller tail risk

These exploratory follow-ups use the same nine physical N-BaIoT devices, 30 paired support windows per device, support sizes n=30/100/300/1000, and target attack rows only for evaluation. C87's shared-marginal OAS and source-attack/target-benign logistic scores are the two endpoints. No target attack labels are used to fit or calibrate either endpoint. Rank fusion is established prior art: Marques et al. (2023) evaluate reciprocal-rank fusion, `RRF(s)=Στ 1/(ε+τ(s))`, in one-class anomaly ensembles ([paper](https://link.springer.com/article/10.1007/s10618-023-00931-x)); these blend POCs are diagnostic applications, not a new fusion principle.

C88 (`pocs/source_attack_oof_rank_blend_poc.py`) cross-fits both models over three target-benign folds, maps held-out benign and test scores to empirical percentiles, and averages the fold percentiles before blending at α=.25/.50/.75. The 7,560-row output is complete, and its full-support OAS/logistic references exactly reproduce C87's paired scores (maximum AUROC difference zero). Relative to full-support shared OAS, the α=.50 blend has mean AUROC deltas −.04044/−.01516/−.00175/+.00176 at n=30/100/300/1000, with paired win rates .104/.163/.300/.567. Its standardized pAUC(.01) deltas are −.23096/−.18163/−.08684/−.00038, and the n=1000 q10 remains −.14701. The apparent low-support failure is partly a comparison cost: each fitted endpoint uses only two thirds of the available support, and the empirical tail ranks are coarse. It is not a fair baseline-preserving blend and is not promoted.

C89 (`pocs/source_attack_support_rank_blend_poc.py`) instead fits both endpoints on full support and maps their in-sample support scores to empirical percentiles before blending. Its 7,560 rows are complete. This avoids the C88 training-data reduction, but an empirical CDF is bounded: test scores beyond the benign support's score range collapse to tied percentiles. At n=1000, the support-rank OAS alone loses .04306 mean AUROC and .28455 pAUC(.01) against raw shared OAS. The α=.50 blend's AUROC gain is only +.00093, while pAUC(.01) is −.02880 on average with q10 −.17446; n≤300 is materially worse. This diagnosed tail saturation rules out the in-sample ECDF map here.

C90 (`pocs/source_attack_zscore_blend_poc.py`) replaces bounded ranks with affine score standardization using either benign-support mean/SD or median/MAD (normal-consistency factor 1.4826), then blends standardized endpoint scores at α=.25/.50/.75. All 12,960 rows are present: 9 devices × 4 support sizes × 30 paired windows × 12 methods. Both standardized OAS-only endpoints reproduce the raw shared-OAS AUROC exactly on every pair because positive affine scaling preserves its ranking. For the standard z-score α=.50 blend, mean AUROC changes over raw shared OAS are +.00020/+.00045/+.00223/+.00128 and standardized pAUC(.01) changes +.00115/+.00094/+.00404/+.01068 for n=30/100/300/1000. At n=1000 the mean device delta is positive for all nine devices and 85.2% of paired AUROC cells improve, but pAUC(.01) still has q05 −.00064 and a worst paired delta −.01153; pAUC(.05) worst is −.00233. The robust median/MAD scale behaves similarly but gives smaller mean lift at n=1000. These blends temper C87's transfer contribution enough to stay near the shared baseline and yield a small average tail-metric lift, yet they do not supply lower-tail protection or a benign-only safety guarantee.

Conclusion: C88 loses support efficiency, C89's bounded ECDF destroys out-of-range score ordering, and C90's affine scaling avoids that failure but shrinks most of C87's average detector gain while retaining occasional low-FPR harm. Do not promote a blend. C87 remains the stronger exploratory mean signal; C90 is evidence that simple score scaling can keep a blend close to the baseline, not that it resolves the target ranking-selection problem. Continue the broader search for an observable operational objective with justified assumptions and an independently evaluable transfer signal. No confirmatory protocol is locked.

## C91 — source-domain maximin validation adapts the blend by support size, but does not eliminate target harm

The ignored `pocs/source_attack_lodo_blend_selection_poc.py` tests whether C90's fixed blend weight can be selected without the actual target's attack labels. For each outer target device, its eight other devices are the eligible labeled sources. Each source device is used as a pseudo-target in turn; its benign support is combined with attack examples from the other seven devices to fit the same C=.01 logistic detector, then that pseudo-target's labeled attacks validate the blend. Ten deterministic support windows per pseudo-target/support size produce eight domain-level validation means. The rule selects `α*=argmax_{α∈{0,.25,.5,.75,1}} min_{j∈S_t} (1/10)Σ_r [pAUC01_{j,r}(α)−pAUC01_{j,r}(0)]`. The outer target's attacks are never used to choose its α. The chosen value is then evaluated on the same 30 target support windows, test benign/attack subsets, and C87/C90 source attack rows as the fixed blend; all final endpoint and fixed-alpha AUROC/pAUC(.01) values reproduce the corresponding C90 outputs exactly (maximum absolute difference zero). The 5,400-row evaluation and 180-row validation CSVs are complete.

This selector has no novelty claim. Leave-one-domain-out validation is a standard model-selection criterion: DomainBed explicitly lists IID, held-out-domain, and test-domain selection, and Gulrajani & Lopez-Paz argue that model-selection assumptions are part of a domain-generalization algorithm ([paper](https://arxiv.org/abs/2007.01434); [DomainBed implementation](https://github.com/facebookresearch/DomainBed)). Lu et al.'s source-domain model selection evaluates candidates on a validation distribution and chooses the highest held-out accuracy; their related time-series/domain-generalization study also discusses leave-one-domain-out validation ([paper and equations](https://proceedings.mlr.press/v218/lu23a.html)). The worst-domain criterion is also established: Mo et al.'s empirical minimax-regret objective is `min_θ max_k { n_k⁻¹Σ_i ℓ_θ(Z_i^k) − inf_β n_k⁻¹Σ_i ℓ_β(Z_i^k) }` ([paper](https://arxiv.org/html/2405.01709v2)). C91 uses relative pAUC improvement against one fixed OAS baseline rather than per-domain optimal-model regret, but it remains ordinary source-domain model selection, with only eight fixed source-device folds and no theorem for a new device.

The selector chooses α=0 for all nine targets at n=30, α=.75 for all targets at n=1000, and mixed values at n=100/300 (n=100 counts: two α=0, three α=.5, four α=.75; n=300: two α=0, one α=.25, one α=.5, five α=.75). Thus at small support it discards C87's mean gains, while at large support it increasingly relies on the source-trained classifier rather than letting external attack information fade. At n=1000 it is exactly the fixed C90 α=.75 blend, so the selector adds no benefit over that fixed choice.

Against exact shared OAS, selected-rule mean AUROC deltas are +0/+.000388/+.002472/+.002109 for n=30/100/300/1000, and standardized pAUC(.01) deltas are +0/+.001934/+.005501/+.022759. AUROC paired-win rates are 0%/49.3%/59.6%/84.1%; positive device means are 0/9, 6/9, 7/9, and 9/9. For pAUC(.01), paired-win rates are 0%/47.4%/58.1%/83.3%, positive device means are 0/9, 6/9, 7/9, and 9/9, and q10 deltas are 0/−.000152/0/−.001068. The n=100 worst device mean is Danmini at −.001268; the worst paired pAUC(.01) losses are 0/−.013445/−.007052/−.012440 and q05 is 0/−.000561/−.001473/−.004020. The positive average at n=1000 equals C90's static α=.75 result and still has a −.01244 paired low-FPR loss; source validation therefore does not establish no-harm transfer. At n=300 it improves the mean pAUC(.01) over fixed α=.50 (+.00550 vs +.00404) but remains below fixed α=.75 (+.00861), while the selected worst paired loss (−.00705) is larger than fixed α=.50's (−.00371). The different criteria expose a mean/tail tradeoff rather than a uniquely supported alpha.

Conclusion: source leave-one-device-out validation produces a support-dependent alpha, but the sequence (α=0 at n=30; mostly α=.5/.75 by n=100–300; α=.75 for every target at n=1000) does not support safe transfer or naturally vanishing source use. It offers no communication reduction from C87 and does not improve on fixed α=.75 at n=1000. Keep C91 as a diagnostic for source-domain validation, not a promoted algorithm or a novelty claim. The C87 supervised-transfer branch remains exploratory, its source-label contract and raw payload remain substantial, and no second physical cohort or confirmatory split exists. Continue the broader search from the unresolved scale/ranking-loss mismatch; keep the protocol unlocked.


## C92 — benign source similarity does not identify useful attack transfer

The ignored `pocs/source_attack_benign_similarity_poc.py` tests whether a simple benign-only descriptor can rank the eight possible labeled-attack sources for each outer target. It fits a separate class-balanced logistic regression (`C=.01`) per source, using the target's sampled benign support and that source's attack rows; every source is scored on the same target benign/attack evaluation rows and compared with the same shared-OAS scores. The two source/target compatibility descriptors are RMS standardized mean distance and RMS featurewise log-SD distance, computed from target benign support and the source benign pool. Across 9 targets × 8 sources × 4 support sizes × 15 paired windows, all 4,320 rows are present. The script reuses the C87/C90/C91 devices, deterministic support windows, target test rows, and source attack construction. No target attack labels enter fitting or descriptor construction; target labels are used only to report transfer utility. This is a source-relevance screen, not an additive decomposition of C87's pooled-source classifier.

For paired pAUC(.01) gain over shared OAS, median within-target/window Spearman correlations of gain with benign mean distance are +.226/+ .202/+.024/+.024 at n=30/100/300/1000; correlations with log-SD distance are +.314/+.357/+.357/+.286. The sign is mostly positive, opposite the simple hypothesis that a closer benign source is more useful for attack transfer. These correlations are modest and heterogeneous across target/window (some correlations are undefined when inputs are constant); they do not support a monotone source selector. Excluding the Provision cameras as both sources and targets leaves the aggregate direction mostly positive, so the pattern is not solely a Provision artifact, but it remains a reused nine-device diagnostic rather than external validation.

Fitting a single source-specific classifier has mean pAUC(.01) deltas +.01776/+.01789/+.02221/+.04342 over shared OAS at n=30/100/300/1000, but only 36/72, 47/72, 57/72, and 56/72 source-target mean pairs are positive. Paired q10 deltas are −.06560/−.08004/−.07330/−.08096, and the worst paired losses range from −.11086 to −.17181. The worst mean source-target pair is Danmini target with SimpleHome XCS7-1002 source at all supports (−.07814/−.09027/−.08734/−.06823). Family effects reverse with support: Gafgyt mean pAUC(.01) transfer is −.00070/+.00394/+.02722/+.05343, while Mirai is +.04150/+.03454/+.00271/+.00760. Source-specific AUROC means are positive but much smaller (+.00644/+.01164/+.00559/+.00348), likewise with device variation. Thus positive average transfer does not translate into a reliable benign-only compatibility signal or tail protection.

Prior art makes no novelty case for C92: source-target similarity weighting and compatibility filtering are established in transfer and federated learning; LOCIT (Vercruyssen et al., AAAI 2020) already gates labeled source anomaly instances using target/source local neighborhood location and covariance, while generic target-dependent source weighting is standard. C92's two benign distances are simple diagnostics, not a new estimator or rule. More fundamentally, benign marginal similarity need not predict whether a source's labeled attack family will transfer. The single-source fit is also not additive: the pooled C87 classifier's coefficients and regularization change when all sources are included, so these per-source deltas cannot be summed into a valid C87 source contribution or communication allocation.

Conclusion: do not select or weight sources by these benign distances, and do not promote one-source logistic transfer. This narrows the useful next question to whether a target-benign, attack-label-free signal can estimate *ranking utility* with credible uncertainty under family shift; the present simple marginal-distance hypothesis does not. C87 remains the strongest positive mean detector-transfer signal, but its harmful tails, source-family dependence, roughly 57× raw attack payload, and one-cohort reuse remain unresolved. The leading scale-based mechanisms still lack detector-ranking advantage over fixed shared marginals. No method is promoted and the confirmation protocol remains unlocked.
