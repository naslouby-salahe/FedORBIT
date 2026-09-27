# Novelty matrix: candidate FedORBIT cold-start collaboration mechanisms vs. closest prior art

Scope: algorithm search through 2026-09-27. Baseline to beat is `shared-marginals` (fixed, 100% partner replacement of local mean/std), not local-only. This is the sole live novelty matrix; the durable checkpoint summarizes the current conclusions.

## Closest prior art found (real sources, access level noted)

| Work | What is shared | Adapts strength? | Explicit uncertainty? | New/unseen clients? | Needs labels? | Partner selection? | Negative-transfer risk? | IoT anomaly detection, physical devices? | Access |
|---|---|---|---|---|---|---|---|---|---|
| Ferber-style weighted model averaging, "Optimal Model Averaging: Towards Personalized Collaborative Learning" (arXiv:2110.12946) | scalar local/global model estimates | yes, in principle (paper studies benefit as function of weight) | bias-variance tradeoff framing, but full derivation not confirmed (abstract only accessible; PDF unreadable here) | not stated | not stated | no | discussed as "possibly negative benefit" | no | abstract only, full text inaccessible |
| DerSimonian-Laird / random-effects meta-analysis (foundational stats, not FL-specific) | N/A (statistical framework) | yes, tau^2-driven inverse-variance weights | yes, explicit between-study heterogeneity tau^2 | N/A | N/A | N/A | tau^2 absorbs it but no "reject" mechanism | no | well-established, secondary sources only (full DL derivation not independently re-fetched) |
| FedCOF, "Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning" (arXiv:2412.14326) | per-client means only; covariance reconstructed from between-client mean dispersion | yes, uncertainty-weighted client aggregation | yes, uncertainty from estimated covariance structure | not confirmed | classification (prototype-style), so implicitly yes | no | not confirmed | no (classification, not anomaly/IoT) | full text fetched via WebFetch |
| FedCollab (Bao et al., ICML 2023) | client cluster membership decided from distribution-distance + data-quantity; standard model updates within a coalition | yes, more collaborators when a client has less data | not framed as explicit uncertainty, more a clustering distance | not confirmed | supervised (classification benchmarks) | yes, hard coalition clustering | explicit target of the method | no | abstract only |
| Adaptive Bayesian Partner Selection (arXiv:2609.16446, submitted 2026-09-15), clinical FL | model updates; Beta-Bernoulli posterior over peer Shapley utility, UCB ranking | yes, bandit-style | yes, explicit posterior uncertainty over peer utility | not addressed | yes (mortality labels) | yes, hard select/abstain, not soft weighting | explicitly handles negative-transfer abstention | no (healthcare) | abstract/full preprint located; exact implementation/equations need audit |
| "Enhanced Federated Anomaly Detection Through Autoencoders Using Summary Statistics-Based Thresholding" (arXiv:2410.09284) | summary statistics aggregated into a global anomaly threshold | not confirmed | not confirmed | not confirmed | unsupervised (autoencoder) | not confirmed | not confirmed | no (Credit Card Fraud, Shuttle, Covertype) | abstract only |
| StatAvg (Bouzinis et al., IEEE TNSM 2025) | client feature means and variances; count-weighted global pooled mean/variance, broadcast for universal normalization | no, all participating clients use pooled stats | no | no; assumes clients participate in the FL normalization/training | yes, supervised IDS model training | no | no explicit abstention; global pooling | IDS on TON-IoT and CIC-IoT-2023; not benign-only new physical-device adaptation | full equations, algorithm, and experiments inspected |
| Federated BatchNorm (Guerraoui et al., arXiv:2405.14670, 2024) | BN statistics aggregated to approximate centralized BN; proposes robustification to erroneous/adversarial stats | global BN stats; no feature-wise cold-start rule identified | robustifies erroneous stats | unseen-client use not established | deep-model training | no target partner selection identified | aims to mitigate erroneous statistics | deep-model normalization, not benign-only physical-device anomaly estimation | abstract/PDF located; exact equations still need audit |
| Population Normalization (CVPR 2025) | population means/variances are trainable parameters with constraints derived from their definitions | shared trainable normalization parameters, no per-target uncertainty weighting identified | injects noise to model BN statistical uncertainty at larger batch sizes | no held-out new-client cold start identified | supervised deep FL | no partner selection | addresses heterogeneity and small-batch estimation | deep image classification, not anomaly detection | CVPR full paper/supplement available; equations need review |
| OASD (IMF WP/23/257, 2023) | local sample covariance shrunk toward `diag(S)` rather than average-variance identity | closed-form/iterative OAS shrinkage intensity | uncertainty in sample covariance through OAS approximation, Gaussian assumptions | no | unlabeled covariance estimation | none | target is local sample diagonal, no transfer safety | general high-dimensional covariance, p>n | full derivation, theorem, and simulations inspected |
| Multiple-target linear covariance shrinkage (Gray et al., 2018; Oriol, 2024) | sample covariance combined with multiple structured covariance target matrices; Gray et al. use external biological sources | yes, target-specific Bayesian/linear shrinkage weights | target uncertainty is explicit in multi-target construction | not a cold-start client method | depends on application; external priors may be unlabeled | multi-source target weighting, not client partner selection | handles target uncertainty; mismatch remains modelled | covariance estimation, not federated anomaly detection | full papers inspected; closest estimator-family prior art |
| Jing et al., *Tuning-Free Covariance-Aware Sequential Shrinkage for Multi-Source Estimation* (arXiv:2606.30615v2, 2026-09-21) | source-specific target estimators and covariance estimates; sequential, not pooled-first | yes; closed-form SURE/risk-bound step size and priority scores | yes; estimator covariance enters shrinkage direction and size | generic target dataset; not specifically new FL clients | mean estimation and smooth M-estimation | sequential source-specific updates | finite-sample risk-improving range; asymptotically discounts heterogeneous sources under stated separation | no IoT anomaly application | full equations/theorems inspected |
| FedBN (Li et al., ICLR 2021) | conv weights shared, BatchNorm running statistics kept strictly local | N/A - opposite design choice | no | not central | supervised | no | avoided by never sharing stats | no | abstract/summary |
| Ledoit-Wolf / OAS shrinkage covariance literature (foundational stats) | N/A | shrinkage intensity adapts to n and dimension | yes (asymptotic MSE-optimal shrinkage) | N/A | N/A | N/A | N/A | no | well-established, not FL-specific |

**Hostile gap statement (revised 2026-09-27).** StatAvg invalidates any broad novelty claim for federated feature-moment aggregation or shared normalization in IDS: it explicitly communicates means and variances and applies pooled normalization on TON-IoT and CIC-IoT-2023. The remaining possible distinction is narrower: benign-only anomaly estimation for a held-out/new target device with small local support, retaining the local target centre while using leave-target-out peer marginal variances as a diagonal target for OAS covariance regularization. This is a problem-setting/mechanism combination, not yet an established algorithmic novelty; OASD and multi-target covariance shrinkage are close estimator-family prior art. The chi-square gate is not a novelty survivor: its calibration assumes Gaussian i.i.d. target rows, contradicted by heavy-tailed/skewed support data, and its underlying lower-tail test is classical. FedCOF remains adjacent (client means recover class covariance under its assumptions) but is not the closest IDS prior; StatAvg is. The distinction from StatAvg and covariance shrinkage must be supported by exact equations, same-window comparisons, and adversarial search for held-out-client benign-only normalization, not wording alone.

| Comparison dimension | Current FedORBIT direction (unselected) | StatAvg (closest IDS prior) |
|---|---|---|
| Target problem | cold-start anomaly detection on a held-out physical device | supervised federated IDS with participating clients |
| New/unseen client | yes, target excluded from peer summaries | no, target clients supply their moments to global pool |
| Benign-only | yes, support and peer summaries benign; attacks evaluation-only | no, feature moments computed on client training data for supervised IDS |
| Shared information | target local mean and a peer population diagonal scale used as OAS covariance shrinkage target; target omitted from peer aggregate | each client's feature mean and variance |
| Adaptation unit | peer diagonal target across features; OAS intensity depends on target support/covariance | global feature normalization for all clients |
| Uncertainty model | standard OAS intensity only; peer-target mismatch not estimated | none; exact pooled-population moments |
| Negative-transfer handling | none in the fixed peer-target arm; feature-wise risk selection failed to match the detector utility | none beyond global pooling |
| Partner handling | equal-device peer aggregate; no target in aggregate | sample-count-weighted aggregation of all participating clients |
| Closed-form / optimization | closed-form OAS in normalized coordinates, equivalent to raw-space diagonal-target shrinkage | closed-form pooled mean and variance |
| Theory | exact algebraic OAS identity; diagonal/multi-target shrinkage are existing prior art | pooled-moment identity under disjoint client datasets |
| IoT anomaly setting | yes, unsupervised detection on physical-device N-BaIoT | IoT IDS but supervised and on TON-IoT/CIC-IoT-2023 |
| Main mathematical difference | uses a leave-target-out peer marginal diagonal as a high-dimensional covariance regularization target, with local target centre | global union moments normalize participating clients before supervised FL |
| Main empirical difference | benign-only held-out-device anomaly AUROC; same-window peer-target arm nearly ties the shared baseline but is not consistently better | classification metrics for federated training; different estimand and protocol |

## Rejected weak-novelty mechanisms (per instructions, explicitly not pursued as the "novel" claim)

- Sample-count weighting alone (`lambda = n/(n+k)` with no uncertainty term) — tested only as an implicit control inside the two candidates below; not proposed as the final mechanism.
- Hand-tuned convex interpolation with a fixed lambda constant — rejected, no principled derivation.
- FedAvg personalization renamed — out of scope; the registered study already tested FedAvg and found small non-significant marginal gain over marginals.
- Fixed n<100 threshold switch — this is structurally what `shared-marginals` already is at the level of "which channel is active"; not novel.
- Plain nearest-neighbor partner selection — already measured and REJECTED by the confirmatory study (deployable-nearest-2 significantly worse than pooling all partners).
- Plain mean/variance sharing with no adaptive mechanism — this is exactly the existing `shared-marginals` baseline; the bar to beat, not a candidate.
- Vanilla empirical Bayes with no new mechanism — tested as Candidate 1 below; found to fail for a diagnosable, mechanism-specific reason (see POC Results).

## Candidates tested (real POCs against `outputs/prepared/nbaiot`, see `POC Results.md` in this directory)

| ID | Mechanism | Family | Status |
|---|---|---|---|
| C1 | Per-feature precision-weighted (empirical-Bayes/meta-analytic) blend of local vs. partner-aggregate mean and log-std, prior variance = between-partner heterogeneity | A + B | REJECTED — loses to fixed shared-marginals at every n (9/9 devices negative), see diagnosis |
| C2 | James-Stein shrinkage of local mean toward partner mean, computed in partner-normalised coordinates, positive-part JS factor from classical d>=3 theorem | A (with a real theorem) | REJECTED — same failure mode as C1, smaller magnitude |
| C3 | Ablation: which half of "replace everything with partner stats" actually carries the benefit — local mean + partner scale, vs. partner mean + local scale, vs. full local, vs. full marginals | diagnostic, not a deployable candidate | Completed — benefit is predominantly associated with scale replacement, but does not establish why |
| C4 | Dimension-anchored local/peer log-scale blend | shrinkage | Rejected — ΔAUROC vs shared −0.0667 at n=30, −0.0104 at n=1000 |
| C5 | Lower-tail chi-square gate for local variance underestimation | classical hypothesis test | Demoted — near-average tie but 69–76% feature borrowing, negative-transfer cells, and severe non-Gaussian miscalibration |
| C6 | Local MAD scale, normal-consistency factor | robust scale | Rejected as drop-in — ΔAUROC −0.2493 at n=30 and −0.2287 at n=1000; scale about 0.30–0.43× sample SD |
| C7 | Forward-blocked Gaussian marginal NLL chooses local vs peer scale, local centre retained | held-out predictive-risk transfer | Rejected for the current detector — 0/9 positive device deltas vs shared-marginals, mean −0.0993 at n=30; borrows 14–20% and nearly reverts to local-only |
| C8 | Peer diagonal covariance target in OAS; factorial decomposition shows target changes cause nearly all AUROC gain while lambda changes do not | OASD and multi-target shrinkage are close mathematical prior art; the unseen-device benign anomaly setting is the remaining possible distinction | 15-replicate paired POC: fixed-local-intensity peer target gains +0.0799 vs local at n=30 and +0.0124 at n=1000 (9/9 device means); vs exact shared it is −0.0049 at n=30 and +0.0012 at n=1000, with 5/9 and 6/9 device means positive; Gaussian stress test shows mismatch SD 0.5 reverses the small n=30 local gain |

## Current novelty status

No algorithm has been selected. StatAvg is the closest IDS prior art and rules out claims to having
introduced shared feature moments or universal normalization in federated IDS. OASD and multi-target
covariance shrinkage are the closest estimator-family prior art; they already cover diagonal and
multiple/external shrinkage targets. A possible remaining distinction is using a peer population's
diagonal as the shrinkage target for a held-out physical IoT device's benign-only anomaly detector.
This is not yet a novelty claim: exact same-window comparison shows a near tie to the stronger
shared-marginals baseline, and a controlled Gaussian test shows peer-scale mismatch can reverse the
small-sample gain. Target-specific negative-transfer behavior on real hostile peer subsets is untested. A 2026 covariance-aware
multi-source SURE shrinkage preprint is especially relevant to any subsequent safe-transfer rule; its
source-specific risk updates and finite-sample guarantees must be compared directly. The chi-square gate is not a
credible novelty/safety claim because its Gaussian pivot is miscalibrated on heavy tails. Blocked
predictive NLL misses the AUROC benefit. Continue adversarial search against StatAvg, OASD,
multi-target covariance estimators, federated BN/population normalization, and unseen-client
normalization/test-time adaptation.
