# Novelty matrix: candidate FedORBIT cold-start collaboration mechanisms vs. closest prior art

Scope: algorithm search through 2026-09-27. Baseline to beat is `shared-marginals` (fixed, 100% partner replacement of local mean/std), not local-only. This is the sole live novelty matrix; the durable checkpoint summarizes the current conclusions.

## Closest prior art found (real sources, access level noted)

| Work | What is shared | Adapts strength? | Explicit uncertainty? | New/unseen clients? | Needs labels? | Partner selection? | Negative-transfer risk? | IoT anomaly detection, physical devices? | Access |
|---|---|---|---|---|---|---|---|---|---|
| Ferber-style weighted model averaging, "Optimal Model Averaging: Towards Personalized Collaborative Learning" (arXiv:2110.12946) | scalar local/global model estimates | yes, in principle (paper studies benefit as function of weight) | bias-variance tradeoff framing, but full derivation not confirmed (abstract only accessible; PDF unreadable here) | not stated | not stated | no | discussed as "possibly negative benefit" | no | abstract only, full text inaccessible |
| DerSimonian-Laird / random-effects meta-analysis (foundational stats, not FL-specific) | N/A (statistical framework) | yes, tau^2-driven inverse-variance weights | yes, explicit between-study heterogeneity tau^2 | N/A | N/A | N/A | tau^2 absorbs it but no "reject" mechanism | no | well-established, secondary sources only (full DL derivation not independently re-fetched) |
| FedCOF, "Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning" (arXiv:2412.14326) | per-client means only; covariance reconstructed from between-client mean dispersion | yes, uncertainty-weighted client aggregation | yes, uncertainty from estimated covariance structure | not confirmed | classification (prototype-style), so implicitly yes | no | not confirmed | no (classification, not anomaly/IoT) | full text fetched via WebFetch |
| FedCollab (Bao et al., ICML 2023) | client cluster membership decided from distribution-distance + data-quantity; standard model updates within a coalition | yes, more collaborators when a client has less data | not framed as explicit uncertainty, more a clustering distance | not confirmed | supervised (classification benchmarks) | yes, hard coalition clustering | explicit target of the method | no | abstract only |
| Adaptive Bayesian Partner Selection (arXiv:2609.16446, 2025), clinical FL | model updates; Beta-Bernoulli posterior over peer Shapley utility, UCB ranking | yes, bandit-style | yes, explicit posterior uncertainty over peer utility | not addressed | yes (mortality labels) | yes, hard select/abstain, not soft weighting | explicitly handled via abstention | no (healthcare) | abstract only |
| "Enhanced Federated Anomaly Detection Through Autoencoders Using Summary Statistics-Based Thresholding" (arXiv:2410.09284) | summary statistics aggregated into a global anomaly threshold | not confirmed | not confirmed | not confirmed | unsupervised (autoencoder) | not confirmed | not confirmed | no (Credit Card Fraud, Shuttle, Covertype) | abstract only |
| StatAvg (Bouzinis et al., IEEE TNSM 2025) | client feature means and variances; count-weighted global pooled mean/variance, broadcast for universal normalization | no, all participating clients use pooled stats | no | no; assumes clients participate in the FL normalization/training | yes, supervised IDS model training | no | no explicit abstention; global pooling | IDS on TON-IoT and CIC-IoT-2023; not benign-only new physical-device adaptation | full equations, algorithm, and experiments inspected |
| FedBN (Li et al., ICLR 2021) | conv weights shared, BatchNorm running statistics kept strictly local | N/A - opposite design choice | no | not central | supervised | no | avoided by never sharing stats | no | abstract/summary |
| Ledoit-Wolf / OAS shrinkage covariance literature (foundational stats) | N/A | shrinkage intensity adapts to n and dimension | yes (asymptotic MSE-optimal shrinkage) | N/A | N/A | N/A | N/A | no | well-established, not FL-specific |

**Hostile gap statement (revised 2026-09-27).** StatAvg invalidates any broad novelty claim for federated feature-moment aggregation or shared normalization in IDS: it explicitly communicates means and variances and applies pooled normalization on TON-IoT and CIC-IoT-2023. The remaining possible distinction is narrower: benign-only anomaly estimation for a held-out/new target device with small local support, preserving the target local centre while selectively replacing uncertain marginal scales from peer devices. This is a problem-setting/mechanism combination, not yet an established algorithmic novelty. The chi-square gate is not a novelty survivor: its calibration assumes Gaussian i.i.d. target rows, contradicted by heavy-tailed/skewed support data, and its underlying lower-tail test is classical. FedCOF remains adjacent (client means recover class covariance under its assumptions) but is not the closest IDS prior; StatAvg is. The distinction from StatAvg must be supported by a direct evaluation/setup comparison and adversarial search for held-out-client benign-only normalization, not wording alone.

| Comparison dimension | Current FedORBIT direction (provisional) | StatAvg (closest IDS prior) |
|---|---|---|
| Target problem | cold-start anomaly detection on a held-out physical device | supervised federated IDS with participating clients |
| New/unseen client | yes, target excluded from peer summaries | no, target clients supply their moments to global pool |
| Benign-only | yes, support and peer summaries benign; attacks evaluation-only | no, feature moments computed on client training data for supervised IDS |
| Shared information | target local mean plus selected peer marginal scales (candidate under falsification) | each client's feature mean and variance |
| Adaptation unit | per target feature and support window | global feature normalization for all clients |
| Uncertainty model | prior chi-square test; currently invalid under non-Gaussian data | none; exact pooled-population moments |
| Negative-transfer handling | local scale abstention unless test fires; current test uncalibrated | none beyond global pooling |
| Partner handling | equal-device peer aggregate; no target in aggregate | sample-count-weighted aggregation of all participating clients |
| Closed-form / optimization | closed-form local/peer statistic selection | closed-form pooled mean and variance |
| Theory | classical chi-square result with unmet normality assumption | pooled-moment identity under disjoint client datasets |
| IoT anomaly setting | yes, unsupervised detection on physical-device N-BaIoT | IoT IDS but supervised and on TON-IoT/CIC-IoT-2023 |
| Main mathematical difference | feature-local cold-start transfer from peer reference into an unseen target | global union moments for normalization of the same participating population |
| Main empirical difference | local-support ladder against local and full peer marginals | classification metrics for federated training; different estimand and protocol |

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
| C7 | Blocked cross-validated predictive-risk comparison of local and peer scales | held-out proper-score transfer decision | Proposed only — not yet implemented or run; novelty and performance unassessed |

## Current novelty status

No candidate survives. StatAvg is now the closest IDS prior art and rules out claims to having
introduced shared feature moments or universal normalization in federated IDS. A possible remaining
distinction is cold-start benign-only detection for a genuinely held-out physical device with local
support and feature-level scale borrowing. That combination is not by itself proof of novelty. The
chi-square gate is not a credible novelty or safety claim because the exact Gaussian pivot is
miscalibrated on heavy-tailed data. Continue adversarial search against StatAvg and papers on
unseen-client normalization/test-time adaptation before making any novelty claim.
