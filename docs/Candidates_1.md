# Re-audited FedORBIT Scientific Decision Report

I re-did the audit with a substantially harsher standard than in my previous answer. The result is materially different.

The earlier **FARCast recommendation should not be treated as the current primary direction**. It assumed that the existing N-BaIoT preparation could support a defensible longitudinal future-risk forecasting experiment. Your project audit explicitly says the prepared row order has **no verified absolute timing**, the source of the future-FPR inflation is unresolved, and no second compatible physical attack cohort has yet been established. status_report(2) Building a journal algorithm around future trajectory forecasting now would therefore violate your requirement that feasibility be established *before* implementation.

The stronger conclusion after extending the review is:

> **Do not lock a journal algorithm yet.**
>
> I audited **170 unique literature contributions in total**, including **70 additional papers beyond the previous 100**, and then audited **57 distinct algorithm candidates** against FedORBIT's actual data contract, previous failed POCs, prior art and likely publication strength.
>
> **Only three candidates currently justify a new POC.**
>
> 1. **A3 — Hierarchical Federated Marginal Distribution Transfer with Target-Local Dependence**
> 2. **B7 — Federated Hierarchical Exceedance / Extreme-Tail Calibration**
> 3. **E1 — Federated Neyman–Pearson Domain-Generalized Anomaly Scoring**
>
> Everything else is either a baseline/component, already contradicted by your experiments, too close to existing literature, too dependent on unvalidated data, or too likely to overfit the nine-device cohort.

That is a much narrower frontier than the previous answer, but it is much safer.

---

# 1. What FedORBIT actually gives us

The current project is not scientifically empty. It gives three unusually useful facts.

At support 30, equal-peer shared marginal statistics improve AUROC over all three local controls on all nine devices, with mean gains of +0.1034 over local standardization, +0.0315 over unscaled local detection and +0.0467 over the local scale-floor control. The benefit decreases as local benign support increases. But the nominal 1% operating point fails badly: shared-marginal future-test FPR is 19.54% at \(n=30\) and remains 8.65% at \(n=1000\). status_report(2)

The second critical finding is negative. More information is **not automatically better**. Covariance sharing has no reliable registered gain, nearest-two partner selection is worse than using all peers at low support, and FedAvg adds little beyond the marginal-normalization effect. The project has already rejected many empirical-Bayes, reliability, robust-pooling, block, dependence and scale-selection mechanisms. status_report(2)

The third fact is methodological: the strongest simple method to beat is therefore not FedAvg. It is:

\[
\boxed{\text{equal-device all-peer marginal sharing}}
\]

Any next algorithm that cannot beat that simple baseline on **pAUC / low-FPR performance and device tails**, not merely average AUROC, is not worth becoming a paper.

The project itself correctly says there is presently **no established new algorithm/framework**, no independent physical replication, and no established explanation for the threshold failure. status_report(2)

That is the correct starting point.

---

# 2. Correction to my previous FARCast recommendation

I would now classify FARCast as:

**REJECT FOR THE CURRENT PHASE / retain only as a future data-dependent direction.**

Why?

FARCast requires legitimate episodes of:

\[
\text{early benign state}
\rightarrow
\text{later benign state}.
\]

Your audit states that row order has no verified absolute timing, and that finite calibration error, serial dependence and benign distribution change cannot currently be separated. status_report(2)

The external datasets previously discussed have also not cleared your project-level contract. The report explicitly says that no second cohort currently establishes, together, the physical target identity, benign support, chronological later benign observations, attacks, leakage-safe schema and rights/access requirements. status_report(2)

There is also much more neighboring prior art than my previous answer gave enough weight to. ColdFusion explicitly studies cold-start anomaly detection, meta-forecasting methods learn from related historical series to predict short target series, and safe-transfer methods already provide theory for deciding whether auxiliary information should be borrowed. [ACL Anthology](https://aclanthology.org/2024.findings-acl.453/)

FARCast could become interesting if a future timestamped multi-device cohort is established. It is **not the algorithm I would spend implementation time on now**.

---

# 3. Extended literature audit methodology

I did not simply add 50 generic FL papers.

The additional searches targeted the mechanisms that could actually rescue FedORBIT:

- cold-start anomaly detection;
- unseen-client FL;
- full marginal distribution transfer;
- quantile/sketch federation;
- copula anomaly detection;
- hierarchical Bayesian borrowing;
- extreme-value / POT anomaly thresholds;
- small-sample extreme quantiles;
- conformal risk control;
- false-positive / FDR guarantees;
- safe transfer and negative-transfer prevention;
- low-FPR / Neyman–Pearson learning;
- federated domain generalization;
- superquantile/CVaR FL;
- multi-task/meta forecasting;
- new-client personalization;
- anomaly-score normalization under domain shift;
- 2025–2026 federated anomaly-detection frameworks.

For candidate-equivalence searches I searched the **complete mechanism**, not merely its keywords. For example:

> hierarchical GPD/POT partial pooling across federated clients + small benign support + new-client anomaly-score calibration

and:

> federated full marginal distribution estimation + probability-integral transform + target-local dependence.

This matters because several superficially attractive candidates disappeared under exact-mechanism searches.

Two recent papers narrow the frontier particularly strongly. **FCPJS (UAI 2026)** already constructs a global nonconformity-score sketch and performs client-local importance-weighted calibration under joint shift. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v337/shi26a.html) **APEW-Fed (2026)** already contains federated anomaly-score calibration using differentially private quantile sketches. [Nature](https://www.nature.com/articles/s41598-026-60574-y)

Likewise, federated normalization now explicitly includes privacy-preserving ranked-element/median computation, so “federated quantile normalization” by itself is no longer a strong algorithmic contribution. [arXiv](https://arxiv.org/abs/2511.11249)

---

# 4. Literature audit: papers 101–170

These are **70 additional unique contributions** beyond the 100 in my previous audit. Preprint/published duplicates are counted once.

| # | Paper / mechanism | What I extracted | Effect on our candidate space |
|---:|---|---|---|
| 101 | **From Zero to Hero: Cold-Start Anomaly Detection**, ACL Findings 2024 | ColdFusion combines prior zero-shot guidance with a few contaminated target observations | Generic “cold-start AD” is occupied |
| 102 | **Few-Shot Time-Series Anomaly Detection with Unsupervised Domain Adaptation**, Information Sciences 2023 | Few-shot target adaptation from related domains | Low target support alone is not novel |
| 103 | **COFT-AD**, IEEE TIP 2024 | Pretraining + contrastive fine-tuning using few normal target examples | Few-normal-sample detector adaptation is crowded |
| 104 | **Task Calibration for Distributional Uncertainty in Few-Shot Classification**, 2021 | Learns to react to support/query distribution mismatch | Meta-calibration is established |
| 105 | **Few-Shot Anomaly Detection in Text with Deviation Learning**, 2023 | Explicit score learning using few anomalies | Few-shot score learning exists |
| 106 | **LaSCal: Label-Shift Calibration without Target Labels**, 2024 | Label-free target calibration under label shift | Unlabeled-target calibration alone insufficient |
| 107 | **Bayesian Estimation of Extreme Quantiles and the Exceedance Distribution for Paretian Tails**, 2025 | Bayesian tail inference greatly improves small-sample extreme-quantile calibration in studied settings | Strong prior-art foundation for EVT candidate |
| 108 | **Bayesian Extreme Value Models: Asymptotic Behavior, Hierarchical Convergence, and Predictive Robustness**, 2024 | Hierarchical EVT borrows strength among heterogeneous groups | Generic hierarchical EVT is occupied |
| 109 | **Bayesian Joint-Quantile Regression**, 2021 | Borrows information among tail quantiles, particularly useful at small \(n\) | Tail sharing requires a federated-specific delta |
| 110 | **Estimation and Uncertainty Quantification for Extreme Quantile Regions**, Extremes 2021 | Bayesian uncertainty for extreme regions | Plain Bayesian q99 estimation insufficient |
| 111 | **On Quantification of Borrowing of Information in Hierarchical Bayesian Models**, 2025 | Analyzes when deeper hierarchical borrowing helps | Gives theoretical framework for transfer audit |
| 112 | **Extreme Conditional Quantile Estimation in High Dimensions**, 2026 | Compares modern estimators under difficult tail regimes | Tail model choice must be justified |
| 113 | **Bayesian Smoothed Quantile Regression**, 2025 | Better extreme-quantile estimation and MCMC efficiency in experiments | Generic Bayesian quantile regression is crowded |
| 114 | **Distributionally Robust FL with Client Drift Minimization**, 2025 | Worst-client min-max FL objective | Generic worst-client FL already established |
| 115 | **Scaff-PD**, 2023 | Communication-efficient distributionally robust FL | DRO alone gives no novelty |
| 116 | **FedCIM**, 2025 | Iterative personalized clustering for heterogeneous FL | Dynamic client clustering crowded |
| 117 | **Robust FL for Heterogeneous Clients and Unreliable Communications**, IEEE TWC 2024 | Constrained optimization under device/data heterogeneity | Generic constrained FL is occupied |
| 118 | **Federated Learning with Superquantile Aggregation for Heterogeneous Data**, Machine Learning 2023 | Optimizes tail of client loss distribution | CVaR/superquantile objective already exists in FL |
| 119 | **Local Gradient Regulation Stabilizes FL under Client Heterogeneity**, 2026 | Regulates local gradients to reduce drift | Yet another heterogeneity optimizer |
| 120 | **Communication-Efficient Federated Group DRO**, 2024 | CVaR and KL group-DRO algorithms | Generic group-DRO candidate rejected |
| 121 | **FLAME: ADMM in Heterogeneous FL**, 2024 | Personalized/global models with robustness/fairness | ADMM personalization already crowded |
| 122 | **FedAWARE**, 2023 | Server optimization via gradient-diversity maximization | Gradient-based heterogeneity handling irrelevant to core FedORBIT gap |
| 123 | **Learn then Test**, Annals of Applied Statistics 2025 | Finite-sample risk calibration including type-I error for outlier detection | Critical novelty threat to “risk-controlled selector” |
| 124 | **Full-Conformal Novelty Detection**, 2025 | Full-conformal e-values with finite-sample FDR control | Generic guaranteed novelty detection occupied |
| 125 | **Conditional Calibration for FDR Control under Dependence**, AoS 2022 | Dependence-aware finite-sample FDR control | Dependence-adjusted calibration already deep field |
| 126 | **Machine Learning Meets FDR — AdaDetect**, 2022 | Flexible ML scores + FDR guarantees | Another novelty-control mechanism |
| 127 | **ART: Distribution-Free Changepoint Detection with Finite-Sample Guarantees**, 2025 | Model-agnostic exact Type-I change detection | Generic shift detection is occupied |
| 128 | **FDR Control for Online Anomaly Detection**, 2023 | Online anomaly detection with mFDR control | Online FPR/FDR control already studied |
| 129 | **PLIS / Conformal Q-values**, 2023 | Structured FDR control under weaker symmetry | Generic conformal multiple testing crowded |
| 130 | **Forecasting Early with Meta Learning**, IJCNN 2023 | Meta-forecasting with very short target histories | Weakens FARCast novelty |
| 131 | **METAFORS**, 2025 | Related longer time series help tailor a predictor for a short target series | Direct FARCast conceptual threat |
| 132 | **Hierarchically Regularized Deep Forecasting**, 2021 | Shares structure among correlated series | Hierarchical forecasting established |
| 133 | **Cross-Frequency Time Series Meta-Forecasting**, 2023 | Generalizes forecasting across unseen series/frequencies | Cold-start forecasting itself occupied |
| 134 | **Multi-Task NARMAX-LSTM Forecasting**, 2022 | Related tasks improve forecasting | Generic MTL forecast not new |
| 135 | **CaReTS**, 2025 | Multi-task trend + regression forecasting | Forecast framework space crowded |
| 136 | **G2G Federated Domain Generalization**, ICASSP 2024 | Explicit unseen-domain generalization in FL | Major threat to unseen-client candidates |
| 137 | **FedCCRL**, 2024 | Cross-client representations for unseen domains | Generic cross-client unseen-target representation occupied |
| 138 | **Federated Adversarial Domain Hallucination**, IEEE TMM 2023 | Synthesizes unseen-domain shifts during FL | Domain-hallucination candidate rejected |
| 139 | **UPFL / FedTTA**, 2023 | Personalizes to a newly joining unlabeled client | Very important new-client novelty killer |
| 140 | **GRACE**, 2023 | Balances source-client personalization and unseen-client generalization | Generic GPFL candidate crowded |
| 141 | **TsmoBN**, 2021 | Uses target-specific BN statistics to generalize to unseen clients | Strong threat to normalization-based new-client claims |
| 142 | **Siffer et al., SPOT: Anomaly Detection in Streams with Extreme Value Theory**, KDD 2017 | Automatic EVT threshold with risk controlling FPs | EVT thresholding itself absolutely not novel |
| 143 | **Adaptive Threshold Estimation via EVT**, IEEE TSP 2010 | GPD tail threshold designed for low false-alarm rates | Tail thresholding has long prior art |
| 144 | **Bayesian Anomaly Detection Using EVT**, 2019 | Bayesian normal/extreme anomaly model | Bayesian EVT alone insufficient |
| 145 | **Multivariate Statistical Process AD Based on POT**, 2025 | POT dynamic threshold on multivariate residual statistic | Another direct POT AD method |
| 146 | **Extreme Value Support Measure Machine**, 2025 | EVT for group anomaly detection | EVT anomaly framework broad |
| 147 | **Extreme Value Modelling of Feature Residuals for Dynamic-Graph AD**, 2024 | Temporal residual model + EVT | Tail calibration after residualization established |
| 148 | **Entropy-Based Validation of POT Threshold Selection**, 2024 | Automated threshold-selection method | POT base threshold is itself a research question |
| 149 | **Estimating Extreme Quantiles with POT**, 2024 | Practical extreme-quantile estimator | Must beat strong classical tail baseline |
| 150 | **CVAE + EVT for Known/Unknown IDS**, IEEE TIFS 2021 | EVT used directly in network intrusion detection | Important IDS-specific novelty threat |
| 151 | **Group-Weighted Conformal Prediction**, 2024 | Covariate/group shift with imperfectly estimated weights | Weighted calibration crowded |
| 152 | **Cross-Validation Conformal Risk Control**, 2024 | Risk control when calibration data are scarce | Direct small-data threat |
| 153 | **Conformal Risk Control**, ICLR 2024 | Distribution-free control of general monotone risks | Generic risk guarantee not novel |
| 154 | **Doubly Robust Calibration under Covariate Shift**, JRSS-B 2024 | Influence-function calibration under shift | Shift calibration already sophisticated |
| 155 | **Conformal Prediction Under Covariate Shift**, 2019 | Importance-weighted CP | Another direct shift prior |
| 156 | **Selective Conformal Risk Control**, 2025 | Joint abstention + risk control | Selective risk candidates crowded |
| 157 | **Adaptive Transfer Learning**, AAAI 2010 | Learns whether source and target are sufficiently related | “Safe peer borrowing” decades old |
| 158 | **Adaptive Sample Sharing for Linear Regression**, 2025 | Finite-sample rule that borrows only when estimated gain is positive | Very close to safe-sharing candidate |
| 159 | **Tessellation Localized Transfer Learning**, 2026 | Transfers only in locally compatible covariate regions | Localized safe transfer already active area |
| 160 | **ART of Transfer Learning**, Stat 2023 | Non-asymptotic adaptive transfer with negative-transfer protection | Generic safe-transfer framework occupied |
| 161 | **Characterizing and Avoiding Negative Transfer**, 2018 | Formal negative-transfer definition and source filtering | Similarity/gating claims must be stronger |
| 162 | **To Transfer or Not to Transfer**, 2005 | Early formulation of detecting harmful transfer from scarce target data | Fundamental novelty boundary |
| 163 | **Bridging Local and Federated Data Normalization**, 2025 | Federated normalization incl. secure ranked/median operations | Quantile-normalization novelty narrowed |
| 164 | **Differentially Private Federated Quantiles with Distributed Discrete Gaussian Mechanism** | Privacy-preserving distributed quantile estimation | Quantile communication already established |
| 165 | **Copula-Driven Unsupervised Learning for Heterogeneous AD**, IEEE MLSP 2021 | Copula dependence modeling for heterogeneous AD | Copulas by themselves not new |
| 166 | **COPOD**, ICDM 2020 | Empirical copula tail probabilities for anomaly detection | Marginal tail-probability scoring already strong baseline |
| 167 | **FCPJS**, UAI 2026 | Global score sketch + client-local ratio weighting under joint shift | Major threat to score-distribution sharing |
| 168 | **APEW-Fed**, Scientific Reports 2026 | Ensemble AD + DP score calibration via quantile sketches | Major threat to quantile-calibration candidate |
| 169 | **FedCIGAR**, IJCAI 2026 | Personalized reconstruction and dynamic client clustering for federated graph anomalies | Personalization/cluster AD very current |
| 170 | **Multi-Source Ensemble Transfer AD for UAVs**, 2025 | Source-similarity weighting + fine-tuning + EVT dynamic threshold | Critically, source weighting + scarce target data + EVT already appears together |

The last item is particularly important. A recent UAV anomaly-detection system already combines **source similarity, multi-source transfer, scarce target data and EVT dynamic thresholding**, so a new candidate cannot merely combine those ingredients and claim the composition as novel. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S147403462500148X)

Likewise SPOT already makes the false-positive risk the main user parameter of an EVT anomaly detector. [kdd.org](https://www.kdd.org/kdd2017/papers/view/anomaly-detection-in-streams-with-extreme-value-theory)

---

# 5. What the 170-paper audit now says is saturated

I would rule these out as the main paper contribution:

- “federated cold-start anomaly detection” generically;
- “share moments/statistics”;
- “share quantiles”;
- “learn a collaboration graph”;
- “find the most similar clients”;
- “trust/reputation weighting”;
- “safe transfer based on similarity”;
- “mixture of experts for a new client”;
- “federated conformal calibration”;
- “conformal risk control”;
- “EVT anomaly thresholding”;
- “hierarchical Bayesian modeling” generically;
- “POT with federation” without a stronger structural contribution;
- “CVaR/worst-client federated learning”;
- “domain generalization to unseen clients”;
- “test-time adaptation of a new client”;
- “forecast the future tail using other time series”;
- “adaptive threshold under drift”;
- “learn whether to switch from local to global” as a heuristic.

Recent work makes this even more crowded. FCPJS directly addresses heterogeneous federated calibration under joint shifts, while APEW-Fed directly shares calibrated score information through quantile sketches. [proceedings.mlr.press](https://proceedings.mlr.press/v337/shi26a.html)

Safe transfer itself also has mature theory guaranteeing or encouraging abstention when transfer is harmful. [Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1002/sta4.582)

---

# 6. What remains scientifically interesting

The narrow gap left by FedORBIT is more specific.

Your experiments suggest:

\[
\text{peer marginal information}
\quad\text{is useful}
\]

while

\[
\text{peer dependence/covariance information}
\quad\text{is not reliably useful}.
\]

That is scientifically unusual.

Most distribution-transfer methods either transfer a complete representation/distribution or train a shared model.

FedORBIT suggests a more surgical decomposition:

\[
\boxed{
\text{share the one-dimensional benign laws}
\quad
\text{but localize dependence}
}
\]

A second independent gap is:

\[
\boxed{
\text{the extreme benign score tail is badly underdetermined
when the target only has 30--100 normal observations}
}
\]

At \(n=30\), a nominal empirical 99th percentile is intrinsically an extreme small-sample problem even before any distribution shift is considered.

A third possible gap is:

\[
\boxed{
\text{peers have labeled attacks that could teach a low-FPR scoring geometry,
while the new target itself remains benign-only}
}
\]

Those are the three mechanisms I would test.

---

# 7. Full 57-candidate algorithm audit

Legend:

**PROMOTE** = worth an immediate POC.  
**SECONDARY** = potentially useful, but only after a promoted parent succeeds.  
**DEFER** = scientifically possible but data/validation insufficient.  
**REJECT** = do not spend POC time now.  
**BASELINE** = implement only as comparator.

| ID | Candidate / actual new computation | Novelty & feasibility audit | Candidate-specific killer POC | Decision |
|---|---|---|---|---|
| **A1** | Peer full-CDF PIT + target-local covariance | Feasible now; closer to FADngs, federated normalization and COPOD than desired | Replace peer mean/std by equal-peer feature ECDFs, PIT-transform target, local-OAS score; compare pAUC | **BASELINE/COMPONENT** |
| **A2** | Wasserstein/quantile barycenter marginals + local dependence | Feasible; OT/quantile barycenters established | Equal-client quantile barycenter vs simple pooled ECDF | **SECONDARY** |
| **A3** | **Hierarchical-Dirichlet full marginal distribution transfer + target-local dependence** | Stronger delta: federation learns *distribution of benign marginals across clients*; no exact federated AD equivalent found. Uses current arrays only | Hierarchical histogram/CDF posterior on held-out target; PIT + local OAS; strict device-level pAUC gate | **PROMOTE** |
| A4 | Median/trimmed peer CDF | Robust aggregation already failed repeatedly in FedORBIT and novelty weak | Median-CDF vs arithmetic CDF | **REJECT** |
| A5 | DKW/KS confidence-band CDF gate | Principled but similar to previous failed compatibility gates; benign fit may not predict attack ranking | Borrow each feature only when peer/local CDF bands overlap | **DEFER** |
| **A6** | Transfer only upper/lower marginal tails, local body | Feasible and closer to anomaly objective than whole-CDF transfer | Replace only q90+ and q10− tails with peer estimate; compare A3 | **SECONDARY / A3 ablation** |
| A7 | Spline-compressed federated CDF | Primarily communication engineering, not scientific mechanism | 16/32/64 knot sketches vs full ECDF | **REJECT main contribution** |
| A8 | Family/block copula transfer | Existing block experiments substantially harmed pAUC | Copula per 23×5 block | **REJECT** |
| A9 | Shared marginal PIT + sparse target precision | \(n=30,p=115\) is difficult and graphical-lasso tuning would be unstable | Fixed lambda sparse precision in PIT space | **DEFER** |
| A10 | Private federated quantile normalization | Direct 2025 prior art plus DP federated quantiles | Secure-quantile simulation | **REJECT as contribution** |
| **B1** | Hierarchical Bayesian GPD/POT across clients | Feasible from current score streams; exact FL cold-start use appears weakly explored but hierarchical EVT itself established | Hierarchical GPD vs local POT vs empirical q99 | **PROMOTE, B7 preferred** |
| B2 | Shared tail shape \(\xi\), target-local tail scale | Lower complexity but restrictive assumption | Common-\(\xi\) GPD vs hierarchical random-\(\xi\) | **SECONDARY baseline** |
| B3 | Full Bayesian posterior-predictive extreme quantile | Statistically sound; novelty mostly application-level | posterior q99 credible bound vs bootstrap/local POT | **SECONDARY** |
| **B4** | Hierarchical EVT + conformal/risk correction | Strong but risks “stacking established methods”; useful only if B7 first works | outer calibration of B7 residual FPR | **SECONDARY** |
| B5 | Federated SPOT initialization | Requires meaningful online sequence/drift interpretation not validated in current data | SPOT on row-order stream | **DEFER — current data unsuitable** |
| B6 | Robust POT under support contamination | Useful if benign support contamination becomes part of problem; current protocol assumes benign support | inject 1–10% contamination, compare robust GPD | **SECONDARY only** |
| **B7** | **Hierarchical joint exceedance-rate + GPD excess model** | Strongest tail candidate; explicitly separates probability of entering tail from excess magnitude | Beta/Binomial random effect + hierarchical GPD on scores; q99/FPR test | **PROMOTE** |
| B8 | Meta-network predicts q99 from q50/q90/q95 | Nine independent devices too few for defensible meta-learning | leave-one-device-out linear/meta regression | **REJECT current cohort** |
| B9 | Cross-validation conformal risk-control threshold | Strong baseline, but CV-CRC already exists | apply CV-CRC to same anomaly score | **BASELINE** |
| B10 | Full-conformal e-value anomaly batches | Strong theory but changes problem from pointwise IDS to batch discoveries | batch FDR comparison | **DEFER / different task** |
| C1 | Learn-then-Test selection over local/shared channels | LTT itself prior art; more importantly prior benign-only gates in FedORBIT failed to predict detector utility | benign-only selection vs target oracle pAUC | **REJECT unless A3 creates clear channel diversity** |
| C2 | Safe transfer using CDF confidence bands | Similar problem to C101–C106: compatibility is not detector utility | DKW gate vs fixed A3 | **REJECT current** |
| C3 | Estimate transfer gain before borrowing | Attractive theory but project's previous predictive-loss/risk gates failed | estimate support-only gain vs actual pAUC gain | **REJECT** |
| C4 | ART safe aggregation of multiple candidate detectors | ART is generic and no credible target-side utility estimator available | ART over local/shared/CDF channels | **REJECT now** |
| C5 | Noninferiority-constrained local/shared score mixture | Requires utility proxy; target attack labels forbidden | source-trained mixture, target benign only | **DEFER** |
| C6 | Localized transfer by feature compatibility | More plausible than whole-client similarity; still close safe-transfer literature | featurewise peer/local CDF risk | **SECONDARY to A3** |
| C7 | Local/shared/abstain controller | Useful deployment feature, not core algorithm | calibrated abstention under unsupported CDF | **COMPONENT** |
| C8 | Generate peer pseudo-benign target samples | Introduces artificial distribution and privacy issues; project pseudo-target POCs weak | sample from peer density and augment local | **REJECT** |
| D1 | Meta-net from support stats to normalization | FedL2P/TsmoBN already very close; only nine tasks | leave-device-out tiny MLP/meta-net | **REJECT** |
| D2 | Hypernetwork produces new-client one-class detector | Too many parameters relative to independent clients | simple linear hypernetwork first | **DEFER** |
| D3 | FedTTA anomaly detector for joining client | FedTTA directly targets new unlabeled clients | port FedTTA to Gaussian/AE | **REJECT novelty** |
| D4 | Federated domain-generalized one-class representation | Scientifically plausible and feasible with peer benign data, but FedDG literature crowded | leave-device-out representation vs FedAvg/local | **DEFER / challenger only** |
| D5 | Cross-client benign contrastive DG | GPU heavier and FLCAD/FedDTCN very close | minimal contrastive encoder | **REJECT for this project** |
| D6 | Distill normality from peer teachers | FADngs and Sentinel threaten novelty | distill peer AE scores | **REJECT** |
| D7 | One-class mixture-of-experts | MoE/PFL heavily saturated; prior partner-selection evidence negative | oracle expert headroom first | **REJECT** |
| D8 | Test-specific BN / normalization geometry | TsmoBN + StatAvg + federated normalization too close | test-specific mean/std | **REJECT** |
| **E1** | **Federated Neyman–Pearson domain-generalized anomaly scorer**: source clients' attacks learn low-FPR global feature/family weights; target contributes only benign calibration | Feasible from N-BaIoT. Strong PhD fit. Main threats: FDG, superquantile FL, source-attack transfer. Exact intersection remains less occupied | leave-one-device-out 23-family weights; no target attacks in fitting; evaluate target pAUC/TPR@FPR | **PROMOTE** |
| E2 | CVaR/superquantile pAUC training | Superquantile FL already exists; could be E1 optimizer, not contribution | replace mean source loss by top-client CVaR | **COMPONENT** |
| E3 | Group-DRO anomaly scorer | Generic FGDRO too close | group-DRO on source clients | **REJECT as standalone** |
| E4 | Low-FPR federated domain generalization | Essentially E1 at representation level; greater complexity | differentiable NP loss | **SECONDARY if E1 passes** |
| E5 | Source attack experts + benign target gate | Past source-selector work had weak similarity→utility relation; MoE prior art strong | oracle expert mixture headroom | **REJECT unless oracle headroom huge** |
| E6 | Complementarity-based attack-knowledge routing | Strong overlap with CTK and previous partner selection failures | cross-source attack-family complementarity | **REJECT for thesis redundancy** |
| E7 | Average + worst-device dual objective | Good robustness component for E1; not novel alone | mean pAUC vs min-device pAUC Pareto | **SECONDARY** |
| E8 | Train detector jointly against FABRID risk budgets | Strong overlap with FABRID | budget-aware source objective | **REJECT as separate paper** |
| F1 | Sequential cross-peer e-evidence fusion | Excellent theory, but requires defensible stream ordering; not established | e-value fusion on chronological external data | **DEFER** |
| F2 | e-process monitor of false-alarm drift | Same data issue; WATCH/e-detectors close prior art | anytime false-alarm monitor | **DEFER** |
| F3 | Evidence-based shared→local switch | Current support ladder supports idea, but sequential evidence interpretation weak and DATP overlap high | handoff decision vs support size | **DEFER** |
| F4 | Change-point-aware POT | No validated target timestamps/change points | POT + ART/BOCPD | **DEFER** |
| F5 | Repository of normality regimes | FedRepo already close; chronology missing | cluster historical regimes | **REJECT** |
| F6 | Federated weighted conformal martingale | WATCH already close and sequence missing | weighted CTM | **REJECT current** |
| G1 | Risk-aligned multi-target covariance shrinkage | Existing MTS-OAS has preliminary signal, but MTS prior art and no target-risk estimator are major obstacles | replace MSE weights with cross-source low-FPR risk; LO-device-out | **DEFERED LEAD** |
| G2 | Transfer precision sparsity pattern only | More interesting than full covariance, but structure unreliable with nine heterogeneous devices | peer edge-frequency → target graphical lasso | **DEFER** |
| G3 | Shared low-rank benign subspace + local residual | PCA/factor transfer common; covariance null result reduces plausibility | shared factors vs local OAS | **REJECT unless trivial POC strong** |
| G4 | HAC-aware MTS | Already tested; mixed/fragile | current POC exists | **REJECT** |
| G5 | Block precision over N-BaIoT feature families | Related block POCs harmed pAUC | current schema | **REJECT** |
| G6 | Covariance shrink only if predictive noninferiority certified | Same target-utility problem as failed gates | benign predictive bound vs attack pAUC | **REJECT** |
| **H1** | FARCast future benign-risk forecast | Methodologically interesting, but requires longitudinal episodes not currently validated; meta-forecasting prior art narrower novelty | requires new timestamped data first | **REJECT CURRENT / future only** |

This is the key outcome of the candidate audit:

### 57 candidates considered.
### 3 deserve an immediate new POC.
### 54 should **not** consume implementation time now.

---

# 8. Ranking of the surviving frontier

Using the same weighted criteria as before:

- Novelty: 25%
- Current-data feasibility: 25%
- Evidence-backed positive-result plausibility: 20%
- Journal depth: 15%
- PhD coherence: 10%
- Practicality: 5%

| Candidate | Novelty | Feasibility | Signal plausibility | Journal | PhD | Practical | Weighted |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A3 Hierarchical marginal-CDF transfer** | 8.5 | 9.5 | 8.7 | 8.8 | 9.2 | 9.0 | **8.93** |
| **B7 Hierarchical exceedance + GPD tail** | 8.7 | 9.6 | 8.5 | 9.0 | 8.4 | 9.3 | **8.93** |
| **B1 Hierarchical POT** | 8.3 | 9.6 | 8.4 | 8.6 | 8.2 | 9.4 | **8.73** |
| **E1 Federated NP domain-generalized scorer** | 8.3 | 9.0 | 8.0 | 9.0 | 9.4 | 8.3 | **8.63** |
| A6 Tail-only marginal transfer | 8.0 | 9.6 | 8.3 | 8.0 | 9.0 | 9.3 | **8.62** |
| A1 simple full-CDF PIT | 7.5 | 9.8 | 8.5 | 7.3 | 9.0 | 9.5 | **8.50** |
| B4 EVT + conformal risk layer | 8.1 | 9.2 | 8.0 | 8.6 | 8.5 | 8.8 | **8.50** |
| G1 Risk-aligned MTS | 7.2 | 9.1 | 8.0 | 7.8 | 8.8 | 8.9 | **8.17** |
| E7 mean + worst-device objective | 7.1 | 8.9 | 7.8 | 7.9 | 9.1 | 8.7 | **8.09** |
| D4 FedDG one-class representation | 7.4 | 8.0 | 7.8 | 8.5 | 9.2 | 7.0 | **7.96** |

A3 and B7 tie numerically, but they attack **different failures**:

- **A3 attacks ranking / detector geometry.**
- **B7 attacks extreme-tail calibration.**

That is useful because FedORBIT currently suffers both.

---

# 9. Candidate A3 — Hierarchical Federated Marginal Distribution Transfer

This is the candidate I would test **first**, because it follows directly from the strongest positive and negative evidence already in FedORBIT.

## 9.1 Scientific hypothesis

FedORBIT says:

> sharing marginal means/scales helps enormously at low support,

but

> sharing covariance/dependence adds little or hurts.

So test the stronger hypothesis:

\[
\boxed{
\text{the transferable object is the benign marginal law,
not merely its first two moments}
}
\]

while the target client's dependence structure should remain local.

This is not “another federated normalizer.”

It is a semiparametric decomposition:

\[
P_i(X_1,\ldots,X_p)
=
C_i
\left(
F_{i1}(X_1),\ldots,F_{ip}(X_p)
\right),
\]

where:

- the federation helps estimate \(F_{ik}\);
- the target estimates \(C_i\) locally.

That separation is exactly the empirical pattern FedORBIT currently suggests.

## 9.2 Hierarchical marginal model

For feature \(k\), define \(B\) quantile bins.

Mature client \(j\) has probability vector

\[
\mathbf p_{jk}
=
(p_{jk1},\ldots,p_{jkB}).
\]

Model:

\[
\mathbf p_{jk}
\sim
\operatorname{Dirichlet}
(\kappa_k\boldsymbol{\pi}_k).
\]

The federation learns:

\[
\boldsymbol{\pi}_k,\kappa_k
\]

from peer clients.

The new client observes \(n_i\) benign observations and obtains bin counts

\[
\mathbf c_{ik}.
\]

Posterior:

\[
\mathbf p_{ik}\mid \mathbf c_{ik}
\sim
\operatorname{Dirichlet}
\left(
\kappa_k\boldsymbol{\pi}_k+\mathbf c_{ik}
\right).
\]

The posterior mean automatically performs support-dependent borrowing:

\[
E[\mathbf p_{ik}\mid \mathbf c_{ik}]
=
\frac{
\kappa_k\boldsymbol{\pi}_k+\mathbf c_{ik}
}{
\kappa_k+n_i
}.
\]

Unlike an ad-hoc mixing coefficient, the degree of borrowing emerges from the model.

Then obtain the target predictive CDF

\[
\widehat F_{ik}(x)
\]

and PIT-transform:

\[
u_{ik}=\widehat F_{ik}(x_k),
\qquad
z_{ik}=
\Phi^{-1}(u_{ik}).
\]

Each transformed marginal is approximately standard Gaussian if the benign marginal estimator is correct.

The target then estimates only its own latent dependence:

\[
R_i
=
\operatorname{OAS}
\left(
\operatorname{Cov}
(z_{i1},\ldots,z_{ip})
\right),
\]

or initially simply \(R_i=I\).

Score:

\[
S_i(x)
=
z_i^\top R_i^{-1}z_i.
\]

## 9.3 Why this is meaningfully different from current FedORBIT

Current shared marginals assume roughly:

\[
X_k
\sim
(\mu_k,\sigma_k).
\]

A3 transfers an entire one-dimensional benign law.

Skew, multimodality, heavy tails and asymmetry that are invisible to mean/std become transferable.

Yet it deliberately refuses to federate cross-feature dependence, because the project has **negative evidence** for that channel.

## 9.4 Closest novelty threats

It must be compared directly against:

- StatAvg;
- 2025 federated normalization;
- FADngs;
- COPOD;
- copula-based heterogeneous AD;
- FCPJS;
- APEW-Fed.

The 2025 federated-normalization work even includes secure ranked/median operations, so the paper cannot claim that quantile-based federation itself is new. [arXiv](https://arxiv.org/abs/2511.11249)

FADngs also explicitly shares processed density information.

What remains potentially distinctive is:

> **hierarchical client-specific estimation of complete benign feature marginals for a new cold-start client, combined with explicitly target-local dependence in federated one-class anomaly detection.**

The exact-mechanism search did not find an equivalent algorithm, although that is not proof of absence.

## 9.5 Killer POC A3

No neural network. No new dataset. No large training.

For every held-out N-BaIoT device:

1. use the same peer set already used by FedORBIT;
2. define 16 fixed bins/feature from peer quantile sketches;
3. estimate \(\pi_k,\kappa_k\) from peers;
4. update with the target's \(n=30\) benign support;
5. transform target support/test via posterior PIT;
6. score first with identity dependence;
7. repeat with target-local OAS;
8. compare against:
   - local standardization;
   - shared mean/std;
   - simple peer-only PIT;
   - local ECDF PIT;
   - COPOD-style marginal scoring.

### Non-negotiable PASS

At \(n=30\) and \(100\):

- mean pAUC gain over shared marginals **≥ 0.015**;
- at least **7/9 devices positive**;
- worst-device pAUC delta **> -0.01**;
- AUROC not lower by >0.005;
- identity-dependence version should already carry much of the gain.

If it only increases mean AUROC but harms pAUC/device tails:

**FAIL and delete the entire A-family.**

That POC should be cheap.

---

# 10. Candidate B7 — Federated Hierarchical Extreme-Tail Model

This is the strongest operating-point candidate.

But unlike my previous answer, I would **not** initially claim it solves temporal drift.

Its first question is much cleaner:

> How much of the 1% operating-point failure is simply caused by trying to estimate an extreme benign tail from 30–100 target samples?

That question is feasible on the current data.

## 10.1 Why ordinary empirical q99 is statistically weak

At \(n=30\), the desired q99 lies beyond what the empirical sample can estimate with precision.

Even under a perfectly stationary benign distribution, the empirical maximum is a poor estimate of the 0.99 quantile.

That makes extreme-value pooling a scientifically natural mechanism rather than an arbitrary addition.

SPOT and older EVT work already use extreme-value theory for false-positive-controlled anomaly thresholds, so **EVT is not the contribution**. [kdd.org](https://www.kdd.org/kdd2017/papers/view/anomaly-detection-in-streams-with-extreme-value-theory)

Hierarchical POT is also established in other disciplines as a way to borrow information across groups when extremes are scarce. [PubMed](https://pubmed.ncbi.nlm.nih.gov/32949863/)

The possible contribution is the federated cold-start structure.

## 10.2 Model

First robustly standardize client scores:

\[
r_{it}
=
\frac{
S_{it}-\operatorname{median}(S_i)
}{
\operatorname{IQR}(S_i)
}.
\]

Choose a moderate federation-defined threshold \(u\), not a 99th percentile.

For target \(i\):

\[
K_i
=
\sum_t I(r_{it}>u).
\]

Model the tail-entry probability:

\[
K_i
\sim
\operatorname{Binomial}
(n_i,\pi_i),
\]

with hierarchical random effect

\[
\operatorname{logit}(\pi_i)
\sim
N(\mu_\pi,\sigma_\pi^2).
\]

For excesses

\[
Y_{it}=r_{it}-u\mid r_{it}>u,
\]

use:

\[
Y_{it}
\sim
\operatorname{GPD}(\xi_i,\beta_i).
\]

Client parameters are hierarchically pooled:

\[
\log\beta_i
\sim
N(\mu_\beta,\sigma_\beta^2),
\]

and use a bounded prior for \(\xi_i\).

The target tail survival is:

\[
P(R_i>r)
=
\pi_i
\left(
1+
\xi_i
\frac{r-u}{\beta_i}
\right)^{-1/\xi_i}.
\]

Solve:

\[
P(R_i>\tau_i)=\alpha.
\]

The resulting \(\tau_i\) is a personalized threshold even when the target itself has very few tail observations.

## 10.3 Why B7 is stronger than ordinary hierarchical GPD

It models both:

1. **how often the client enters the tail**, and
2. **how severe the tail is after entering it**.

That separates two forms of client heterogeneity that a pooled q99 cannot distinguish.

## 10.4 Why this could still fail

This is critical.

The project observes FPR near 20%, not merely 3–4%.

A perfect small-sample EVT estimator may correct finite-support calibration yet still fail on the project's later benign set if the dominant cause is distribution shift.

Your report explicitly says this decomposition is currently unknown. status_report(2)

Therefore B7 gets **two-stage POC evaluation**.

## 10.5 Killer POC B7

### Stage 1 — stationary/finite-support test

Use benign observations from the same parent benign pool.

For each support size:

- target empirical q99;
- local POT;
- pooled-peer POT;
- common-shape GPD;
- hierarchical B7.

Evaluate on held-out benign observations from the **same pool**.

Measure:

\[
|\widehat{\mathrm{FPR}}-0.01|.
\]

### PASS-1

At \(n=30,100\):

- ≥50% reduction in absolute FPR error relative to empirical q99;
- ≥25% reduction relative to local POT;
- at least 7/9 devices better than empirical q99.

If this fails:

**kill B7 immediately.**

### Stage 2 — existing later-test set

Only after Stage 1 passes.

Apply thresholds unchanged to the existing later benign test.

Then evaluate:

- mean FPR;
- worst-device FPR;
- attack TPR;
- pAUC unchanged, since score ranking is fixed.

### PASS-2

A serious journal signal requires:

- future mean FPR reduced by **at least 30% relative**;
- no device with >5 percentage-point additional FPR;
- attack TPR loss <5 percentage points.

If Stage 1 works but Stage 2 does not, the result itself demonstrates:

> finite-support tail estimation is not the main source of FedORBIT's deployment failure.

That would be scientifically useful, but B7 would **not** become the paper algorithm.

---

# 11. Candidate E1 — Federated Neyman–Pearson Domain-Generalized Scoring

A3 and B7 use only benign target information.

E1 tests a different collaboration hypothesis:

> **The target has no attacks, but its peers do. Can the federation learn what anomaly geometry is consistently useful at very low FPR across peer devices?**

That is highly aligned with the thesis.

## 11.1 Why this is different from CTK

CTK asks whether peers carry complementary threat knowledge.

E1 would ask:

> can source-client threat evidence be distilled into a **device-general scoring geometry** whose operating point is subsequently calibrated only from target benign traffic?

No target attack labels are ever used for learning or selection.

## 11.2 Avoiding overfitting

Do not learn 115 arbitrary feature weights from eight source devices.

The project already has a natural 23-family × 5-window structure.

Start with **23 nonnegative family weights**:

\[
w_g\ge 0,
\qquad
\sum_g w_g=23.
\]

For standardized target vector \(z\):

\[
S_w(x)
=
\sum_{g=1}^{23}
w_g
D_g(x),
\]

where \(D_g\) can initially be the mean standardized squared deviation within family \(g\).

Train \(w\) only on source clients.

## 11.3 Objective

A standard average objective is not enough.

Use low-FPR client risk:

\[
L_j(w)
=
1-
\operatorname{pAUC}_{j,[0,\alpha_{\max}]}(w).
\]

Then:

\[
\min_w
\operatorname{CVaR}_q
\{L_1(w),\ldots,L_m(w)\}
+
\lambda\|w-\mathbf 1\|_2^2.
\]

This prevents one source device from dominating.

Superquantile FL already exists, so CVaR itself is not novel. [arXiv](https://arxiv.org/abs/2112.09429)

The potential novelty is its role in:

> **low-FPR source-attack knowledge distillation for an unseen benign-only target device.**

## 11.4 Strong prior-art threats

This candidate must beat:

- federated domain generalization;
- FedCCRL;
- UPFL/FedTTA;
- anomaly detection with domain adaptation;
- Learning from Others;
- MSETL-AD;
- superquantile FL;
- your own CTK work.

This is why its novelty score is lower than A3/B7.

## 11.5 Killer POC E1

Perform strict leave-one-device-out evaluation.

For target \(i\):

- all attack rows from \(i\) are **evaluation only**;
- source devices \(j\ne i\) may use their attacks;
- tune weights using source-device internal leave-one-source-out validation only;
- target provides only benign support for normalization/calibration.

Baselines:

- uniform 23-family weights;
- current shared-marginal Mahalanobis scorer;
- source-average supervised weights;
- mean-risk training;
- CVaR training.

### PASS

At \(n=30\):

- mean pAUC improvement ≥0.02;
- ≥7/9 targets improve;
- **worst-device difference must be non-negative**;
- no AUROC loss >0.01;
- gains remain at \(n=100\);
- no target labels involved anywhere before final evaluation.

If one or two devices suffer large losses:

**do not rescue it with target-specific selection. Reject it.**

That avoids repeating the partner-selection rabbit hole.

---

# 12. Why I am not promoting the other seemingly attractive algorithms

### Risk-controlled channel selection

Learn-then-Test is powerful and provides finite-sample risk guarantees. [arXiv](https://arxiv.org/abs/2110.01052)

But your own C101–C106 family already demonstrates the deeper problem:

> benign-fit/borrowing evidence was not reliably aligned with actual anomaly detection risk.

A mathematically valid selector of a bad proxy remains scientifically unhelpful.

So I would not implement it before a genuinely better alternative channel exists.

### Safe-transfer methods

ART and adaptive sample sharing already provide strong frameworks for avoiding negative transfer. [Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1002/sta4.582)

Adding them around existing FedORBIT channels would likely be a wrapper paper.

### Sequential e-values / monitoring

Scientifically strong, but the current data contract does not establish the chronology needed to make it convincing.

### Graphs and MoE

Already very crowded, and FedORBIT has direct negative evidence that naive client similarity fails to predict utility.

### More covariance variants

The project's preliminary MTS-OAS result is worth preserving, but a new covariance estimator is not yet a PhD-level direction. Multi-target shrinkage is established, and the current positive mean still has negative device tails. status_report(2)

---

# 13. Candidate-specific POC policy

This is where I would change your workflow most.

**Do not implement a candidate framework and then discover whether it has signal.**

Each candidate must first survive one deliberately minimal test of its *unique mechanism*.

For the three survivors:

| Candidate | Unique hypothesis | Cheapest falsification | Implementation before PASS |
|---|---|---|---|
| **A3** | More of the *marginal distribution* is transferable than mean/std, while dependence is local | posterior-CDF PIT on current arrays | ~one standalone POC |
| **B7** | federation can stabilize a target's scarce extreme benign tail | hierarchical POT on existing scores | score-level script only |
| **E1** | source attacks contain device-general low-FPR geometry | 23-family LO-device-out optimization | tiny optimizer only |

No production wiring.

No architecture refactor.

No new CLI.

No GPU first.

No external dataset first.

No 20-variant hyperparameter sweep.

---

# 14. Exact POC order I recommend

## POC 1 — A3

Why first?

Because the project already provides direct evidence that **marginals are the useful collaboration channel**.

If A3 works, it directly upgrades the current positive FedORBIT result into an algorithmic mechanism.

If it fails, we learn that the existing mean/std effect is probably already capturing almost everything useful in the marginal channel.

Estimated scientific information yield: **very high**.

---

## POC 2 — B7

Run independently of A3 using the existing shared-marginal score stream.

This isolates:

\[
\text{scoring problem}
\neq
\text{threshold-estimation problem}.
\]

That separation is essential.

If B7 fixes same-distribution FPR but not future FPR, you have strong evidence that the main deployment problem is nonstationarity rather than sampling error.

---

## POC 3 — E1

Only then test whether peer attacks can improve target ranking.

This changes the information model more substantially, so it belongs after the benign-only candidates.

---

# 15. Do not combine algorithms prematurely

A tempting framework would be:

\[
\text{A3 scoring}
+
\text{B7 calibration}
+
\text{E1 attack-aware weighting}.
\]

Do **not** build that.

It would be impossible to know where any gain comes from and reviewers could reasonably call it an assembly of known components.

The only acceptable path is:

```text
A3 alone
   ↓
mechanism proven?

B7 alone
   ↓
mechanism proven?

E1 alone
   ↓
mechanism proven?

only then ask whether two successful mechanisms
solve orthogonal problems and interact positively.
```

A combined framework should exist only if the factorial experiment shows a genuine interaction or clearly additive independent benefits.

---

# 16. Future composite candidate — only conditional on two successful POCs

If both A3 and B7 pass strongly, there is a genuinely coherent framework:

## Federated semiparametric benign-distribution transfer

The federation would share:

\[
\text{feature marginal structure}
\]

for score geometry,

while retaining:

\[
\text{target-local dependence},
\]

and separately share:

\[
\text{cross-client extreme-tail structure}
\]

for operating-point calibration.

Conceptually:

\[
\boxed{
\text{share what is statistically transferable;
localize what FedORBIT shows is device-specific}
}
\]

That is a stronger scientific philosophy than “share more information.”

But this **must not be named or promoted as the final algorithm until both mechanisms independently survive**.

---

# 17. Strong-result gates for every future candidate

I recommend putting these rules into the candidate audit itself.

A candidate cannot be called “positive” merely because its mean AUROC rises.

For FedORBIT, require:

### Ranking candidate

At minimum:

- positive device count ≥7/9;
- mean pAUC improvement ≥0.015–0.02;
- worst-device loss no worse than −0.01;
- AUROC noninferiority;
- effect at ≥2 support sizes;
- stronger than trivial current shared marginals.

### Threshold candidate

At minimum:

- substantial reduction of absolute FPR error;
- worst-client FPR improves;
- TPR does not collapse;
- no test-informed threshold tuning;
- comparison against empirical quantile, POT, FedCP/CRC-style controls.

### Source-attack candidate

At minimum:

- target attacks used once, for final evaluation only;
- leave-one-device-out outer loop;
- ≥7/9 targets positive;
- worst-device non-negative;
- source-only hyperparameters.

### Any adaptive selector

It must beat:

\[
\text{fixed all-peer policy}
\]

not merely random selection.

That is especially important because the existing nearest-peer policy already failed.

---

# 18. External validation should come **after** mechanism survival

Your report is right that external validation remains essential. There is presently no second compatible physical attack cohort in the project. status_report(2)

But I would no longer search for a dataset **before** deciding whether A3/B7/E1 has signal.

That reverses the order that repeatedly wastes time:

```text
old:
find dataset
→ adapt lots of code
→ discover method has no signal

new:
current N-BaIoT killer POC
→ mechanism either dies or survives
→ freeze algorithm
→ THEN pay cost of external validation
```

For publication, independent validation is still mandatory before making a broad claim.

---

# 19. How the thesis story changes depending on which candidate survives

### If A3 wins

The contribution becomes:

> **Collaborative normality modeling should transfer client marginal laws while preserving device-specific dependence.**

Thesis role:

**what statistical structure should be shared?**

This is probably the cleanest continuation of FedORBIT.

### If B7 wins

The contribution becomes:

> **Federations can stabilize extreme operating-point estimation for data-scarce clients by partially pooling benign tail behavior.**

Thesis role:

**how should clients calibrate rare-event decisions when their own benign evidence is insufficient?**

Strong, but closer to DATP.

### If E1 wins

The contribution becomes:

> **Attack knowledge from mature peers can learn a device-general low-FPR detection geometry while a newly joining device remains attack-label-free.**

Thesis role:

**how can threat knowledge be transferred without requiring attack examples at the target?**

This is probably the strongest malware-security narrative, although it needs the most careful separation from CTK.

---

# 20. Final revised decision

My previous answer tried to pick the journal identity too early.

After the 170-paper literature audit and the 57-candidate feasibility/novelty audit, I would **not** tell your agent to develop FARCast, a graph selector, a MoE, another covariance shrinkage rule, or a generic conformal/EVT framework.

I would freeze the current project and run exactly **three surgical candidate POCs**:

1. **A3 — hierarchical full-marginal transfer with local dependence**
2. **B7 — hierarchical exceedance/GPD tail calibration**
3. **E1 — source-attack Neyman–Pearson domain-generalized scoring**

And I would run them **in that order**.

A3 is the most direct algorithmic extrapolation from the strongest FedORBIT evidence. B7 is the most principled explanation/repair candidate for the operating-point problem. E1 is the strongest alternative if benign-only collaboration has reached its ceiling.

Most importantly, each has an explicit kill condition.

So this time the workflow is not:

> invent algorithm → implement → run full experiments → discover it was weak.

It is:

> **literature gate → data-contract gate → trivial-baseline gate → one-mechanism POC → device-tail gate → only then implementation.**

If all three fail their stated thresholds, my recommendation would be **not to invent candidate 58**. At that point the scientifically correct decision would be that the existing N-BaIoT/FedORBIT evidence does not support a strong new algorithm, and the project should change either its information model or its dataset before more algorithm search.