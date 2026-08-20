# Global Roadmap Implementation Audit

## 1. Purpose & Audit Contract

Verify end-to-end that the implemented repository completely and faithfully realizes every mandatory obligation in the authoritative roadmap.

This audit is the final repository-wide acceptance gate.

Issue completion, milestone completion, code presence, experiment execution, or artifact existence alone is not sufficient evidence of roadmap completion.

A final `PASS` means that the repository state represented by the audited commit is fully traceable to the roadmap, complete, internally consistent, validated, reproducible where required, and free of unresolved mandatory implementation gaps.

### 1.1 What This Audit Proves

A `PASS` proves that:

- every mandatory roadmap obligation is represented in the coverage inventory;
- every mandatory roadmap obligation is mapped through implementation and validation to inspectable evidence;
- every triggered conditional obligation is satisfied;
- every milestone is complete and has a current valid milestone audit;
- roadmap-defined datasets, preprocessing, algorithms, mathematics, configurations, experiments, metrics, statistics, execution paths, artifacts, and claim boundaries are faithfully implemented;
- no unauthorized scientific functionality, experiment, comparator, metric, configuration default, or claim has entered the repository;
- all required repository quality gates pass;
- the implementation remains consistent with the roadmap after a fresh independent second-pass derivation.

### 1.2 What This Audit Does Not Require

The audit does not require a scientifically positive outcome unless the roadmap explicitly requires one.

Correctly executed results may include:

- null effects;
- failed hypotheses;
- negative results;
- unsupported claims;
- unavailable operating points;
- infeasible experimental cells;
- insufficient evidence;
- non-significant comparisons.

Such outcomes must remain visible and must be represented according to roadmap-defined semantics.

Scientific failure is not implementation failure when the required experiment was executed correctly.

---

## 2. Audit Status Model

Every mandatory audit item must resolve to exactly one of:

```text
PASS
FAIL
NOT_APPLICABLE
```

Rules:

- `PASS` requires inspectable evidence.
- `FAIL` means the implementation, validation, artifact, or evidence does not satisfy the roadmap.
- `NOT_APPLICABLE` is allowed only when the roadmap or repository state makes the check genuinely irrelevant.
- Every `NOT_APPLICABLE` result requires a written justification.
- `UNKNOWN`, `TBD`, `TODO`, `PENDING`, unresolved ambiguity, probabilistic wording, or equivalent states are forbidden at final audit.

---

## 3. Final PASS / FAIL Rules

The global audit may be marked `PASS` only if all of the following are true:

- [ ] Mandatory roadmap coverage is exactly `100%`.
- [ ] Unmapped mandatory requirements = `0`.
- [ ] Unresolved blocking ambiguities = `0`.
- [ ] Every triggered conditional requirement is satisfied.
- [ ] Every milestone is complete.
- [ ] Every milestone audit has a current `PASS`.
- [ ] No downstream change has invalidated a milestone audit.
- [ ] Forward traceability is complete for every mandatory requirement.
- [ ] Reverse traceability contains no unauthorized implementation.
- [ ] All roadmap-required dataset and preprocessing checks pass.
- [ ] All roadmap-required algorithmic and mathematical checks pass.
- [ ] All roadmap-required configuration and default checks pass.
- [ ] The complete required experiment matrix is accounted for.
- [ ] All required metrics and statistical procedures are validated.
- [ ] All required execution and reproducibility checks pass.
- [ ] All required artifacts exist and validate semantically.
- [ ] Cross-artifact consistency checks pass.
- [ ] Claim and scope boundaries are respected.
- [ ] All configured repository quality gates pass.
- [ ] Required documentation and operational contracts match implementation.
- [ ] Independent fresh roadmap derivation finds no missing mandatory obligation.
- [ ] Cross-section consistency checks reveal no unresolved contradiction.
- [ ] Blocking findings = `0`.
- [ ] No mandatory roadmap violation is classified as non-blocking.
- [ ] Repository state matches the audited commit.

If any mandatory condition above fails, the global audit result is:

```text
FAIL
```

There is no partial, provisional, approximate, or discretionary global `PASS`.

---

## 4. Global Non-Negotiable Rules

These rules apply to every section of this audit.

- [ ] No unresolved ambiguity remains.
- [ ] No undocumented assumption is required to understand or execute mandatory behavior.
- [ ] No silent scientific or implementation substitution occurred.
- [ ] No roadmap requirement was weakened because implementation was difficult.
- [ ] No mandatory requirement was omitted.
- [ ] No undocumented scientific default exists.
- [ ] No hidden fallback changes scientific behavior.
- [ ] No post-hoc scientific decision was introduced after inspecting results unless explicitly authorized by the roadmap.
- [ ] No unauthorized experiment, metric, comparator, dataset, algorithm, feature, or claim exists.
- [ ] No roadmap-defined exclusion was violated.
- [ ] No diagnostic-only result was promoted to confirmatory evidence unless explicitly authorized.
- [ ] No issue is considered complete solely because it is closed.
- [ ] No milestone is considered complete solely because its issues are closed.
- [ ] No artifact is considered valid solely because the file exists.
- [ ] No experiment is considered valid solely because a command completed successfully.
- [ ] No claim is considered supported solely because a metric was produced.
- [ ] Negative, null, failed, infeasible, or unsupported scientific outcomes remain visible.
- [ ] Conditional obligations are evaluated whenever their triggering conditions are satisfied.
- [ ] Every mandatory `PASS` is supported by inspectable evidence.

Administrative state is not implementation evidence.

---

## 5. Preconditions

Before substantive audit execution:

- [ ] The authoritative roadmap has been identified.
- [ ] The authoritative roadmap has been read from beginning to end.
- [ ] The current Roadmap Coverage Inventory exists.
- [ ] The current milestone decomposition exists.
- [ ] All roadmap-derived implementation issues exist.
- [ ] Milestone audits exist for all milestones.
- [ ] The repository is on the intended implementation state.
- [ ] The audited commit is known.
- [ ] No known blocking implementation issue is being intentionally deferred.
- [ ] Required external inputs needed for audit execution are available.
- [ ] Required repository commands can be invoked in the audit environment.

Precondition result:

```text
PASS / FAIL
```

Evidence:

- ...

---

## 6. Roadmap Coverage & Classification

Reread the entire roadmap from beginning to end.

Verify that every roadmap statement has been correctly classified as one of:

```text
MANDATORY_ACTIONABLE
CONDITIONAL_ACTIONABLE
NON_IMPLEMENTATION
```

### 6.1 Coverage Checks

- [ ] Coverage inventory reflects the complete roadmap.
- [ ] Every mandatory actionable requirement is represented.
- [ ] Every conditional actionable requirement is represented.
- [ ] Every conditional requirement has an explicit trigger condition.
- [ ] Non-implementation statements are correctly classified.
- [ ] No mandatory requirement remains unmapped.
- [ ] No triggered conditional requirement remains unsatisfied.
- [ ] No roadmap section was skipped because it appeared descriptive.
- [ ] No blocking ambiguity remains unresolved.

### 6.2 Coverage Summary

| Measure | Value | Required |
|---|---:|---:|
| Total roadmap requirements | ... | — |
| Mandatory actionable | ... | — |
| Conditional actionable | ... | — |
| Non-implementation | ... | — |
| Mapped mandatory | ... | = Mandatory actionable |
| Unmapped mandatory | ... | 0 |
| Triggered conditional | ... | — |
| Satisfied triggered conditional | ... | = Triggered conditional |
| Mandatory coverage | ...% | 100% |
| Blocking ambiguities | ... | 0 |

Coverage result:

```text
PASS / FAIL
```

---

## 7. Roadmap Fidelity

Coverage alone is insufficient. Verify that implemented behavior preserves the roadmap's intended meaning.

- [ ] Roadmap terminology is preserved where terminology is semantically meaningful.
- [ ] Algorithm identities are preserved.
- [ ] Formula definitions are preserved.
- [ ] Numerical constants are preserved.
- [ ] Dataset semantics are preserved.
- [ ] Client/population/partition semantics are preserved.
- [ ] Split semantics are preserved.
- [ ] Experiment identities are preserved.
- [ ] Metric definitions are preserved.
- [ ] Statistical procedures are preserved.
- [ ] Artifact semantics are preserved.
- [ ] Claim boundaries are preserved.
- [ ] Scope exclusions are preserved.
- [ ] Optional work remains optional unless its roadmap trigger is satisfied.
- [ ] Diagnostic work remains diagnostic unless the roadmap authorizes stronger use.
- [ ] No roadmap requirement was silently reinterpreted.
- [ ] No scientifically meaningful behavior differs from the roadmap without explicit authorization.

Roadmap fidelity result:

```text
PASS / FAIL
```

Evidence / findings:

- ...

---

## 8. Milestone Completion

Every milestone must be evaluated against the current repository state.

Do not rely only on historical milestone audit results.

| Milestone | Required Issues Complete | Audit Status | Revalidated After Downstream Changes | Open Findings | Final Status |
|---|---|---|---|---:|---|
| ... | PASS / FAIL | PASS / FAIL | PASS / FAIL / N/A | ... | PASS / FAIL |
| ... | ... | ... | ... | ... | ... |

For every milestone:

- [ ] All mandatory implementation issues are complete.
- [ ] All issue acceptance criteria are satisfied by evidence.
- [ ] Milestone audit has a current `PASS`.
- [ ] Shared downstream changes have been assessed for invalidation.
- [ ] Any invalidated milestone audit was rerun.
- [ ] Reopened findings have been resolved.
- [ ] No milestone has unresolved mandatory debt.

### 8.1 Downstream Invalidation Triggers

A previous milestone audit must be reconsidered when later changes affect shared:

- source code;
- scientific algorithms;
- configuration;
- preprocessing;
- schemas or data contracts;
- public/internal APIs;
- experiment orchestration;
- metric computation;
- statistical analysis;
- artifact generation;
- reporting logic;
- shared tests or validation infrastructure.

Milestone completion result:

```text
PASS / FAIL
```

---

## 9. Forward Traceability

For every mandatory roadmap requirement, verify the complete chain:

```text
Roadmap requirement
→ Coverage Inventory entry
→ Milestone
→ GitHub issue
→ Acceptance criteria
→ Implementation
→ Test / validation
→ Evidence / artifact
```

Every mandatory requirement must have an inspectable traceability row.

| Requirement ID | Roadmap Requirement | Milestone | Issue | Implementation Location | Test / Validation | Evidence / Artifact | Result |
|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | PASS / FAIL |
| ... | ... | ... | ... | ... | ... | ... | PASS / FAIL |

Checks:

- [ ] Every mandatory roadmap requirement has a row.
- [ ] Every row maps to the correct coverage entry.
- [ ] Every row maps to the correct milestone.
- [ ] Every row maps to one or more implementation issues.
- [ ] Every issue has explicit acceptance criteria.
- [ ] Every acceptance criterion maps to actual implementation.
- [ ] Every implementation obligation has validation.
- [ ] Every validation produces or points to inspectable evidence.
- [ ] No traceability chain terminates at an administrative state.
- [ ] No required traceability link is missing.

Forward traceability result:

```text
PASS / FAIL
```

---

## 10. Reverse Traceability

Verify the opposite direction:

```text
Implementation / configuration / experiment / metric / artifact
→ authorized roadmap requirement
```

Audit all scientifically or operationally meaningful repository elements.

- [ ] Every source module is roadmap-authorized or necessary supporting infrastructure.
- [ ] Every scientific algorithm is roadmap-authorized.
- [ ] Every scientific configuration value is roadmap-authorized or derived by an authorized deterministic rule.
- [ ] Every scientific default is documented.
- [ ] Every CLI command or execution path is roadmap-authorized or required supporting infrastructure.
- [ ] Every dataset used is authorized.
- [ ] Every experiment is authorized.
- [ ] Every baseline is authorized.
- [ ] Every comparator is authorized.
- [ ] Every ablation is authorized.
- [ ] Every robustness condition is authorized.
- [ ] Every metric is authorized.
- [ ] Every statistical procedure is authorized.
- [ ] Every table is authorized.
- [ ] Every figure is authorized.
- [ ] Every report is authorized.
- [ ] Every claim-bearing artifact is authorized.
- [ ] No unexplained scientific functionality exists.
- [ ] No undocumented experiment exists.
- [ ] No unsupported comparator exists.
- [ ] No undocumented scientific default exists.

Reverse-traceability violations:

| Element | Type | Roadmap Authorization | Result | Finding |
|---|---|---|---|---|
| ... | ... | ... | PASS / FAIL | ... |

Reverse traceability result:

```text
PASS / FAIL
```

---

## 11. Dataset & Preprocessing

For every roadmap-authorized dataset and data path:

### 11.1 Dataset Identity & Provenance

- [ ] Actual raw source files were inspected.
- [ ] Dataset identity matches the roadmap.
- [ ] Dataset version/release assumptions are validated where applicable.
- [ ] Source provenance is recorded.
- [ ] Expected-vs-observed file inventory is recorded where required.
- [ ] Expected-vs-observed schema is recorded.
- [ ] Dataset-specific assumptions are validated against actual raw data.

### 11.2 Schema & Semantics

- [ ] Required columns/features exist or are adapted according to roadmap rules.
- [ ] Labels are validated.
- [ ] Client identities / partitions match roadmap semantics.
- [ ] Device/site/population identity logic matches the roadmap.
- [ ] Counts used by implementation come from actual data where required.
- [ ] Dataset-dependent decisions are deterministic.
- [ ] Legitimate raw-data differences are handled according to an explicit deterministic rule.
- [ ] Adaptations are recorded in evidence/manifests where required.

### 11.3 Splits & Leakage

- [ ] Train/calibration/validation/test split rules match the roadmap.
- [ ] Split ordering semantics match.
- [ ] Split boundaries are deterministic.
- [ ] Duplicate leakage checks pass where applicable.
- [ ] Cross-split identity leakage checks pass where applicable.
- [ ] Test data do not influence training.
- [ ] Test data do not influence calibration.
- [ ] Test data do not influence model selection unless explicitly authorized.
- [ ] Preprocessing statistics are fitted only on roadmap-authorized data.
- [ ] Reservoir/candidate pools obey roadmap boundaries where applicable.
- [ ] Attack or evaluation labels cannot leak into forbidden stages.

### 11.4 Preprocessing

- [ ] Raw-data handling matches the roadmap.
- [ ] Feature derivation matches the roadmap.
- [ ] Normalization/scaling semantics match.
- [ ] Categorical handling matches.
- [ ] Missing-value handling matches.
- [ ] Duplicate handling matches.
- [ ] Ordering/chronology handling matches.
- [ ] Preprocessing is deterministic where required.
- [ ] Preprocessing outputs carry sufficient provenance.

Dataset and preprocessing result:

```text
PASS / FAIL
```

Evidence / findings:

- ...

---

## 12. Algorithms & Mathematics

For every roadmap-defined algorithm, equation, decision rule, and scientific transformation:

### 12.1 Algorithm Presence & Identity

- [ ] Every required algorithm is implemented.
- [ ] No required algorithm is substituted.
- [ ] Algorithm naming matches roadmap semantics.
- [ ] Inputs and outputs match the defined contract.
- [ ] Algorithm composition/order matches the roadmap.

### 12.2 Formula Fidelity

- [ ] Formula implementations match the roadmap exactly.
- [ ] Aggregation formulas match.
- [ ] Weighting rules match.
- [ ] Normalization rules match.
- [ ] Threshold/calibration rules match where applicable.
- [ ] Statistical transformations match.
- [ ] Support / abstention conditions match.
- [ ] Eligibility logic matches.

### 12.3 Numerical Conventions

- [ ] Numerical precision conventions match.
- [ ] Rounding conventions match.
- [ ] Quantile/percentile conventions match.
- [ ] Inclusivity/exclusivity boundaries match.
- [ ] Tie-breaking rules match.
- [ ] Sorting/order dependence is deterministic.
- [ ] Numerical stabilization constants match.
- [ ] Degenerate-input behavior matches.
- [ ] Missing/empty input behavior matches.
- [ ] NaN/Inf behavior is explicit and correct.

### 12.4 Randomness

- [ ] Randomness is introduced only where authorized.
- [ ] Required seeds are used.
- [ ] Seed namespaces/roles match.
- [ ] Paired randomness semantics match where required.
- [ ] Deterministic components remain deterministic.
- [ ] Repeated execution with identical identity/configuration reproduces required outputs.

Algorithms and mathematics result:

```text
PASS / FAIL
```

Evidence / findings:

- ...

---

## 13. Configuration & Scientific Defaults

Audit every scientifically meaningful configuration value.

- [ ] Every scientific parameter comes from the roadmap or an explicitly authorized deterministic data-derived rule.
- [ ] No magic scientific value exists only in implementation code.
- [ ] No hidden scientific fallback exists.
- [ ] No environment-dependent scientific behavior exists unless roadmap-authorized.
- [ ] Invalid parameter combinations fail explicitly.
- [ ] Required enumerations reject unsupported values.
- [ ] Scientific defaults are explicit.
- [ ] Default values match the roadmap.
- [ ] CLI/configuration interfaces cannot silently change scientific semantics.
- [ ] Executed values are recoverable from run provenance.
- [ ] Serialized configuration is sufficient to identify the scientific condition.
- [ ] Configuration identity prevents incompatible runs from sharing an experiment identity.
- [ ] Configuration changes trigger correct artifact invalidation.

Configuration and defaults result:

```text
PASS / FAIL
```

Evidence / findings:

- ...

---

## 14. Experiment Matrix

Derive the expected experiment matrix directly from the roadmap.

The matrix must account for every roadmap-defined factor, including as applicable:

- dataset;
- population/client definition;
- model;
- FL algorithm;
- threshold/policy;
- baseline;
- comparator;
- ablation;
- robustness condition;
- attack condition;
- objective;
- fraction/intensity;
- seed;
- scalability setting;
- hardware/deployment setting;
- optional/conditional stage.

### 14.1 Matrix Accounting

| Measure | Value |
|---|---:|
| Expected experiment cells | ... |
| Executable experiment cells | ... |
| Completed valid cells | ... |
| Legitimately infeasible cells | ... |
| Legitimately skipped conditional cells | ... |
| Failed technical cells | ... |
| Missing unexplained cells | ... |

Requirements:

- [ ] All mandatory experiments are represented.
- [ ] All required baselines are represented.
- [ ] All required comparators are represented.
- [ ] All required ablations are represented.
- [ ] All required robustness conditions are represented.
- [ ] All required scalability conditions are represented.
- [ ] All required seeds are represented.
- [ ] All required datasets/populations are represented.
- [ ] Every expected cell has an explicit status.
- [ ] Every infeasible cell satisfies roadmap-defined infeasibility semantics.
- [ ] Every skipped conditional cell documents why its trigger was not satisfied.
- [ ] Failed technical cells are not mistaken for scientific outcomes.
- [ ] Missing unexplained cells = `0`.
- [ ] No post-hoc experiment substitution occurred.

### 14.2 Experiment Identity

- [ ] Scientifically distinct conditions have distinct experiment identities.
- [ ] Equivalent identities cannot refer to different scientific conditions.
- [ ] Outputs from one experiment cannot overwrite another.
- [ ] Results cannot be silently reused across incompatible conditions.
- [ ] Resume/reuse logic verifies full scientific compatibility before reuse.

Experiment matrix result:

```text
PASS / FAIL
```

---

## 15. Metrics & Statistics

For every roadmap-defined metric, statistical procedure, decision rule, and claim gate:

### 15.1 Metric Fidelity

- [ ] Every required metric is implemented.
- [ ] Formula definitions match the roadmap.
- [ ] Averaging/aggregation semantics match.
- [ ] Macro/micro/weighted semantics match where applicable.
- [ ] Per-client/global semantics match.
- [ ] Worst-client/best-client semantics match.
- [ ] Missing-value semantics match.
- [ ] Failed-cell semantics match.
- [ ] Infeasible-cell semantics match.
- [ ] Denominators are correct.
- [ ] Metric source rows are traceable.

### 15.2 Statistical Fidelity

- [ ] Required statistical tests are implemented.
- [ ] Paired/unpaired semantics match.
- [ ] Analysis unit matches the roadmap.
- [ ] Seed-level aggregation matches.
- [ ] Multiplicity handling matches.
- [ ] Confidence interval procedures match.
- [ ] Bootstrap semantics match where applicable.
- [ ] Equivalence criteria match.
- [ ] Superiority criteria match.
- [ ] Materiality criteria match.
- [ ] Claim-gate logic matches.
- [ ] Missing-data handling matches.
- [ ] Statistical implementation is validated against a known/reference case where practical.

### 15.3 Result Provenance

- [ ] Every claim-bearing metric can be traced to source experiment rows/cells.
- [ ] Every statistical output can be traced to its exact input set.
- [ ] Tables and reports do not recompute metrics using different semantics.
- [ ] No post-hoc exclusion changes claim-bearing results.
- [ ] No technical failure is silently dropped from statistics.
- [ ] No negative or null result is suppressed.

Metrics and statistics result:

```text
PASS / FAIL
```

Evidence / findings:

- ...

---

## 16. Execution & Reproducibility

### 16.1 Required Execution Paths

Every roadmap-required workflow must exist and be executable.

Audit as applicable:

- [ ] validation;
- [ ] planning;
- [ ] preprocessing;
- [ ] synthetic/smoke execution;
- [ ] baseline execution;
- [ ] main experiment execution;
- [ ] optional/conditional execution;
- [ ] reporting;
- [ ] status inspection;
- [ ] resume;
- [ ] artifact validation;
- [ ] final analysis.

For each required path:

- [ ] The command/interface exists.
- [ ] Invocation matches documented usage.
- [ ] Required inputs are validated.
- [ ] Invalid inputs fail explicitly.
- [ ] Outputs are produced in roadmap-defined locations.
- [ ] Scientific identity is preserved through execution.

### 16.2 Reproducibility

- [ ] Runs are deterministic where required.
- [ ] Randomized runs are reproducible from recorded seeds.
- [ ] Code provenance is recorded.
- [ ] Configuration provenance is recorded.
- [ ] Dataset/input provenance is recorded.
- [ ] Environment/dependency provenance is sufficient.
- [ ] Artifact dependency tracking works.
- [ ] Resume/reuse semantics work.
- [ ] Selective invalidation works.
- [ ] Stale artifacts are detected.
- [ ] Failed scientific outcomes remain distinguishable from technical failure.
- [ ] Interrupted executions cannot masquerade as complete runs.

### 16.3 Clean-Environment Execution

Where required by the roadmap or project quality contract:

- [ ] Dependencies can be installed/resolved from repository-controlled metadata.
- [ ] Required environment assumptions are documented.
- [ ] No undocumented local path is required.
- [ ] No undocumented manual mutation is required.
- [ ] Repository plus declared external inputs is sufficient to reproduce required workflows.

Execution and reproducibility result:

```text
PASS / FAIL
```

---

## 17. Artifacts & Evidence

For every roadmap-required artifact:

- [ ] Artifact exists.
- [ ] Artifact path/name matches the expected contract.
- [ ] Artifact schema is correct.
- [ ] Artifact producer is correct.
- [ ] Artifact scientific identity is correct.
- [ ] Artifact contains all expected rows/records/fields.
- [ ] Artifact upstream dependencies are correct.
- [ ] Artifact provenance is complete.
- [ ] Artifact is not stale.
- [ ] Artifact validation passes.
- [ ] Artifact contents are semantically correct.
- [ ] Artifact can be regenerated where required.

Audit all required artifact classes as applicable:

- [ ] raw results;
- [ ] processed results;
- [ ] summaries;
- [ ] tables;
- [ ] figures;
- [ ] manifests;
- [ ] statistical outputs;
- [ ] audit outputs;
- [ ] claim/evidence outputs;
- [ ] reporting artifacts;
- [ ] execution plans;
- [ ] status/provenance records.

### 17.1 Artifact Inventory

| Artifact | Required By | Producer | Validation | Provenance | Status |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | PASS / FAIL |
| ... | ... | ... | ... | ... | PASS / FAIL |

Artifact result:

```text
PASS / FAIL
```

---

## 18. Cross-Artifact Consistency

Verify that downstream artifacts faithfully reflect their upstream evidence.

Required consistency paths as applicable:

```text
raw results
→ processed results
→ summaries
→ statistical outputs
→ tables / figures
→ claims / reports
```

Checks:

- [ ] Raw results match processed results.
- [ ] Processed results match summaries.
- [ ] Summaries match statistical outputs.
- [ ] Statistical outputs match tables.
- [ ] Statistical outputs match figures.
- [ ] Table values match source evidence.
- [ ] Figure values match source evidence.
- [ ] Reports match tables/figures/statistical outputs.
- [ ] Manifests match actual files.
- [ ] Artifact counts match experiment-matrix accounting.
- [ ] No stale downstream artifact reflects an earlier run.
- [ ] No claim-bearing artifact combines incompatible experiment identities.

Cross-artifact consistency result:

```text
PASS / FAIL
```

Evidence / findings:

- ...

---

## 19. Claims & Scope Boundaries

### 19.1 Claim → Evidence Audit

For every roadmap-authorized claim:

```text
Claim
→ required experiment(s)
→ required metric(s)
→ required statistical gate(s)
→ required artifact(s)
→ actual result
→ supported / unsupported status
```

| Claim ID | Claim | Required Evidence | Actual Evidence | Result | Final Claim Status |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | SUPPORTED / UNSUPPORTED |
| ... | ... | ... | ... | ... | SUPPORTED / UNSUPPORTED |

Checks:

- [ ] Every supported claim has the required evidence.
- [ ] Unsupported claims remain explicitly unsupported.
- [ ] Claim wording does not exceed evidence.
- [ ] Forbidden extrapolations have not entered outputs.
- [ ] Negative scientific results are preserved.
- [ ] Unavailable operating points are reported according to roadmap semantics.
- [ ] Insufficient-evidence outcomes remain visible.
- [ ] No diagnostic result is used as confirmatory evidence unless authorized.
- [ ] No optional extension result is generalized beyond its roadmap claim boundary.
- [ ] No dataset supports a stronger claim than its roadmap role permits.

### 19.2 Scope Guardrails

- [ ] Excluded experiments were not introduced.
- [ ] Excluded algorithms were not introduced.
- [ ] Excluded datasets were not promoted into unauthorized evidence.
- [ ] Excluded metrics were not introduced as claim-bearing outputs.
- [ ] Diagnostic-only stages remain diagnostic.
- [ ] Optional work is not represented as mandatory unless triggered.
- [ ] No broad security/privacy/deployment claim exceeds the roadmap.
- [ ] No raw-data or operational claim is inferred from proxy evidence unless explicitly authorized.

Claims and scope result:

```text
PASS / FAIL
```

---

## 20. Repository Quality

All configured repository quality gates are mandatory unless the roadmap explicitly states otherwise.

### 20.1 Automated Quality Gates

- [ ] Full test suite passes.
- [ ] Full linting passes.
- [ ] Full formatting checks pass.
- [ ] Full type checking passes.
- [ ] Architecture checks pass.
- [ ] Static-analysis gates pass where configured.
- [ ] Dependency validation passes.
- [ ] Security/secret scanning passes where configured.

### 20.2 Structural Quality

- [ ] No stale required code remains.
- [ ] No duplicate scientific implementation remains.
- [ ] No dead required path remains.
- [ ] No unresolved repository failure remains.
- [ ] No relevant `TODO`, `FIXME`, placeholder, stub, or unimplemented branch remains.
- [ ] No debug-only scientific behavior remains active.
- [ ] No generated/cache/transient junk is improperly committed.
- [ ] No secret or credential is committed.
- [ ] Repository structure matches required architecture.
- [ ] Repository working tree is clean at audit completion, unless explicitly documented and irrelevant to the audited state.

Repository quality result:

```text
PASS / FAIL
```

Evidence / commands:

- ...

---

## 21. Documentation & Operational Contract

Where the roadmap or project requires documented usage:

- [ ] README instructions match actual implementation.
- [ ] CLI help matches accepted commands/options.
- [ ] Configuration documentation matches accepted values.
- [ ] Setup instructions work.
- [ ] Environment requirements are accurate.
- [ ] Dataset/input placement instructions are accurate.
- [ ] Execution examples are valid.
- [ ] Artifact paths match actual outputs.
- [ ] Reporting instructions match actual commands.
- [ ] Resume/reuse instructions match implementation.
- [ ] No stale command or removed option remains documented.
- [ ] No undocumented mandatory manual step exists.

Documentation result:

```text
PASS / FAIL / NOT_APPLICABLE
```

Justification if `NOT_APPLICABLE`:

- ...

---

## 22. Independent Fresh Roadmap Pass

Reread the authoritative roadmap from beginning to end without relying on the existing coverage inventory.

For every roadmap section ask:

> What implementation, validation, experiment, configuration, dataset, artifact, statistical, reporting, or claim obligation does this section create?

Build a fresh requirement set independently.

Then compare:

```text
fresh_requirements - inventory_requirements
```

Required result:

```text
∅
```

and:

```text
inventory_requirements - roadmap_authorized_requirements
```

Required result:

```text
∅
```

Checks:

- [ ] Fresh derivation completed without relying on inventory structure.
- [ ] No previously missed mandatory requirement was found.
- [ ] No previously missed triggered conditional requirement was found.
- [ ] No inventory requirement lacks roadmap authorization.
- [ ] Any newly discovered requirement was added to coverage and fully implemented before continuing.
- [ ] Audit was rerun after any newly discovered requirement changed implementation.

Independent fresh-pass result:

```text
PASS / FAIL
```

Newly discovered obligations:

- None / ...

---

## 23. Cross-Section Consistency Audit

Verify that audit conclusions are mutually consistent.

Check for contradictions such as:

- coverage says complete while an artifact is missing;
- milestone audit says `PASS` while shared downstream code invalidated it;
- experiment matrix says complete while required seed cells are absent;
- metric audit says valid while reporting uses different semantics;
- artifact audit says valid while manifest identity differs;
- claim audit says supported while required statistical gate failed;
- repository quality says clean while required tests fail;
- reverse traceability says authorized while scope audit identifies unauthorized work.

Checks:

- [ ] Coverage findings are consistent with traceability.
- [ ] Milestone findings are consistent with repository state.
- [ ] Dataset findings are consistent with experiment identity.
- [ ] Algorithm findings are consistent with metric outputs.
- [ ] Experiment matrix is consistent with artifact inventory.
- [ ] Statistics are consistent with claim status.
- [ ] Claim status is consistent with reporting.
- [ ] Repository quality is consistent with execution evidence.
- [ ] No unresolved contradiction remains across audit sections.

Cross-section consistency result:

```text
PASS / FAIL
```

Contradictions found:

- None / ...

---

## 24. Findings & Remediation

Every finding must be structured.

Allowed severity:

```text
BLOCKING
NON_BLOCKING
```

Rules:

- Any violation of a mandatory roadmap requirement is `BLOCKING`.
- Any failed mandatory audit gate is `BLOCKING`.
- Any unresolved ambiguity affecting mandatory behavior is `BLOCKING`.
- Any unauthorized scientific behavior is `BLOCKING`.
- `NON_BLOCKING` may be used only for genuinely optional quality observations that do not violate the roadmap, acceptance contract, or configured mandatory repository gates.
- A blocking finding cannot remain unresolved under a global `PASS`.

### 24.1 Findings Table

| Finding ID | Severity | Affected Requirement(s) | Description | Evidence | Required Remediation | Issue | Resolution Evidence | Status |
|---|---|---|---|---|---|---|---|---|
| ... | BLOCKING / NON_BLOCKING | ... | ... | ... | ... | ... | ... | OPEN / RESOLVED |

### 24.2 Remediation Rules

Every blocking finding must either:

1. be fixed during the audit and fully revalidated; or
2. have a concrete implementation issue created and force the global audit to `FAIL`.

No blocking finding may remain as prose-only technical debt.

Blocking findings:

- None / ...

Non-blocking findings:

- None / ...

New implementation issues created:

- None / ...

---

## 25. Final Quantitative Scorecard

| Measure | Actual | Required | Result |
|---|---:|---:|---|
| Mandatory requirements mapped | ... / ... | 100% | PASS / FAIL |
| Triggered conditional requirements satisfied | ... / ... | 100% | PASS / FAIL |
| Milestones passed | ... / ... | 100% | PASS / FAIL |
| Current milestone audits passed | ... / ... | 100% | PASS / FAIL |
| Mandatory forward traceability rows complete | ... / ... | 100% | PASS / FAIL |
| Reverse-traceability violations | ... | 0 | PASS / FAIL |
| Expected experiment cells accounted for | ... / ... | 100% | PASS / FAIL |
| Missing unexplained experiment cells | ... | 0 | PASS / FAIL |
| Required artifacts validated | ... / ... | 100% | PASS / FAIL |
| Required claim gates evaluated | ... / ... | 100% | PASS / FAIL |
| Repository quality gates passed | ... / ... | 100% | PASS / FAIL |
| Fresh second-pass omissions | ... | 0 | PASS / FAIL |
| Cross-section contradictions | ... | 0 | PASS / FAIL |
| Blocking findings | ... | 0 | PASS / FAIL |

---

## 26. Final Audit Verdict

### Mandatory Final Checks

- [ ] All global PASS conditions in Section 3 are satisfied.
- [ ] All mandatory audit sections are `PASS`.
- [ ] Every `NOT_APPLICABLE` result has a valid written justification.
- [ ] Blocking findings = `0`.
- [ ] No unresolved ambiguity remains.
- [ ] No unresolved mandatory implementation issue remains.
- [ ] Final scorecard satisfies every required threshold.
- [ ] Audited repository state matches the commit recorded below.

### Audit Result

- [ ] PASS
- [ ] FAIL

### Final Failure Reasons

Required when result is `FAIL`.

- None / ...

### Final Non-Blocking Observations

- None / ...

### Audit Provenance

Audited commit:

Audit date:
