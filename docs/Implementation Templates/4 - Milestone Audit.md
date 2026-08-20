# Milestone Audit — <Milestone ID>: <Milestone Name>

## Purpose

Repeatedly verify that this milestone remains complete, correct, roadmap-faithful, scientifically valid, tested, integrated, reproducible, provenance-safe, and compliant with all mandatory project-wide engineering and research rules.

Rerun this audit after milestone work, bug fixes, refactors, shared-infrastructure changes, dependency changes, dataset/preprocessing changes, artifact-schema changes, downstream integration, CI changes, experiment failures, or resolution of prior findings.

This audit has two scopes:

- **Milestone scope** — requirements, issues, implementation, tests, evidence, and scientific behavior owned or affected by this milestone.
- **Global scope** — the entire repository. Global failures are never ignored because they are pre-existing, unrelated to the milestone, or outside changed files.

---

# 1. Audit Identity & Authority

## Audit Identity

- Milestone ID:
- Milestone name:
- Roadmap sections:
- Covered requirement IDs:
  - REQ-...
- Milestone issues:
  - #...
- Prerequisite milestones:
  - ...
- Audit trigger / reason:
- Implementation revision:
- Audit date:
- Agent / auditor:

## Authority Order

Unless the roadmap explicitly defines otherwise:

1. Roadmap.
2. Roadmap Coverage Inventory.
3. Milestone definition.
4. Issue specifications and acceptance criteria.
5. Project Engineering & Research Rules.
6. Implementation, tests, artifacts, reports, and runtime behavior.

- [ ] No lower-authority source silently overrides a higher-authority source.
- [ ] Existing implementation is not treated as authoritative merely because it exists.
- [ ] Closed issues are verified against actual acceptance criteria rather than trusted administratively.
- [ ] Contradictions between authoritative sources are recorded as findings.
- [ ] Ambiguities affecting correctness are resolved before PASS.
- [ ] No agent or implementation invents missing scientific behavior, values, experiments, or semantics.

---

# 2. Audit Contract

## Mandatory Rules

- [ ] Every applicable mandatory item is checked.
- [ ] `N/A` is used only with a written justification.
- [ ] Unknown, unchecked, unresolved, or unverified mandatory items prohibit PASS.
- [ ] Pre-existing or unrelated repository failures discovered during the audit are fixed or recorded as blocking findings.
- [ ] No mandatory global-rule violation is dismissed because another milestone introduced it.
- [ ] Previous findings are reconciled on every rerun; none disappear silently.

## Finding Classes

- `BLOCKER` — prevents PASS.
- `NON-BLOCKING` — genuine observation that does not affect roadmap coverage, correctness, scientific validity, evidence validity, required quality gates, integration, or readiness.

A mandatory rule violation cannot be downgraded merely to allow PASS.

## Verdict Rules

### `PASS`

Requires:

- all mandatory roadmap requirements covered and verified;
- all required milestone issues genuinely complete;
- roadmap/scientific fidelity preserved;
- required tests and scientific validation passing;
- required artifacts and provenance valid;
- integration/regression passing;
- all applicable global engineering/research rules passing;
- all mandatory quality gates passing;
- zero open blockers;
- zero unchecked or unknown mandatory items.

### `PASS WITH NON-BLOCKING FINDINGS`

Requires every `PASS` condition, with only explicitly documented non-blocking findings remaining.

### `FAIL`

Required when any mandatory requirement, issue, scientific behavior, evidence, test, integration check, global rule, or quality gate is missing, invalid, unverified, or failed, or when any blocker remains open.

---

# 3. Roadmap Coverage & Bidirectional Traceability

## Roadmap → Milestone → Issue

- [ ] Every roadmap requirement assigned to this milestone appears in the Roadmap Coverage Inventory.
- [ ] Every mandatory requirement is intentionally assigned to implementation work.
- [ ] Every mapped requirement has one or more corresponding issues.
- [ ] Every required issue exists and is genuinely complete.
- [ ] No mandatory requirement remains `UNMAPPED`.
- [ ] No unresolved `AMBIGUOUS` requirement can affect correctness.
- [ ] Conditional requirements are handled exactly according to their activation conditions.
- [ ] Requirement ownership does not conflict across milestones.

## Requirement → Evidence

Verify the applicable traceability chain:

```text
Roadmap requirement
  → Coverage Inventory
  → Milestone
  → Issue(s)
  → Implementation
  → Test(s)
  → Artifact(s) / verified evidence
  → Report / claim consumer, if applicable
```

- [ ] Every mandatory requirement reaches the implementation evidence it requires.
- [ ] Every required test maps to the behavior it validates.
- [ ] Every required artifact maps to the requirement or experiment requiring it.
- [ ] Every claim-bearing output maps to verified evidence.

## Reverse Traceability / Scope Control

- [ ] Every substantive implementation path introduced or changed by the milestone maps back to an authorized issue and requirement.
- [ ] No unauthorized algorithm, experiment, metric, scientific decision, configuration path, fallback, or alternative execution path has been introduced.
- [ ] No issue adds scientific behavior unsupported by the roadmap.
- [ ] No unnecessary compatibility behavior preserves superseded semantics.

---

# 4. Roadmap & Scientific Fidelity

## Algorithms, Mathematics & Semantics

- [ ] Algorithms match the roadmap.
- [ ] Mathematical definitions match the roadmap.
- [ ] Numerical constants come from the correct authority.
- [ ] Equality semantics, thresholds, tolerances, and boundary behavior match the roadmap.
- [ ] Failure, unavailable, infeasible, insufficient-evidence, suppressed, and abstention semantics match the roadmap.
- [ ] No implementation shortcut changes scientific meaning.

## Configuration

- [ ] Configuration authority matches the roadmap.
- [ ] No scientific value has been duplicated or independently redefined.
- [ ] No unspecified scientific value has been invented.
- [ ] Invalid scientific/configuration combinations fail before execution.

## Datasets

- [ ] Dataset identity, schema, clients, features, labels, chronology, preprocessing, and split semantics match the roadmap and validated raw data.
- [ ] Dataset-specific assumptions were validated against actual data where applicable.
- [ ] No unavailable labels, timestamps, client identities, features, or evidence were fabricated.

## Experiments & Claims

- [ ] Experiment purpose, experimental unit, seeds, variables, controls, comparators, metrics, statistical analysis, artifacts, and completion criteria match the roadmap.
- [ ] Expected experiment coordinates are predefined rather than invented during execution.
- [ ] Confirmatory, robustness, mechanism, external-validation, and exploratory roles remain distinct where applicable.
- [ ] Claim boundaries remain unchanged.

---

# 5. Implementation Completeness

## Issues

For every milestone issue:

- [ ] Acceptance criteria are satisfied in the implementation.
- [ ] Required code paths exist.
- [ ] Required tests exist and pass.
- [ ] Required artifacts/outputs exist and validate.
- [ ] Referenced requirements are actually satisfied.
- [ ] Closing the issue did not leave hidden correctness work behind.

## Implementation

- [ ] All required execution paths exist.
- [ ] Required edge cases and invalid states are handled.
- [ ] Unsupported conditions fail, abstain, or report unavailable exactly as specified.
- [ ] Deterministic behavior is preserved where required.
- [ ] No placeholder or temporary implementation remains.
- [ ] No milestone-required TODO or FIXME remains.
- [ ] No silent fallback or swallowed exception changes required behavior.
- [ ] No duplicate, stale, or undocumented alternative implementation path remains active.

---

# 6. Testing & Scientific Validation

## Test Coverage

- [ ] Required unit, integration, architecture, configuration, dataset, artifact/provenance, execution/completion, regression, and smoke tests exist where applicable.
- [ ] Boundary conditions, invalid states, failure paths, serialization round trips, corruption handling, stale evidence, resume/reuse/rebuild, and unsupported states are tested where applicable.
- [ ] Obsolete tests were removed rather than preserved unnecessarily after redesigns.

## Scientific Validation

- [ ] Scientific formulas are tested numerically against independently derived expected behavior.
- [ ] Required synthetic validation passes.
- [ ] Mathematical invariants hold.
- [ ] Expected null/control behavior is validated where applicable.
- [ ] Equality/tolerance semantics are tested.
- [ ] Ties, degenerate samples, small samples, NaN, infinity, and undefined cases are handled and tested where applicable.
- [ ] Missing/unavailable metrics are never silently converted to zero.
- [ ] Paired-analysis provenance is established before paired tests are used.
- [ ] No pseudo-replication is introduced.
- [ ] Confirmatory and exploratory analyses cannot be silently mixed.
- [ ] No test merely asserts the implementation against itself.

## Reproducibility

- [ ] Every stochastic process has an explicit RNG/seed owner.
- [ ] Seed derivation is deterministic.
- [ ] Independent stochastic domains remain independent where required.
- [ ] Hidden global RNG state is absent where deterministic ownership is required.
- [ ] Reproducibility tolerances are defined and validated where exact equality is inappropriate.

## Result

- [ ] Full milestone-relevant test suite passes.
- [ ] Full repository suite passes where mandated by project policy.
- [ ] No failing test is dismissed as unrelated or pre-existing.

---

# 7. Artifacts, Provenance, Completion & Reuse

## Artifacts

- [ ] Required artifacts are defined before execution and all are produced.
- [ ] Artifact paths, serialization, hashing, manifests, and integrity rules use their authoritative infrastructure.
- [ ] Schemas validate on read.
- [ ] Writes are atomic where partial output could otherwise appear valid.
- [ ] Deterministic serialization is used where identity depends on content.
- [ ] Structured artifacts carry schema versions.
- [ ] Artifact ownership is explicit.

## Provenance

- [ ] Important outputs record the required experiment, dataset/split, seed, configuration/protocol, preprocessing, model/checkpoint, upstream evidence, schema, code, and content-digest provenance where applicable.
- [ ] Every important result can be traced to its required inputs.
- [ ] Scientifically relevant upstream changes invalidate dependent evidence.

## Completion

- [ ] Completion derives from verified evidence, not process exit status or file existence.
- [ ] Expected evidence is defined by protocol/design rather than directory scanning.
- [ ] Verification is read-only and never trains models, scores data, or creates missing artifacts.
- [ ] Not-started, execution-incomplete, evidence-incomplete, stale, failed, blocked, and passed states remain distinguishable where applicable.
- [ ] `status`, `report`, and publication use the same completion definition.

## Resume / Reuse

- [ ] Reuse occurs only for valid, current, provenance-compatible evidence.
- [ ] Stale, malformed, incomplete, or failed evidence is never silently reused.
- [ ] Reruns deterministically reuse verified evidence or explicitly rebuild owned artifacts.
- [ ] Rebuild/overwrite never deletes unrelated or shared evidence.

---

# 8. Integration & Regression

- [ ] Integration with every prerequisite milestone passes.
- [ ] Existing earlier functionality still passes.
- [ ] Shared-infrastructure changes did not invalidate earlier assumptions.
- [ ] Configuration, dataset, preprocessing, or schema changes invalidate dependent evidence correctly.
- [ ] Downstream consumers remain correct where required.
- [ ] No unrelated repository failure remains unresolved.
- [ ] Smoke execution reaches the expected boundary through the real execution path.
- [ ] No milestone-only shortcut bypasses normal application wiring.

---

# 9. Mandatory Global Engineering & Research Rules

Every check in this section is **repository-wide**.

## Architecture

- [ ] Layered package responsibilities are clear; dependencies flow one way; lower layers do not depend on higher layers; `core` remains independent of execution, reporting, application, and CLI.
- [ ] Scientific logic is outside the CLI; reporting/presentation concerns are outside scientific modules.
- [ ] Import Linter and targeted architecture tests enforce dependency rules automatically.
- [ ] Modules are focused; no large unrelated-responsibility modules or empty template-only layers exist.

## Domain Types

- [ ] Enums represent finite identities/policies/statuses/dataset IDs/experiment IDs/algorithm choices where appropriate; frozen dataclasses represent meaningful immutable domain values/aggregates.
- [ ] Pydantic/`Annotated` constrained scalars are used for validation-only constraints; nominal types are used where accidental value mixing would create realistic errors; primitives are not wrapped merely for style.
- [ ] `Any`, anonymous domain dictionaries, and untyped domain payloads are absent unless narrowly justified.
- [ ] CLI/YAML/JSON/boundary primitives become validated domain types immediately; configuration and persisted structured artifacts use frozen Pydantic models where applicable.

## Configuration

- [ ] One authoritative immutable production configuration hierarchy exists; unknown fields are rejected and cross-field constraints are validated before execution.
- [ ] Scientific values are not duplicated across configuration, CLI defaults, constants, tests, and implementation; scientific constants are explicit and centrally owned.
- [ ] No implementation agent invents unspecified scientific values.
- [ ] Deterministic configuration/protocol digests support provenance and stale-evidence detection.

## Naming

- [ ] Packages, modules, classes, functions, experiments, and artifacts use descriptive domain names.
- [ ] Opaque aliases, unexplained abbreviations, unjustified generic names (`utils`, `helpers`, `common`, `manager`, `processor`, `base`), artificial versions (`v2`, `final2`, etc.), and obsolete terminology are absent.

## CLI

- [ ] CLI remains thin: parse, validate, invoke application logic, render results, translate errors.
- [ ] CLI contains no scientific algorithms, statistics, dataset construction, or artifact-layout ownership.
- [ ] Command surface is small and descriptive; typed experiment/policy identities replace free-form strings; runtime/dependency/GPU/environment readiness is exposed where useful.

## Datasets

- [ ] Dataset identity, source, schema, layout, labels, features, clients, chronology, preprocessing, splitting, and limitations are explicit.
- [ ] Raw data is never modified; expected files/columns/schemas/populations are validated before experiments.
- [ ] Train/calibration/validation/test roles are explicit and non-overlapping; leakage checks exist where scientifically relevant.
- [ ] Dataset/preprocessing provenance is persisted and changes invalidate dependents.
- [ ] Unavailable labels, metrics, clients, timestamps, or evidence are never fabricated.

## Experiments

- [ ] Every experiment predefines purpose/role, dataset, experimental unit, seeds, variables, controls, comparators, metrics, statistics, artifacts, completion criteria, and enumerable coordinates.
- [ ] Execution never discovers or invents scientific design dynamically.
- [ ] Upstream evidence is reused only when scientifically valid.
- [ ] Confirmatory, robustness, mechanism, external-validation, and exploratory experiments remain distinct.

## Artifacts

- [ ] Artifact path construction, serialization, hashing, manifests, and integrity checks are centralized.
- [ ] Deterministic serialization and atomic writes are used where required; artifacts validate when read; schema versions and clear ownership exist.
- [ ] JSON, Parquet, SafeTensors, or other formats are used according to artifact semantics rather than convenience.

## Provenance

- [ ] Important results are traceable to protocol, configuration, data, model, seed, code, and upstream evidence using content digests where relevant.
- [ ] Experiment/dataset/seed/config/protocol/preprocessing/model/upstream/schema/code identities are persisted where applicable.
- [ ] Scientifically relevant provenance changes automatically invalidate downstream evidence.

## Idempotence & Resume

- [ ] File existence never equals completion; missing/valid/stale/malformed/incomplete/failed/blocked states are distinguishable.
- [ ] Only valid current provenance-compatible evidence is reused; stale or malformed artifacts are never silently reused.
- [ ] Reruns deterministically reuse verified evidence or explicitly rebuild owned artifacts without deleting unrelated/shared evidence.

## Completion & Validation

- [ ] Completion derives from verified expected evidence, not process exit codes or directory contents.
- [ ] Verification is read-only and never trains, scores, or creates evidence.
- [ ] Not-started, execution-incomplete, evidence-incomplete, stale, failed, and passed states remain distinct.
- [ ] Mandatory evidence, schema validity, provenance validity, and scientific gates are required before PASS; `status`, `report`, and publication share that definition.

## Scientific & Statistical Rules

- [ ] Every scientific gate explicitly defines formula, threshold, equality semantics, tolerance, and insufficient-evidence behavior.
- [ ] Failure is distinct from unavailable, infeasible, insufficient-evidence, and suppressed; missing metrics never become zero.
- [ ] Estimands, experimental units, pairing keys, tests, effect sizes, confidence intervals, multiplicity correction, alpha, and fallback behavior are predefined where applicable.
- [ ] Paired provenance is proven before paired testing; pseudo-replication is avoided.
- [ ] Ties, degeneracy, small samples, NaN, infinity, and undefined cases are explicit; confirmatory and exploratory analyses remain separated.

## Randomness & Reproducibility

- [ ] Every stochastic process has an explicit RNG/seed owner with deterministic seed derivation and independent seed domains where required.
- [ ] Hidden global RNG state is avoided; seed/reproducibility configuration is persisted.
- [ ] Deterministic CUDA behavior and numerical reproducibility tolerances are defined where scientifically required.

## Runtime & Observability

- [ ] Required hardware/runtime dependencies fail fast; mandatory GPU execution never silently falls back to CPU.
- [ ] Logs expose active experiment, dataset, seed, stage, coordinate, reuse decision, status, and elapsed time where applicable.
- [ ] CPU/RAM/GPU/VRAM/stage duration are monitored for expensive workloads where useful; long stages report progress without excessive noise; useful telemetry is persisted.

## Testing

- [ ] Tests cover architecture, configuration, datasets, scientific logic, artifacts, provenance, execution, validation, integration, and smoke workflows where applicable.
- [ ] Scientific formulas, boundaries, failure paths, serialization, corruption, staleness, provenance invalidation, resume/reuse/rebuild, and mathematical invariants are tested where applicable.
- [ ] Property-based tests are used where valuable; tests run in parallel where safe; smoke tests remain small but representative; obsolete tests are removed.

## Static Quality

- [ ] Ruff formatting and linting pass repository-wide.
- [ ] Strict Pyright passes repository-wide including tests, with only narrow justified third-party-stub exceptions.
- [ ] Import Linter passes; custom AST architecture checks are limited to project-specific rules standard tools cannot express cleanly.
- [ ] Blanket lint/type suppressions are absent and quality rules have not been weakened merely to make CI pass.

## CI

- [ ] Relevant pushes/PRs run mandatory gates from the locked dependency environment and verify the package builds.
- [ ] CI runs formatting, linting, strict typing, dependency architecture, architecture tests, parallel tests, and coverage gates; the built package is tested where practical.
- [ ] CI and local mandatory validation commands remain aligned.
- [ ] Rules that can be mechanically enforced are not left only to prompts or developer discipline.

## Code Hygiene

- [ ] Dead code, stale modules, obsolete redirects, duplicate implementations, superseded architecture, and unnecessary compatibility shims are absent.
- [ ] Constants, serialization logic, path construction, and domain models are not duplicated unnecessarily.
- [ ] Temporary code, commented-out implementations, unresolved required TODOs, unexplained FIXMEs, hidden fallbacks, and silent exception swallowing are absent.
- [ ] Comments explain scientific/architectural reasoning rather than obvious code; documentation and implementation terminology remain synchronized.

## Reporting & Publication

- [ ] Tables, figures, reports, and publication evidence are generated from persisted verified artifacts rather than manually copied scientific values.
- [ ] Missing, stale, malformed, or unverified evidence cannot reach reporting/publication.
- [ ] Every important claim maps to supporting experiments, metrics, analyses, and validity conditions; failed gates block claims; exploratory findings are never presented as confirmatory.

## Global Readiness Gate

- [ ] Architecture boundaries are enforced; strict typing passes; one authoritative validated configuration exists.
- [ ] No unexplained scientific constants, agent-invented values, dataset-contract violations, leakage, stale evidence, duplicate logic, stale terminology, or unnecessary compatibility code remains.
- [ ] Experiment coordinates/evidence are predefined; artifact/provenance infrastructure is centralized; execution is deterministic, resumable, and safely idempotent.
- [ ] Scientific/statistical rules and all required tests pass; Ruff, Pyright, Import Linter, architecture checks, tests, coverage, and CI requirements pass.
- [ ] Reports use only verified evidence; every important result is traceable; no important behavior depends on guessing what was intended.

---

# 10. Hostile Review

Actively attempt to prove the milestone or repository incomplete or incorrect. Successful tests do not eliminate this step.

## Coverage Attack

- [ ] Re-read assigned roadmap sections and search for omitted obligations.
- [ ] Compare roadmap, coverage inventory, milestone, and issues for mismatches, missing ownership, unauthorized work, or activated conditional requirements not implemented.

## Scientific Attack

- [ ] Search for altered algorithms, formulas, constants, thresholds, tolerances, equality semantics, statistical units, fallbacks, missing-value coercion, pseudo-replication, invalid pairing, or exploratory evidence presented as confirmatory.

## Configuration Attack

- [ ] Search for hardcoded/duplicated scientific values, unauthorized defaults, CLI overrides, test values defining production behavior, and invalid combinations accepted silently.

## Architecture & Type Attack

- [ ] Search for dependency violations, scientific logic in CLI/reporting, schema/primitive leaks, `Any`, anonymous domain dictionaries, free-form typed identities, duplicate domain models, and generic dumping-ground modules.

## Evidence & Provenance Attack

- [ ] Search for file-exists completion, stale/malformed/partial evidence reuse, untraceable outputs, incomplete digests/provenance, failed invalidation, or verification code that mutates/trains/scores/repairs evidence.

## Reproducibility Attack

- [ ] Search for hidden RNG, unowned stochastic operations, seed collisions, nondeterministic ordering/identity, and silent hardware-mode differences where execution mode matters scientifically.

## Validation Attack

- [ ] Search for self-confirming tests, missing boundary/failure/corruption/staleness/resume tests, smoke paths bypassing real execution, and disabled/skipped/weakened quality gates.

## Execution-Path Attack

- [ ] Search for stale/alternative paths, unnecessary compatibility shims, silent fallbacks, swallowed exceptions, forbidden CPU fallback, and runtime-discovered scientific design.

## Reporting Attack

- [ ] Search for manually copied scientific values, validation bypasses, claims emitted after failed gates, and invalid evidence reaching reports/publication.

## Repository Hygiene Sweep

Search repository-wide for relevant instances of:

- [ ] `TODO`, `FIXME`, placeholders, `pass`, `NotImplemented`, or `NotImplementedError` in completed required paths.
- [ ] `Any`, lint/type suppressions, broad/silent exception handling, hardcoded scientific constants, deprecated terminology, obsolete paths, generic module names, artificial version suffixes, duplicate implementations, and hidden fallbacks.

Every relevant match is justified, fixed, or recorded as a finding.

---

# 11. Findings & Reconciliation

This section persists across reruns. Never silently delete previous findings.

## Previous Findings

- [ ] Every previous open finding was re-evaluated.
- [ ] Every previous resolved finding was checked for regression.
- [ ] No finding disappeared without explicit resolution evidence.
- [ ] Regressions were reopened or recorded as new findings.

## Open Findings

If none:

```text
None.
```

Otherwise:

### FINDING-<ID> — <Short title>

- Scope: `MILESTONE` / `GLOBAL`
- Class: `BLOCKER` / `NON-BLOCKING`
- Violated requirement / rule:
- Evidence:
- Impact:
- Corrective issue / action:
- Status: `OPEN`

## Resolved Findings

### FINDING-<ID> — <Short title>

- Scope: `MILESTONE` / `GLOBAL`
- Previous class:
- Resolution:
- Resolution evidence:
- Regression check:
- Status: `RESOLVED`

---

# 12. Audit Evidence

A checked box without supporting evidence is insufficient when evidence can reasonably be recorded.

## Sources Inspected

- Roadmap sections:
- Coverage Inventory entries:
- Milestone definition:
- Issues:
- Engineering & Research Rules:
- Configurations:
- Dataset contracts / raw-data validations:
- Artifacts / manifests:
- Reports / publication outputs:

## Commands / Quality Gates

| Check | Command / Method | Result | Evidence / Notes |
| --- | --- | --- | --- |
| Formatting |  |  |  |
| Linting |  |  |  |
| Strict typing |  |  |  |
| Import architecture |  |  |  |
| Architecture tests |  |  |  |
| Unit tests |  |  |  |
| Integration tests |  |  |  |
| Smoke tests |  |  |  |
| Coverage |  |  |  |
| Package build |  |  |  |
| Other mandatory gates |  |  |  |

## Test Summary

- Tests collected:
- Tests passed:
- Tests failed:
- Tests skipped:
- Expected skips and justification:
- Coverage:

## Scientific Validation Evidence

- Synthetic/reference validations:
- Invariants:
- Boundary/degenerate cases:
- Reproducibility checks:
- Statistical checks:

## Artifact / Provenance Evidence

- Expected artifacts:
- Valid artifacts:
- Missing artifacts:
- Stale artifacts:
- Malformed/incomplete artifacts:
- Provenance checks:
- Resume/reuse checks:

## Repository State

Repository state is operational evidence, not a substitute for correctness.

- Working tree state:
- Relevant uncommitted changes:
- Temporary/generated files requiring cleanup:
- Notes:

---

# 13. Audit Summary

## Coverage

- Mandatory requirements expected:
- Mandatory requirements verified:
- Mandatory requirements missing:
- Conditional requirements activated / verified:
- Milestone issues expected:
- Milestone issues verified:
- Milestone issues incomplete:

## Validation

- Required test groups:
- Scientific gates:
- Required artifacts:
- Integration / regression:
- Global engineering & research rules:
- Mandatory quality gates:

## Findings

- Open blockers:
- Open non-blocking findings:
- Resolved findings revalidated:
- Regressions detected:

---

# 14. Final Verdict

Result:

- [ ] `PASS`
- [ ] `PASS WITH NON-BLOCKING FINDINGS`
- [ ] `FAIL`

Verdict basis:

- Roadmap coverage:
- Scientific fidelity:
- Implementation completeness:
- Testing / scientific validation:
- Artifacts / provenance / completion:
- Integration / regression:
- Repository-wide engineering & research rules:
- Hostile review:
- Blocking findings:

Final statement:

> <One concise statement explaining why the milestone passes or fails.>
