# Issue Template

> Every issue is an executable unit of roadmap work.
>
> The roadmap is the scientific and implementation authority. This issue may clarify execution and verification, but it must not silently expand, reinterpret, redesign, or weaken roadmap requirements.
>
> The mandatory engineering and research checklist in this template applies to **every issue**. It is a compliance gate, not additional scientific scope.

## Issue Summary

- **Milestone:** ...
- **Primary deliverable:** ...
- **Issue type:** Implementation / Experiment / Infrastructure / Validation / Reporting / Other

## Roadmap Authority

### Roadmap Sections

- §...
- §...

### Requirements

- REQ-...
- REQ-...

Every scope item, acceptance criterion, required output, and claim-bearing behavior in this issue must trace to the referenced roadmap authority or to prerequisite implementation strictly necessary to satisfy it.

## Objective

State the concrete outcome that must exist when this issue is complete.

The objective must describe the required result, not a vague activity or an alternative design.

## Scope

Implement only the work required to satisfy the referenced roadmap requirements:

- ...
- ...
- ...

No requirement may exist only implicitly in acceptance criteria, tests, or implementation notes.

## Out of Scope

- ...
- ...

Anything not required by the referenced roadmap authority or strictly necessary prerequisite implementation is out of scope.

Do not introduce speculative features, alternative scientific behavior, unrelated refactors, new experiments, additional abstractions, or convenience functionality unless required to complete the issue correctly.

## Dependencies

### Blocked By

- #... — reason this dependency must complete first
- #... — ...

### Blocks

- #... — what this issue provides to the downstream issue
- #... — ...

Dependencies must describe the actual contract or evidence exchanged between issues, not only issue numbers.

## Non-Negotiable Implementation Contracts

The implementation must preserve exactly, where defined by the roadmap:

- roadmap-defined algorithmic semantics;
- formulas, thresholds, equality semantics, tolerances, and numerical conventions;
- configuration authority and ownership;
- dataset, split, eligibility, chronology, and preprocessing semantics;
- experiment coordinates, controls, comparators, seeds, and statistical procedures;
- deterministic and reproducibility requirements;
- failure, unavailable, infeasible, insufficient-evidence, suppression, and abstention semantics;
- artifact, integrity, completion, and provenance requirements;
- scientific claim boundaries.

Do not introduce:

- undocumented defaults;
- hidden fallbacks;
- alternative algorithms or scientific behavior;
- post-hoc scientific choices;
- duplicated scientific constants;
- silent compatibility behavior;
- agent-invented values or assumptions.

If the roadmap is genuinely missing or contradictory on a decision that materially affects scientific or implementation behavior, do not guess. Record it under **Roadmap Deviations / Follow-Ups** and create a dedicated clarification or correction issue.

## Implementation Surface

Expected areas affected by this issue:

### Production Code

- ...

### Configuration / Domain Contracts

- ...

### CLI / Application Surface

- ...

### Tests

- ...

### Artifacts / Reporting

- ...

Remove non-applicable subsections when instantiating the issue.

Exact file placement follows repository architecture unless the roadmap explicitly mandates a path.

## Acceptance Criteria

Acceptance criteria must be observable, testable, and trace directly to the issue scope. They may verify scope but must not silently introduce new requirements.

- [ ] REQ-... — ...
- [ ] REQ-... — ...
- [ ] All scope items are implemented completely.
- [ ] Roadmap semantics are preserved exactly.
- [ ] Configuration values are consumed from their authoritative source.
- [ ] Invalid, unavailable, infeasible, unsupported, and insufficient-evidence states follow the declared behavior.
- [ ] Determinism and reproducibility requirements are satisfied where required.
- [ ] Integration with required upstream and downstream contracts works.
- [ ] Required artifacts and evidence are produced with the correct identity, schema, integrity, and provenance.
- [ ] No unrelated scientific or product behavior has been introduced.

## Required Tests

Specify the concrete tests required for this issue. A category that genuinely does not apply must be recorded as `N/A — <reason>` in the completion record rather than silently ignored.

- [ ] Unit tests
- [ ] Boundary and edge-case tests
- [ ] Invalid-input and failure-path tests
- [ ] Numerical / formula tests where scientific calculations are involved
- [ ] Determinism / reproducibility tests where stochastic or deterministic behavior is specified
- [ ] Serialization / integrity / corruption tests where persisted artifacts are involved
- [ ] Provenance / stale-evidence tests where evidence dependencies are involved
- [ ] Resume / reuse / rebuild tests where execution state is involved
- [ ] Integration tests
- [ ] CLI / end-to-end tests where an execution surface is involved
- [ ] Smoke tests where the real execution path must be exercised cheaply
- [ ] Regression tests for every bug discovered while implementing this issue

Each test must prove an acceptance criterion, invariant, failure path, or previously observed defect. Avoid tests that merely duplicate implementation details.

## Required Outputs

List every artifact, persisted evidence item, generated report, manifest, checkpoint, table, figure, CLI result, or other deliverable required from this issue:

- ...
- ...

For each required output, define its expected identity or location when the roadmap or repository contract specifies one.

## Quality Gates

All repository-defined mandatory quality gates must pass before this issue can close.

At minimum, run every gate applicable to the repository, including formatting, linting, strict typing, dependency architecture, architecture tests, automated tests, coverage, package/build validation, and any project-specific scientific or artifact verification gates.

**No required check may be skipped, weakened, suppressed, or dismissed because the failure is believed to be pre-existing, unrelated, flaky, or outside the immediate files changed by this issue. Fix the failure before closure.**

## Mandatory Engineering & Research Checklist

> This checklist is mandatory in **every issue**.
>
> Checking an item means the implementation introduced by this issue complies with the rule and does not leave the repository violating it in the affected execution path. The checklist does not require unrelated features to be added when they are genuinely inapplicable.

### Architecture

- [ ] Package/module responsibilities remain clear and layered.
- [ ] Dependency direction remains one-way; lower layers do not depend on higher layers.
- [ ] `core` remains independent from execution, reporting, application, and CLI layers.
- [ ] Scientific logic is not implemented in the CLI.
- [ ] Reporting or presentation concerns are not implemented inside scientific modules.
- [ ] Dependency rules are automatically enforced with Import Linter and targeted architecture tests where applicable.
- [ ] Modules remain small and focused rather than accumulating unrelated responsibilities.
- [ ] No empty architectural layer is introduced merely to satisfy a template.

### Domain Types

- [ ] Finite identities, policies, statuses, dataset IDs, experiment IDs, and algorithm choices use enums where appropriate.
- [ ] Meaningful immutable domain values and aggregates use frozen dataclasses where appropriate.
- [ ] Validation-only scalar constraints use constrained Pydantic / `Annotated` types where appropriate.
- [ ] Nominal domain types are used where mixing values could create a realistic scientific or implementation error.
- [ ] Primitives are not wrapped unnecessarily for style alone.
- [ ] `Any`, anonymous dictionaries, and untyped domain payloads are avoided.
- [ ] CLI, YAML, JSON, and other boundary primitives are converted immediately into validated domain types.
- [ ] Configuration and persisted structured artifacts use frozen Pydantic models where appropriate.

### Configuration

- [ ] One authoritative production configuration hierarchy remains in force.
- [ ] Scientific values are not duplicated across configuration, CLI defaults, constants, tests, or implementation.
- [ ] Configuration is immutable and rejects unknown fields.
- [ ] Cross-field constraints are validated before execution begins.
- [ ] Scientific constants are explicit and centrally owned.
- [ ] No unspecified scientific value has been invented by the implementation agent.
- [ ] Deterministic configuration/protocol digests are generated where required for provenance or stale-evidence detection.

### Naming

- [ ] Packages, modules, classes, functions, experiments, and artifacts use descriptive domain names.
- [ ] No opaque experiment aliases or unexplained abbreviations are introduced.
- [ ] Generic names such as `utils`, `helpers`, `common`, `manager`, `processor`, or `base` are avoided unless genuinely appropriate.
- [ ] No artificial version names such as `v2`, `final2`, or equivalent naming drift are introduced.
- [ ] Obsolete terminology affected by the change is removed rather than preserved alongside the new terminology.

### CLI

- [ ] CLI code remains limited to parsing, validation, application invocation, result rendering, and error translation.
- [ ] Scientific algorithms, statistics, dataset construction, and artifact-layout logic are kept out of CLI commands.
- [ ] Command names remain small, descriptive, and domain-meaningful.
- [ ] Experiment and policy identities use typed values rather than free-form strings.
- [ ] Runtime/dependency/GPU readiness is exposed through `doctor` or equivalent repository-defined validation where useful.

### Datasets

- [ ] Dataset identity, source, schema, layout, labels, features, clients, chronology, preprocessing, splitting, and limitations are explicit wherever this issue touches them.
- [ ] Raw data is never modified.
- [ ] Expected files, columns, schemas, and population assumptions are validated before experiments run.
- [ ] Train, calibration, validation, and test roles remain explicit and non-overlapping.
- [ ] Scientifically relevant leakage checks exist and pass.
- [ ] Preprocessing and dataset provenance are persisted where required.
- [ ] Dataset or preprocessing changes invalidate dependent artifacts when scientifically relevant.
- [ ] Unavailable labels, metrics, clients, timestamps, or evidence are never fabricated.

### Experiments

- [ ] Every affected experiment has an explicit purpose and scientific role.
- [ ] Dataset, experimental unit, seeds, variables, controls, comparators, metrics, statistical analysis, artifacts, and completion criteria are defined before execution.
- [ ] Expected experiment coordinates are enumerable before execution begins.
- [ ] Execution code does not dynamically discover or invent the scientific design.
- [ ] Valid upstream artifacts are reused only where scientifically permitted.
- [ ] Confirmatory, robustness, mechanism, external-validation, and exploratory experiments remain explicitly distinguished.

### Artifacts

- [ ] Artifact path construction is centralized.
- [ ] Serialization, hashing, manifests, and integrity checks are centralized.
- [ ] Serialization is deterministic when artifact identity depends on content.
- [ ] Writes are atomic so partial files cannot appear valid.
- [ ] Persisted artifacts are validated when read.
- [ ] Structured metadata, large tabular data, and tensor/model data use the repository-approved formats, such as JSON, Parquet, and SafeTensors, where appropriate.
- [ ] Persisted structured artifacts record schema versions.
- [ ] Every artifact type has a clear owner.

### Provenance

- [ ] Important results remain traceable to protocol, configuration, data, model, seed, code, and upstream evidence.
- [ ] Content digests are recorded instead of relying only on filenames.
- [ ] Relevant experiment identity, dataset identity, seed, config/protocol digest, preprocessing digest, checkpoint/model digest, upstream digests, schema version, and code revision are persisted.
- [ ] Downstream evidence is automatically invalidated when scientifically relevant provenance changes.

### Idempotence and Resume

- [ ] `file exists` is never treated as equivalent to `experiment complete`.
- [ ] Evidence is reused only when valid, current, and provenance-compatible.
- [ ] Missing, valid, stale, malformed, incomplete, failed, and blocked states are distinguishable where applicable.
- [ ] Rerunning a completed operation either reuses verified evidence or explicitly rebuilds artifacts owned by that operation.
- [ ] Stale or malformed artifacts are never silently reused.
- [ ] Overwrite/rebuild behavior never deletes unrelated or shared evidence.
- [ ] Resume and rebuild behavior is deterministic and testable.

### Completion and Validation

- [ ] Completion is derived from verified evidence rather than process exit codes alone.
- [ ] Expected evidence is derived from protocol/configuration, not from scanning whatever files happen to exist.
- [ ] Completion verification is read-only.
- [ ] Verification never trains models, scores data, or creates missing artifacts.
- [ ] Not-started, execution-incomplete, evidence-incomplete, stale, failed, and passed states remain distinguishable where applicable.
- [ ] Mandatory evidence, schema validity, provenance validity, and scientific gates are required before declaring an experiment passed.
- [ ] `status`, `report`, and publication paths rely on the same completion definition.

### Scientific and Statistical Rules

- [ ] Every affected scientific gate defines its formula, threshold, equality semantics, tolerance, and insufficient-evidence behavior explicitly.
- [ ] Failure is distinct from unavailable, infeasible, insufficient-evidence, and suppressed outcomes.
- [ ] Missing or unavailable metrics are never converted to zero.
- [ ] Estimands, experimental units, pairing keys, statistical tests, effect sizes, confidence intervals, multiplicity correction, alpha, and fallback behavior are predefined where relevant.
- [ ] Paired-analysis provenance is proven before paired statistical tests are used.
- [ ] Pseudo-replication is avoided.
- [ ] Ties, degenerate samples, small samples, NaN, infinity, and undefined numerical cases are handled explicitly.
- [ ] Confirmatory and exploratory analyses remain clearly separated.

### Randomness and Reproducibility

- [ ] Every stochastic process has an explicit RNG/seed owner.
- [ ] Seed derivation is deterministic.
- [ ] Independent stochastic processes use distinct seed domains where required.
- [ ] Hidden global RNG state is avoided.
- [ ] Seed and reproducibility configuration is persisted.
- [ ] Deterministic CUDA behavior is configured where scientifically required.
- [ ] Numerical reproducibility tolerances are defined where exact reproducibility is not expected.

### Runtime and Observability

- [ ] Required hardware and runtime dependencies fail fast when unavailable.
- [ ] Mandatory GPU execution never silently falls back to CPU.
- [ ] Logs expose active experiment, dataset, seed, stage, coordinate, reuse decision, status, and elapsed time where applicable.
- [ ] CPU, RAM, GPU, VRAM, and stage duration are monitored for expensive workloads where useful.
- [ ] Long-running stages expose useful progress without excessive log noise.
- [ ] Useful runtime telemetry is persisted where required for debugging or reproducibility.

### Testing

- [ ] Tests cover affected architecture, configuration, datasets, scientific logic, artifacts, provenance, execution, validation, integration, and smoke workflows as applicable.
- [ ] Scientific formulas are tested numerically.
- [ ] Boundary cases and failure paths are tested.
- [ ] Serialization round trips and artifact corruption handling are tested where applicable.
- [ ] Stale-evidence detection and provenance invalidation are tested where applicable.
- [ ] Resume, reuse, and rebuild behavior is tested where applicable.
- [ ] Property-based tests cover valuable mathematical invariants where appropriate.
- [ ] Tests run in parallel where safe.
- [ ] Smoke tests remain small but representative of the real execution path.
- [ ] Obsolete tests are removed after redesigns rather than preserved solely for unnecessary compatibility.

### Static Quality

- [ ] Ruff formatting passes.
- [ ] Ruff linting passes.
- [ ] Repository-wide strict Pyright passes across source and tests, with only narrow justified third-party-stub exceptions.
- [ ] Import Linter dependency architecture checks pass.
- [ ] Custom AST architecture tests are used only for project-specific rules standard tools cannot express cleanly.
- [ ] No blanket lint or type-check suppressions are introduced.
- [ ] Quality rules are not weakened merely to make CI pass.

### CI

- [ ] Mandatory quality gates run on every relevant push and pull request.
- [ ] CI installs from the locked dependency environment.
- [ ] CI verifies that the package builds successfully.
- [ ] CI runs formatting, linting, strict typing, dependency architecture, architecture tests, parallel tests, and coverage gates as defined by the repository.
- [ ] The built package is tested where practical, not only the source checkout.
- [ ] CI behavior remains aligned with local validation commands.
- [ ] Enforceable rules are automated in CI rather than relying only on prompts or developer discipline.

### Code Hygiene

- [ ] Dead code, stale modules, obsolete redirects, duplicate implementations, and superseded architecture affected by the change are removed.
- [ ] Compatibility shims are not introduced unless explicitly required.
- [ ] Constants, serialization logic, path construction, and domain models are not duplicated.
- [ ] Completed work contains no temporary code, commented-out implementations, TODOs, or unexplained FIXMEs.
- [ ] Hidden fallbacks and silent exception swallowing are absent.
- [ ] Comments, when necessary, explain scientific or architectural reasoning rather than narrating obvious code.
- [ ] Documentation and implementation terminology remain synchronized.

### Reporting and Publication

- [ ] Tables, figures, reports, and publication evidence are generated from persisted verified artifacts.
- [ ] Scientific result values are never manually copied into reporting code.
- [ ] Reporting/publication generation rejects missing, stale, malformed, or unverified evidence.
- [ ] Every important scientific claim maps to the experiments, metrics, analyses, and validity conditions that support it.
- [ ] Claims are blocked when required evidence or scientific gates fail.
- [ ] Exploratory findings are never presented as confirmatory evidence.

## Roadmap Deviations / Follow-Ups

### Roadmap Deviations

`None.`

If a genuine deviation is unavoidable, replace `None` with one entry per deviation containing:

- roadmap section / requirement;
- exact deviation;
- reason;
- scientific or implementation impact;
- linked clarification/correction issue.

No deviation may be hidden in implementation notes, comments, commits, or completion evidence.

### Follow-Up Issues

- None.

Record only genuinely separate work discovered during implementation. Do not move unfinished scope from this issue into a follow-up merely to close the issue.

## Completion Record

Complete this section before closing the issue.

### Implementation

- **Implementation locations:** ...
- **Configuration/domain changes:** ...
- **CLI/application changes:** ...

### Tests

- **Tests added/updated:** ...
- **Required-test categories marked N/A with reasons:** ...

### Validation

- **Commands executed:** ...
- **Quality-gate results:** ...
- **Test results:** ...

### Outputs / Evidence

- **Artifacts generated:** ...
- **Artifact/provenance verification:** ...
- **Scientific gates / completion verification:** ...

### Traceability

- **Requirements satisfied:** ...
- **Relevant commit SHA(s):** ...
- **Linked clarification/correction/follow-up issues:** ...

## Definition of Done

This issue is complete only when **all** of the following are true:

- [ ] Every in-scope roadmap requirement is implemented completely.
- [ ] Every acceptance criterion passes.
- [ ] Every required test is implemented and passes, or a genuinely inapplicable category is explicitly justified.
- [ ] Every required output exists and passes schema, integrity, provenance, and validity checks.
- [ ] Every mandatory repository quality gate passes with no ignored pre-existing or unrelated failures.
- [ ] The Mandatory Engineering & Research Checklist is fully satisfied.
- [ ] Roadmap semantics and scientific claim boundaries remain unchanged unless an explicit linked correction authorizes otherwise.
- [ ] No unresolved ambiguity requires the implementer or downstream issue to guess a scientific or implementation decision.
- [ ] Roadmap deviations are `None` or explicitly documented and linked to an authorized clarification/correction issue.
- [ ] The completion record is filled with reproducible evidence.
- [ ] The issue leaves the repository in a valid, tested, clean, and internally consistent state.
