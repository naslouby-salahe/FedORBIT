# CLAUDE.md

Behavioral and engineering guidelines for implementing this project correctly, minimizing common LLM coding mistakes, and preserving scientific, architectural, and repository integrity.

**Tradeoff:** These guidelines bias toward correctness, restraint, and maintainability over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Do not assume. Do not hide uncertainty. Surface meaningful tradeoffs.**

Before implementing:

* State assumptions explicitly when they materially affect implementation.
* Never invent missing scientific values, thresholds, metrics, experiments, dataset properties, protocol decisions, or architectural requirements.
* If multiple materially different interpretations exist, surface them rather than choosing silently.
* If a simpler approach fully satisfies the requirement, prefer it.
* Push back on unnecessary complexity when warranted.
* If something required for correctness is genuinely unclear, identify exactly what is unclear before proceeding.

For multi-step work, establish a short implementation plan with verifiable outcomes:

```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

## 2. Simplicity First

**Minimum code that correctly solves the requested problem. Nothing speculative.**

* Do not add features beyond what was requested or specified by the roadmap.
* Do not create abstractions for single-use behavior without a concrete architectural reason.
* Do not add speculative flexibility, configurability, hooks, factories, adapters, extension points, or fallback mechanisms.
* Do not add error handling for impossible states already excluded by validated invariants.
* Prefer straightforward domain-specific code over generic frameworks.
* Avoid duplicate implementations of the same concept.
* If substantially less code can express the same behavior without harming clarity or correctness, use the simpler implementation.
* Ask whether a senior engineer would consider the design unnecessarily complicated. If yes, simplify it.

## 3. Surgical Changes

**Keep requested changes focused, but never leave the repository knowingly broken.**

When editing existing code:

* Do not opportunistically redesign or refactor unrelated working code.
* Do not change unrelated APIs, architecture, naming, formatting, or behavior without a concrete reason.
* Match established project conventions unless the task explicitly changes them.
* Every feature or behavioral change should have a clear reason.
* Remove imports, variables, functions, files, and other artifacts made obsolete by your changes.
* If you encounter a pre-existing test failure, type error, lint violation, formatting failure, architecture violation, dead-code violation, dependency problem, or any other mandatory quality-gate failure, **fix it even when your current change did not cause it**.
* Never dismiss a failure as pre-existing, unrelated, outside your changes, or "not caused by me."
* Never leave a known repository failure behind merely to keep the diff smaller.
* When the full mandatory quality suite exposes additional existing violations, continue fixing them until the relevant mandatory gates pass.
* Keep those fixes minimal and targeted. Fix the violation without using it as justification for unrelated redesign or feature work.

Do not preserve obsolete code through redirects, wrappers, aliases, compatibility shims, transitional APIs, or duplicate implementations unless compatibility is explicitly required.

## 4. Goal-Driven Execution

**Define success criteria and continue until they are verified.**

Transform implementation requests into concrete checks:

* "Add validation" → test valid and invalid cases, then verify both.
* "Fix the bug" → reproduce the failure, implement the correction, and verify regression coverage.
* "Refactor X" → establish passing behavior before and after the refactor.
* "Add an experiment" → verify coordinates, configuration, execution, artifacts, and completion evidence.
* "Add architecture rules" → create executable checks and verify prohibited structures fail while valid structures pass.

Do not treat implementation as complete merely because code was written or a command exited successfully.

Run the relevant repository validation and continue resolving failures until the required quality gates pass. A failure is actionable regardless of whether it existed before the current task.

## 5. Core Engineering Rules

* Follow the project roadmap and scientific protocol exactly.
* Do not invent missing scientific values, thresholds, experiments, metrics, assumptions, dataset properties, or methodological decisions.
* Keep architecture layered and dependency direction one-way.
* Lower layers must never depend on CLI, reporting, or other higher layers.
* Keep `core` and scientific/domain logic independent from execution, reporting, application orchestration, and CLI concerns where required by the repository architecture.
* Keep CLI thin: parse, validate, invoke application logic, render results, and translate errors.
* Do not place scientific algorithms, statistics, dataset construction, experimental design, or artifact-layout logic inside CLI commands.

### Domain Types and Boundaries

* Use enums for finite identities, policies, statuses, dataset IDs, experiment IDs, algorithm choices, and other closed domain sets.
* An enum must be genuinely used wherever it is the authoritative representation of its domain concept.
* Do not define an enum and continue passing equivalent free-form strings.
* Use frozen dataclasses for meaningful immutable domain values and aggregates.
* Use frozen Pydantic models for configuration and persisted structured schemas.
* Use constrained or nominal domain types where semantic separation prevents realistic scientific or implementation mistakes.
* Do not wrap every primitive merely for style.
* Avoid inappropriate primitive leakage across scientific, domain, application, and public boundaries.
* Public/domain/application inputs and outputs representing meaningful domain concepts should use explicit domain types where raw primitives would permit accidental mixing.
* Normal primitives remain appropriate inside low-level numerical and implementation code.
* Avoid `Any`.
* Avoid `object` as a generic typing escape hatch.
* Avoid anonymous dictionaries and untyped payloads for meaningful domain/configuration/artifact data.
* Avoid duplicated domain models.
* Convert CLI, JSON, YAML, environment, and other boundary primitives into validated domain/configuration types immediately.

### Configuration

* Maintain one authoritative validated configuration hierarchy.
* Do not duplicate scientific or governed values across configuration, constants, CLI defaults, tests, implementation, or parallel configuration structures.
* Do not hardcode governed scientific, experimental, statistical, dataset, seed, threshold, algorithm, protocol, or similar values outside their authoritative owner.
* Tests should consume authoritative configuration where appropriate instead of reproducing production values independently, except where an independent expected value is deliberately necessary to verify correctness.
* Reject unknown configuration fields.
* Validate cross-field constraints before execution starts.

### Naming

* Use descriptive domain names for packages, modules, classes, functions, methods, variables, parameters, experiments, policies, and artifacts.
* Avoid vague or generic names such as `utils`, `helpers`, `common`, `manager`, `processor`, or `base` unless genuinely appropriate.
* Avoid artificial names such as `v2`, `final2`, obsolete aliases, opaque experiment labels, and unexplained abbreviations.
* Avoid vague, misleading, strange, or unjustifiably short method, function, variable, and parameter names.
* Use canonical project terminology consistently.
* Remove stale terminology when the owning implementation changes.

### Datasets

* Keep raw datasets immutable.
* Validate schemas, client identities, labels, features, splits, chronology, preprocessing assumptions, population assumptions, and leakage boundaries before experiments run.
* Never fabricate unavailable dataset information, clients, timestamps, labels, metrics, or evidence.
* Persist dataset and preprocessing provenance.
* Detect dataset or preprocessing changes that invalidate downstream evidence.

### Experiments and Statistics

* Every experiment must explicitly define its dataset, experimental unit, seeds, variables, controls, comparators, metrics, statistical analysis, expected artifacts, and completion criteria before execution.
* Execution code must not dynamically invent scientific design.
* Define estimands, pairing keys, statistical tests, effect sizes, confidence intervals, multiplicity handling, alpha, numerical tolerances, fallback rules, and insufficient-evidence behavior explicitly.
* Never convert unavailable, undefined, infeasible, or missing scientific evidence into zero or a successful result.
* Handle ties, degenerate samples, small samples, NaN, infinity, and undefined numerical cases explicitly.
* Prove paired-analysis provenance before applying paired statistical tests.
* Avoid pseudo-replication.

### Artifacts, Provenance, and Completion

* Centralize artifact path construction, serialization, hashing, manifests, integrity checks, and provenance.
* Use deterministic serialization when artifact identity depends on content.
* Use atomic writes.
* Validate persisted artifacts when reading them.
* Never treat file existence as experiment completion.
* Reuse only evidence that is valid, current, complete, and provenance-compatible.
* Distinguish missing, stale, malformed, incomplete, failed, blocked, and valid evidence.
* Completion must be derived from verified evidence, not process exit codes.
* Verification must remain read-only and must not train models, regenerate missing evidence, or mutate scientific artifacts.
* Persist enough provenance to trace every important result to protocol, configuration, dataset, preprocessing, model/checkpoint, seed, code revision, upstream artifacts, and relevant content digests.
* Automatically invalidate downstream evidence when scientifically relevant provenance changes.

### Randomness and Runtime

* Make randomness explicit and deterministic.
* Every stochastic process must have an explicit RNG/seed owner.
* Persist seeds, seed domains, and deterministic seed derivation where relevant.
* Avoid hidden global RNG state.
* Fail fast when mandatory runtime requirements are unavailable.
* Never silently fall back from required GPU execution to CPU.
* Use structured logging and expose experiment, dataset, seed, stage, coordinate, status, reuse decision, elapsed time, and useful runtime/resource telemetry where appropriate.

### Code Hygiene

* Remove dead code, stale modules, obsolete tests, stale terminology, duplicate implementations, superseded architecture, temporary code, and unnecessary compatibility behavior whenever encountered.
* Do not leave known pre-existing code-quality or architecture violations unfixed merely because they were not introduced by the current task.
* Do not leave redirect modules, re-export-only modules, legacy aliases, compatibility wrappers, or production code that exists solely to satisfy tests unless explicitly required.
* Do not leave TODO, FIXME, HACK, XXX, commented-out implementations, temporary markers, or unfinished development residue.
* **NEVER add comments to Python source code.**
* **NEVER add docstrings to Python source code.**
* **Remove existing Python comments and docstrings when encountered while working in the repository.**
* Do not use comments or docstrings as a workaround for unclear naming or architecture. Make the code itself explicit and descriptive.
* Keep scientific and architectural rationale in the roadmap and project documentation, not inside Python source.
* Do not silently swallow exceptions.
* Do not introduce hidden fallbacks.
* Do not weaken validation, scientific gates, tests, typing rules, architecture checks, or quality rules merely to make implementation pass.
* Never bypass quality rules using broad ignores, type suppressions, aliases, wrappers, duplicate implementations, fallback paths, or special-case test code.

### Git and Commit Integrity

* **NEVER add Claude, Anthropic, ChatGPT, OpenAI, Copilot, another AI system, an AI agent, or any AI-related identity as a commit author or co-author.**
* **NEVER add ****`Co-authored-by:`**** trailers identifying Claude or any other AI system.**
* Do not add AI attribution, AI-generated-by notices, assistant signatures, or similar metadata to commit messages unless the user explicitly requests such text.
* Preserve the repository's normal human authorship and commit conventions.
* Do not modify Git author or committer identity to represent an AI system.

### Reporting and Publication

* Generate reports, tables, figures, and publication outputs only from persisted verified evidence.
* Never manually copy scientific result values into reporting code.
* Reports and publication outputs must reject missing, stale, malformed, incomplete, or unverified evidence.
* Claims must remain within the evidence and scientific boundaries defined by the roadmap.

## 6. Static Analysis and Automated Enforcement

Important engineering rules must be enforced mechanically whenever practical.

Do not rely on this file, prompts, or developer discipline for rules that can be checked automatically.

Use the repository's architecture and quality tests together with appropriate specialized tools:

* **Ruff** for formatting, linting, supported naming rules, unused imports/variables, and general Python hygiene.
* **strict Pyright** for static typing across production and test code.
* **Semgrep** for project-specific structural source rules and forbidden patterns that would otherwise require repetitive custom AST traversal.
* **Vulture** for dead and unused production-code analysis.
* **deptry** for dependency hygiene, including unused, missing, and incorrectly declared dependencies.
* **Import Linter** and architecture tests for dependency direction and architectural boundaries.
* **pytest** for behavioral, scientific, integration, architecture, repository-invariant, and quality-gateway enforcement.
* **coverage** for configured test-coverage gates.

Prefer mature existing analysis tools over large custom analyzers when they can express the rule correctly.

Keep repository-specific architecture tests focused on meaningful invariants that standard tooling cannot reliably express.

Automated enforcement should detect, where applicable:

* dead, obsolete, unreachable, and superseded production code;
* unused enums, classes, functions, methods, modules, and constants;
* enums that exist but are bypassed by free-form strings;
* production code referenced only by tests;
* inappropriate `Any`, `object`, anonymous `dict`, or untyped payload usage;
* inappropriate primitive leakage through public/domain/application inputs and outputs;
* hardcoded governed values;
* configuration values duplicated outside their authoritative owner;
* duplicate constants and duplicate domain models;
* redirect modules, compatibility shims, legacy aliases, transitional wrappers, and unnecessary re-export-only modules;
* dependency-direction and architectural-responsibility violations;
* stale, inconsistent, strange, generic, misleading, or unjustifiably short names;
* comments and docstrings;
* TODO, FIXME, HACK, XXX, commented-out implementations, and temporary code residue;
* formatting, linting, typing, and dependency-hygiene failures.

Any violation exposed by these mandatory checks must be fixed, including violations that predate the current task. Do not classify mandatory failures as somebody else's problem or leave them unresolved because they are unrelated to the original requested change.

## 7. Testing Rules

* Test architecture, configuration, scientific formulas, datasets, artifacts, provenance, resume/rebuild behavior, failure paths, integration, and representative smoke workflows.
* Test scientific formulas numerically.
* Test meaningful boundary and degenerate cases.
* Test serialization round trips and corrupted-artifact handling.
* Test stale-evidence detection and provenance invalidation.
* Test resume, reuse, rebuild, and idempotence behavior.
* Use property-based tests for mathematical invariants where valuable.
* Keep tests subject to the same strict typing and architecture expectations as production code.
* Remove obsolete tests after redesigns rather than maintaining unnecessary compatibility.
* Run tests in parallel where safe.
* Keep smoke tests small but representative of the actual execution path.
* When a test or mandatory quality gate fails, fix the failure regardless of whether it was introduced by the current task or already existed.
* Never dismiss a failure as pre-existing, unrelated, external to the current diff, or outside responsibility.
* Continue until the relevant mandatory test and quality suite passes without weakening the checks.

## 8. Definition of Done

A task or implementation is not complete merely because the requested code exists. The relevant checks for its scope must pass.

For project readiness:

* Architecture boundaries and responsibilities are clear and automatically enforced.
* Strict typing passes across source and tests.
* No inappropriate `Any`, `object`, anonymous dictionary, untyped payload, or primitive boundary leakage remains.
* Configuration is authoritative and validated.
* No governed scientific/configuration values are hardcoded or duplicated outside their authoritative owner.
* Dataset, schema, preprocessing, client-identity, split, and leakage contracts pass.
* Experiments are fully specified before execution.
* Artifact paths, serialization, hashing, integrity, and provenance are centralized and validated.
* Artifacts are typed, atomic, hashed, provenance-aware, and validated when consumed.
* Stale, malformed, partial, incomplete, and missing evidence are detectable.
* Execution is deterministic, resumable, safely reusable, and idempotent.
* Scientific and statistical rules are explicit and tested.
* Canonical terminology and descriptive naming are enforced across packages, modules, classes, methods, functions, variables, parameters, experiments, policies, and artifacts.
* No dead, obsolete, superseded, test-only, redirect, compatibility-shim, legacy-wrapper, or unnecessary re-export code remains.
* **No Python comments or docstrings remain.**
* No TODOs, FIXMEs, HACKs, XXX markers, commented-out implementations, or temporary implementation residue remain.
* No known pre-existing mandatory quality, architecture, typing, linting, dependency, or test failure remains unresolved.
* Ruff formatting and linting pass.
* Strict Pyright passes.
* Semgrep rules pass.
* Vulture passes under the repository's configured policy.
* deptry passes.
* Import Linter and architecture tests pass.
* pytest passes, including scientific, architecture, integration, failure-path, and smoke tests as applicable.
* Coverage gates pass.
* The package builds successfully where packaging is part of the repository.
* CI enforces all mandatory gates.
* Reports and publication outputs consume only verified persisted evidence.
* Every important result can be traced to its data, protocol, configuration, preprocessing, model/checkpoint, seed, code revision, and upstream evidence.
* No important scientific or implementation behavior depends on guessing.
* No validation, test, type, architecture, or scientific rule has been weakened merely to achieve a passing state.
* Git commits contain no Claude or other AI co-author attribution unless explicitly requested by the user.

---

**These guidelines are working if:** implementation remains focused without knowingly leaving the repository broken, recurring architecture and typing mistakes are caught automatically, pre-existing mandatory failures are fixed rather than dismissed, Python source remains completely free of comments and docstrings, scientific decisions are never invented by the implementation agent, Git history remains free of unsolicited AI authorship attribution, and failures are discovered and resolved through executable repository checks rather than deferred to later audits.
