# Milestones
## Authority and Allocation Contract
The **Roadmap Coverage Inventory** is the traceability authority for milestone allocation. The authoritative roadmap remains scientifically controlling if a genuine conflict is later identified. This document organizes execution only; it does not redefine algorithms, mathematics, datasets, experiments, thresholds, statistical procedures, claim criteria, exclusions, or reporting semantics.
All `3184` inventory requirements are allocated exactly once across the milestones below: `3070` implementation-bearing requirements and `114` `NON_IMPLEMENTATION` constraints. `NON_IMPLEMENTATION` requirements are retained as governing scope, terminology, theorem-assumption, exclusion, or claim constraints and must not be converted into fictitious implementation issues.
Implementation issue references and milestone-audit issue references are intentionally unassigned because those issues do not yet exist. They remain `—` until the later issue-planning and audit phases.
## Milestone Index
| Milestone | Outcome Boundary | Allocated Requirements | Implementation-Bearing | NON_IMPLEMENTATION | Upstream Milestones |
|---|---|---|---|---|---|
| M01 | Authoritative Scientific Contract and Repository Foundation | 774 | 774 | 0 | None |
| M02 | Deterministic Execution, Artifact, and Provenance Backbone | 592 | 582 | 10 | M01 |
| M03 | Data Preparation and Strict Resource Interface | 411 | 405 | 6 | M01, M02 |
| M04 | Local Models and Procedural Response Packets | 138 | 138 | 0 | M02, M03 |
| M05 | Robust Correspondence, Selection, and Confirmation Engine | 245 | 225 | 20 | M02, M04 |
| M06 | Comparator, Metric, and Statistical Evaluation Framework | 360 | 352 | 8 | M02, M05 |
| M07 | Pre-Confirmatory Validation and Solver Benchmarking | 220 | 219 | 1 | M03, M04, M05, M06 |
| M08 | Confirmatory Transfer, Robustness, and Applicability Experiments | 200 | 200 | 0 | M07 |
| M09 | Statistical Synthesis, Claim Adjudication, and Manuscript Evidence | 244 | 175 | 69 | M06, M07, M08 |

---

# M01 — Authoritative Scientific Contract and Repository Foundation
> **Outcome:** The repository, complete typed scientific configuration, locked environment, structural test boundaries, and scientific-contract readiness checks enforce the roadmap as the sole source of scientific and execution constants before downstream implementation.
## At a Glance
| Field | Value |
|---|---|
| Roadmap scope | Document authority; configuration-YAML contracts across §§4, 9, 12, 14, 15, 17–21; §§4.22, 4.27, 4.29, 19.1, 26 |
| Requirement ownership | REQ-0001–REQ-0003, REQ-0097–REQ-0126, REQ-0132–REQ-0143, REQ-0174–REQ-0193, REQ-0277, REQ-0318–REQ-0328, REQ-0355–REQ-0357, REQ-0376–REQ-0384, REQ-0405–REQ-0412, REQ-0438–REQ-0442, REQ-0446–REQ-0456, REQ-0485–REQ-0497, REQ-0507–REQ-0511, REQ-0548–REQ-0555, REQ-0608, REQ-0653, REQ-0658–REQ-0660, REQ-0685–REQ-0705, REQ-0728–REQ-0750, REQ-0785–REQ-0790, REQ-0799–REQ-0821, REQ-0827–REQ-0840, REQ-1101–REQ-1102, REQ-1159–REQ-1161, REQ-1334–REQ-1363, REQ-1425–REQ-1458, REQ-1909–REQ-1922, REQ-1990, REQ-2019–REQ-2474, REQ-3172–REQ-3173, REQ-3182 |
| Allocated requirement count | `774` total (`774` implementation-bearing; `0` `NON_IMPLEMENTATION`) |
| Upstream milestones | None |
| Implementation issues | `I01`, `I02`, `I03`, `I04`, `I05`, `I06` |
| Coverage authority | Roadmap Coverage Inventory |
| Audit issue | None — no dedicated per-milestone audit issue; verified via `Milestone Audit.md` |
| Audit status | `PENDING` |
## Coverage
The Roadmap Coverage Inventory is the traceability authority for this milestone. The work packages below partition this milestone's allocation; no requirement listed in another milestone is owned here.
| Roadmap Section | Scope / Work Package | Requirement IDs | Implementation Issue(s) | Verification / Evidence |
|---|---|---|---|---|
| Document authority; §§4–4.1 | Roadmap authority, configuration-data ownership, and typed-contract boundary | REQ-0001–REQ-0003, REQ-0097–REQ-0104 | `I01` | Typed-contract/config ownership tests reject any scientific value or definition that is not sourced from the roadmap-owned configuration contract. |
| Configuration YAML across §§4.2–4.21 | Action, split, dataset, training, response, confirmation, solver, baseline, target-importance, randomness, statistics, and claim-criterion constants | REQ-0105–REQ-0126, REQ-0132–REQ-0143, REQ-0174–REQ-0193, REQ-0277, REQ-0318–REQ-0328, REQ-0355–REQ-0357, REQ-0376–REQ-0384, REQ-0405–REQ-0412, REQ-0438–REQ-0442, REQ-0446–REQ-0456, REQ-0485–REQ-0497, REQ-0507–REQ-0511, REQ-0548–REQ-0555, REQ-0608, REQ-0653, REQ-0658–REQ-0660, REQ-0697–REQ-0705, REQ-0728–REQ-0750 | `I01` | Typed configuration snapshot and schema tests prove every registered value is present exactly once and downstream code cannot redefine it. |
| §§4.22, 4.26–4.28 | CUDA/reference-hardware, environment, dependency-lock, and reporting-precision configuration | REQ-0685–REQ-0696, REQ-0785–REQ-0790, REQ-0799–REQ-0821 | `I02` | Runtime/environment checks validate CUDA policy, reference hardware identity, exact Python/dependency lock, and registered reporting precision values. |
| §§9, 12, 14–15 | Multi-source, metric, synthetic-generator, and experiment-registry configuration | REQ-1101–REQ-1102, REQ-1159–REQ-1161, REQ-1334–REQ-1363, REQ-1425–REQ-1458 | `I03` | Config-schema and experiment-catalogue tests prove exact registered constants, seeds, grids, and experiment membership controls. |
| §§17–18; §§19–21 | Kill-rule, retry, artifact-root, and runtime-layout configuration | REQ-1909–REQ-1922, REQ-1990, REQ-2019–REQ-2040 | `I04` | Config validation proves falsification thresholds, retry count, canonical roots, and runtime layout cannot drift from the authoritative values. |
| §4.29 | Nonclaim test and smoke configuration controls | REQ-0827–REQ-0840 | `I06` | Schema/architecture tests reject scientific values in test/smoke configs and enforce only registered nonclaim fixture controls. |
| §19.1 | Repository architecture, package boundaries, canonical paths, and structural enforcement | REQ-2041–REQ-2474 | `I05` | Repository-tree, import-boundary, AST/static, naming, dead-code, primitive-leak, and architecture tests enforce the required layout and responsibility boundaries. |
| §26 | Scientific-contract implementation-readiness gates | REQ-3172–REQ-3173, REQ-3182 | `I06` | Readiness tests prove the typed scientific contract matches the roadmap, all scientific decisions are bound, and no implementer must invent a scientific value or rule. |
### Coverage Rules

- Every requirement listed in this milestone remains traceable to its exact Roadmap Coverage Inventory ID.
- Every mandatory implementation-bearing requirement must map to at least one real implementation issue before implementation of this milestone begins.
- Every conditional requirement remains traceable and must be implemented only when its roadmap-defined condition applies.
- `NON_IMPLEMENTATION` requirements are governing constraints only; they must remain traceable but must not be converted into fictitious implementation work.
- Every mapped implementation-bearing requirement must have objective verification or evidence.
- Every implementation issue must reference the exact requirement IDs it satisfies.
- A requirement is not considered covered merely because it falls inside a roadmap section associated with the milestone; the explicit requirement IDs in this document control milestone allocation.
- The current inventory-wide `UNMAPPED` state is expected before issue creation. No blocking requirement owned by this milestone may remain issue-unmapped when implementation of the milestone begins.
- No issue may redefine, weaken, silently reinterpret, duplicate, or extend the authoritative roadmap requirement it implements.
## Dependencies
### Milestone Dependencies
| Milestone | Required Input / Contract | Entry Gate |
|---|---|---|
| — | No upstream milestone dependency | Begin from the authoritative Roadmap Coverage Inventory and supplied milestone contract. |
### Artifact / Interface Dependencies
| Dependency | Produced By | Required Validation |
|---|---|---|
| Roadmap Coverage Inventory | Authoritative input | All 3,184 requirement IDs present; all requirements `READY`; no `AMBIGUOUS` or `BLOCKED` entries. |
Dependency completion alone is not sufficient. Every consumed dependency must be present, valid, provenance-compatible where applicable, current under its dependency fingerprint, and compatible with the active roadmap contract.
## Implementation Issues
Implementation issues for this milestone are listed below; each issue's detailed task checklist, acceptance criteria, and required tests are defined in `Issues.md`.
| Order | Issue | Work Package | Roadmap Scope | Requirement Coverage | Depends On |
|---:|---|---|---|---|---|
| 1 | `I01` — Establish Authoritative Scientific Configuration Contract | Immutable typed `configs/fedorbit.yaml` contract with single-source scientific authority. | Document authority; §§4–4.1; Configuration YAML and configuration-adjacent label identity across §§4.2–4.4, 4.8–4.21, 4.23, 4.25, and §6.4 | 176 atomic requirements | None (foundational within this milestone chain) |
| 2 | `I02` — Lock Runtime, CUDA, Environment, and Reporting Configuration | Validated runtime/environment/reference-hardware and reporting-precision configuration. | §§4.22, 4.26–4.28 | 41 atomic requirements | `I01` |
| 3 | `I03` — Register Metrics, Generators, Multi-Source, and Experiment Configuration | Typed registered experiment, metric, generator, and multi-source configuration catalogue. | §§9, 12, 14–15 | 69 atomic requirements | `I01` |
| 4 | `I04` — Bind Kill Rules, Retry Policy, Artifact Roots, and Runtime Layout | Typed execution-policy configuration for kill rules, retries, roots, and canonical runtime layout. | §§17–18; §§19–21 | 37 atomic requirements | `I01` |
| 5 | `I05` — Enforce Canonical Repository Architecture and Package Boundaries | Roadmap-mandated repository tree, dependency boundaries, and structural enforcement tests. | §19.1 | 434 atomic requirements | `I01` |
| 6 | `I06` — Enforce Nonclaim Test Boundaries and Scientific-Contract Readiness | Nonclaim test/smoke boundary and executable M01 scientific-contract readiness gates. | §4.29; §26 | 17 atomic requirements | `I01`, `I02`, `I03`, `I04`, `I05` |
### Issue Contract

Every future milestone issue must:

- reference its exact roadmap section(s);
- list every covered implementation-bearing requirement ID;
- contain a detailed implementation checklist;
- define objective acceptance criteria;
- identify required tests;
- identify required artifacts, outputs, or interfaces;
- identify required provenance or manifest updates where applicable;
- identify explicit dependencies on upstream milestones, issues, artifacts, or interfaces;
- preserve roadmap terminology and semantics;
- avoid converting `NON_IMPLEMENTATION` constraints into artificial implementation tasks;
- close only when every mapped requirement and acceptance criterion is satisfied.
## Deliverables
| Deliverable | Source Issue(s) | Required Validation | Downstream Consumer |
|---|---|---|---|
| Complete typed scientific configuration contract | `I01`, `I02`, `I03`, `I04`, `I05`, `I06` | Exact schema/value validation and single-authority tests | M02–M09 |
| Canonical repository/package/file layout | `I01`, `I02`, `I03`, `I04`, `I05`, `I06` | Repository-tree, import-boundary and architecture tests | M02–M09 |
| Locked Python/dependency/CUDA/reference-hardware contract | `I01`, `I02`, `I03`, `I04`, `I05`, `I06` | Environment, lockfile, package-hash and runtime validation | M02–M09 |
| Test/smoke configuration boundary | `I01`, `I02`, `I03`, `I04`, `I05`, `I06` | Schema/static tests prove no scientific-value ownership in nonclaim fixtures | M02–M09 |
| Scientific-contract readiness checks | `I01`, `I02`, `I03`, `I04`, `I05`, `I06` | M01-owned §26 readiness gates pass | M02 |
All roadmap-required deliverables for this milestone must appear in this table or remain explicitly traceable through the Roadmap Coverage Inventory. Source issue references remain `—` until real implementation issues exist.
## Entry Criteria
Implementation of this milestone may begin only when all applicable conditions below are true:
- there is no upstream milestone dependency;
- every required upstream artifact, interface, schema, manifest, or authoritative input exists;
- every consumed dependency passes its applicable validation and is provenance-compatible where provenance applies;
- the Roadmap Coverage Inventory is available as the traceability authority and reports `3184` total requirements, `3070` implementation-bearing requirements, `114` `NON_IMPLEMENTATION` requirements, and all `3184` requirements as `READY`;
- the supplied milestone structure is available and no implementation or audit issue IDs are assumed;
- all roadmap requirements owned by this milestone remain present in the Roadmap Coverage Inventory with unchanged IDs unless the authoritative inventory itself has been updated;
- after issue decomposition, every mandatory implementation-bearing requirement is mapped to at least one real milestone issue with explicit verification evidence;
- no blocking requirement is `AMBIGUOUS` or `BLOCKED`;
- no unresolved roadmap ambiguity would force an implementer to invent a scientific, mathematical, methodological, numerical, architectural, configuration, artifact, or execution decision.
## Exit Criteria
The milestone is complete only when all applicable conditions below are true:
- every implementation-bearing requirement owned by this milestone is satisfied;
- every applicable conditional requirement is satisfied;
- every governing `NON_IMPLEMENTATION` constraint assigned to this milestone is preserved;
- every mapped implementation issue is closed;
- the complete typed configuration snapshot contains every roadmap-owned registered constant represented by this milestone and repository code has no competing scientific-value authority;
- all canonical repository/package/path, environment-lock, smoke/test-boundary and M01-owned scientific-contract readiness checks pass;
- all required unit, integration, numerical, structural, schema, CLI/E2E, scientific, statistical, provenance, and reproducibility validations applicable to this milestone pass;
- all required deliverables are generated and all required artifacts, interfaces, schemas, and manifests pass validation;
- required provenance is complete and valid, and no required evidence is stale or incompatible with its material dependencies;
- the milestone audit is `PASS` with no unresolved blocking finding.
## Acceptance Evidence
| Evidence Area | Required Evidence | Pass Condition |
|---|---|---|
| Requirement coverage | Roadmap Coverage Inventory + milestone allocation | Every allocated requirement is accounted for exactly once; implementation-bearing requirements have completed issue evidence and governing constraints remain preserved. |
| Implementation | Closed real milestone issues linked to exact implementation-bearing requirements | Every implementation-bearing requirement has completed implementation evidence. |
| Scientific configuration | Typed config snapshot + ownership/static tests | Every owned configuration value exactly matches the authoritative contract and no duplicate scientific constant authority exists. |
| Repository architecture | Tree/import/AST/architecture test suite | All required paths/responsibility boundaries exist and prohibited structure/primitive leaks fail tests. |
| Environment | Lockfile, package hashes, Python/CUDA/reference-hardware checks | Registered runtime identities validate and are reproducibly resolvable. |
| Scientific-contract readiness | M01-owned §26 readiness tests | The typed contract matches the roadmap, all scientific decisions are bound, and no implementation agent must invent a scientific value or rule. |
| Deliverables | Required milestone outputs and artifacts | Complete, readable, schema-valid where applicable, and consistent with the roadmap. |
| Provenance | Required manifests, dependency identity, checksums, compatibility and staleness evidence | Complete and sufficient to verify origin, compatibility, reuse/invalidation and freshness where applicable. |
| Audit | Milestone audit issue | Final result is `PASS` with no unresolved blocking findings. |
## Milestone Audit
**Audit issue:** `—`

**Status:** `PENDING`

The milestone audit is the final completion gate. It must independently verify:

- complete roadmap coverage for all requirements owned by the milestone;
- exact requirement-to-issue traceability for implementation-bearing requirements;
- correct treatment of `NON_IMPLEMENTATION` constraints;
- closure of every mandatory implementation issue;
- completion and passing status of all required tests;
- completion and passing status of all required validations;
- existence and validity of all required deliverables;
- completeness and validity of required provenance and manifests;
- absence of stale or incompatible evidence;
- absence of unresolved blocking findings;
- readiness of milestone outputs for all declared downstream consumers.

The audit must end in exactly one result:

- `PASS` — every completion condition is satisfied.
- `FAIL` — one or more blocking conditions remain.

A milestone is not complete until the audit result is `PASS`.
## Scope Boundary
- This milestone implements only the implementation-bearing roadmap requirements explicitly allocated to it and preserves only the governing `NON_IMPLEMENTATION` constraints allocated to it.
- The authoritative roadmap remains the source of scientific, mathematical, methodological, architectural, numerical, configuration, artifact, execution, statistical, reporting, and claim requirements.
- This milestone may organize implementation work but may not redefine, weaken, strengthen, extend, or silently reinterpret the roadmap.
- Detailed implementation checklists belong in future implementation issues; no issue has been invented in this document.
- Detailed verification checklists belong in the future milestone audit issue; no audit issue has been invented in this document.
- Work outside this milestone's explicit requirement allocation must not be added unless the authoritative roadmap or Roadmap Coverage Inventory is explicitly updated first.

---

# M02 — Deterministic Execution, Artifact, and Provenance Backbone
> **Outcome:** FedORBIT can execute idempotently through canonical semantic cells with deterministic RNG, validated artifacts, provenance-aware reuse and invalidation, explicit failure recovery, and reproducible dependency identities.
## At a Glance
| Field | Value |
|---|---|
| Roadmap scope | §§4.21, 15 experiment-artifact reuse, 18, 19.2, 20–22.9, 24–26 |
| Requirement ownership | REQ-0661–REQ-0684, REQ-1750, REQ-1832–REQ-1841, REQ-1991–REQ-2018, REQ-2475–REQ-2884, REQ-3015–REQ-3047, REQ-3093–REQ-3171, REQ-3174, REQ-3176–REQ-3181 |
| Allocated requirement count | `592` total (`582` implementation-bearing; `10` `NON_IMPLEMENTATION`) |
| Upstream milestones | M01 |
| Implementation issues | `I07`, `I08`, `I09`, `I10`, `I11`, `I12`, `I13` |
| Coverage authority | Roadmap Coverage Inventory |
| Audit issue | None — no dedicated per-milestone audit issue; verified via `Milestone Audit.md` |
| Audit status | `PENDING` |
## Coverage
The Roadmap Coverage Inventory is the traceability authority for this milestone. The work packages below partition this milestone's allocation; no requirement listed in another milestone is owned here.
| Roadmap Section | Scope / Work Package | Requirement IDs | Implementation Issue(s) | Verification / Evidence |
|---|---|---|---|---|
| §4.21 | Deterministic RNG behavior, seed namespaces, and replay | REQ-0661–REQ-0684 | `I07` | Seed-contract and replay tests reproduce identical stochastic choices and reject post-lock confirmatory-seed changes. |
| §15 — Experiment dependency and artifact map | Shared canonical artifact reuse across experiments | REQ-1750, REQ-1832–REQ-1841 | `I08` | Artifact-reference tests prove dependency-fingerprint reuse without duplicate prepared splits, checkpoints, predictions, packets, solver results, confirmations, or other immutable payloads. |
| §18 | Failure taxonomy, retries, scientific-failure handling, and recovery behavior | REQ-1991–REQ-2018 | `I09` | Failure-injection tests distinguish infrastructure from scientific algorithmic failure, enforce retry limits, and preserve paired missingness semantics. |
| §19.2 | Public CLI execution semantics, idempotency, status, and overwrite behavior | REQ-2475–REQ-2544 | `I10` | CLI/E2E tests prove exact command semantics, no-op reuse, selective overwrite, and descendant validity behavior. |
| §20 | Semantic-cell identity, fingerprints, provenance, staleness, invalidation, and recovery | REQ-2545–REQ-2727 | `I11` | Fingerprint/provenance tests detect meaningful dependency changes, preserve unaffected ancestors, and recompute from the nearest valid boundary. |
| §21 | Canonical computational workspace and active artifact layout | REQ-2728–REQ-2884 | `I12` | Artifact-layout/schema/checksum tests prove canonical paths, payload identity, manifest completeness, and prohibition of active alternate artifacts. |
| §22.8–§22.9 | Completion and reusable-artifact manifests | REQ-3015–REQ-3047 | `I08` | Typed-schema tests validate completion/reusable manifests, payload checksums, dependency references, and stable artifact identity. |
| §24 | Layered execution ordering and restart semantics | REQ-3093–REQ-3136 | `I13` | Execution-plan/E2E tests enforce the registered layer order while permitting reuse of valid upstream boundaries and isolating downstream failures. |
| §25 | Reproducibility, environment identity, statistical identity, and stale-evidence prevention | REQ-3137–REQ-3171 | `I07` | Reproduction/provenance checks prove scientific identity preservation and reject silent replacement under incompatible environment, statistic, or claim criteria. |
| §26 | Execution and provenance implementation-readiness gates | REQ-3174, REQ-3176–REQ-3181 | `I13` | Automated readiness tests prove deterministic experiment expansion, dependency fingerprints, selective invalidation, atomic completion, semantic execution, nearest-valid-boundary recovery, and stale-descendant protection are implemented. |
### Coverage Rules

- Every requirement listed in this milestone remains traceable to its exact Roadmap Coverage Inventory ID.
- Every mandatory implementation-bearing requirement must map to at least one real implementation issue before implementation of this milestone begins.
- Every conditional requirement remains traceable and must be implemented only when its roadmap-defined condition applies.
- `NON_IMPLEMENTATION` requirements are governing constraints only; they must remain traceable but must not be converted into fictitious implementation work.
- Every mapped implementation-bearing requirement must have objective verification or evidence.
- Every implementation issue must reference the exact requirement IDs it satisfies.
- A requirement is not considered covered merely because it falls inside a roadmap section associated with the milestone; the explicit requirement IDs in this document control milestone allocation.
- The current inventory-wide `UNMAPPED` state is expected before issue creation. No blocking requirement owned by this milestone may remain issue-unmapped when implementation of the milestone begins.
- No issue may redefine, weaken, silently reinterpret, duplicate, or extend the authoritative roadmap requirement it implements.
## Dependencies
### Milestone Dependencies
| Milestone | Required Input / Contract | Entry Gate |
|---|---|---|
| M01 — Authoritative Scientific Contract and Repository Foundation | Typed configuration contract, canonical repository paths, locked environment, and structural enforcement | Complete + audit PASS |
### Artifact / Interface Dependencies
| Dependency | Produced By | Required Validation |
|---|---|---|
| Typed scientific configuration and path contract | M01 | Schema-valid; exact registered values; no duplicate scientific ownership. |
| Repository/package skeleton and architecture tests | M01 | Canonical paths/import boundaries valid under the locked environment. |
Dependency completion alone is not sufficient. Every consumed dependency must be present, valid, provenance-compatible where applicable, current under its dependency fingerprint, and compatible with the active roadmap contract.
## Implementation Issues
Implementation issues for this milestone are listed below; each issue's detailed task checklist, acceptance criteria, and required tests are defined in `Issues.md`.
| Order | Issue | Work Package | Roadmap Scope | Requirement Coverage | Depends On |
|---:|---|---|---|---|---|
| 1 | `I07` — Implement Deterministic RNG and Reproducibility Identity | Deterministic seed namespaces, replay behavior, and reproducibility/environment identity. | §4.21; §25 | 59 atomic requirements | `I01`, `I02`, `I05`, `I06` |
| 2 | `I08` — Implement Canonical Artifact Reuse and Completion Manifests | Reusable-artifact and completion manifests with dependency-aware immutable artifact reuse. | §15 — Experiment dependency and artifact map; §22.8–§22.9 | 44 atomic requirements | `I04`, `I05`, `I06` |
| 3 | `I09` — Implement Failure Taxonomy, Retry, and Scientific Failure Semantics | Typed failure states, retry behavior, and scientific/infrastructure failure handling. | §18 | 28 atomic requirements | `I04`, `I07`, `I08` |
| 4 | `I10` — Implement Public CLI Idempotency, Status, and Overwrite Semantics | Roadmap-defined public CLI execution surface with validated idempotent status/overwrite behavior. | §19.2 | 70 atomic requirements | `I05`, `I08`, `I09` |
| 5 | `I11` — Implement Semantic Cells, Fingerprints, Selective Invalidation, and Recovery | Semantic-cell execution graph with dependency fingerprints, staleness, invalidation, crash safety, and recovery. | §20 | 183 atomic requirements | `I07`, `I08`, `I09` |
| 6 | `I12` — Implement Canonical Workspace, Serialization, and Provenance Lifecycle | Canonical computational workspace and validated artifact/provenance lifecycle. | §21 | 157 atomic requirements | `I05`, `I08`, `I11` |
| 7 | `I13` — Implement Ordered Execution Pipeline and Execution-Readiness Gates | Layered restartable execution pipeline and M02 readiness verification. | §24; §26 | 51 atomic requirements | `I10`, `I11`, `I12` |
### Issue Contract

Every future milestone issue must:

- reference its exact roadmap section(s);
- list every covered implementation-bearing requirement ID;
- contain a detailed implementation checklist;
- define objective acceptance criteria;
- identify required tests;
- identify required artifacts, outputs, or interfaces;
- identify required provenance or manifest updates where applicable;
- identify explicit dependencies on upstream milestones, issues, artifacts, or interfaces;
- preserve roadmap terminology and semantics;
- avoid converting `NON_IMPLEMENTATION` constraints into artificial implementation tasks;
- close only when every mapped requirement and acceptance criterion is satisfied.
## Deliverables
| Deliverable | Source Issue(s) | Required Validation | Downstream Consumer |
|---|---|---|---|
| Deterministic RNG and seed-replay substrate | `I07`, `I08`, `I09`, `I10`, `I11`, `I12`, `I13` | Seed namespace/replay tests | M03–M09 |
| Semantic-cell identity, dependency fingerprint and provenance engine | `I07`, `I08`, `I09`, `I10`, `I11`, `I12`, `I13` | Fingerprint/staleness/invalidation/recovery tests | M03–M09 |
| Canonical `outputs/` artifact workspace and reusable/completion manifest support | `I07`, `I08`, `I09`, `I10`, `I11`, `I12`, `I13` | Layout, schema, checksum and manifest validation | M03–M09 |
| Public CLI idempotency/overwrite/status/recovery semantics | `I07`, `I08`, `I09`, `I10`, `I11`, `I12`, `I13` | CLI/E2E reuse and selective-recompute tests | M03–M09 |
| Failure classification and recovery boundary behavior | `I07`, `I08`, `I09`, `I10`, `I11`, `I12`, `I13` | Failure-injection and retry/missingness tests | M03–M09 |
| Reproducibility and stale-evidence enforcement | `I07`, `I08`, `I09`, `I10`, `I11`, `I12`, `I13` | Replay/environment/statistical-identity compatibility checks | M03–M09 |
| Execution/provenance readiness gates | `I07`, `I08`, `I09`, `I10`, `I11`, `I12`, `I13` | M02-owned §26 readiness tests pass | M03–M09 |
All roadmap-required deliverables for this milestone must appear in this table or remain explicitly traceable through the Roadmap Coverage Inventory. Source issue references remain `—` until real implementation issues exist.
## Entry Criteria
Implementation of this milestone may begin only when all applicable conditions below are true:
- all required upstream milestones are complete;
- every required upstream milestone audit is `PASS`;
- every required upstream artifact, interface, schema, manifest, or authoritative input exists;
- every consumed dependency passes its applicable validation and is provenance-compatible where provenance applies;
- M01 has established the complete typed scientific configuration, canonical repository layout, environment lock, and structural test boundaries;
- artifact/provenance work consumes the M01 canonical path and configuration identities rather than defining parallel roots or constants;
- all roadmap requirements owned by this milestone remain present in the Roadmap Coverage Inventory with unchanged IDs unless the authoritative inventory itself has been updated;
- after issue decomposition, every mandatory implementation-bearing requirement is mapped to at least one real milestone issue with explicit verification evidence;
- no blocking requirement is `AMBIGUOUS` or `BLOCKED`;
- no unresolved roadmap ambiguity would force an implementer to invent a scientific, mathematical, methodological, numerical, architectural, configuration, artifact, or execution decision.
## Exit Criteria
The milestone is complete only when all applicable conditions below are true:
- every implementation-bearing requirement owned by this milestone is satisfied;
- every applicable conditional requirement is satisfied;
- every governing `NON_IMPLEMENTATION` constraint assigned to this milestone is preserved;
- every mapped implementation issue is closed;
- unchanged semantic cells deterministically reuse canonical valid artifacts, while meaningful dependency changes invalidate only the correct descendants;
- failure/retry/recovery, overwrite and restart behavior matches the registered semantics and all completion/reusable manifests validate;
- deterministic experiment expansion, dependency fingerprints, selective invalidation, atomic completion, semantic execution, nearest-valid-boundary recovery, and stale-descendant protection satisfy the M02-owned §26 readiness gates;
- all required unit, integration, numerical, structural, schema, CLI/E2E, scientific, statistical, provenance, and reproducibility validations applicable to this milestone pass;
- all required deliverables are generated and all required artifacts, interfaces, schemas, and manifests pass validation;
- required provenance is complete and valid, and no required evidence is stale or incompatible with its material dependencies;
- the milestone audit is `PASS` with no unresolved blocking finding.
## Acceptance Evidence
| Evidence Area | Required Evidence | Pass Condition |
|---|---|---|
| Requirement coverage | Roadmap Coverage Inventory + milestone allocation | Every allocated requirement is accounted for exactly once; implementation-bearing requirements have completed issue evidence and governing constraints remain preserved. |
| Implementation | Closed real milestone issues linked to exact implementation-bearing requirements | Every implementation-bearing requirement has completed implementation evidence. |
| Determinism | Seed replay records | Repeated identical semantic cells reproduce stochastic choices and artifact identities. |
| Artifact/provenance | Fingerprint, checksum, manifest and staleness tests | Valid artifacts are reused; only true dependency changes invalidate correct descendants. |
| CLI/recovery | CLI E2E and failure-injection suite | Idempotency, overwrite, retry, status and nearest-valid-boundary recovery follow the registered semantics. |
| Reproducibility | Environment/statistical/claim identity compatibility records | No incompatible evidence is silently retained under an unchanged active identity. |
| Execution readiness | M02-owned §26 readiness test suite | Every execution/provenance readiness gate owned by M02 passes before M03 can establish the overall implementation-readiness state. |
| Deliverables | Required milestone outputs and artifacts | Complete, readable, schema-valid where applicable, and consistent with the roadmap. |
| Provenance | Required manifests, dependency identity, checksums, compatibility and staleness evidence | Complete and sufficient to verify origin, compatibility, reuse/invalidation and freshness where applicable. |
| Audit | Milestone audit issue | Final result is `PASS` with no unresolved blocking findings. |
## Milestone Audit
**Audit issue:** `—`

**Status:** `PENDING`

The milestone audit is the final completion gate. It must independently verify:

- complete roadmap coverage for all requirements owned by the milestone;
- exact requirement-to-issue traceability for implementation-bearing requirements;
- correct treatment of `NON_IMPLEMENTATION` constraints;
- closure of every mandatory implementation issue;
- completion and passing status of all required tests;
- completion and passing status of all required validations;
- existence and validity of all required deliverables;
- completeness and validity of required provenance and manifests;
- absence of stale or incompatible evidence;
- absence of unresolved blocking findings;
- readiness of milestone outputs for all declared downstream consumers.

The audit must end in exactly one result:

- `PASS` — every completion condition is satisfied.
- `FAIL` — one or more blocking conditions remain.

A milestone is not complete until the audit result is `PASS`.
## Scope Boundary
- This milestone implements only the implementation-bearing roadmap requirements explicitly allocated to it and preserves only the governing `NON_IMPLEMENTATION` constraints allocated to it.
- The authoritative roadmap remains the source of scientific, mathematical, methodological, architectural, numerical, configuration, artifact, execution, statistical, reporting, and claim requirements.
- This milestone may organize implementation work but may not redefine, weaken, strengthen, extend, or silently reinterpret the roadmap.
- Detailed implementation checklists belong in future implementation issues; no issue has been invented in this document.
- Detailed verification checklists belong in the future milestone audit issue; no audit issue has been invented in this document.
- Work outside this milestone's explicit requirement allocation must not be added unless the authoritative roadmap or Roadmap Coverage Inventory is explicitly updated first.

---

# M03 — Data Preparation and Strict Resource Interface
> **Outcome:** All registered datasets and clients are deterministically validated, split, preprocessed, and exposed through the strict no-common-interface resource contract with typed transfer and semantic manifests, completing the roadmap implementation-readiness gate for downstream scientific execution.
## At a Glance
| Field | Value |
|---|---|
| Roadmap scope | §§1.2, 4.3–4.6, 5–6, 22.1–22.3, 26 |
| Requirement ownership | REQ-0012, REQ-0144–REQ-0173, REQ-0194–REQ-0276, REQ-0278–REQ-0295, REQ-0841–REQ-1047, REQ-2885–REQ-2954, REQ-3175, REQ-3183 |
| Allocated requirement count | `411` total (`405` implementation-bearing; `6` `NON_IMPLEMENTATION`) |
| Upstream milestones | M01, M02 |
| Implementation issues | `I14`, `I15`, `I16`, `I17`, `I18` |
| Coverage authority | Roadmap Coverage Inventory |
| Audit issue | None — no dedicated per-milestone audit issue; verified via `Milestone Audit.md` |
| Audit status | `PENDING` |
## Coverage
The Roadmap Coverage Inventory is the traceability authority for this milestone. The work packages below partition this milestone's allocation; no requirement listed in another milestone is owned here.
| Roadmap Section | Scope / Work Package | Requirement IDs | Implementation Issue(s) | Verification / Evidence |
|---|---|---|---|---|
| §1.2 | Strict no-common-interface information regime | REQ-0012 | `I14` | Architecture/resource tests prove that only the roadmap-whitelisted source/target resources are accessible. |
| §4.3 | Duplicate-safe chronological splits, transfer eligibility, and split failure semantics | REQ-0144–REQ-0173 | `I15` | Split/property tests validate exact intervals from configuration, deduplication, eligibility thresholds, deterministic ordering, and failure states. |
| §4.4 | Dataset/client registry behavior, immutable raw-data authority, label canonicalization, and observed-value validation | REQ-0194–REQ-0276 | `I16` | Dataset validators compare configured identities to immutable raw bytes and record observed checksums/counts/schema facts without hardcoding literature values. |
| §§4.5–4.6 | Feature selection, identifier removal, duplicate hashing, and leakage-safe preprocessing | REQ-0278–REQ-0295 | `I17` | Preprocessing tests prove forbidden identity/provenance fields are excluded before hashing/model input and deterministic transformations preserve split semantics. |
| §5 | Strict resource whitelist, access-control boundary, dynamic access logging, and invalid-cell semantics | REQ-0841–REQ-0924 | `I14` | Static and dynamic resource-access scans prove each method reads only permitted resources; any violation marks the scientific cell invalid. |
| §6 | Dataset-specific adapters, client construction, preprocessing outputs, validation, and observed dataset facts | REQ-0925–REQ-1047 | `I16` | `fedorbit preprocess` outputs schema-valid client artifacts and records all roadmap-required observed values/checksums with provenance. |
| §22.1–§22.3 | Typed dataset, transfer-eligibility, and semantic-cell manifests | REQ-2885–REQ-2954 | `I18` | Schema tests enforce every required field, identity coordinate, provenance reference, eligibility decision, and semantic-cell contract. |
| §26 | Strict-resource readiness and overall implementation-readiness state | REQ-3175, REQ-3183 | `I14` | The strict-resource enforcement readiness test passes; together with valid M01/M02 readiness evidence, the roadmap implementation-readiness state remains `Pass` before downstream scientific execution. |
### Coverage Rules

- Every requirement listed in this milestone remains traceable to its exact Roadmap Coverage Inventory ID.
- Every mandatory implementation-bearing requirement must map to at least one real implementation issue before implementation of this milestone begins.
- Every conditional requirement remains traceable and must be implemented only when its roadmap-defined condition applies.
- `NON_IMPLEMENTATION` requirements are governing constraints only; they must remain traceable but must not be converted into fictitious implementation work.
- Every mapped implementation-bearing requirement must have objective verification or evidence.
- Every implementation issue must reference the exact requirement IDs it satisfies.
- A requirement is not considered covered merely because it falls inside a roadmap section associated with the milestone; the explicit requirement IDs in this document control milestone allocation.
- The current inventory-wide `UNMAPPED` state is expected before issue creation. No blocking requirement owned by this milestone may remain issue-unmapped when implementation of the milestone begins.
- No issue may redefine, weaken, silently reinterpret, duplicate, or extend the authoritative roadmap requirement it implements.
## Dependencies
### Milestone Dependencies
| Milestone | Required Input / Contract | Entry Gate |
|---|---|---|
| M01 — Authoritative Scientific Contract and Repository Foundation | Dataset/split/preprocessing configuration and repository adapters | Complete + audit PASS |
| M02 — Deterministic Execution, Artifact, and Provenance Backbone | Canonical artifact writer, semantic identity, provenance/fingerprint, CLI reuse and failure semantics | Complete + audit PASS |
### Artifact / Interface Dependencies
| Dependency | Produced By | Required Validation |
|---|---|---|
| Dataset/split/preprocessing configuration | M01 | Exact registered values and identifiers; typed validation passes. |
| Artifact/provenance/manifests substrate | M02 | Canonical identity, checksum, dependency, completion, and reusable-artifact validation passes. |
| Immutable raw dataset inputs | External data inputs | Readable at registered paths; checksums and observed facts are derived rather than assumed. |
Dependency completion alone is not sufficient. Every consumed dependency must be present, valid, provenance-compatible where applicable, current under its dependency fingerprint, and compatible with the active roadmap contract.
## Implementation Issues
Implementation issues for this milestone are listed below; each issue's detailed task checklist, acceptance criteria, and required tests are defined in `Issues.md`.
| Order | Issue | Work Package | Roadmap Scope | Requirement Coverage | Depends On |
|---:|---|---|---|---|---|
| 1 | `I14` — Enforce Strict No-Common-Interface Resource Boundary | Whitelisted strict-resource interface with dynamic access validation and readiness gates. | §1.2; §5; §26 | 87 atomic requirements | `I05`, `I11`, `I13` |
| 2 | `I15` — Implement Duplicate-Safe Chronological Splits and Transfer Eligibility | Deterministic duplicate-safe chronological splitting and source/target transfer eligibility. | §4.3 | 30 atomic requirements | `I07`, `I14` |
| 3 | `I16` — Implement Dataset Registry, Client Construction, Ontology, and Adapters | Immutable raw-data-backed dataset/client registry, hidden ontology, adapters, and observed-value validation. | §4.4; §6 | 206 atomic requirements | `I12`, `I14`, `I15` |
| 4 | `I17` — Implement Leakage-Safe Feature Selection, Deduplication, and Preprocessing | Leakage-safe feature exclusion, duplicate hashing, and deterministic preprocessing pipeline. | §§4.5–4.6 | 18 atomic requirements | `I15`, `I16` |
| 5 | `I18` — Persist Dataset, Eligibility, and Semantic-Cell Manifests | Typed dataset, transfer-eligibility, and semantic-cell manifests with provenance. | §22.1–§22.3 | 70 atomic requirements | `I12`, `I15`, `I16`, `I17` |
### Issue Contract

Every future milestone issue must:

- reference its exact roadmap section(s);
- list every covered implementation-bearing requirement ID;
- contain a detailed implementation checklist;
- define objective acceptance criteria;
- identify required tests;
- identify required artifacts, outputs, or interfaces;
- identify required provenance or manifest updates where applicable;
- identify explicit dependencies on upstream milestones, issues, artifacts, or interfaces;
- preserve roadmap terminology and semantics;
- avoid converting `NON_IMPLEMENTATION` constraints into artificial implementation tasks;
- close only when every mapped requirement and acceptance criterion is satisfied.
## Deliverables
| Deliverable | Source Issue(s) | Required Validation | Downstream Consumer |
|---|---|---|---|
| Validated dataset/client registry and raw-data observations | `I14`, `I15`, `I16`, `I17`, `I18` | Dataset validation, checksum and observed-value evidence | M04, M07, M08 |
| Deterministic leakage-safe splits and fitted preprocessors | `I14`, `I15`, `I16`, `I17`, `I18` | Split/preprocessing property and leakage tests | M04, M07, M08 |
| Strict no-common-interface resource layer with access logs | `I14`, `I15`, `I16`, `I17`, `I18` | Static/dynamic resource validation | M04–M09 |
| Transfer-eligibility and semantic-cell decisions | `I14`, `I15`, `I16`, `I17`, `I18` | Typed schema and eligibility validation | M04, M07, M08 |
| Dataset, transfer-eligibility and semantic manifests | `I14`, `I15`, `I16`, `I17`, `I18` | Schema, checksum and provenance validation | M04–M09 |
| Roadmap implementation-readiness gate evidence | `I14`, `I15`, `I16`, `I17`, `I18` | Strict-resource readiness test plus validated M01/M02 readiness evidence establish `Pass` | M04–M09 |
All roadmap-required deliverables for this milestone must appear in this table or remain explicitly traceable through the Roadmap Coverage Inventory. Source issue references remain `—` until real implementation issues exist.
## Entry Criteria
Implementation of this milestone may begin only when all applicable conditions below are true:
- all required upstream milestones are complete;
- every required upstream milestone audit is `PASS`;
- every required upstream artifact, interface, schema, manifest, or authoritative input exists;
- every consumed dependency passes its applicable validation and is provenance-compatible where provenance applies;
- M01 configuration for datasets, clients, splits and preprocessing is valid;
- M02 semantic identity, manifest, checksum, failure and reuse machinery is operational;
- registered raw inputs are available at the configured locations; literature counts are not substituted for observed raw-data facts;
- all roadmap requirements owned by this milestone remain present in the Roadmap Coverage Inventory with unchanged IDs unless the authoritative inventory itself has been updated;
- after issue decomposition, every mandatory implementation-bearing requirement is mapped to at least one real milestone issue with explicit verification evidence;
- no blocking requirement is `AMBIGUOUS` or `BLOCKED`;
- no unresolved roadmap ambiguity would force an implementer to invent a scientific, mathematical, methodological, numerical, architectural, configuration, artifact, or execution decision.
## Exit Criteria
The milestone is complete only when all applicable conditions below are true:
- every implementation-bearing requirement owned by this milestone is satisfied;
- every applicable conditional requirement is satisfied;
- every governing `NON_IMPLEMENTATION` constraint assigned to this milestone is preserved;
- every mapped implementation issue is closed;
- all registered dataset/client adapters produce validated observed facts, splits and preprocessed artifacts with no prohibited leakage;
- strict-resource static/dynamic scans pass and all dataset/eligibility/semantic manifests validate;
- the strict-resource readiness test passes and, with current valid M01/M02 readiness evidence, the roadmap implementation-readiness state is `Pass` before M04 begins;
- all required unit, integration, numerical, structural, schema, CLI/E2E, scientific, statistical, provenance, and reproducibility validations applicable to this milestone pass;
- all required deliverables are generated and all required artifacts, interfaces, schemas, and manifests pass validation;
- required provenance is complete and valid, and no required evidence is stale or incompatible with its material dependencies;
- the milestone audit is `PASS` with no unresolved blocking finding.
## Acceptance Evidence
| Evidence Area | Required Evidence | Pass Condition |
|---|---|---|
| Requirement coverage | Roadmap Coverage Inventory + milestone allocation | Every allocated requirement is accounted for exactly once; implementation-bearing requirements have completed issue evidence and governing constraints remain preserved. |
| Implementation | Closed real milestone issues linked to exact implementation-bearing requirements | Every implementation-bearing requirement has completed implementation evidence. |
| Dataset validation | Observed dataset/client facts + checksums | Observed values come from immutable raw inputs and required validation rules pass. |
| Splits/preprocessing | Split/preprocessor artifacts + property/leakage tests | Chronology/deduplication/eligibility and forbidden-field rules pass deterministically. |
| Strict resources | Static/dynamic resource access logs | Every valid cell uses only whitelisted resources; violations become `Invalid`. |
| Schemas | Dataset/eligibility/semantic manifest validation | All required fields, provenance, identities and decisions validate. |
| Implementation readiness | M01/M02 readiness evidence + M03 strict-resource readiness test | All roadmap §26 implementation-readiness conditions are represented by their owning milestones and the resulting readiness state is `Pass` before downstream scientific execution. |
| Deliverables | Required milestone outputs and artifacts | Complete, readable, schema-valid where applicable, and consistent with the roadmap. |
| Provenance | Required manifests, dependency identity, checksums, compatibility and staleness evidence | Complete and sufficient to verify origin, compatibility, reuse/invalidation and freshness where applicable. |
| Audit | Milestone audit issue | Final result is `PASS` with no unresolved blocking findings. |
## Milestone Audit
**Audit issue:** `—`

**Status:** `PENDING`

The milestone audit is the final completion gate. It must independently verify:

- complete roadmap coverage for all requirements owned by the milestone;
- exact requirement-to-issue traceability for implementation-bearing requirements;
- correct treatment of `NON_IMPLEMENTATION` constraints;
- closure of every mandatory implementation issue;
- completion and passing status of all required tests;
- completion and passing status of all required validations;
- existence and validity of all required deliverables;
- completeness and validity of required provenance and manifests;
- absence of stale or incompatible evidence;
- absence of unresolved blocking findings;
- readiness of milestone outputs for all declared downstream consumers.

The audit must end in exactly one result:

- `PASS` — every completion condition is satisfied.
- `FAIL` — one or more blocking conditions remain.

A milestone is not complete until the audit result is `PASS`.
## Scope Boundary
- This milestone implements only the implementation-bearing roadmap requirements explicitly allocated to it and preserves only the governing `NON_IMPLEMENTATION` constraints allocated to it.
- The authoritative roadmap remains the source of scientific, mathematical, methodological, architectural, numerical, configuration, artifact, execution, statistical, reporting, and claim requirements.
- This milestone may organize implementation work but may not redefine, weaken, strengthen, extend, or silently reinterpret the roadmap.
- Detailed implementation checklists belong in future implementation issues; no issue has been invented in this document.
- Detailed verification checklists belong in the future milestone audit issue; no audit issue has been invented in this document.
- Work outside this milestone's explicit requirement allocation must not be added unless the authoritative roadmap or Roadmap Coverage Inventory is explicitly updated first.

---

# M04 — Local Models and Procedural Response Packets
> **Outcome:** Each modality-specific client can train its registered local model and deterministically produce seed-bound anonymous procedural-response packets with simultaneous uncertainty bands and freshness guarantees.
## At a Glance
| Field | Value |
|---|---|
| Roadmap scope | §§1.1–1.2, 4.7–4.13, 7 |
| Requirement ownership | REQ-0006, REQ-0013, REQ-0296–REQ-0317, REQ-0329–REQ-0354, REQ-0358–REQ-0375, REQ-0385–REQ-0404, REQ-0413–REQ-0437, REQ-0443–REQ-0445, REQ-1048–REQ-1069 |
| Allocated requirement count | `138` total (`138` implementation-bearing; `0` `NON_IMPLEMENTATION`) |
| Upstream milestones | M02, M03 |
| Implementation issues | `I19`, `I20`, `I21`, `I22`, `I23`, `I24` |
| Coverage authority | Roadmap Coverage Inventory |
| Audit issue | None — no dedicated per-milestone audit issue; verified via `Milestone Audit.md` |
| Audit status | `PENDING` |
## Coverage
The Roadmap Coverage Inventory is the traceability authority for this milestone. The work packages below partition this milestone's allocation; no requirement listed in another milestone is owned here.
| Roadmap Section | Scope / Work Package | Requirement IDs | Implementation Issue(s) | Verification / Evidence |
|---|---|---|---|---|
| §§1.1–1.2 | Source packet export and simultaneous response-band estimation | REQ-0006, REQ-0013 | `I22` | Integration tests demonstrate anonymous packet production and simultaneous lower/upper response bands from permitted source-local resources. |
| §4.7 | Modality-specific local model architectures | REQ-0296–REQ-0317 | `I19` | Architecture/numerical tests validate exact layer order, dimensions derived from fitted preprocessors, modality branching, and deterministic initialization. |
| §4.8 | Base training loop, optimizer behavior, checkpoints, and training artifacts | REQ-0329–REQ-0354 | `I19` | Training tests validate configured optimizer settings, budgets, checkpoint semantics, and artifact/provenance outputs. |
| §§4.9–4.10 | Base-model pilot selection and class-weight/minibatch objective behavior | REQ-0358–REQ-0375 | `I20` | Pilot matrix/selection artifacts and numeric tests validate the registered grid consumption, VALID-only deterministic selection, class weights, and minibatch loss. |
| §4.11 | Source-response estimator pilot | REQ-0385–REQ-0404 | `I21` | Pilot artifacts cover every configured intervention/horizon candidate and validate derivative/diagnostic selection logic. |
| §4.12 | Final source-response estimation and simultaneous uncertainty bands | REQ-0413–REQ-0437 | `I22` | Paired-replicate/bootstrap tests validate shadow isolation, simultaneous coverage construction, serialization, and packet eligibility. |
| §4.13 | Target-local response diagnostic | REQ-0443–REQ-0445 | `I23` | Diagnostic artifact uses only registered target-local inputs and statistical settings and passes deterministic replay. |
| §7 | Model instantiation, local training/resource behavior, response packet schema, seed binding, and zero staleness grace | REQ-1048–REQ-1069 | `I24` | Model/packet integration tests validate modality input dimensions, exact seed identity, packet integrity, and immediate invalidation on material dependency changes. |
### Coverage Rules

- Every requirement listed in this milestone remains traceable to its exact Roadmap Coverage Inventory ID.
- Every mandatory implementation-bearing requirement must map to at least one real implementation issue before implementation of this milestone begins.
- Every conditional requirement remains traceable and must be implemented only when its roadmap-defined condition applies.
- `NON_IMPLEMENTATION` requirements are governing constraints only; they must remain traceable but must not be converted into fictitious implementation work.
- Every mapped implementation-bearing requirement must have objective verification or evidence.
- Every implementation issue must reference the exact requirement IDs it satisfies.
- A requirement is not considered covered merely because it falls inside a roadmap section associated with the milestone; the explicit requirement IDs in this document control milestone allocation.
- The current inventory-wide `UNMAPPED` state is expected before issue creation. No blocking requirement owned by this milestone may remain issue-unmapped when implementation of the milestone begins.
- No issue may redefine, weaken, silently reinterpret, duplicate, or extend the authoritative roadmap requirement it implements.
## Dependencies
### Milestone Dependencies
| Milestone | Required Input / Contract | Entry Gate |
|---|---|---|
| M02 — Deterministic Execution, Artifact, and Provenance Backbone | Deterministic execution, artifact identity, checkpoint/provenance and reuse semantics | Complete + audit PASS |
| M03 — Data Preparation and Strict Resource Interface | Validated client splits, fitted preprocessors, eligibility/resource contracts, and manifests | Complete + audit PASS |
### Artifact / Interface Dependencies
| Dependency | Produced By | Required Validation |
|---|---|---|
| Prepared TRAIN/VALID/META client artifacts and fitted preprocessors | M03 | Schema-valid, leakage-safe, resource-valid, provenance-compatible. |
| Dataset/transfer/semantic manifests | M03 | Identity, eligibility, checksums, split and provenance fields validate. |
| Canonical checkpoint/artifact interfaces | M02 | Content-addressed identity and dependency fingerprints validate. |
Dependency completion alone is not sufficient. Every consumed dependency must be present, valid, provenance-compatible where applicable, current under its dependency fingerprint, and compatible with the active roadmap contract.
## Implementation Issues
Implementation issues for this milestone are listed below; each issue's detailed task checklist, acceptance criteria, and required tests are defined in `Issues.md`.
| Order | Issue | Work Package | Roadmap Scope | Requirement Coverage | Depends On |
|---:|---|---|---|---|---|
| 1 | `I19` — Implement Modality-Specific Local Models and Base Training | Deterministic modality-specific classifiers, training loop, checkpoints, and training artifacts. | §4.7; §4.8 | 48 atomic requirements | `I07`, `I12`, `I17`, `I18` |
| 2 | `I20` — Implement Base-Model Pilot, Class Weights, and Minibatch Objective | Registered base-model pilot and final class-weight/minibatch objective behavior. | §§4.9–4.10 | 18 atomic requirements | `I19` |
| 3 | `I21` — Implement Source-Response Estimator Pilot | Registered source-response estimator pilot and selection evidence. | §4.11 | 20 atomic requirements | `I20` |
| 4 | `I22` — Implement Final Source Response Bands and Anonymous Packet Export | Simultaneous response-band estimator and strict anonymous procedural-response packet. | §§1.1–1.2; §4.12 | 27 atomic requirements | `I14`, `I18`, `I21` |
| 5 | `I23` — Implement Target-Local Response Diagnostics | Target-local response diagnostic computed only from permitted target-local evidence. | §4.13 | 3 atomic requirements | `I19`, `I22` |
| 6 | `I24` — Integrate Local Model, Response Packet, Seed, and Resource Contracts | Integrated §7 local-model and response-packet execution contract with zero-support and seed/resource behavior. | §7 | 22 atomic requirements | `I18`, `I22`, `I23` |
### Issue Contract

Every future milestone issue must:

- reference its exact roadmap section(s);
- list every covered implementation-bearing requirement ID;
- contain a detailed implementation checklist;
- define objective acceptance criteria;
- identify required tests;
- identify required artifacts, outputs, or interfaces;
- identify required provenance or manifest updates where applicable;
- identify explicit dependencies on upstream milestones, issues, artifacts, or interfaces;
- preserve roadmap terminology and semantics;
- avoid converting `NON_IMPLEMENTATION` constraints into artificial implementation tasks;
- close only when every mapped requirement and acceptance criterion is satisfied.
## Deliverables
| Deliverable | Source Issue(s) | Required Validation | Downstream Consumer |
|---|---|---|---|
| Modality-specific local model implementations and canonical checkpoints | `I19`, `I20`, `I21`, `I22`, `I23`, `I24` | Architecture, training, checkpoint and replay tests | M05, M07, M08 |
| Base-model pilot selection artifacts | `I19`, `I20`, `I21`, `I22`, `I23`, `I24` | Exact grid, VALID-only selection and provenance checks | M07 |
| Source-response estimator pilot artifacts | `I19`, `I20`, `I21`, `I22`, `I23`, `I24` | Candidate-grid, diagnostic and deterministic-selection validation | M07 |
| Final anonymous procedural-response packets with simultaneous bands | `I19`, `I20`, `I21`, `I22`, `I23`, `I24` | Numeric/statistical, schema, seed, freshness and resource validation | M05, M07, M08 |
| Target-local response diagnostic artifacts | `I19`, `I20`, `I21`, `I22`, `I23`, `I24` | Target-only resource and statistical/replay validation | M05, M07 |
All roadmap-required deliverables for this milestone must appear in this table or remain explicitly traceable through the Roadmap Coverage Inventory. Source issue references remain `—` until real implementation issues exist.
## Entry Criteria
Implementation of this milestone may begin only when all applicable conditions below are true:
- all required upstream milestones are complete;
- every required upstream milestone audit is `PASS`;
- every required upstream artifact, interface, schema, manifest, or authoritative input exists;
- every consumed dependency passes its applicable validation and is provenance-compatible where provenance applies;
- M03 has produced valid eligible client data, leakage-safe splits, fitted preprocessors, resource contracts and manifests;
- M03 has established the roadmap implementation-readiness state as `Pass` before local-model and procedural-response execution begins;
- M02 checkpoint/artifact/provenance interfaces are available for all pilot and confirmatory model outputs;
- all roadmap requirements owned by this milestone remain present in the Roadmap Coverage Inventory with unchanged IDs unless the authoritative inventory itself has been updated;
- after issue decomposition, every mandatory implementation-bearing requirement is mapped to at least one real milestone issue with explicit verification evidence;
- no blocking requirement is `AMBIGUOUS` or `BLOCKED`;
- no unresolved roadmap ambiguity would force an implementer to invent a scientific, mathematical, methodological, numerical, architectural, configuration, artifact, or execution decision.
## Exit Criteria
The milestone is complete only when all applicable conditions below are true:
- every implementation-bearing requirement owned by this milestone is satisfied;
- every applicable conditional requirement is satisfied;
- every governing `NON_IMPLEMENTATION` constraint assigned to this milestone is preserved;
- every mapped implementation issue is closed;
- registered local models train/replay under the locked contracts and all required pilot/checkpoint artifacts validate;
- final source-response packets and target diagnostics are schema-valid, statistically valid, seed-bound, fresh and strict-resource compliant;
- all required unit, integration, numerical, structural, schema, CLI/E2E, scientific, statistical, provenance, and reproducibility validations applicable to this milestone pass;
- all required deliverables are generated and all required artifacts, interfaces, schemas, and manifests pass validation;
- required provenance is complete and valid, and no required evidence is stale or incompatible with its material dependencies;
- the milestone audit is `PASS` with no unresolved blocking finding.
## Acceptance Evidence
| Evidence Area | Required Evidence | Pass Condition |
|---|---|---|
| Requirement coverage | Roadmap Coverage Inventory + milestone allocation | Every allocated requirement is accounted for exactly once; implementation-bearing requirements have completed issue evidence and governing constraints remain preserved. |
| Implementation | Closed real milestone issues linked to exact implementation-bearing requirements | Every implementation-bearing requirement has completed implementation evidence. |
| Model training | Architecture/training/checkpoint test results | Registered modality models, optimizer behavior and checkpoints pass deterministic integration tests. |
| Pilot selection | Base-model and response-pilot selection artifacts | Exact registered candidate grids are evaluated and deterministic selection rules pass. |
| Response bands | Final paired-replicate/simultaneous-band artifacts | Numeric/statistical band construction and serialization pass. |
| Packet validity | Source packet schema/seed/freshness/resource checks | Packets are anonymous, seed-bound, fresh, strict-resource valid and provenance-compatible. |
| Deliverables | Required milestone outputs and artifacts | Complete, readable, schema-valid where applicable, and consistent with the roadmap. |
| Provenance | Required manifests, dependency identity, checksums, compatibility and staleness evidence | Complete and sufficient to verify origin, compatibility, reuse/invalidation and freshness where applicable. |
| Audit | Milestone audit issue | Final result is `PASS` with no unresolved blocking findings. |
## Milestone Audit
**Audit issue:** `—`

**Status:** `PENDING`

The milestone audit is the final completion gate. It must independently verify:

- complete roadmap coverage for all requirements owned by the milestone;
- exact requirement-to-issue traceability for implementation-bearing requirements;
- correct treatment of `NON_IMPLEMENTATION` constraints;
- closure of every mandatory implementation issue;
- completion and passing status of all required tests;
- completion and passing status of all required validations;
- existence and validity of all required deliverables;
- completeness and validity of required provenance and manifests;
- absence of stale or incompatible evidence;
- absence of unresolved blocking findings;
- readiness of milestone outputs for all declared downstream consumers.

The audit must end in exactly one result:

- `PASS` — every completion condition is satisfied.
- `FAIL` — one or more blocking conditions remain.

A milestone is not complete until the audit result is `PASS`.
## Scope Boundary
- This milestone implements only the implementation-bearing roadmap requirements explicitly allocated to it and preserves only the governing `NON_IMPLEMENTATION` constraints allocated to it.
- The authoritative roadmap remains the source of scientific, mathematical, methodological, architectural, numerical, configuration, artifact, execution, statistical, reporting, and claim requirements.
- This milestone may organize implementation work but may not redefine, weaken, strengthen, extend, or silently reinterpret the roadmap.
- Detailed implementation checklists belong in future implementation issues; no issue has been invented in this document.
- Detailed verification checklists belong in the future milestone audit issue; no audit issue has been invented in this document.
- Work outside this milestone's explicit requirement allocation must not be added unless the authoritative roadmap or Roadmap Coverage Inventory is explicitly updated first.

---

# M05 — Robust Correspondence, Selection, and Confirmation Engine
> **Outcome:** The FedORBIT mathematical action model, exact-sparse correspondence solver, exact-QAP truth path, dense non-exact fallback, multi-source selection, and target confirmation operate under the registered theorem assumptions and compute budgets.
## At a Glance
| Field | Value |
|---|---|
| Roadmap scope | §§1.1–1.4, 3, 4.2, 4.14–4.18, 4.20, 8–10 |
| Requirement ownership | REQ-0007–REQ-0011, REQ-0014–REQ-0016, REQ-0018–REQ-0019, REQ-0022–REQ-0029, REQ-0064–REQ-0096, REQ-0127–REQ-0131, REQ-0457–REQ-0484, REQ-0498–REQ-0506, REQ-0512–REQ-0547, REQ-0556–REQ-0607, REQ-0654–REQ-0657, REQ-1070–REQ-1100, REQ-1103–REQ-1131 |
| Allocated requirement count | `245` total (`225` implementation-bearing; `20` `NON_IMPLEMENTATION`) |
| Upstream milestones | M02, M04 |
| Implementation issues | `I25`, `I26`, `I27`, `I28`, `I29`, `I30`, `I31`, `I32` |
| Coverage authority | Roadmap Coverage Inventory |
| Audit issue | None — no dedicated per-milestone audit issue; verified via `Milestone Audit.md` |
| Audit status | `PENDING` |
## Coverage
The Roadmap Coverage Inventory is the traceability authority for this milestone. The work packages below partition this milestone's allocation; no requirement listed in another milestone is owned here.
| Roadmap Section | Scope / Work Package | Requirement IDs | Implementation Issue(s) | Verification / Evidence |
|---|---|---|---|---|
| §§1.1–1.4 | Core correspondence/action semantics, solver naming, dense-fallback scope, confirmation requirement, and theorem assumptions | REQ-0007–REQ-0011, REQ-0014–REQ-0016, REQ-0018–REQ-0019, REQ-0022–REQ-0029 | `I25` | Canonical-vocabulary, theorem-assumption, algorithm, and scope tests enforce joint correspondence, robust action selection, target confirmation, and dense-fallback non-exactness. |
| §3 | Padded block correspondence space, action polytope, robust objective, orbit/rectangularization, map value, null-node semantics, and work-count mathematics | REQ-0064–REQ-0096 | `I25` | Hand/exhaustive numerical and property tests match every registered equation and invariant within roadmap tolerances. |
| §4.2 | Executable action-set construction from the authoritative action configuration | REQ-0127–REQ-0131 | `I25` | Action-feasibility tests enforce coordinate caps, aggregate budget, support restrictions, and null-node action exclusions. |
| §§4.14–4.15 | Paired target confirmation behavior, bootstrap decision logic, and target-local compute budgets | REQ-0457–REQ-0484 | `I26` | Confirmation-shadow tests prove exact pairing, pre-TEST isolation, configured bootstrap decisions, and hard optimizer-step budget enforcement. |
| §4.16; §8.3 | Exact-sparse solver backend behavior, deterministic ties, master/separator loop, and certificates | REQ-0498–REQ-0506 | `I27` | Solver unit tests and certificate checks validate master feasibility, LAP/action tie semantics, cut generation, and configured stopping tolerance. |
| §4.17 | Generic Exact QAP truth/comparator solver behavior | REQ-0512–REQ-0547 | `I28` | SCIP/PySCIPOpt tests validate exact-QAP formulation, tie rules, truth availability, timeout/failure semantics, and no oracle leakage. |
| §4.18 | FedORBIT Dense-CCP Fallback behavior | REQ-0556–REQ-0607 | `I29` | Dense-CCP numerical tests validate full dense action use, configured penalties/restarts/stopping, deterministic selection, diagnostics, and explicit non-exact metadata. |
| §4.20 | Target-importance construction | REQ-0654–REQ-0657 | `I30` | Target-META-only construction tests reproduce the configured nonnegative importance vector and risk floor. |
| §8 | Exact sparse separator and robust-action optimization | REQ-1070–REQ-1100 | `I27` | Exhaustive/synthetic fixtures validate active-image enumeration, blockwise LAP decomposition, exact worst correspondence, action tie-breaking, and certificates. |
| §9 | Multi-source proposal ranking and sequential selection behavior | REQ-1103–REQ-1113 | `I31` | Selection tests enforce the configured ranking objective, deterministic proposal order, and prohibition on principal packet averaging. |
| §10 | Paired target confirmation and live assimilation | REQ-1114–REQ-1131 | `I32` | Shadow/live-assimilation E2E tests enforce target-local paired confirmation, TEST read barrier, acceptance/rejection semantics, and invalidation on early TEST access. |
### Coverage Rules

- Every requirement listed in this milestone remains traceable to its exact Roadmap Coverage Inventory ID.
- Every mandatory implementation-bearing requirement must map to at least one real implementation issue before implementation of this milestone begins.
- Every conditional requirement remains traceable and must be implemented only when its roadmap-defined condition applies.
- `NON_IMPLEMENTATION` requirements are governing constraints only; they must remain traceable but must not be converted into fictitious implementation work.
- Every mapped implementation-bearing requirement must have objective verification or evidence.
- Every implementation issue must reference the exact requirement IDs it satisfies.
- A requirement is not considered covered merely because it falls inside a roadmap section associated with the milestone; the explicit requirement IDs in this document control milestone allocation.
- The current inventory-wide `UNMAPPED` state is expected before issue creation. No blocking requirement owned by this milestone may remain issue-unmapped when implementation of the milestone begins.
- No issue may redefine, weaken, silently reinterpret, duplicate, or extend the authoritative roadmap requirement it implements.
## Dependencies
### Milestone Dependencies
| Milestone | Required Input / Contract | Entry Gate |
|---|---|---|
| M02 — Deterministic Execution, Artifact, and Provenance Backbone | Canonical solver/action/confirmation artifact identity, reuse, failure and provenance contracts | Complete + audit PASS |
| M03 — Data Preparation and Strict Resource Interface | Processed dataset splits, transfer-eligibility, and semantic-cell manifests consumed directly by `I25`/`I30` | Complete + audit PASS |
| M04 — Local Models and Procedural Response Packets | Validated local checkpoints, target diagnostics, and anonymous source response packets | Complete + audit PASS |
### Artifact / Interface Dependencies
| Dependency | Produced By | Required Validation |
|---|---|---|
| Source procedural-response packets with simultaneous bands | M04 | Seed-bound, schema-valid, fresh, strict-resource-valid, provenance-compatible. |
| Target local checkpoints and response diagnostics | M04 | Checkpoint identity and target-local provenance validate. |
| Scientific action/solver/confirmation configuration | M01 | Exact typed values and tolerances validate. |
Dependency completion alone is not sufficient. Every consumed dependency must be present, valid, provenance-compatible where applicable, current under its dependency fingerprint, and compatible with the active roadmap contract.
## Implementation Issues
Implementation issues for this milestone are listed below; each issue's detailed task checklist, acceptance criteria, and required tests are defined in `Issues.md`.
| Order | Issue | Work Package | Roadmap Scope | Requirement Coverage | Depends On |
|---:|---|---|---|---|---|
| 1 | `I25` — Implement Formal Correspondence, Orbit, Action, and Null-Node Semantics | Typed formal problem objects and executable action/correspondence semantics preserving theorem assumptions. | §§1.1–1.4; §3; §4.2 | 56 atomic requirements | `I01`, `I18`, `I24` |
| 2 | `I26` — Implement Paired Confirmation Decisions and Target Compute Budgets | Roadmap-exact paired confirmation bootstrap logic and target-local optimizer-step budgets. | §§4.14–4.15 | 28 atomic requirements | `I19`, `I25` |
| 3 | `I27` — Implement FedORBIT Exact-Sparse Solver and Certificates | Exact sparse separator/master implementation with deterministic ties, work counters, and validity certificates. | §4.16; §8.3; §8 | 40 atomic requirements | `I07`, `I24`, `I25` |
| 4 | `I28` — Implement Generic Exact-QAP Truth and Comparator Solver | Generic exact-QAP truth/comparator backend with exactness and failure semantics. | §4.17 | 36 atomic requirements | `I07`, `I25` |
| 5 | `I29` — Implement FedORBIT Dense-CCP Fallback | Non-exact Dense-CCP fallback with deterministic starts, projection, CCP/master behavior, and scope labeling. | §4.18 | 52 atomic requirements | `I07`, `I25` |
| 6 | `I30` — Implement Target-Importance Construction | Roadmap-defined nonnegative target-importance vector construction. | §4.20 | 4 atomic requirements | `I18`, `I25` |
| 7 | `I31` — Implement Multi-Source Proposal Ranking and Sequential Selection | Deterministic multi-source source-proposal ranking, limits, and sequential selection. | §9 | 11 atomic requirements | `I22`, `I27`, `I30` |
| 8 | `I32` — Implement Target Confirmation and Live Assimilation | Paired proposal confirmation, reject/accept behavior, TEST opening, and live assimilation. | §10 | 18 atomic requirements | `I26`, `I31` |
### Issue Contract

Every future milestone issue must:

- reference its exact roadmap section(s);
- list every covered implementation-bearing requirement ID;
- contain a detailed implementation checklist;
- define objective acceptance criteria;
- identify required tests;
- identify required artifacts, outputs, or interfaces;
- identify required provenance or manifest updates where applicable;
- identify explicit dependencies on upstream milestones, issues, artifacts, or interfaces;
- preserve roadmap terminology and semantics;
- avoid converting `NON_IMPLEMENTATION` constraints into artificial implementation tasks;
- close only when every mapped requirement and acceptance criterion is satisfied.
## Deliverables
| Deliverable | Source Issue(s) | Required Validation | Downstream Consumer |
|---|---|---|---|
| Executable correspondence/action mathematical primitives | `I25`, `I26`, `I27`, `I28`, `I29`, `I30`, `I31`, `I32` | Numeric/property/exhaustive fixture validation | M06, M07, M08 |
| FedORBIT Exact-Sparse Solver and certificates | `I25`, `I26`, `I27`, `I28`, `I29`, `I30`, `I31`, `I32` | Exact truth, LAP/master/separator, tie and certificate tests | M06–M09 |
| Generic Exact QAP truth/comparator path | `I25`, `I26`, `I27`, `I28`, `I29`, `I30`, `I31`, `I32` | Exact formulation, ACL, timeout and truth-availability validation | M06–M09 |
| FedORBIT Dense-CCP Fallback | `I25`, `I26`, `I27`, `I28`, `I29`, `I30`, `I31`, `I32` | Dense-action, convergence/restart/diagnostic and non-exactness validation | M06, M08, M09 |
| Multi-source proposal ranking/selection engine | `I25`, `I26`, `I27`, `I28`, `I29`, `I30`, `I31`, `I32` | Deterministic ranking and no-packet-averaging tests | M07, M08 |
| Paired target confirmation and live-assimilation engine | `I25`, `I26`, `I27`, `I28`, `I29`, `I30`, `I31`, `I32` | Shadow pairing, compute-budget and TEST-barrier E2E tests | M06–M09 |
All roadmap-required deliverables for this milestone must appear in this table or remain explicitly traceable through the Roadmap Coverage Inventory. Source issue references remain `—` until real implementation issues exist.
## Entry Criteria
Implementation of this milestone may begin only when all applicable conditions below are true:
- all required upstream milestones are complete;
- every required upstream milestone audit is `PASS`;
- every required upstream artifact, interface, schema, manifest, or authoritative input exists;
- every consumed dependency passes its applicable validation and is provenance-compatible where provenance applies;
- M04 response packets/checkpoints required for real-data integration are valid and fresh;
- the mathematical and solver configuration owned by M01 is loaded through the typed contract without local overrides;
- all roadmap requirements owned by this milestone remain present in the Roadmap Coverage Inventory with unchanged IDs unless the authoritative inventory itself has been updated;
- after issue decomposition, every mandatory implementation-bearing requirement is mapped to at least one real milestone issue with explicit verification evidence;
- no blocking requirement is `AMBIGUOUS` or `BLOCKED`;
- no unresolved roadmap ambiguity would force an implementer to invent a scientific, mathematical, methodological, numerical, architectural, configuration, artifact, or execution decision.
## Exit Criteria
The milestone is complete only when all applicable conditions below are true:
- every implementation-bearing requirement owned by this milestone is satisfied;
- every applicable conditional requirement is satisfied;
- every governing `NON_IMPLEMENTATION` constraint assigned to this milestone is preserved;
- every mapped implementation issue is closed;
- exact-sparse solver results agree with available exact truth where required and carry valid deterministic certificates/counters;
- dense fallback is never represented as exact, source selection obeys registered ranking semantics, and confirmation enforces budget/TEST barriers;
- all required unit, integration, numerical, structural, schema, CLI/E2E, scientific, statistical, provenance, and reproducibility validations applicable to this milestone pass;
- all required deliverables are generated and all required artifacts, interfaces, schemas, and manifests pass validation;
- required provenance is complete and valid, and no required evidence is stale or incompatible with its material dependencies;
- the milestone audit is `PASS` with no unresolved blocking finding.
## Acceptance Evidence
| Evidence Area | Required Evidence | Pass Condition |
|---|---|---|
| Requirement coverage | Roadmap Coverage Inventory + milestone allocation | Every allocated requirement is accounted for exactly once; implementation-bearing requirements have completed issue evidence and governing constraints remain preserved. |
| Implementation | Closed real milestone issues linked to exact implementation-bearing requirements | Every implementation-bearing requirement has completed implementation evidence. |
| Mathematics | Hand/exhaustive primitive tests | Every action/correspondence/map-value equation and invariant matches the roadmap tolerances. |
| Exact sparse | Truth comparisons + solver certificates/counters | Available exact truth agrees and certificates/tie/counter semantics validate. |
| Dense fallback | Dense-CCP diagnostic artifacts | Registered dense procedure completes with required diagnostics and non-exact classification. |
| Selection/confirmation | Sequential-selection and paired-confirmation E2E traces | Ranking, budgets, TEST barrier, accept/reject and live assimilation semantics pass. |
| Deliverables | Required milestone outputs and artifacts | Complete, readable, schema-valid where applicable, and consistent with the roadmap. |
| Provenance | Required manifests, dependency identity, checksums, compatibility and staleness evidence | Complete and sufficient to verify origin, compatibility, reuse/invalidation and freshness where applicable. |
| Audit | Milestone audit issue | Final result is `PASS` with no unresolved blocking findings. |
## Milestone Audit
**Audit issue:** `—`

**Status:** `PENDING`

The milestone audit is the final completion gate. It must independently verify:

- complete roadmap coverage for all requirements owned by the milestone;
- exact requirement-to-issue traceability for implementation-bearing requirements;
- correct treatment of `NON_IMPLEMENTATION` constraints;
- closure of every mandatory implementation issue;
- completion and passing status of all required tests;
- completion and passing status of all required validations;
- existence and validity of all required deliverables;
- completeness and validity of required provenance and manifests;
- absence of stale or incompatible evidence;
- absence of unresolved blocking findings;
- readiness of milestone outputs for all declared downstream consumers.

The audit must end in exactly one result:

- `PASS` — every completion condition is satisfied.
- `FAIL` — one or more blocking conditions remain.

A milestone is not complete until the audit result is `PASS`.
## Scope Boundary
- This milestone implements only the implementation-bearing roadmap requirements explicitly allocated to it and preserves only the governing `NON_IMPLEMENTATION` constraints allocated to it.
- The authoritative roadmap remains the source of scientific, mathematical, methodological, architectural, numerical, configuration, artifact, execution, statistical, reporting, and claim requirements.
- This milestone may organize implementation work but may not redefine, weaken, strengthen, extend, or silently reinterpret the roadmap.
- Detailed implementation checklists belong in future implementation issues; no issue has been invented in this document.
- Detailed verification checklists belong in the future milestone audit issue; no audit issue has been invented in this document.
- Work outside this milestone's explicit requirement allocation must not be added unless the authoritative roadmap or Roadmap Coverage Inventory is explicitly updated first.

---

# M06 — Comparator, Metric, and Statistical Evaluation Framework
> **Outcome:** All registered methods can be compared under equal resource budgets using canonical predictions, metrics, paired statistical procedures, claim criteria, and typed evaluation schemas without downstream reimplementation.
## At a Glance
| Field | Value |
|---|---|
| Roadmap scope | §§1.2, 4.19, 4.23–4.25, 11–13, 22.4–22.7 |
| Requirement ownership | REQ-0017, REQ-0609–REQ-0652, REQ-0706–REQ-0727, REQ-0751–REQ-0784, REQ-1132–REQ-1158, REQ-1162–REQ-1333, REQ-2955–REQ-3014 |
| Allocated requirement count | `360` total (`352` implementation-bearing; `8` `NON_IMPLEMENTATION`) |
| Upstream milestones | M02, M05 |
| Implementation issues | `I33`, `I34`, `I35`, `I36` |
| Coverage authority | Roadmap Coverage Inventory |
| Audit issue | None — no dedicated per-milestone audit issue; verified via `Milestone Audit.md` |
| Audit status | `PENDING` |
## Coverage
The Roadmap Coverage Inventory is the traceability authority for this milestone. The work packages below partition this milestone's allocation; no requirement listed in another milestone is owned here.
| Roadmap Section | Scope / Work Package | Requirement IDs | Implementation Issue(s) | Verification / Evidence |
|---|---|---|---|---|
| §1.2 | Map-identifiability versus action-certifiability diagnostics | REQ-0017 | `I33` | Canonical evaluation records separately expose map-identification and action-certification diagnostics without conflation. |
| §4.19 | Baseline and oracle implementation behavior | REQ-0609–REQ-0652 | `I34` | Comparator tests validate Local-Only, Local-SIR, rectangular, point-correspondence, oracle, and related registered behavior under strict resources. |
| §§4.23–4.25 | Statistical behavior, multiplicity families, materiality/equivalence rules, and mechanical claim criteria | REQ-0706–REQ-0727, REQ-0751–REQ-0784 | `I35` | Statistical tests reproduce configured alpha/CI/bootstrap/randomization/Holm/materiality criteria and claim-gate decisions without redefining constants. |
| §11 | Comparator catalogue, descriptive names, equal compute/resource budgets, and checkpoint fairness | REQ-1132–REQ-1158 | `I34` | Comparator catalogue and fairness tests reject opaque names, excess target compute, favorable checkpoints, or resource/interface advantages. |
| §12 | Canonical predictive, transfer, coupling, confirmation, map-value, fairness, and efficiency metrics | REQ-1162–REQ-1242 | `I33` | Metric unit tests on hand fixtures reproduce every formula, denominator rule, aggregation scope, and efficiency field exactly. |
| §13 | Paired experimental unit, BCa intervals, exact randomization, equivalence, materiality, multiplicity, missingness, and synthesis rules | REQ-1243–REQ-1333 | `I35` | Statistical golden tests and exhaustive small fixtures match registered procedures, effective sample handling, family sizes, and correction rules. |
| §22.4–§22.7 | Typed prediction, metric, paired-comparison, and statistical-metadata schemas | REQ-2955–REQ-3014 | `I36` | Schema validation ensures canonical field sets, identities, provenance links, statistical metadata, and reusable source records. |
### Coverage Rules

- Every requirement listed in this milestone remains traceable to its exact Roadmap Coverage Inventory ID.
- Every mandatory implementation-bearing requirement must map to at least one real implementation issue before implementation of this milestone begins.
- Every conditional requirement remains traceable and must be implemented only when its roadmap-defined condition applies.
- `NON_IMPLEMENTATION` requirements are governing constraints only; they must remain traceable but must not be converted into fictitious implementation work.
- Every mapped implementation-bearing requirement must have objective verification or evidence.
- Every implementation issue must reference the exact requirement IDs it satisfies.
- A requirement is not considered covered merely because it falls inside a roadmap section associated with the milestone; the explicit requirement IDs in this document control milestone allocation.
- The current inventory-wide `UNMAPPED` state is expected before issue creation. No blocking requirement owned by this milestone may remain issue-unmapped when implementation of the milestone begins.
- No issue may redefine, weaken, silently reinterpret, duplicate, or extend the authoritative roadmap requirement it implements.
## Dependencies
### Milestone Dependencies
| Milestone | Required Input / Contract | Entry Gate |
|---|---|---|
| M02 — Deterministic Execution, Artifact, and Provenance Backbone | Canonical prediction/metric/statistical artifact identity and provenance substrate | Complete + audit PASS |
| M05 — Robust Correspondence, Selection, and Confirmation Engine | Method outputs, solver certificates, source-selection traces, and confirmation semantics | Complete + audit PASS |
### Artifact / Interface Dependencies
| Dependency | Produced By | Required Validation |
|---|---|---|
| Canonical method/action/confirmation records | M05 | Method identity, solver/certificate and confirmation provenance validate. |
| Prediction and model outputs from strict-resource execution | M04 / M05 | Semantic-cell identity and resource/access validation pass. |
| Statistical and claim-criterion configuration | M01 | All registered thresholds, multiplicity families, confidence and materiality settings validate. |
Dependency completion alone is not sufficient. Every consumed dependency must be present, valid, provenance-compatible where applicable, current under its dependency fingerprint, and compatible with the active roadmap contract.
## Implementation Issues
Implementation issues for this milestone are listed below; each issue's detailed task checklist, acceptance criteria, and required tests are defined in `Issues.md`.
| Order | Issue | Work Package | Roadmap Scope | Requirement Coverage | Depends On |
|---:|---|---|---|---|---|
| 1 | `I33` — Implement Canonical Predictive, Mechanism, Confirmation, and Efficiency Metrics | Single canonical metric layer including map/action diagnostics and solver/confirmation/efficiency measures. | §1.2; §12 | 82 atomic requirements | `I24`, `I27`, `I28`, `I29`, `I30`, `I32` |
| 2 | `I34` — Implement Fair Baselines, Oracles, and Matched Resource Budgets | Complete comparator/oracle catalogue with descriptive identities and matched checkpoint/compute/resource fairness. | §4.19; §11 | 71 atomic requirements | `I19`, `I27`, `I28`, `I29`, `I33` |
| 3 | `I35` — Implement Statistical Analysis, Multiplicity, Materiality, and Equivalence | Predefined paired statistical framework, multiplicity families, equivalence/materiality, and insufficient-evidence behavior. | §§4.23–4.25; §13 | 147 atomic requirements | `I07`, `I33`, `I34` |
| 4 | `I36` — Persist Prediction, Metric, Paired-Comparison, and Statistical Schemas | Typed canonical evaluation/statistical result schemas and metadata. | §22.4–§22.7 | 60 atomic requirements | `I12`, `I33`, `I35` |
### Issue Contract

Every future milestone issue must:

- reference its exact roadmap section(s);
- list every covered implementation-bearing requirement ID;
- contain a detailed implementation checklist;
- define objective acceptance criteria;
- identify required tests;
- identify required artifacts, outputs, or interfaces;
- identify required provenance or manifest updates where applicable;
- identify explicit dependencies on upstream milestones, issues, artifacts, or interfaces;
- preserve roadmap terminology and semantics;
- avoid converting `NON_IMPLEMENTATION` constraints into artificial implementation tasks;
- close only when every mapped requirement and acceptance criterion is satisfied.
## Deliverables
| Deliverable | Source Issue(s) | Required Validation | Downstream Consumer |
|---|---|---|---|
| Registered comparator suite with equal-resource enforcement | `I33`, `I34`, `I35`, `I36` | Comparator correctness/fairness tests | M07–M09 |
| Canonical metric computation library | `I33`, `I34`, `I35`, `I36` | Golden numerical tests for every registered metric | M07–M09 |
| Paired statistical analysis library | `I33`, `I34`, `I35`, `I36` | BCa/randomization/equivalence/Holm/materiality golden tests | M07–M09 |
| Mechanical claim-criterion evaluators | `I33`, `I34`, `I35`, `I36` | Threshold/materiality/family and decision-rule tests | M09 |
| Prediction, metric, paired-comparison and statistical metadata schemas | `I33`, `I34`, `I35`, `I36` | Typed schema/provenance validation | M07–M09 |
All roadmap-required deliverables for this milestone must appear in this table or remain explicitly traceable through the Roadmap Coverage Inventory. Source issue references remain `—` until real implementation issues exist.
## Entry Criteria
Implementation of this milestone may begin only when all applicable conditions below are true:
- all required upstream milestones are complete;
- every required upstream milestone audit is `PASS`;
- every required upstream artifact, interface, schema, manifest, or authoritative input exists;
- every consumed dependency passes its applicable validation and is provenance-compatible where provenance applies;
- M05 exposes canonical method/action/confirmation outputs and exact/non-exact solver metadata;
- the statistical/claim-criterion configuration from M01 is fixed before evaluation code or result-dependent analysis begins;
- all roadmap requirements owned by this milestone remain present in the Roadmap Coverage Inventory with unchanged IDs unless the authoritative inventory itself has been updated;
- after issue decomposition, every mandatory implementation-bearing requirement is mapped to at least one real milestone issue with explicit verification evidence;
- no blocking requirement is `AMBIGUOUS` or `BLOCKED`;
- no unresolved roadmap ambiguity would force an implementer to invent a scientific, mathematical, methodological, numerical, architectural, configuration, artifact, or execution decision.
## Exit Criteria
The milestone is complete only when all applicable conditions below are true:
- every implementation-bearing requirement owned by this milestone is satisfied;
- every applicable conditional requirement is satisfied;
- every governing `NON_IMPLEMENTATION` constraint assigned to this milestone is preserved;
- every mapped implementation issue is closed;
- all comparator implementations pass correctness/fairness checks and all canonical metrics/statistics match golden fixtures;
- prediction/metric/comparison/statistical schemas and mechanical claim-criterion evaluators validate without reimplementing scientific definitions downstream;
- all required unit, integration, numerical, structural, schema, CLI/E2E, scientific, statistical, provenance, and reproducibility validations applicable to this milestone pass;
- all required deliverables are generated and all required artifacts, interfaces, schemas, and manifests pass validation;
- required provenance is complete and valid, and no required evidence is stale or incompatible with its material dependencies;
- the milestone audit is `PASS` with no unresolved blocking finding.
## Acceptance Evidence
| Evidence Area | Required Evidence | Pass Condition |
|---|---|---|
| Requirement coverage | Roadmap Coverage Inventory + milestone allocation | Every allocated requirement is accounted for exactly once; implementation-bearing requirements have completed issue evidence and governing constraints remain preserved. |
| Implementation | Closed real milestone issues linked to exact implementation-bearing requirements | Every implementation-bearing requirement has completed implementation evidence. |
| Comparators | Correctness/fairness validation suite | All registered baselines/oracles use allowed resources and no favorable compute/checkpoint advantage. |
| Metrics | Golden metric fixtures | Every canonical metric formula and aggregation scope matches expected values. |
| Statistics | Golden/exhaustive statistical fixtures | BCa, randomization, equivalence, materiality, Holm, missingness and family registration match the roadmap. |
| Schemas | Prediction/metric/comparison/statistical schema checks | All canonical fields, identities and provenance links validate. |
| Deliverables | Required milestone outputs and artifacts | Complete, readable, schema-valid where applicable, and consistent with the roadmap. |
| Provenance | Required manifests, dependency identity, checksums, compatibility and staleness evidence | Complete and sufficient to verify origin, compatibility, reuse/invalidation and freshness where applicable. |
| Audit | Milestone audit issue | Final result is `PASS` with no unresolved blocking findings. |
## Milestone Audit
**Audit issue:** `—`

**Status:** `PENDING`

The milestone audit is the final completion gate. It must independently verify:

- complete roadmap coverage for all requirements owned by the milestone;
- exact requirement-to-issue traceability for implementation-bearing requirements;
- correct treatment of `NON_IMPLEMENTATION` constraints;
- closure of every mandatory implementation issue;
- completion and passing status of all required tests;
- completion and passing status of all required validations;
- existence and validity of all required deliverables;
- completeness and validity of required provenance and manifests;
- absence of stale or incompatible evidence;
- absence of unresolved blocking findings;
- readiness of milestone outputs for all declared downstream consumers.

The audit must end in exactly one result:

- `PASS` — every completion condition is satisfied.
- `FAIL` — one or more blocking conditions remain.

A milestone is not complete until the audit result is `PASS`.
## Scope Boundary
- This milestone implements only the implementation-bearing roadmap requirements explicitly allocated to it and preserves only the governing `NON_IMPLEMENTATION` constraints allocated to it.
- The authoritative roadmap remains the source of scientific, mathematical, methodological, architectural, numerical, configuration, artifact, execution, statistical, reporting, and claim requirements.
- This milestone may organize implementation work but may not redefine, weaken, strengthen, extend, or silently reinterpret the roadmap.
- Detailed implementation checklists belong in future implementation issues; no issue has been invented in this document.
- Detailed verification checklists belong in the future milestone audit issue; no audit issue has been invented in this document.
- Work outside this milestone's explicit requirement allocation must not be added unless the authoritative roadmap or Roadmap Coverage Inventory is explicitly updated first.

---

# M07 — Pre-Confirmatory Validation and Solver Benchmarking
> **Outcome:** All primitive, theorem, coupling, data/resource, model/response, baseline/oracle, solver, and controlled map-action validation gates produce reusable verified evidence required before claim-bearing transfer experiments.
## At a Glance
| Field | Value |
|---|---|
| Roadmap scope | §§14–15 through Exact Map-Value Bound Validation; §15 dependency map |
| Requirement ownership | REQ-1364–REQ-1424, REQ-1459–REQ-1572, REQ-1751–REQ-1795 |
| Allocated requirement count | `220` total (`219` implementation-bearing; `1` `NON_IMPLEMENTATION`) |
| Upstream milestones | M03, M04, M05, M06 |
| Implementation issues | `I37`, `I38`, `I39`, `I40`, `I41`, `I42` |
| Coverage authority | Roadmap Coverage Inventory |
| Audit issue | None — no dedicated per-milestone audit issue; verified via `Milestone Audit.md` |
| Audit status | `PENDING` |
## Coverage
The Roadmap Coverage Inventory is the traceability authority for this milestone. The work packages below partition this milestone's allocation; no requirement listed in another milestone is owned here.
| Roadmap Section | Scope / Work Package | Requirement IDs | Implementation Issue(s) | Verification / Evidence |
|---|---|---|---|---|
| §14 | Synthetic generators, exhaustive fixtures, coupling/map-action worlds, scalability instances, and deterministic serialization behavior | REQ-1364–REQ-1424 | `I37` | Generator/property tests consume the authoritative configured values and reproduce exact fixture classes, truth labels, generation semantics, and serialization invariants. |
| §15 | Experiment catalogue execution contract and pre-confirmatory experiment membership | REQ-1459–REQ-1462 | `I38` | Experiment catalogue/plan output materializes the registered execution semantics without duplicate or unregistered cells. |
| §15 — Mathematical Primitive / Exact Sparse Theorem / Coupling and Map-Bound Validation | Primitive, exhaustive exactness, and coupling/map-bound validation | REQ-1463–REQ-1493 | `I39` | Completed validation manifests prove numerical primitives, exhaustive orbit/QAP truth agreement, coupling inequalities, and map-bound fixtures pass. |
| §15 — Dataset, Client, Strict-Resource / Base-Model / Source-Response validations | Real-data readiness, base-model pilot, response-estimator pilot, and final response-band validation | REQ-1494–REQ-1519 | `I40` | Validation/pilot artifacts prove dataset/resource admissibility, deterministic model selection, response-estimator selection, and final packet-band correctness. |
| §15 — Baseline and Oracle Correctness / Exact-Sparse Solver Benchmark | Comparator/oracle correctness and exact-sparse benchmark | REQ-1520–REQ-1539 | `I41` | Correctness certificates and benchmark manifests demonstrate baseline semantics, oracle ACL separation, exact truth agreement, counters, runtime/memory, and valid solver certificates. |
| §15 — Synthetic / Real-Packet Coupling-Mechanism Validation | Designed and real-packet coupling mechanism evidence | REQ-1540–REQ-1555 | `I42` | Per-instance/per-pair coupling artifacts compute orbit versus rectangular behavior and registered destruction/mechanism checks. |
| §15 — Common Action / Robust Compromise / Map-Dependent Boundary / Exact Map-Value Bound | Controlled action-certifiability and map-value validation | REQ-1556–REQ-1572 | `I42` | Controlled fixtures demonstrate common-optimum, robust-compromise, map-dependent failure, and exact map-value-bound behavior with exact solver truth. |
| §15 — Experiment dependency and artifact map | Canonical dependencies, outputs, and downstream exposure for all pre-confirmatory validation experiments | REQ-1751–REQ-1795 | `I38` | Dependency-map/manifests prove every listed experiment consumes and emits exactly its registered immutable artifacts with correct downstream exposure. |
### Coverage Rules

- Every requirement listed in this milestone remains traceable to its exact Roadmap Coverage Inventory ID.
- Every mandatory implementation-bearing requirement must map to at least one real implementation issue before implementation of this milestone begins.
- Every conditional requirement remains traceable and must be implemented only when its roadmap-defined condition applies.
- `NON_IMPLEMENTATION` requirements are governing constraints only; they must remain traceable but must not be converted into fictitious implementation work.
- Every mapped implementation-bearing requirement must have objective verification or evidence.
- Every implementation issue must reference the exact requirement IDs it satisfies.
- A requirement is not considered covered merely because it falls inside a roadmap section associated with the milestone; the explicit requirement IDs in this document control milestone allocation.
- The current inventory-wide `UNMAPPED` state is expected before issue creation. No blocking requirement owned by this milestone may remain issue-unmapped when implementation of the milestone begins.
- No issue may redefine, weaken, silently reinterpret, duplicate, or extend the authoritative roadmap requirement it implements.
## Dependencies
### Milestone Dependencies
| Milestone | Required Input / Contract | Entry Gate |
|---|---|---|
| M02 — Deterministic Execution, Artifact, and Provenance Backbone | Canonical execution/provenance/readiness substrate consumed directly by `I37`/`I38` | Complete + audit PASS |
| M03 — Data Preparation and Strict Resource Interface | Validated real-data inputs and strict-resource evidence | Complete + audit PASS |
| M04 — Local Models and Procedural Response Packets | Base-model and response-estimation capabilities | Complete + audit PASS |
| M05 — Robust Correspondence, Selection, and Confirmation Engine | Exact/dense solver, action, map-value and confirmation capabilities | Complete + audit PASS |
| M06 — Comparator, Metric, and Statistical Evaluation Framework | Baseline correctness framework, metrics and canonical evaluation/statistical schemas | Complete + audit PASS |
### Artifact / Interface Dependencies
| Dependency | Produced By | Required Validation |
|---|---|---|
| Synthetic generator and experiment configuration | M01 | Registered seeds/grids/fixture controls and experiment settings validate. |
| Prepared real-data and source/target model artifacts | M03 / M04 | Eligibility, strict resources, checkpoint and response-packet provenance validate. |
| Solver/comparator/metric interfaces | M05 / M06 | Exactness/certificate interfaces and metric/comparator contracts validate. |
Dependency completion alone is not sufficient. Every consumed dependency must be present, valid, provenance-compatible where applicable, current under its dependency fingerprint, and compatible with the active roadmap contract.
## Implementation Issues
Implementation issues for this milestone are listed below; each issue's detailed task checklist, acceptance criteria, and required tests are defined in `Issues.md`.
| Order | Issue | Work Package | Roadmap Scope | Requirement Coverage | Depends On |
|---:|---|---|---|---|---|
| 1 | `I37` — Implement Deterministic Synthetic and Controlled Generators | Registered synthetic/exhaustive/coupling/map/scalability generators with deterministic serialization. | §14 | 61 atomic requirements | `I01`, `I07`, `I12`, `I25`, `I33` |
| 2 | `I38` — Materialize Pre-Confirmatory Experiment Catalogue and Dependency Map | Enumerated pre-confirmatory experiment cells, canonical dependencies, and output exposure contracts. | §15; §15 — Experiment dependency and artifact map | 49 atomic requirements | `I03`, `I13`, `I36`, `I37` |
| 3 | `I39` — Validate Mathematical Primitives, Exact-Sparse Theorem, Coupling, and Map Bounds | Completed primitive/exhaustive exactness/coupling/map-bound validation evidence. | §15 — Mathematical Primitive / Exact Sparse Theorem / Coupling and Map-Bound Validation | 31 atomic requirements | `I27`, `I28`, `I33`, `I37`, `I38` |
| 4 | `I40` — Validate Real Data, Local Models, and Source-Response Bands | Dataset/resource readiness, base-model pilot, response-estimator pilot, and final band validation evidence. | §15 — Dataset, Client, Strict-Resource / Base-Model / Source-Response validations | 26 atomic requirements | `I18`, `I24`, `I38` |
| 5 | `I41` — Validate Baselines, Oracles, and Exact-Sparse Solver Work Structure | Comparator/oracle correctness and exact-sparse benchmark evidence. | §15 — Baseline and Oracle Correctness / Exact-Sparse Solver Benchmark | 20 atomic requirements | `I27`, `I28`, `I29`, `I34`, `I38` |
| 6 | `I42` — Validate Coupling Mechanism, Action Certifiability, and Map-Value Boundaries | Synthetic/real coupling and unresolved-map controlled-world evidence with exact map-value boundaries. | §15 — Synthetic / Real-Packet Coupling-Mechanism Validation; §15 — Common Action / Robust Compromise / Map-Dependent Boundary / Exact Map-Value Bound | 33 atomic requirements | `I33`, `I37`, `I38`, `I39`, `I41` |
### Issue Contract

Every future milestone issue must:

- reference its exact roadmap section(s);
- list every covered implementation-bearing requirement ID;
- contain a detailed implementation checklist;
- define objective acceptance criteria;
- identify required tests;
- identify required artifacts, outputs, or interfaces;
- identify required provenance or manifest updates where applicable;
- identify explicit dependencies on upstream milestones, issues, artifacts, or interfaces;
- preserve roadmap terminology and semantics;
- avoid converting `NON_IMPLEMENTATION` constraints into artificial implementation tasks;
- close only when every mapped requirement and acceptance criterion is satisfied.
## Deliverables
| Deliverable | Source Issue(s) | Required Validation | Downstream Consumer |
|---|---|---|---|
| Deterministic synthetic/theorem/coupling/map-action fixture corpus | `I37`, `I38`, `I39`, `I40`, `I41`, `I42` | Generator/property/serialization validation | M08, M09 |
| Primitive, exact-sparse theorem and coupling/map-bound validation evidence | `I37`, `I38`, `I39`, `I40`, `I41`, `I42` | Registered numerical/exhaustive pass criteria | M08, M09 |
| Dataset/resource/model/response readiness validation evidence | `I37`, `I38`, `I39`, `I40`, `I41`, `I42` | All real-data pilot/readiness gates pass | M08 |
| Baseline/oracle correctness certificates | `I37`, `I38`, `I39`, `I40`, `I41`, `I42` | Comparator correctness and oracle ACL validation | M08, M09 |
| Exact-sparse solver benchmark artifacts | `I37`, `I38`, `I39`, `I40`, `I41`, `I42` | Exactness, counters, certificates, runtime/memory evidence | M08, M09 |
| Coupling and unresolved-map action-certifiability validation artifacts | `I37`, `I38`, `I39`, `I40`, `I41`, `I42` | Synthetic/real mechanism and map-value pass criteria | M09 |
All roadmap-required deliverables for this milestone must appear in this table or remain explicitly traceable through the Roadmap Coverage Inventory. Source issue references remain `—` until real implementation issues exist.
## Entry Criteria
Implementation of this milestone may begin only when all applicable conditions below are true:
- all required upstream milestones are complete;
- every required upstream milestone audit is `PASS`;
- every required upstream artifact, interface, schema, manifest, or authoritative input exists;
- every consumed dependency passes its applicable validation and is provenance-compatible where provenance applies;
- M03–M06 integration paths are complete enough to execute validation cells without bypassing strict resources, canonical artifacts, solver certificates, comparator contracts, metrics or statistics;
- all registered synthetic and validation experiment configuration is fixed before any result-bearing validation run;
- all roadmap requirements owned by this milestone remain present in the Roadmap Coverage Inventory with unchanged IDs unless the authoritative inventory itself has been updated;
- after issue decomposition, every mandatory implementation-bearing requirement is mapped to at least one real milestone issue with explicit verification evidence;
- no blocking requirement is `AMBIGUOUS` or `BLOCKED`;
- no unresolved roadmap ambiguity would force an implementer to invent a scientific, mathematical, methodological, numerical, architectural, configuration, artifact, or execution decision.
## Exit Criteria
The milestone is complete only when all applicable conditions below are true:
- every implementation-bearing requirement owned by this milestone is satisfied;
- every applicable conditional requirement is satisfied;
- every governing `NON_IMPLEMENTATION` constraint assigned to this milestone is preserved;
- every mapped implementation issue is closed;
- every pre-confirmatory validation experiment required as a downstream prerequisite has its registered completed artifact set and passing validation status;
- all validation dependency-map references point to canonical immutable artifacts and no failed prerequisite is silently bypassed;
- all required unit, integration, numerical, structural, schema, CLI/E2E, scientific, statistical, provenance, and reproducibility validations applicable to this milestone pass;
- all required deliverables are generated and all required artifacts, interfaces, schemas, and manifests pass validation;
- required provenance is complete and valid, and no required evidence is stale or incompatible with its material dependencies;
- the milestone audit is `PASS` with no unresolved blocking finding.
## Acceptance Evidence
| Evidence Area | Required Evidence | Pass Condition |
|---|---|---|
| Requirement coverage | Roadmap Coverage Inventory + milestone allocation | Every allocated requirement is accounted for exactly once; implementation-bearing requirements have completed issue evidence and governing constraints remain preserved. |
| Implementation | Closed real milestone issues linked to exact implementation-bearing requirements | Every implementation-bearing requirement has completed implementation evidence. |
| Synthetic/theorem validation | Generator and exhaustive validation artifacts | Primitive, theorem, coupling/map-bound and exactness criteria pass. |
| Real-data readiness | Dataset/resource/model/response pilot validation manifests | All required real-data prerequisites are valid before claim-bearing transfer. |
| Solver/baseline correctness | Correctness certificates and benchmark artifacts | Exact truth, oracle ACL, counters/certificates and registered benchmark outputs validate. |
| Dependency map | Validation experiment manifests | Every validation experiment consumes/produces only its registered canonical artifacts. |
| Deliverables | Required milestone outputs and artifacts | Complete, readable, schema-valid where applicable, and consistent with the roadmap. |
| Provenance | Required manifests, dependency identity, checksums, compatibility and staleness evidence | Complete and sufficient to verify origin, compatibility, reuse/invalidation and freshness where applicable. |
| Audit | Milestone audit issue | Final result is `PASS` with no unresolved blocking findings. |
## Milestone Audit
**Audit issue:** `—`

**Status:** `PENDING`

The milestone audit is the final completion gate. It must independently verify:

- complete roadmap coverage for all requirements owned by the milestone;
- exact requirement-to-issue traceability for implementation-bearing requirements;
- correct treatment of `NON_IMPLEMENTATION` constraints;
- closure of every mandatory implementation issue;
- completion and passing status of all required tests;
- completion and passing status of all required validations;
- existence and validity of all required deliverables;
- completeness and validity of required provenance and manifests;
- absence of stale or incompatible evidence;
- absence of unresolved blocking findings;
- readiness of milestone outputs for all declared downstream consumers.

The audit must end in exactly one result:

- `PASS` — every completion condition is satisfied.
- `FAIL` — one or more blocking conditions remain.

A milestone is not complete until the audit result is `PASS`.
## Scope Boundary
- This milestone implements only the implementation-bearing roadmap requirements explicitly allocated to it and preserves only the governing `NON_IMPLEMENTATION` constraints allocated to it.
- The authoritative roadmap remains the source of scientific, mathematical, methodological, architectural, numerical, configuration, artifact, execution, statistical, reporting, and claim requirements.
- This milestone may organize implementation work but may not redefine, weaken, strengthen, extend, or silently reinterpret the roadmap.
- Detailed implementation checklists belong in future implementation issues; no issue has been invented in this document.
- Detailed verification checklists belong in the future milestone audit issue; no audit issue has been invented in this document.
- Work outside this milestone's explicit requirement allocation must not be added unless the authoritative roadmap or Roadmap Coverage Inventory is explicitly updated first.

---

# M08 — Confirmatory Transfer, Robustness, and Applicability Experiments
> **Outcome:** The full registered claim-bearing and boundary experiment matrix executes from validated prerequisites, covering primary/secondary transfer, source selection, ablations, sparsity, confirmation, robustness frontiers, map applicability, and scalability with verified artifacts.
## At a Glance
| Field | Value |
|---|---|
| Roadmap scope | §§1.3, 4.26, 15 from Primary Strict Cross-Telemetry Transfer through Scalability and Efficiency |
| Requirement ownership | REQ-0020–REQ-0021, REQ-0791–REQ-0798, REQ-1573–REQ-1728, REQ-1796–REQ-1825, REQ-1842–REQ-1844, REQ-3184 |
| Allocated requirement count | `200` total (`200` implementation-bearing; `0` `NON_IMPLEMENTATION`) |
| Upstream milestones | M07 |
| Implementation issues | `I43`, `I44`, `I45`, `I46`, `I47`, `I48`, `I49` |
| Coverage authority | Roadmap Coverage Inventory |
| Audit issue | None — no dedicated per-milestone audit issue; verified via `Milestone Audit.md` |
| Audit status | `PENDING` |
## Coverage
The Roadmap Coverage Inventory is the traceability authority for this milestone. The work packages below partition this milestone's allocation; no requirement listed in another milestone is owned here.
| Roadmap Section | Scope / Work Package | Requirement IDs | Implementation Issue(s) | Verification / Evidence |
|---|---|---|---|---|
| §1.3 | Sparse-support and dense-support sensitivity obligations | REQ-0020–REQ-0021 | `I43` | Robustness manifests demonstrate exactly the registered sparse support alternatives and Dense-CCP sensitivity path. |
| §4.26 | Reference-GPU efficiency execution behavior | REQ-0791–REQ-0798 | `I44` | Efficiency manifests prove runs use and record the configured reference hardware and required comparable execution conditions. |
| §15 — Primary Strict Cross-Telemetry Transfer / Multi-Source Selection Validation | Primary transfer and multi-source confirmatory experiments | REQ-1573–REQ-1600 | `I45` | Completed cell manifests cover exact pair×seed×method membership, valid strict-resource inputs, actions/confirmations, TEST predictions, and metrics. |
| §15 — Mechanism Ablations / Sparsity and Dense Fallback | Mechanism ablations and sparse/dense sensitivity | REQ-1601–REQ-1620 | `I43` | Ablation/robustness artifacts isolate each registered mechanism and support condition while reusing dependency-identical principal cells. |
| §15 — Target Confirmation and Portability / Secondary Cross-Modality Generalization | Confirmation safety/portability and secondary generalization | REQ-1621–REQ-1636, REQ-1842 | `I46` | Experiment manifests cover exact primary/secondary pair membership, methods, all ten confirmatory seeds, and scope-qualified secondary evidence. |
| §15 — Semantic Sufficiency Frontier / Weak-Signal, Support, and Heterogeneity Boundaries | Semantic-partition and one-factor-at-a-time failure-boundary studies | REQ-1637–REQ-1670, REQ-1843–REQ-1844 | `I47` | Boundary manifests implement exact condition grids, deterministic support subsampling/packet transforms, eligibility semantics, and all ten seeds. |
| §15 — Map-Availability Applicability Audit | Packet-only recovery and controlled two-researcher human map-availability audit | REQ-1671–REQ-1710, REQ-3184 | `I48` | CLI/E2E plus audit-schema checks enforce missing-template blocking, timed sessions, resource/oracle ACLs, submission hashes, exact comparison gating, and applicability classification. |
| §15 — Scalability and Efficiency | Synthetic and real solver scalability/work-structure study | REQ-1711–REQ-1728 | `I44` | Timing/counter artifacts cover registered cell counts, warmups/repetitions, timeouts/resource limits, exact work counters, and descriptive trend criteria. |
| §15 — Experiment dependency and artifact map | Canonical dependencies, outputs, and downstream exposure for confirmatory/robustness experiments | REQ-1796–REQ-1825 | `I49` | Experiment manifests reference exactly the registered validated inputs and emit canonical action, prediction, metric, timing, audit, and diagnostic artifacts. |
### Coverage Rules

- Every requirement listed in this milestone remains traceable to its exact Roadmap Coverage Inventory ID.
- Every mandatory implementation-bearing requirement must map to at least one real implementation issue before implementation of this milestone begins.
- Every conditional requirement remains traceable and must be implemented only when its roadmap-defined condition applies.
- `NON_IMPLEMENTATION` requirements are governing constraints only; they must remain traceable but must not be converted into fictitious implementation work.
- Every mapped implementation-bearing requirement must have objective verification or evidence.
- Every implementation issue must reference the exact requirement IDs it satisfies.
- A requirement is not considered covered merely because it falls inside a roadmap section associated with the milestone; the explicit requirement IDs in this document control milestone allocation.
- The current inventory-wide `UNMAPPED` state is expected before issue creation. No blocking requirement owned by this milestone may remain issue-unmapped when implementation of the milestone begins.
- No issue may redefine, weaken, silently reinterpret, duplicate, or extend the authoritative roadmap requirement it implements.
## Dependencies
### Milestone Dependencies
| Milestone | Required Input / Contract | Entry Gate |
|---|---|---|
| M07 — Pre-Confirmatory Validation and Solver Benchmarking | All prerequisite validation gates, correctness certificates, solver benchmarks, and reusable validation artifacts | Complete + audit PASS |
### Artifact / Interface Dependencies
| Dependency | Produced By | Required Validation |
|---|---|---|
| Valid pair-seed/resource manifest and response packets | M03 / M04 / M07 | Eligibility, strict-resource, seed, freshness and provenance checks pass. |
| Validated solver/comparator/confirmation stack | M05 / M06 / M07 | Pre-confirmatory exactness/correctness gates pass. |
| Principal and boundary experiment configuration | M01 | Registered cell grids, support levels, seeds, hardware and audit controls validate. |
| Human-audit input templates/submissions when applicable | Researcher input through registered audit interface | Schema, timing, resource-access, hash and oracle-gating validation passes before comparison. |
Dependency completion alone is not sufficient. Every consumed dependency must be present, valid, provenance-compatible where applicable, current under its dependency fingerprint, and compatible with the active roadmap contract.
## Implementation Issues
Implementation issues for this milestone are listed below; each issue's detailed task checklist, acceptance criteria, and required tests are defined in `Issues.md`.
| Order | Issue | Work Package | Roadmap Scope | Requirement Coverage | Depends On |
|---:|---|---|---|---|---|
| 1 | `I43` — Run Mechanism Ablations and Sparse/Dense Sensitivity | Registered mechanism ablations plus principal sparse-support alternatives and Dense-CCP sensitivity evidence. | §1.3; §15 — Mechanism Ablations / Sparsity and Dense Fallback | 22 atomic requirements | `I29`, `I42` |
| 2 | `I44` — Run Reference-GPU Scalability, Efficiency, and Work-Structure Study | Reference-hardware solver scalability, active-image/LAP work counters, runtime, and efficiency evidence. | §4.26; §15 — Scalability and Efficiency | 26 atomic requirements | `I02`, `I27`, `I29`, `I41`, `I42` |
| 3 | `I45` — Run Primary Strict Cross-Telemetry Transfer and Multi-Source Validation | Primary directed-pair confirmatory transfer and multi-source selection evidence under strict resources. | §15 — Primary Strict Cross-Telemetry Transfer / Multi-Source Selection Validation | 28 atomic requirements | `I31`, `I32`, `I34`, `I35`, `I36`, `I42` |
| 4 | `I46` — Run Target Confirmation, Portability, and Secondary Generalization | Confirmation-safety/portability and secondary cross-modality generalization evidence. | §15 — Target Confirmation and Portability / Secondary Cross-Modality Generalization | 17 atomic requirements | `I32`, `I43`, `I45` |
| 5 | `I47` — Run Semantic Sufficiency, Weak-Signal, Support, and Heterogeneity Boundaries | Registered semantic-partition frontier and one-factor-at-a-time applicability boundary evidence. | §15 — Semantic Sufficiency Frontier / Weak-Signal, Support, and Heterogeneity Boundaries | 36 atomic requirements | `I45` |
| 6 | `I48` — Run Map-Availability Applicability Audit | Packet-only recovery and controlled two-researcher map-availability audit evidence. | §15 — Map-Availability Applicability Audit | 41 atomic requirements | `I42`, `I45` |
| 7 | `I49` — Finalize Confirmatory Experiment Dependencies and Evidence Exposure | Canonical dependency/output/downstream-exposure map for all confirmatory and robustness experiments. | §15 — Experiment dependency and artifact map | 30 atomic requirements | `I43`, `I44`, `I45`, `I46`, `I47`, `I48` |
### Issue Contract

Every future milestone issue must:

- reference its exact roadmap section(s);
- list every covered implementation-bearing requirement ID;
- contain a detailed implementation checklist;
- define objective acceptance criteria;
- identify required tests;
- identify required artifacts, outputs, or interfaces;
- identify required provenance or manifest updates where applicable;
- identify explicit dependencies on upstream milestones, issues, artifacts, or interfaces;
- preserve roadmap terminology and semantics;
- avoid converting `NON_IMPLEMENTATION` constraints into artificial implementation tasks;
- close only when every mapped requirement and acceptance criterion is satisfied.
## Deliverables
| Deliverable | Source Issue(s) | Required Validation | Downstream Consumer |
|---|---|---|---|
| Primary strict cross-telemetry transfer and multi-source result artifacts | `I43`, `I44`, `I45`, `I46`, `I47`, `I48`, `I49` | Complete registered pair×seed×method cells with valid metrics/provenance | M09 |
| Mechanism ablation and sparse/dense sensitivity evidence | `I43`, `I44`, `I45`, `I46`, `I47`, `I48`, `I49` | Registered conditions completed; principal reuse identity validated | M09 |
| Target-confirmation safety/portability and secondary-generalization evidence | `I43`, `I44`, `I45`, `I46`, `I47`, `I48`, `I49` | Registered primary/secondary scope and seed completeness checks | M09 |
| Semantic-sufficiency and weak-signal/support/heterogeneity boundary evidence | `I43`, `I44`, `I45`, `I46`, `I47`, `I48`, `I49` | Exact condition-grid and expected ineligibility/failure semantics | M09 |
| Map-availability applicability audit evidence | `I43`, `I44`, `I45`, `I46`, `I47`, `I48`, `I49` | Researcher submission schema/timing/resource/hash/oracle-gating checks | M09 |
| Scalability and efficiency evidence | `I43`, `I44`, `I45`, `I46`, `I47`, `I48`, `I49` | Registered timing/counter cells and work-structure/trend validation | M09 |
All roadmap-required deliverables for this milestone must appear in this table or remain explicitly traceable through the Roadmap Coverage Inventory. Source issue references remain `—` until real implementation issues exist.
## Entry Criteria
Implementation of this milestone may begin only when all applicable conditions below are true:
- all required upstream milestones are complete;
- every required upstream milestone audit is `PASS`;
- every required upstream artifact, interface, schema, manifest, or authoritative input exists;
- every consumed dependency passes its applicable validation and is provenance-compatible where provenance applies;
- M07 validation gates required by each experiment are complete and `PASS`, including exactness/correctness prerequisites where applicable;
- principal, sensitivity, boundary, efficiency and audit experiment grids remain exactly those registered in the M01 configuration contract;
- all roadmap requirements owned by this milestone remain present in the Roadmap Coverage Inventory with unchanged IDs unless the authoritative inventory itself has been updated;
- after issue decomposition, every mandatory implementation-bearing requirement is mapped to at least one real milestone issue with explicit verification evidence;
- no blocking requirement is `AMBIGUOUS` or `BLOCKED`;
- no unresolved roadmap ambiguity would force an implementer to invent a scientific, mathematical, methodological, numerical, architectural, configuration, artifact, or execution decision.
## Exit Criteria
The milestone is complete only when all applicable conditions below are true:
- every implementation-bearing requirement owned by this milestone is satisfied;
- every applicable conditional requirement is satisfied;
- every governing `NON_IMPLEMENTATION` constraint assigned to this milestone is preserved;
- every mapped implementation issue is closed;
- every mandatory registered confirmatory, robustness, boundary, applicability and efficiency cell has a valid completed or roadmap-defined admissible terminal state;
- all experiment outputs, shared-artifact references, human-audit evidence where applicable, timing/counters, predictions and metrics validate and are ready for synthesis;
- all required unit, integration, numerical, structural, schema, CLI/E2E, scientific, statistical, provenance, and reproducibility validations applicable to this milestone pass;
- all required deliverables are generated and all required artifacts, interfaces, schemas, and manifests pass validation;
- required provenance is complete and valid, and no required evidence is stale or incompatible with its material dependencies;
- the milestone audit is `PASS` with no unresolved blocking finding.
## Acceptance Evidence
| Evidence Area | Required Evidence | Pass Condition |
|---|---|---|
| Requirement coverage | Roadmap Coverage Inventory + milestone allocation | Every allocated requirement is accounted for exactly once; implementation-bearing requirements have completed issue evidence and governing constraints remain preserved. |
| Implementation | Closed real milestone issues linked to exact implementation-bearing requirements | Every implementation-bearing requirement has completed implementation evidence. |
| Primary/secondary experiments | Completed experiment manifests and predictions/metrics | Registered pair×seed×method membership and scope restrictions are complete and valid. |
| Robustness/boundaries | Ablation/sparsity/semantic/weak-signal boundary artifacts | Exact registered condition grids and transformation/eligibility semantics are respected. |
| Map applicability | Human-audit templates/submissions + packet-only recovery artifacts | Timing/resource/hash/oracle-gating and classification rules pass; missing human input remains explicitly blocked. |
| Scalability/efficiency | Timing, memory, counters, timeout/resource-limit and trend artifacts | Registered cell counts/protocol and exact work-structure criteria validate. |
| Deliverables | Required milestone outputs and artifacts | Complete, readable, schema-valid where applicable, and consistent with the roadmap. |
| Provenance | Required manifests, dependency identity, checksums, compatibility and staleness evidence | Complete and sufficient to verify origin, compatibility, reuse/invalidation and freshness where applicable. |
| Audit | Milestone audit issue | Final result is `PASS` with no unresolved blocking findings. |
## Milestone Audit
**Audit issue:** `—`

**Status:** `PENDING`

The milestone audit is the final completion gate. It must independently verify:

- complete roadmap coverage for all requirements owned by the milestone;
- exact requirement-to-issue traceability for implementation-bearing requirements;
- correct treatment of `NON_IMPLEMENTATION` constraints;
- closure of every mandatory implementation issue;
- completion and passing status of all required tests;
- completion and passing status of all required validations;
- existence and validity of all required deliverables;
- completeness and validity of required provenance and manifests;
- absence of stale or incompatible evidence;
- absence of unresolved blocking findings;
- readiness of milestone outputs for all declared downstream consumers.

The audit must end in exactly one result:

- `PASS` — every completion condition is satisfied.
- `FAIL` — one or more blocking conditions remain.

A milestone is not complete until the audit result is `PASS`.
## Scope Boundary
- This milestone implements only the implementation-bearing roadmap requirements explicitly allocated to it and preserves only the governing `NON_IMPLEMENTATION` constraints allocated to it.
- The authoritative roadmap remains the source of scientific, mathematical, methodological, architectural, numerical, configuration, artifact, execution, statistical, reporting, and claim requirements.
- This milestone may organize implementation work but may not redefine, weaken, strengthen, extend, or silently reinterpret the roadmap.
- Detailed implementation checklists belong in future implementation issues; no issue has been invented in this document.
- Detailed verification checklists belong in the future milestone audit issue; no audit issue has been invented in this document.
- Work outside this milestone's explicit requirement allocation must not be added unless the authoritative roadmap or Roadmap Coverage Inventory is explicitly updated first.

---

# M09 — Statistical Synthesis, Claim Adjudication, and Manuscript Evidence
> **Outcome:** Verified experiment artifacts are synthesized with the registered statistical and falsification rules into bounded claim states, then rendered into manuscript-facing evidence without recomputation or scope expansion.
## At a Glance
| Field | Value |
|---|---|
| Roadmap scope | §§1.1, 1.5, 2, 4.28, 15 Statistical Synthesis/Claim-Evidence Adjudication, 16–17, 23 |
| Requirement ownership | REQ-0004–REQ-0005, REQ-0030–REQ-0063, REQ-0822–REQ-0826, REQ-1729–REQ-1749, REQ-1826–REQ-1831, REQ-1845–REQ-1908, REQ-1923–REQ-1989, REQ-3048–REQ-3092 |
| Allocated requirement count | `244` total (`175` implementation-bearing; `69` `NON_IMPLEMENTATION`) |
| Upstream milestones | M06, M07, M08 |
| Implementation issues | `I50`, `I51`, `I52`, `I53` |
| Coverage authority | Roadmap Coverage Inventory |
| Audit issue | None — no dedicated per-milestone audit issue; verified via `Milestone Audit.md` |
| Audit status | `PENDING` |
## Coverage
The Roadmap Coverage Inventory is the traceability authority for this milestone. The work packages below partition this milestone's allocation; no requirement listed in another milestone is owned here.
| Roadmap Section | Scope / Work Package | Requirement IDs | Implementation Issue(s) | Verification / Evidence |
|---|---|---|---|---|
| §§1.1, 1.5; §2 | Contribution identity, strict-scope constraints, forbidden claims, research questions, hypotheses, and falsification boundaries | REQ-0004–REQ-0005, REQ-0030–REQ-0063 | `I52` | Claim catalogue and scope tests preserve exact allowed/excluded wording and encode every registered support/falsification condition mechanically. |
| §4.28 | Reporting precision and presentation behavior | REQ-0822–REQ-0826 | `I53` | Renderer golden tests enforce configured decimal/threshold formatting without changing source scientific values. |
| §15 — Statistical Synthesis / Claim-Evidence Adjudication | Registered statistical synthesis and final evidence adjudication workflows | REQ-1729–REQ-1749 | `I50` | Synthesis artifacts contain paired contrasts, BCa CIs, exact randomization tests, Holm/equivalence/materiality/completeness results, followed by immutable claim-state outputs. |
| §15 — Experiment dependency and artifact map | Synthesis/adjudication dependency, output, and downstream-exposure contracts | REQ-1826–REQ-1831 | `I50` | Dependency manifests prove synthesis consumes only verified metrics/comparisons and adjudication consumes verified synthesis plus the claim catalogue. |
| §16 | Allowed claim states, evidence sufficiency, manuscript wording, and scope restriction | REQ-1845–REQ-1908 | `I52` | Claim-adjudication tests map evidence to only registered states and mechanically constrain manuscript wording to the adjudicated empirical scope. |
| §17 | Simplification/kill-rule behavior and non-neutralizable falsification semantics | REQ-1923–REQ-1989 | `I51` | Kill-rule fixtures prove configured thresholds, comparator outcomes, failure classifications, and prohibition on post-result threshold/reclassification changes. |
| §23 | Verified results publication and manuscript-facing tables/figures/source data | REQ-3048–REQ-3092 | `I53` | `fedorbit report` tests prove read-only scientific consumption, schema-valid results outputs, shared verified sources for tables/figures, and no scientific recomputation. |
### Coverage Rules

- Every requirement listed in this milestone remains traceable to its exact Roadmap Coverage Inventory ID.
- Every mandatory implementation-bearing requirement must map to at least one real implementation issue before implementation of this milestone begins.
- Every conditional requirement remains traceable and must be implemented only when its roadmap-defined condition applies.
- `NON_IMPLEMENTATION` requirements are governing constraints only; they must remain traceable but must not be converted into fictitious implementation work.
- Every mapped implementation-bearing requirement must have objective verification or evidence.
- Every implementation issue must reference the exact requirement IDs it satisfies.
- A requirement is not considered covered merely because it falls inside a roadmap section associated with the milestone; the explicit requirement IDs in this document control milestone allocation.
- The current inventory-wide `UNMAPPED` state is expected before issue creation. No blocking requirement owned by this milestone may remain issue-unmapped when implementation of the milestone begins.
- No issue may redefine, weaken, silently reinterpret, duplicate, or extend the authoritative roadmap requirement it implements.
## Dependencies
### Milestone Dependencies
| Milestone | Required Input / Contract | Entry Gate |
|---|---|---|
| M06 — Comparator, Metric, and Statistical Evaluation Framework | Canonical metrics, paired comparisons, statistical procedures and claim criteria | Complete + audit PASS |
| M07 — Pre-Confirmatory Validation and Solver Benchmarking | Validated theorem/mechanism/correctness evidence used by claims | Complete + audit PASS |
| M08 — Confirmatory Transfer, Robustness, and Applicability Experiments | Completed valid confirmatory, boundary, applicability and efficiency evidence | Complete + audit PASS |
### Artifact / Interface Dependencies
| Dependency | Produced By | Required Validation |
|---|---|---|
| Verified metric/comparison artifacts | M06 / M08 | Current, complete, schema-valid, provenance-compatible and not stale. |
| Pre-confirmatory theorem/mechanism validation artifacts | M07 | All required validation gates and truth/certificate checks pass. |
| Claim catalogue and falsification/kill-rule configuration | M01 / M09 | Exact registered wording, thresholds and claim-state rules validate. |
| Statistical synthesis artifacts | M09 | Complete registered contrasts, CIs, tests, adjustments, equivalence/materiality and completeness checks before adjudication. |
Dependency completion alone is not sufficient. Every consumed dependency must be present, valid, provenance-compatible where applicable, current under its dependency fingerprint, and compatible with the active roadmap contract.
## Implementation Issues
Implementation issues for this milestone are listed below; each issue's detailed task checklist, acceptance criteria, and required tests are defined in `Issues.md`.
| Order | Issue | Work Package | Roadmap Scope | Requirement Coverage | Depends On |
|---:|---|---|---|---|---|
| 1 | `I50` — Perform Statistical Synthesis and Claim-Evidence Aggregation | Registered paired statistical synthesis and adjudication-ready evidence dependency outputs. | §15 — Statistical Synthesis / Claim-Evidence Adjudication; §15 — Experiment dependency and artifact map | 27 atomic requirements | `I35`, `I36`, `I49` |
| 2 | `I51` — Implement Immutable Kill and Simplification Rules | Mechanical non-neutralizable kill/simplification rules over verified experiment evidence. | §17 | 67 atomic requirements | `I49`, `I50` |
| 3 | `I52` — Implement Claim Scope and Mechanical Claim-State Adjudication | Claim-state engine preserving contribution scope, research questions, hypotheses, falsification boundaries, evidence sufficiency, and allowed wording. | §§1.1, 1.5; §2; §16 | 100 atomic requirements | `I35`, `I42`, `I49`, `I50`, `I51` |
| 4 | `I53` — Generate Verified Publication Tables, Figures, and Manuscript Evidence | Precision-controlled manuscript-facing tables, figures, source data, and publication evidence from verified artifacts. | §4.28; §23 | 50 atomic requirements | `I36`, `I49`, `I52` |
### Issue Contract

Every future milestone issue must:

- reference its exact roadmap section(s);
- list every covered implementation-bearing requirement ID;
- contain a detailed implementation checklist;
- define objective acceptance criteria;
- identify required tests;
- identify required artifacts, outputs, or interfaces;
- identify required provenance or manifest updates where applicable;
- identify explicit dependencies on upstream milestones, issues, artifacts, or interfaces;
- preserve roadmap terminology and semantics;
- avoid converting `NON_IMPLEMENTATION` constraints into artificial implementation tasks;
- close only when every mapped requirement and acceptance criterion is satisfied.
## Deliverables
| Deliverable | Source Issue(s) | Required Validation | Downstream Consumer |
|---|---|---|---|
| Verified statistical synthesis artifacts | `I50`, `I51`, `I52`, `I53` | Registered contrasts, BCa CIs, exact tests, Holm, equivalence, materiality and completeness checks | Claim adjudication |
| Final claim-state and scope-adjudication artifact | `I50`, `I51`, `I52`, `I53` | Only registered states; supporting references and forbidden extrapolations present | `fedorbit report` / manuscript |
| Kill-rule and simplification outcomes | `I50`, `I51`, `I52`, `I53` | Configured falsification criteria evaluated without post-result mutation | Claim adjudication / manuscript |
| Manuscript-facing `results/` tables, figures, metrics/statistics and source data | `I50`, `I51`, `I52`, `I53` | Report-only rendering from verified sources; no scientific recomputation | Manuscript |
| Claim-scope and reporting validation evidence | `I50`, `I51`, `I52`, `I53` | Rendered wording/precision constrained to adjudicated state and roadmap claim boundaries | Milestone audit |
All roadmap-required deliverables for this milestone must appear in this table or remain explicitly traceable through the Roadmap Coverage Inventory. Source issue references remain `—` until real implementation issues exist.
## Entry Criteria
Implementation of this milestone may begin only when all applicable conditions below are true:
- all required upstream milestones are complete;
- every required upstream milestone audit is `PASS`;
- every required upstream artifact, interface, schema, manifest, or authoritative input exists;
- every consumed dependency passes its applicable validation and is provenance-compatible where provenance applies;
- all claim-relevant M07 and M08 experiment artifacts are complete or carry their roadmap-defined valid missing/failure state;
- no statistical synthesis or claim adjudication consumes stale, invalid, resource-violating or provenance-incompatible evidence;
- all roadmap requirements owned by this milestone remain present in the Roadmap Coverage Inventory with unchanged IDs unless the authoritative inventory itself has been updated;
- after issue decomposition, every mandatory implementation-bearing requirement is mapped to at least one real milestone issue with explicit verification evidence;
- no blocking requirement is `AMBIGUOUS` or `BLOCKED`;
- no unresolved roadmap ambiguity would force an implementer to invent a scientific, mathematical, methodological, numerical, architectural, configuration, artifact, or execution decision.
## Exit Criteria
The milestone is complete only when all applicable conditions below are true:
- every implementation-bearing requirement owned by this milestone is satisfied;
- every applicable conditional requirement is satisfied;
- every governing `NON_IMPLEMENTATION` constraint assigned to this milestone is preserved;
- every mapped implementation issue is closed;
- Statistical Synthesis is complete using only current valid registered evidence and all registered completeness/multiplicity/equivalence/materiality procedures have been applied;
- Claim-Evidence Adjudication has emitted one permitted state, scope, forbidden extrapolations and supporting artifact references for each governed claim;
- `fedorbit report` has produced manuscript-facing evidence exclusively from verified source artifacts, with no scientific recomputation and no scope overreach;
- all required unit, integration, numerical, structural, schema, CLI/E2E, scientific, statistical, provenance, and reproducibility validations applicable to this milestone pass;
- all required deliverables are generated and all required artifacts, interfaces, schemas, and manifests pass validation;
- required provenance is complete and valid, and no required evidence is stale or incompatible with its material dependencies;
- the milestone audit is `PASS` with no unresolved blocking finding.
## Acceptance Evidence
| Evidence Area | Required Evidence | Pass Condition |
|---|---|---|
| Requirement coverage | Roadmap Coverage Inventory + milestone allocation | Every allocated requirement is accounted for exactly once; implementation-bearing requirements have completed issue evidence and governing constraints remain preserved. |
| Implementation | Closed real milestone issues linked to exact implementation-bearing requirements | Every implementation-bearing requirement has completed implementation evidence. |
| Statistical synthesis | Paired contrasts, BCa CIs, exact tests, Holm, equivalence/materiality and completeness artifacts | All registered analyses are complete using only current valid evidence. |
| Claim adjudication | Claim-state/scope/evidence-reference artifacts | Only registered claim states are emitted and wording scope never exceeds adjudicated evidence. |
| Kill rules | Mechanical falsification/simplification test records | Configured rules fire exactly when conditions hold and cannot be neutralized post hoc. |
| Reporting | `fedorbit report` outputs and renderer golden tests | Tables/figures/source data come from the same verified source artifacts with no scientific recomputation. |
| Deliverables | Required milestone outputs and artifacts | Complete, readable, schema-valid where applicable, and consistent with the roadmap. |
| Provenance | Required manifests, dependency identity, checksums, compatibility and staleness evidence | Complete and sufficient to verify origin, compatibility, reuse/invalidation and freshness where applicable. |
| Audit | Milestone audit issue | Final result is `PASS` with no unresolved blocking findings. |
## Milestone Audit
**Audit issue:** `—`

**Status:** `PENDING`

The milestone audit is the final completion gate. It must independently verify:

- complete roadmap coverage for all requirements owned by the milestone;
- exact requirement-to-issue traceability for implementation-bearing requirements;
- correct treatment of `NON_IMPLEMENTATION` constraints;
- closure of every mandatory implementation issue;
- completion and passing status of all required tests;
- completion and passing status of all required validations;
- existence and validity of all required deliverables;
- completeness and validity of required provenance and manifests;
- absence of stale or incompatible evidence;
- absence of unresolved blocking findings;
- readiness of milestone outputs for all declared downstream consumers.

The audit must end in exactly one result:

- `PASS` — every completion condition is satisfied.
- `FAIL` — one or more blocking conditions remain.

A milestone is not complete until the audit result is `PASS`.
## Scope Boundary
- This milestone implements only the implementation-bearing roadmap requirements explicitly allocated to it and preserves only the governing `NON_IMPLEMENTATION` constraints allocated to it.
- The authoritative roadmap remains the source of scientific, mathematical, methodological, architectural, numerical, configuration, artifact, execution, statistical, reporting, and claim requirements.
- This milestone may organize implementation work but may not redefine, weaken, strengthen, extend, or silently reinterpret the roadmap.
- Detailed implementation checklists belong in future implementation issues; no issue has been invented in this document.
- Detailed verification checklists belong in the future milestone audit issue; no audit issue has been invented in this document.
- Work outside this milestone's explicit requirement allocation must not be added unless the authoritative roadmap or Roadmap Coverage Inventory is explicitly updated first.
