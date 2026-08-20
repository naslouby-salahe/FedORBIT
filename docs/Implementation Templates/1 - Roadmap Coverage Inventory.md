# Roadmap Coverage Inventory

## 1. Authority

**Roadmap:**  
**Repository:**  

This document is the canonical traceability index between the authoritative roadmap and GitHub implementation work.

It does not redefine, replace, weaken, extend, or reinterpret the roadmap.

---

## 2. Inventory Rules

1. Every roadmap obligation must appear in this inventory.
2. One row must represent one atomic, independently verifiable requirement.
3. Requirement IDs are immutable once referenced by a GitHub issue.
4. Every implementation-bearing requirement must map to at least one GitHub issue and one milestone.
5. Every implementation-bearing requirement must define concrete acceptance evidence.
6. Scope constraints, exclusions, rationale, and claim boundaries must remain traceable as `NON_IMPLEMENTATION` requirements even when they create no direct implementation task.
7. Requirement wording must preserve the roadmap meaning and must not introduce new scientific, mathematical, architectural, experimental, or execution decisions.
8. The roadmap remains authoritative whenever this inventory and the roadmap disagree.

---

## 3. Classification

### 3.1 Coverage

- `MAPPED` — the requirement is mapped to one or more GitHub issues.
- `UNMAPPED` — the requirement has not yet been mapped to implementation work.
- `NON_IMPLEMENTATION` — the requirement defines scope, rationale, exclusions, assumptions, terminology, or claim boundaries and creates no direct implementation task.

### 3.2 Readiness

- `READY` — the requirement is sufficiently specified for implementation or verification.
- `AMBIGUOUS` — the requirement contains an unresolved detail that must be resolved before correct implementation.
- `BLOCKED` — implementation or verification cannot proceed because a required dependency, decision, artifact, or prerequisite is unavailable.

### 3.3 Requirement Types

Use the single most appropriate primary type:

- Architecture
- Algorithm
- Mathematics
- Configuration
- Dataset
- Preprocessing
- Training
- Evaluation
- Experiment
- Comparator
- Ablation
- Robustness
- Metric
- Statistical Analysis
- Artifact
- Reporting
- Provenance
- Reproducibility
- Failure Semantics
- Claim Boundary
- CLI / Execution
- Testing

---

## 4. Requirement Inventory

| ID | Roadmap Reference | Atomic Requirement | Type | Milestone | Issue(s) | Acceptance Evidence | Coverage | Readiness |
|---|---|---|---|---|---|---|---|---|
| REQ-001 | §X.X | ... | Architecture | M01 | #12 | Exact test, artifact, command, or observable evidence proving completion | MAPPED | READY |
| REQ-002 | §X.X | ... | Configuration | M01 | #13 | Exact config validation or execution evidence | MAPPED | READY |
| REQ-003 | §X.X | ... | Experiment | M06 | #47 | Exact experiment artifact and required result outputs | MAPPED | READY |
| REQ-004 | §X.X | ... | Claim Boundary | — | — | Roadmap traceability only | NON_IMPLEMENTATION | READY |

---

## 5. Coverage Summary

### 5.1 Requirement Counts

- Total roadmap requirements:
- Implementation-bearing requirements:
- Non-implementation requirements:

### 5.2 Implementation Mapping

- Mapped:
- Unmapped:
- Ambiguous:
- Blocked:

### 5.3 Coverage Metric

```text
Implementation Coverage % =
  Mapped implementation-bearing requirements
  ------------------------------------------------ × 100
  Total implementation-bearing requirements
```

**Implementation Coverage:**  

A roadmap is fully mapped only when:

```text
Implementation Coverage = 100%
Unmapped = 0
```

A roadmap is implementation-ready only when:

```text
Implementation Coverage = 100%
Unmapped = 0
Ambiguous = 0
Blocked = 0
```

`NON_IMPLEMENTATION` requirements are excluded from the implementation-coverage denominator but must still be present in the inventory for traceability.

---

## 6. Unresolved Items

Only requirements with `AMBIGUOUS` or `BLOCKED` readiness belong here.

| Requirement ID | Problem | Impact | Blocking | Resolution Issue | Resolution Evidence |
|---|---|---|---|---|---|
| REQ-XXX | ... | Implementation / Scientific validity / Reproducibility / Claim boundary | Yes / No | #... | ... |

Rules:

- Reference the requirement by ID; do not duplicate its full wording.
- Every unresolved implementation detail must have a resolution issue.
- Resolution must be reflected back into the roadmap when the roadmap itself requires clarification.
- Mark the inventory requirement `READY` only after the authoritative source and implementation mapping are consistent.

---

## 7. Audit Invariants

The inventory is valid only if all of the following hold:

- Every roadmap obligation is represented by exactly one atomic inventory requirement, unless the roadmap statement contains multiple independently verifiable obligations that must be split into separate requirements.
- Every inventory requirement points back to an authoritative roadmap reference.
- Every implementation-bearing requirement has a milestone.
- Every `MAPPED` requirement references at least one GitHub issue.
- Every implementation-bearing requirement defines concrete acceptance evidence.
- Every `AMBIGUOUS` or `BLOCKED` requirement appears in **Unresolved Items**.
- Every unresolved implementation detail has a resolution issue.
- No `NON_IMPLEMENTATION` requirement contributes to the implementation-coverage percentage.
- No GitHub issue is treated as authoritative over the roadmap.
- No roadmap requirement is silently omitted because it is descriptive, restrictive, negative, or non-code.
- No requirement is considered complete merely because an issue exists; completion must be demonstrated by its declared acceptance evidence.
- `Implementation Coverage = 100%` means every implementation-bearing roadmap requirement is mapped, not that implementation itself is complete.
- Implementation may begin only when all requirements intended for the implementation phase are `MAPPED` and `READY`.
- The final global roadmap audit must verify both directions of traceability:

```text
Roadmap → Inventory → Milestones → Issues
Issues → Inventory → Roadmap
```
