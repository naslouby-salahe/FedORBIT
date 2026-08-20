# MXX — <Milestone Name>

> **Outcome:** <One concise sentence describing the concrete end state this milestone must achieve.>

## At a Glance

| Field | Value |
|---|---|
| Roadmap scope | `<§X.X–§X.X>` |
| Requirement ownership | `<REQ-XXX–REQ-YYY>` |
| Upstream milestones | `<MXX, MXX>` |
| Implementation issues | `<#XXX, #XXX, ...>` |
| Coverage authority | `Roadmap Coverage Inventory` |
| Audit issue | `<#XXX>` |
| Audit status | `PENDING` |

## Coverage

The Roadmap Coverage Inventory is the traceability authority for this milestone. Every roadmap requirement owned by this milestone must be explicitly mapped to its implementation issue(s) and objective verification evidence.

| Roadmap Section | Scope / Work Package | Requirement IDs | Implementation Issue(s) | Verification / Evidence |
|---|---|---|---|---|
| `<§X.X>` | `<Work package>` | `<REQ-XXX–REQ-XXX>` | `<#XXX>` | `<Tests / artifacts / validation>` |
| `<§X.X>` | `<Work package>` | `<REQ-XXX–REQ-XXX>` | `<#XXX>` | `<Tests / artifacts / validation>` |
| `<§X.X>` | `<Work package>` | `<REQ-XXX–REQ-XXX>` | `<#XXX>` | `<Tests / artifacts / validation>` |

### Coverage Rules

- Every mandatory requirement owned by this milestone must be present in the Roadmap Coverage Inventory.
- Every mandatory implementation requirement must map to at least one implementation issue.
- Every conditional requirement must remain traceable and must be implemented when its roadmap-defined condition applies.
- Every mapped requirement must have objective verification or evidence.
- Every implementation issue must reference the exact requirement IDs it satisfies.
- A requirement is not considered covered merely because it falls inside a requirement range or roadmap section assigned to the milestone.
- No blocking requirement may remain `UNMAPPED` or `AMBIGUOUS` when implementation begins.
- No issue may redefine, weaken, silently reinterpret, duplicate, or extend the authoritative roadmap requirement it implements.

## Dependencies

### Milestone Dependencies

| Milestone | Required Input / Contract | Entry Gate |
|---|---|---|
| `<MXX — Milestone Name>` | `<Exact upstream capability, interface, output, or guarantee consumed>` | `Complete + audit PASS` |
| `<MXX — Milestone Name>` | `<Exact upstream capability, interface, output, or guarantee consumed>` | `Complete + audit PASS` |

### Artifact / Interface Dependencies

| Dependency | Produced By | Required Validation |
|---|---|---|
| `<Artifact / interface / schema / manifest>` | `<MXX / #XXX>` | `<Schema, provenance, compatibility, integrity, or other validation>` |
| `<Artifact / interface / schema / manifest>` | `<MXX / #XXX>` | `<Schema, provenance, compatibility, integrity, or other validation>` |

Dependency completion alone is not sufficient. Every consumed dependency must be present, valid, provenance-compatible where applicable, and compatible with the active roadmap contract.

## Implementation Issues

Implementation issues are the executable work units for this milestone. Detailed task checklists belong in the issues, not in this milestone document.

| Order | Issue | Work Package | Roadmap Scope | Requirement Coverage | Depends On |
|---:|---|---|---|---|---|
| 1 | `<#XXX — Issue Name>` | `<Work package>` | `<§X.X>` | `<REQ-XXX–REQ-XXX>` | `<MXX / #XXX / artifact>` |
| 2 | `<#XXX — Issue Name>` | `<Work package>` | `<§X.X>` | `<REQ-XXX–REQ-XXX>` | `<#XXX / artifact>` |
| 3 | `<#XXX — Issue Name>` | `<Work package>` | `<§X.X>` | `<REQ-XXX–REQ-XXX>` | `<#XXX / artifact>` |

### Issue Contract

Every milestone issue must:

- reference its exact roadmap section(s);
- list every covered requirement ID;
- contain a detailed implementation checklist;
- define objective acceptance criteria;
- identify required tests;
- identify required artifacts, outputs, or interfaces;
- identify required provenance or manifest updates where applicable;
- identify explicit dependencies on upstream milestones, issues, artifacts, or interfaces;
- preserve roadmap terminology and semantics;
- close only when every mapped requirement and acceptance criterion is satisfied.

## Deliverables

| Deliverable | Source Issue(s) | Required Validation | Downstream Consumer |
|---|---|---|---|
| `<Implementation component / capability>` | `<#XXX>` | `<Unit / integration / numerical / structural validation>` | `<MXX / #XXX / none>` |
| `<Artifact / dataset / report / manifest>` | `<#XXX>` | `<Schema / content / provenance validation>` | `<MXX / #XXX / none>` |
| `<Test / validation evidence>` | `<#XXX>` | `<Pass condition>` | `<Milestone audit>` |

All roadmap-required deliverables for this milestone must appear in this table or be explicitly referenced through the Roadmap Coverage Inventory.

## Entry Criteria

Implementation may begin only when all of the following are true:

- all required upstream milestones are complete;
- every required upstream milestone audit is `PASS`;
- every required upstream artifact, interface, schema, or manifest exists;
- every consumed dependency passes its applicable validation;
- consumed evidence is provenance-compatible and not stale where provenance applies;
- all roadmap requirements owned by this milestone are present in the Roadmap Coverage Inventory;
- every mandatory implementation requirement is mapped to at least one milestone issue;
- every mapped requirement has an explicit verification or evidence target;
- no blocking requirement is `UNMAPPED`;
- no blocking requirement is `AMBIGUOUS`;
- no unresolved roadmap ambiguity would force the implementer to invent a scientific, mathematical, methodological, numerical, architectural, configuration, artifact, or execution decision.

## Exit Criteria

The milestone is complete only when all of the following are true:

- every mandatory requirement owned by this milestone is satisfied;
- every applicable conditional requirement is satisfied;
- every mapped implementation issue is closed;
- the Roadmap Coverage Inventory contains no unresolved `UNMAPPED` or blocking `AMBIGUOUS` requirement owned by this milestone;
- all required unit tests pass;
- all required integration tests pass;
- all required validation procedures pass;
- all required deliverables are generated;
- all required artifacts, interfaces, schemas, and manifests pass validation;
- required provenance is complete and valid;
- no required evidence is stale or incompatible with its material dependencies;
- the milestone audit is `PASS`;
- no unresolved blocking finding remains.

## Acceptance Evidence

| Evidence Area | Required Evidence | Pass Condition |
|---|---|---|
| Requirement coverage | Roadmap Coverage Inventory | All mandatory and applicable requirements accounted for with no blocking coverage gaps |
| Implementation | Closed milestone issues linked to exact requirements | Every mapped requirement has completed implementation evidence |
| Unit validation | Required unit-test results | All required tests pass |
| Integration validation | Required integration-test results | All required integration paths pass |
| Scientific / functional validation | `<Required validation outputs>` | All roadmap-defined validation conditions pass |
| Deliverables | Required outputs and artifacts | Complete, readable, valid, and consistent with the roadmap |
| Provenance | Required manifests / dependency identity / compatibility evidence | Complete and sufficient to verify origin, compatibility, and staleness where applicable |
| Audit | Milestone audit issue | Final result is `PASS` with no unresolved blocking findings |

## Milestone Audit

**Audit issue:** `<#XXX>`

**Status:** `PENDING`

The milestone audit is the final completion gate. It must independently verify:

- complete roadmap coverage for all requirements owned by the milestone;
- exact requirement-to-issue traceability;
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

- This milestone implements only the roadmap requirements explicitly assigned to it.
- The authoritative roadmap remains the source of scientific, mathematical, methodological, architectural, numerical, configuration, artifact, and execution requirements.
- This milestone may organize implementation work but may not redefine, weaken, extend, or silently reinterpret the roadmap.
- Detailed implementation checklists belong in implementation issues.
- Detailed verification checklists belong in the milestone audit issue.
- Work outside this milestone's mapped roadmap scope must not be added unless the roadmap or coverage inventory is explicitly updated first.
