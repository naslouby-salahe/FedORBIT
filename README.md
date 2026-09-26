# FedORBIT

FedORBIT is research code for robust action selection when a directed response matrix is known only
up to a block-constrained fine-label correspondence. The solver preserves one admissible map jointly
across both matrix axes and selects an action robust to every admissible correspondence.

No publishable study is currently selected (the synthetic QAP candidate was retired on 2026-09-26; the identity-shortcut alternative was retired the same day; see `docs/audit/`). The original cross-schema federated-transfer application
is not supported by current evidence: the available benchmarks do not establish naturally unavailable
fine-label maps or an organization-level use case. A controlled synthetic optimization study remains a
conditional candidate, but it is **not currently recommended as publishable**. Its exact-separator novelty is weakened by prior sparse-flow QAP branch-and-bound methods; equivalence to
those methods and the full QAP-R1 relation remain unresolved, as does the instance-level confirmatory protocol. A ToN benchmark-integrity study is a conditional alternative: a census of the configured 22.34M-row release found 11,179 normalized feature groups with conflicting labels (217,382 rows; 0.9731%), but normalization-level overlap with prior audits and multiclass evaluation impact have not been established (the local ransomware vector counts do not reproduce the 2026 abstract); under the current production key (which retains source/destination ports), exact-feature lookup implies 0.0125% minimum binary disagreements. Dropping only those two ports yields conflicts in 56.24% of rows. An exact-feature majority lookup then has a 91.20% multiclass accuracy ceiling over all rows (84.35% within conflict rows), with 8.80% minimum multiclass disagreements and 0.678% minimum binary disagreements. This is closer to the 2026 address/port ablation because the production key already excludes IP addresses, but 52 versus 322 ransomware vectors means the exact feature definition is still not reproduced. The SSRN abstract already reports Random Forest accuracy rising from 0.9694 without addresses/ports to 0.9981 when retained; the local exact-key lookup ceilings are not directly comparable model scores. These are key-specific deterministic ceilings, not trained-model results. No confirmatory workload is
authorized by the current roadmap. The configured pair-based experiments are legacy
exploratory/diagnostic runs and must not be presented as evidence for the synthetic candidate.

The principal implementation is the **FedORBIT Exact-Sparse Solver**; the secondary dense relaxation
is the **FedORBIT Dense-CCP Fallback**, which is explicitly non-exact.

## Scope

The scientific disposition and provisional study contract are in `docs/FedORBIT_Roadmap.md` and
`docs/audit/Scientific Decisions.md`. The roadmap explicitly marks its former applied-transfer
sections as superseded; `configs/fedorbit.yaml` and the seed-based synthesis have not yet been migrated
to the synthetic candidate. `configs/tests.yml` and `configs/smoke.yml` contain execution-fixture
controls.

## Setup

```text
uv sync --extra dev
```

The repository must contain a fully resolved lockfile (`uv.lock`) with transitive package versions
and hashes. The registered environment contract is defined in `configs/fedorbit.yaml` under
`environment`.

## Public CLI

```text
fedorbit doctor
fedorbit preprocess [DATASET NAME] [--overwrite]
fedorbit plan
fedorbit smoke [--overwrite]
fedorbit run "EXPERIMENT NAME" [--overwrite]
fedorbit status [EXPERIMENT NAME]
fedorbit report [EXPERIMENT NAME] [--overwrite]
```

`fedorbit run` accepts registered experiment names exactly as quoted. Runs using the current legacy
configuration are exploratory/diagnostic only and do not establish readiness for a confirmatory
synthetic study.

## Reproducibility

Scientific identity is semantic. Every reusable artifact carries a dependency fingerprint, payload
checksums, completion manifest, and provenance record. Valid artifacts are reused; changed
dependencies invalidate only affected descendants. All mutable computation is staged under
`outputs/cache/staging/` and promoted atomically after validation.

## Layout

```text
configs/    authoritative configuration YAML and scientific-contract snapshot
data/raw    symlink to immutable external raw datasets
src/fedorbit/   typed domain, configuration, dataset, model, solver, and execution packages
outputs/    complete generated computational workspace (Git-ignored)
results/    terminal manuscript-facing evidence (Git-ignored)
tests/      architecture, unit, scientific, integration, e2e, and smoke suites
```

## Quality Gates

The repository enforces, via `noxfile.py` and the `Makefile`:

- Ruff formatting and linting
- strict Pyright typing across source and tests
- repository architecture and dependency-boundary tests
- pytest unit, scientific, integration, e2e, and smoke suites
- scientific-contract snapshot conformance
