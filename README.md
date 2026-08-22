# FedORBIT

FedORBIT is a cross-schema federated procedural-transfer method for organizations that share only a
coarse semantic partition and do not expose an exact fine-label correspondence. A source exports an
anonymous fine-node procedural-response packet; a target preserves one admissible block-constrained
correspondence jointly across both axes of the directed response operator and selects a curriculum
robust to every admissible correspondence.

The principal solver is the **FedORBIT Exact-Sparse Solver**; the secondary dense relaxation is the
**FedORBIT Dense-CCP Fallback**, which is explicitly non-exact.

## Scope

The authoritative scientific, mathematical, architectural, dataset, configuration, experiment,
statistical, and claim contract is `docs/FedORBIT_Roadmap.md`. `configs/fedorbit.yaml` is the sole
authority for retained numerical parameters, thresholds, seeds, experiment grids, dataset/path
identifiers, and other genuine configurable selections. `configs/tests.yml` and `configs/smoke.yml`
contain only nonclaim execution-fixture controls.

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

`fedorbit run` accepts every registered descriptive experiment name from the roadmap experiment
catalogue exactly as quoted. The operator supplies no scientific parameters.

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
