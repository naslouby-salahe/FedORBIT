# FedORBIT

Research code for a measured decomposition of cold-start collaboration in benign-only IoT anomaly detection: when a new device has only a few benign windows, which shared information (per-feature marginals, covariance, model weights) recovers detection quality, and when does sharing more hurt?

The protocol is in `docs/FedORBIT_Roadmap.md` and the executable scientific contract is `configs/fedorbit.yaml`. The retired earlier direction (cross-schema transfer, sparse-QAP robust action) and the audit trail are in `docs/audit/`; the historical implementation ledger for it is `docs/Audit Matrix.md`.

## Setup

```text
uv sync --extra dev
```

`data/raw` is a symlink to the external raw datasets (N-BaIoT and Gotham2025 are used). The autoencoder experiment requires a CUDA device and fails without one.

## Public CLI

```text
fedorbit doctor
fedorbit preprocess [DATASET NAME] [--overwrite]
fedorbit plan
fedorbit smoke [--overwrite]
fedorbit run "EXPERIMENT NAME" [--overwrite]
fedorbit status [EXPERIMENT NAME]
fedorbit report [--overwrite]
```

Experiments: `cold-start-ladder`, `partner-selection`, `deep-detector`, `simulated-boundary`. `report` writes tables and figures under `results/` only from valid persisted evidence.

## Reproducibility

Every artifact has a manifest with payload digest, provenance (configuration digest, source digest, input digests) and code revision. Valid artifacts are reused; changed configuration, code or inputs mark dependent artifacts stale; corrupted or partial artifacts are rejected. All writes are atomic. Every stochastic step derives its seed from one configured base seed.

## Layout

```text
configs/         authoritative configuration
data/raw         symlink to immutable external raw datasets
src/fedorbit/    config, datasets, detection, study, analysis, pipeline, infrastructure, reporting, cli
outputs/         generated workspace (Git-ignored)
results/         manuscript-facing tables and figures (Git-ignored)
tests/           architecture, unit, scientific, integration, e2e and smoke suites
docs/            roadmap, historical ledger, audit workspace
```

## Quality gates

`make audit-all` runs Ruff formatting and lint, strict Pyright, Vulture, deptry, architecture tests and the unit, scientific, integration, e2e and smoke suites; `make coverage` enforces the coverage gate.
