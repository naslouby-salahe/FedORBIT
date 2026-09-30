# FedORBIT

Research code for a measured decomposition of cold-start collaboration in benign-only IoT anomaly detection: when a new device has only a few benign windows, which shared information (per-feature marginals, covariance, model weights) recovers detection quality, and when does sharing more hurt?

The protocol is in `docs/FedORBIT_Roadmap.md` and the executable scientific contract is `configs/fedorbit.yaml`. The active implementation matrix, progress, reviewer audit, scientific decisions, and consolidated POC findings are in `docs/audit/`. The retired QAP direction is summarized in Git history and is not part of the active study.

The contribution is a focused empirical comparison of peer-information channels at low benign support, not a new detector or a broad claim to federated IoT anomaly detection. Prior work includes FedGroup, which studies functional-group federated attack detection on the same UNSW IoT Analytics cohort using supervised features and random splits; it does not test this study's benign-only support and future-test design. Its public notebook derives labels from timestamp intervals, retains `Timestamp` among predictors, and uses randomized per-device splits, so it does not supply chronological transfer evidence. The roadmap and reviewer audit detail this boundary ([FedGroup paper](https://doi.org/10.1007/s10922-023-09782-9); [notebook](https://github.com/BasemSuleiman/2023_Anomaly_Detection_IoT/blob/main/Project%202%20Privacy-Aware%20Anomaly%20Detection%20in%20IoT%20Environments%20using%20FedGroup%20A%20Group-Based%20Federated%20Learning%20Approach/%5Bupdate%5DFL.ipynb)).

The locked physical-device evidence is limited to nine N-BaIoT devices and Mirai/Gafgyt detection; Gotham contributes descriptive evidence from two eligible simulated devices. The nominal 1% support-calibrated threshold does not transfer to later benign traffic (mean test FPR 8.65–19.54% for shared marginals and 9.67–15.88% for local-standardised). The study makes no achieved low-FPR, deployment, or broad generalisation claim, and the journal-level contribution remains developing.

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

## Experiment runtimes

Measured wall-clock duration of each confirmatory run, from the `elapsed` value each experiment logs on completion, on a 10-core CPU with one CUDA GPU:

| Experiment | Purpose | Measured runtime |
| --- | --- | --- |
| `cold-start-ladder` | primary dose-response and covariance comparison, 9 N-BaIoT devices, Gaussian detector | ≈3h 20m |
| `partner-selection` | deployable-versus-oracle partner selection, 9 N-BaIoT devices, Gaussian detector | ≈1h 34m |
| `deep-detector` | autoencoder replication of the channel ordering, 9 N-BaIoT devices, requires CUDA | ≈20 min |
| `simulated-boundary` | descriptive boundary check on the 2 eligible Gotham devices, Gaussian detector | ≈4 s |

The two Gaussian experiments above ran concurrently with each other and with `deep-detector`, so wall-clock time reflects CPU contention between them rather than isolated cost. `fedorbit smoke` runs a reduced version of `cold-start-ladder` on 2 synthetic devices and completes in under a minute.

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
