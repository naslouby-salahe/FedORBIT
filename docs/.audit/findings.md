# Findings and dispositions

Updated 2026-09-25. The matrix contains 250 rows: 246 PASS, two
`JUSTIFIED_ARCHITECTURE_DEVIATION` (TYPE-003, ART-010), one
`JUSTIFIED_EMPIRICAL_DEVIATION` (DATA-017), and one
`BLOCKED_BY_PRE_EXPERIMENT_STATE` (EXEC-008). No FAIL, PARTIAL, MISSING,
UNWIRED, or DEAD_CODE_CANDIDATE rows remain.

## Historical gate state and active scientific decision

The implementation audit closed with EXEC-008 blocked. The preregistered
primary design requires six directed pairs among ToN Windows 10, Linux Process,
and Network. Windows is timestamp-resolvable and Ready to materialize (35,975
rows), but a later split probe found that class-wise intervals overlap globally:
Password TRAIN begins later than DDoS TEST, so this is not one future-only test
period. Linux and Network
each contain normalized behavioral-feature vectors with conflicting labels
(`ddos` / `injection` for Linux; `ddos` / `scanning` for Network). Roadmap §4.6
rejects such sources. Edge-IoTset also has unparseable event times. A later
complete group census quantified an exclusion policy: it would discard 91.67%
of Linux rows, but 0.973% of Network rows, with substantial class selectivity
in both (especially Network scanning). Thus Linux is not rescued by excluding
every ambiguous group; any filtered Network use is a new, narrower population.
A later 22.34M-row scan found nonzero support for all Windows-shared labels in
all five existing per-class partitions after Network filtering. It does not
resolve the global chronology problem or establish post-purge support under a
future-time protocol.

The configured sources are the `Processed_datasets/...` files. The alternative
`Train_Test_*` files are explicitly excluded by roadmap §4.4 because they lack
`ts` for the chronological split. Choosing a label, dropping conflicting rows,
or substituting those files would change the preregistered science, so none was
done. Real preprocessing and its reuse run agreed on these classifications;
the zero resource-blocked count means no dataset was blocked for memory or
resource limits, not that invalid inputs are usable.

To unblock the primary claims:

1. Prefer a corrected authoritative release of the exact configured Linux
   Process and Network components. Record provenance and hashes, rerun
   production preprocessing and reuse validation, and retain the locked
   design only if all registered validity checks pass.
2. If no corrected release exists, obtain an explicit preregistration/scope
   amendment defining the surviving pair population and revising dependent
   cells, inferential coverage, claims, tables, and evidence rules. Do not
   treat fewer pairs as the original confirmatory design.

The subsequent scientific audit recommends a controlled synthetic optimization
study as the candidate direction and defers ToN, DIAD, and CICIoMT. This is not
a locked protocol amendment: production eligibility rules, pair lists,
thresholds, estimators, statistical synthesis, and run gates still encode the
former real-data study. The candidate scope is mathematical, not applied target
utility, so it does not require a target operator or target-local utility
baseline. Registered synthetic experiments still need action-level contrasts,
a confirmatory unit, instance replication, and synthesis aligned to that scope.
No workload is confirmatory evidence for the candidate study until the
novelty case and roadmap/config/catalogue/implementation are reconciled.
The full design and reviewer objections are recorded in
`docs/audit/Scientific Decisions.md` and `docs/audit/Reviewer Audit.md`.

## Justified non-PASS rows

- TYPE-003: `SourceClientName` intentionally remains a generic algorithm
  boundary so synthetic ranking/tie-break fixtures can use identities beyond
  the closed production `DatasetId` enum; production callers convert from a
  registered dataset ID.
- ART-010: no producer emits a bare dense NumPy array, so there is no NPY
  payload to serialize. Existing deterministic outputs are CSV/Parquet/SVG
  and torch-native checkpoints; no empty NPY was added.
- DATA-017: empirical counts and checksums reflect measured inputs rather
  than literature estimates. Network is invalid because of conflicting
  duplicate labels, not resource exhaustion.

## Completed engineering remediations

- Passed configured HiGHS feasibility and optimality tolerances into exact
  sparse, baseline, and Dense-CCP LPs; added wiring coverage.
- Made raw-file identity depend on SHA-256 of file bytes, including a
  same-size/same-mtime replacement regression test.
- Hardened prepared-data reuse: completion records bind input/contract and
  validation/duplicate identities, every prepared payload checksum, schema,
  split and manifest; incomplete or stale outputs are rebuilt.
- Guarded changed generated project exports with `--overwrite`; prevented
  empty primary results/figures without verified records.
- Extended architecture alias checks to PEP 695 and made raw tabular public
  aliases immutable/read-only (`Mapping`, tuple, `Sequence`).
- Replaced substring manifest selection with exact parsed experiment identity
  (and dataset coordinate where needed); malformed records fail closed.
- Corrected Beneficial Rejected Rate to use the paired no-confirm
  counterfactual, routed production confirmation through the central seeded
  schedule helper, and wired existing objective-error/coverage metric helpers.
- Removed five confirmed unused APIs: raw-digest `matches`/`entry_for`,
  `DatasetPreparationResult.blocked_datasets`, `PreTestLifecycle.opened`,
  and obsolete `analysis.statistics.nominal_alpha`.
- Kept the two unreferenced Pydantic properties used as tested label-redaction
  sentinels; they are intentional, not dead code.

See [Evidence](evidence.md) for verification results and retained raw records.
