# Audit evidence

Updated 2026-09-25. Matrix dispositions and roadmap contracts are authoritative
in [Audit Matrix](../Audit%20Matrix.md); this note summarizes the checks that
support them. Raw logs, JSON inventories, and Graphify graphs remain alongside
this file where available.

## Data and execution gate

Production preprocessing ran twice. Both runs classified ToN Windows 10 as
Ready; ToN Linux Process and Network as Invalid Data for conflicting-label
duplicate groups; and Edge-IoTset as Invalid Data for unparseable event time.
The first run took about 2m53s and the reuse run about 34s, with identical
terminal classifications. Resource-blocked count was zero. The paired run
record is `preprocess-2026-09-25.log`.

Safe `doctor`, `plan`, `status`, `smoke`, and `report` commands exited 0.
Status showed all 27 registered experiments Missing. Report exported zero
verified scientific evidence and emitted no result-bearing primary outputs.
Doctor found the raw data root and a matching GPU, while overall execution
identity compatibility was false. No `fedorbit run` or scientific experiment
was executed.

## Source and call-graph coverage

Fresh forced Graphify output is under `graphify-completion-audit/graphify-out/`
(3,913 nodes, 20,579 raw edges). The earlier comparison graph is under
`graphify-before/graphify-out/` (2,537 nodes, 9,834 raw edges). The current
source-resolved inventory reports 1,183 production callables by independent
AST traversal (904 functions, including 27 nested functions, and 279 methods);
Graphify reports 1,182. CLI closure has 887 callables, 285 leaves, and maximum
depth 18. Twenty-seven registered producer roots cover 724 callables; all are
within the CLI closure. The remaining 295 callables are reconciled against
symbol references and conventional framework hooks. Two unreferenced
properties are the tested label-redaction sentinels described in Findings;
there is no unexplained scientifically required orphan.

| CLI command | Direct | Unique closure | Leaves | Max depth |
| --- | ---: | ---: | ---: | ---: |
| doctor | 7 | 24 | 4 | 8 |
| preprocess | 3 | 152 | 62 | 13 |
| plan | 6 | 24 | 11 | 7 |
| smoke | 2 | 97 | 37 | 12 |
| run | 3 | 735 | 238 | 18 |
| status | 5 | 23 | 11 | 7 |
| report | 62 | 134 | 49 | 9 |

The `run` and producer paths are prospective static reachability, not runtime
experiment traces. The inventory can be regenerated with `callable_counts.py`;
the Graphify JSON is retained for detailed graph inspection.

## Tests and static checks

- Full completion-audit suite: `uv run pytest -q -ra` — 849 passed in
  2472.49s, exit 0; no failure or skip/xfail report.
- Focused final regression suite: 47 passed.
- Ruff check and format passed; Pyright reported 0 diagnostics; deptry found
  no issues; Vulture found no candidates; `git diff --check` passed.
- Earlier intermediate test failures were resolved and the final full suite
  was rerun successfully. The retained completion summary is
  `verification-completion.txt`; `pytest-final-2026-09-25.log` records the
  preceding full rerun (843 passed).

## Retained artifacts

Only the current and before Graphify JSON/manifest, `callable_counts.py`, the
paired final preprocessing log, final verification summary, safe-command
outputs, and the audit matrix's row data/tooling are retained as supporting
records. Reusable extractor caches and superseded snapshots were removed.
