# FedORBIT — Authoritative Research Roadmap

**Project / algorithm:** **FedORBIT**
**Document role:** single authoritative, standalone scientific and implementation specification

This roadmap is the sole authority for scientific definitions, numerical values, experiment membership, statistical procedures, claim criterions, and implementation semantics. No implementation mechanism may override it, and no TEST outcome may alter a scientific definition, parameter, metric, test, experiment, baseline budget, claim threshold, or reporting rule.

# 1. Contribution Boundary

## 1.1 Contribution identity

FedORBIT is a cross-schema federated procedural-transfer method for organizations that:

* have incompatible raw telemetry;
* may use different feature spaces and arbitrary local models;
* share only a coarse semantic partition;
* do not expose an exact fine-label correspondence;
* do not share raw samples, model parameters, embeddings, aligned entities, or executable common queries.

A source exports an anonymous fine-node procedural-response packet. A target preserves one admissible block-constrained correspondence jointly across both axes of the directed response operator and selects a curriculum robust to every admissible correspondence.

The principal solver is **FedORBIT Exact-Sparse Solver**.

The secondary dense solver is **FedORBIT Dense-CCP Fallback**. It is an explicitly non-exact dense relaxation/heuristic and must never be described as a general dense exact solver.

## 1.2 Fixed scientific contribution

The contribution consists of:

1. a strict no-common-interface information regime;
2. source-local procedural-response estimation with simultaneous uncertainty bands;
3. one unresolved block-constrained fine correspondence acting jointly on both axes of a directed response matrix;
4. robust target action selection over the complete admissible correspondence orbit rather than a point estimate;
5. the exact sparse correspondence-separation result: after fixing images of active intervention labels, the remaining correspondence decomposes into independent blockwise linear-assignment problems;
6. correspondence/action diagnostics separating map identifiability from action certifiability;
7. target-local paired confirmation before live assimilation.

## 1.3 Principal scientific regime

The principal confirmatory method uses `scientific.action.principal_sparse_support`, defined exactly once in the Configuration YAML.

Sensitivity analyses use the registered sparse-support alternatives in `scientific.action.sparse_support_sensitivity` and the FedORBIT Dense-CCP Fallback defined by Sections 1.1 and 4.18.

## 1.4 Fixed assumptions

The principal theorem and method require:

* block-constrained bijection after explicit null-node padding;
* one correspondence acting jointly on response rows and columns;
* nonnegative target importance $w$;
* nonnegative curriculum increment $\alpha$;
* nonnegative linear curriculum cost $c$;
* a non-Byzantine source packet;
* target-local confirmation before live assimilation;
* no information outside the strict resource contract.

## 1.5 Explicit non-goals

The paper must not claim:

* generic action-identifiability theory as new;
* generic non-rectangular robust optimization as new;
* generic QAP, RLT, McCormick, CCP, assignment-polytope, or linear-assignment machinery as new;
* dense exactness;
* privacy guarantees from anonymity alone;
* Byzantine-source robustness;
* natural unavailability of fine labels in the public benchmark datasets;
* universal cross-schema transfer;
* deployment readiness beyond measured evidence;
* validity outside the tested telemetry families, support regimes, semantic partitions, and strict information interface.

# 2. Research Questions, Hypotheses, and Claim Intent

## Exact Sparse Separator Exactness

**Research question:** Does sparse correspondence separation return the exact worst admissible correspondence?

**Hypothesis:** Exact-sparse and available exact truth agree within the configured exact validation tolerance.

**Falsified by:** Any reproducible discrepancy above tolerance or an invalid correspondence certificate.

## Joint Correspondence Avoids Rectangular Pessimism

**Research question:** Does preserving one joint correspondence avoid impossible marginal worst-case combinations?

**Hypothesis:** Designed incompatible worlds and a nontrivial fraction of real packets show material positive coupling gap, and coupling destruction removes the mechanism.

**Falsified by:** Failure of the real/synthetic mechanism criteria or mechanism retention after destruction.

## Action Certification Without Fine-Map Identification

**Research question:** Can useful action be certified while the fine map remains unresolved?

**Hypothesis:** Controlled worlds exist with multiple admissible maps and either a common useful optimum or a positive robust compromise.

**Falsified by:** Positive action consistently requiring exact map recovery or failure of the map-value bound.

## Strict Cross-Telemetry Transfer Utility

**Research question:** Does FedORBIT improve target utility under the strict interface?

**Hypothesis:** Principal FedORBIT materially improves macro cross-entropy relative to Local-Only on the registered majority of primary directed pairs without material harm on another pair.

**Falsified by:** Failure of the registered utility criteria or strict-resource validation.

## Value of External Procedural Evidence

**Research question:** Does the source response packet add value beyond matched target-local curriculum search?

**Hypothesis:** FedORBIT materially improves over Local-SIR on the registered majority of primary pairs.

**Falsified by:** Local-SIR equivalence or superiority under the registered criteria.

## Operational Relevance of Sparse Support

**Research question:** Is bounded support practically useful rather than only theoretically convenient?

**Hypothesis:** The registered sparse condition remains close to dense utility on the required proportion of units while retaining useful realized transfer.

**Falsified by:** The registered sparse-irrelevance rule.

## Target Confirmation Safety

**Research question:** Does target confirmation reduce harmful transfer without destroying coverage?

**Hypothesis:** Confirmation meets the registered harm-reduction threshold and coverage-loss ceiling.

**Falsified by:** Failure of the safety or coverage criteria.

## Sparse Solver Work-Structure Agreement

**Research question:** Does implementation behavior follow active-image plus LAP work structure?

**Hypothesis:** Active-image and LAP counts match the exact formulas and runtime follows the predicted operational trend without sacrificing exactness.

**Falsified by:** Counter mismatch or approximation required for speed.

# 3. Formal Problem and Notation

For each coarse group $g$, let the source and target have respectively $m_g^{(s)}$ and $m_g^{(t)}$ real eligible transfer nodes.

Define the padded block size

$$
n_g=\max\left(m_g^{(s)},m_g^{(t)}\right).
$$

The smaller endpoint is padded with explicit null nodes until both endpoints have $n_g$ nodes in the group.

The total padded node count is

$$
K=\sum_g n_g.
$$

Let $\mathcal G_g$ denote padded node indices in coarse group $g$. The admissible correspondence set is

$$
\Pi=\prod_g S_{n_g}.
$$

Each $P\in\Pi$ is one block-preserving permutation matrix.

Let:

* $L\in\mathbb R^{K\times K}$ be the source simultaneous lower response matrix;
* $U\in\mathbb R^{K\times K}$ be the corresponding upper matrix;
* $w\in\mathbb R_{\ge0}^K$ be target importance;
* $\alpha\in\mathbb R_{\ge0}^K$ be the target curriculum increment;
* $c\in\mathbb R_{\ge0}^K$ be curriculum cost;
* $B_\alpha$ be the total action budget;
* $\bar\alpha_j$ be the coordinate cap.

The action polytope is

$$
\mathcal A=
\left\lbrace
\alpha\ge0:
\mathbf 1^T\alpha\le B_\alpha,;
0\le\alpha_j\le\bar\alpha_j
\right\rbrace.
$$

For support limit $s$,

$$
\mathcal A^{(s)} =
\left\lbrace
\alpha\in\mathcal A:|\alpha|_0\le s
\right\rbrace.
$$

The map-conditioned objective is

$$
J(\alpha;P)=w^TP^TLP\alpha-c^T\alpha.
$$

The principal robust action is

$$
\alpha^\star
\in
\arg\max_{\alpha\in\mathcal A^{(s)}}
\min_{P\in\Pi}
J(\alpha;P),
$$

using the principal support defined in the Configuration YAML.

For a permitted support $S$,

$$
s_g=|S\cap\mathcal G_g|,
$$

and the number of active-image assignments is

$$
N_S=
\prod_g(n_g)_{s_g},
\qquad
(n)_r=\frac{n!}{(n-r)!}.
$$

The exact orbit is

$$
\mathcal O(L)=\lbrace P^TLP:P\in\Pi\rbrace.
$$

The entrywise rectangular hull is

$$
\mathrm{Rect}(\mathcal O) =
\prod_{k,j}[\ell_{kj},u_{kj}],
$$

where

$$
\ell_{kj} =
\min_{P\in\Pi}(P^TLP)_{kj},
\qquad
u_{kj} =
\max_{P\in\Pi}(P^TUP)_{kj}.
$$

For a fixed feasible action,

$$
h_{\rm orb}(\alpha) =
\min_{P\in\Pi}w^TP^TLP\alpha,
$$

$$
h_{\rm rect}(\alpha) =
w^T\ell\alpha,
$$

$$
\Gamma(\alpha) =
h_{\rm orb}(\alpha)-h_{\rm rect}(\alpha)\ge0.
$$

For any experiment-specific action set $\mathcal B\subseteq\mathcal A$,

$$
V_{\rm exact}(\mathcal B) =
\max_{\alpha\in\mathcal B}
h_{\rm orb}(\alpha)-c^T\alpha,
$$

$$
V_{\rm rect}(\mathcal B) =
\max_{\alpha\in\mathcal B}
h_{\rm rect}(\alpha)-c^T\alpha,
$$

$$
G_{\rm coupling}(\mathcal B) =
V_{\rm exact}(\mathcal B)-V_{\rm rect}(\mathcal B).
$$

For map-value diagnostics,

$$
V(P;\mathcal B) =
\max_{\alpha\in\mathcal B}J(\alpha;P),
$$

$$
V_{\rm pre}(\mathcal B) =
\max_{\alpha\in\mathcal B}
\min_{P\in\Pi}J(\alpha;P),
$$

$$
V_{\rm post}(\mathcal B) =
\min_{P\in\Pi}V(P;\mathcal B),
$$

$$
\Delta_{\rm map}(\mathcal B) =
V_{\rm post}(\mathcal B)-V_{\rm pre}(\mathcal B).
$$

Unless an experiment explicitly requests the dense action set, all mechanism and map-value diagnostics use the principal sparse action set.

Null source nodes have:

* zero response row;
* zero response column;
* zero source support counts.

Null target nodes have:

* $w=0$;
* $\bar\alpha=0$;
* no action eligibility.

# 4. Configuration Contract

`configs/fedorbit.yaml` contains configuration data only. It is the sole authority for numerical parameters, thresholds, tolerances, counts, fractions, seeds, experiment-grid values, dataset and path identifiers, and other genuine configurable selections retained in the **Configuration YAML** section.

Fixed mathematical definitions, algorithm procedures, validation rules, execution semantics, deterministic ordering, architecture definitions, failure behavior, provenance rules, and reporting semantics are authoritative in the relevant roadmap sections rather than encoded as YAML prose or pseudo-expressions. When a fixed rule uses a configurable numerical value, the rule is defined in prose and references the corresponding YAML key.

Observed raw-data facts, derived quantities, and runtime measurements remain outside YAML.

## 4.1 Authority classes

Every roadmap quantity or rule belongs to one of the following authority classes.

### Configuration data

Primitive values that must be supplied, selected, or varied. These values are authoritative in `configs/fedorbit.yaml`. Their presence in YAML does not imply that arbitrary changes are scientifically valid: principal values, confirmatory thresholds, seeds, and registered grids are scientifically locked except where this roadmap explicitly defines a sensitivity or experiment grid.

### Fixed scientific and implementation rules

Mathematical definitions, architectures, procedures, ordering rules, validation conditions, execution semantics, failure behavior, provenance requirements, and reporting semantics fixed by this roadmap. They are authoritative in the corresponding prose or equations and are not independently configurable.

### Observed raw-data facts

Measured from the immutable raw dataset and recorded in dataset manifests. Observed facts may not be replaced with literature-reported expectations.

### Derived quantities

Deterministically computed from configuration data, fixed rules, or observed inputs and never independently configurable.

### Runtime measurements

Measured outcomes such as execution time, memory consumption, selected pilot winner, solver iteration count, or observed packet size. They are never predetermined.

## 4.2 Core scientific configuration

| Semantic configuration                   |                           Locked value |
| ---------------------------------------- | -------------------------------------: |
| scientific.action.principal_sparse_support                 |                                      2 |
| scientific.action.sparse_support_sensitivity             |                               `{1, 3}` |
| `scientific.action.total_curriculum_budget`                |                                   0.50 |
| `scientific.action.coordinate_cap`                  |                                   0.25 |
| `scientific.action.linear_cost_per_actionable_node`                     |               0.01 per actionable node |
| `scientific.action.positive_source_value_threshold`        |                strictly greater than 0 |
| `scientific.action.maximum_source_proposals_per_target`    |                                      3 |
| `scientific.materiality.coupling_objective_units`         |                  0.005 objective units |
| `scientific.materiality.realized_relative_macro_ce` |                                   0.01 |
| `scientific.materiality.macro_f1_absolute`       |                         0.005 absolute |
| `scientific.materiality.equivalence_relative_macro_ce.lower`               |                -0.01 relative macro-CE |
| `scientific.materiality.equivalence_relative_macro_ce.upper`               |                +0.01 relative macro-CE |
| `scientific.materiality.harmful_transfer_relative_macro_ce_gain`             | TEST relative macro-CE gain $\le-0.01$ |
| `scientific.materiality.useful_transfer_relative_macro_ce_gain`              | TEST relative macro-CE gain $\ge+0.01$ |
| scientific.transfer_support.minimum_actionable_target_concepts     |                                      4 |
| scientific.transfer_support.minimum_nontrivial_block_size          |                                      2 |

For target actionable node $j$,

$$
\bar\alpha_j=\texttt{scientific.action.coordinate＿cap},
\qquad
c_j=\texttt{scientific.action.linear＿cost＿per＿actionable＿node}.
$$

For normal or null nodes,

$$
\bar\alpha_j=0,\qquad c_j=0.
$$

The explicit zero-action candidate has objective exactly 0 by the definition of $J$ with $\alpha=0$; this is a mathematical consequence, not a configurable value.

## 4.3 Dataset and transfer-support configuration

### Transfer support thresholds

| Requirement                                            | Locked count |
| ------------------------------------------------------ | -----------: |
| source TRAIN support                                   |          200 |
| source META support                                    |           40 |
| target META support                                    |           40 |
| target CONFIRM support                                 |           40 |
| target TEST support                                    |           40 |
| minimum total rows for a local prediction attack class |          200 |

A transfer concept is source-eligible only when both source TRAIN and source META thresholds pass.

A transfer concept is target-eligible only when target META, CONFIRM, and TEST thresholds pass.

A concept absent or below the applicable threshold becomes a null transfer node for that endpoint.

### Split fractions

Per retained local prediction class, duplicate groups are assigned chronologically according to group midpoint fraction:

| Split   | Interval      |
| ------- | ------------- |
| TRAIN   | $[0,0.55)$    |
| META    | $[0.55,0.70)$ |
| VALID   | $[0.70,0.80)$ |
| CONFIRM | $[0.80,0.90)$ |
| TEST    | $[0.90,1.00]$ |

### Missingness and preprocessing

| Parameter / fixed preprocessing rule        |                            Locked value |
| ------------------------------------------- | --------------------------------------: |
| missing-indicator TRAIN-rate threshold      |                                   0.001 |
| rare-category TRAIN-frequency threshold     |                                   0.001 |
| feature missing/nonfinite drop threshold    |                       greater than 0.05 |
| client invalidity dropped-feature threshold | greater than 0.20 of candidate features |
| scaled numeric clip lower bound             |                                     -10 |
| scaled numeric clip upper bound             |                                     +10 |
| zero-IQR replacement scale                  |                                       1 |
| quantile interpolation                      |           Hyndman-Fan type 7 / `linear` |
| categorical unseen token                    |                                 `<UNK>` |
| categorical rare token                      |                                `<RARE>` |
| categorical missing token                   |                              `<ABSENT>` |
| one-hot dropped category                    |                                    none |

The numerical thresholds in this table are configured in `scientific.preprocessing`. Their fixed interpretation is:

* a **candidate feature** is one raw semantic feature remaining after label, timestamp, identity, payload/provenance, and Edge-specific exclusions, before imputation, missing-indicator creation, scaling, categorical expansion, or constant-feature removal;
* compute each candidate feature's missing/nonfinite fraction on TRAIN at that raw semantic-feature level before imputation or encoding;
* drop a candidate feature when that TRAIN missing/nonfinite fraction is strictly greater than the configured feature-quality threshold;
* invalidate the client when `number_of_candidate_features_dropped_by_quality / number_of_candidate_features_before_quality_filtering` is strictly greater than the configured client-invalidity threshold;
* construct a missingness indicator only for a retained feature when its pre-imputation TRAIN missing rate is at least the configured missing-indicator threshold;
* map a TRAIN categorical level to `<RARE>` when its TRAIN frequency is below the configured rare-category threshold.

One raw candidate feature contributes exactly one unit to the client-invalidity denominator regardless of how many one-hot columns or missingness indicators it would later generate. A client with zero candidate features after the mandatory semantic exclusions is Invalid Data.

The token identities, comparison directions, quantile rule, categorical ordering, and one-hot semantics are fixed preprocessing rules and are not configurable.

The case-insensitive missing-token vocabulary is exactly:

```text
empty string
0
0.0
nan
none
null
```

The string forms `0` and `0.0` are treated as missing **only for fields declared categorical by the dataset adapter**. Numeric zero remains numeric zero.

### Categorical vocabulary ordering

For each categorical feature:

1. `<ABSENT>`;
2. `<RARE>`;
3. `<UNK>`;
4. every remaining TRAIN category in ascending UTF-8 byte order.

No frequency-based ordering is permitted.

### Numeric robust scaling

TRAIN-only quartiles use Hyndman-Fan type 7 with linear interpolation. This quantile rule is fixed and is not configurable.

For numeric feature $x$,

$$
x'=
\mathrm{clip}
\left(
\frac{x-\mathrm{median}_{TRAIN}(x)}
{\max(IQR_{TRAIN}(x),1\text{ when }IQR=0)},
-10,
10
\right).
$$

A zero-IQR feature is subsequently removed if constant across TRAIN after imputation.

## 4.4 Dataset component authority

Expected source structure is established from the original dataset documentation, but actual repository data are authoritative for execution.

The expected components are:

* Edge-IIoTset network/IoT traffic;
* ToN_IoT Windows 10 OS telemetry;
* ToN_IoT Linux process telemetry;
* ToN_IoT network traffic.

The source documentation establishes Edge-IIoTset as network-derived IoT/IIoT telemetry and ToN_IoT as heterogeneous network, Windows, Linux, and IoT telemetry; the Windows and Linux papers document their host-telemetry components. Execution remains authoritative to the actual immutable repository files; literature counts are validation references only.

### Component selection rules

Edge-IIoTset Network Client:

* retain official tabular Edge-IIoTset traffic records only for optional external robustness work;
* expected event-time semantic field: `frame.time` or release-equivalent event timestamp;
* binary and multiclass label fields are builder-only and removed from model input.

ToN-IoT Windows 10 Host Client:

* use only the Windows 10 host telemetry component;
* Windows 7 records are excluded;
* select `Processed_datasets/Processed_Windows_dataset/windows10_dataset.csv`;
* expected event-time semantic field: `ts` or release-equivalent timestamp.

ToN-IoT Linux Process Host Client:

* use `Linux_process_1.csv` and `Linux_process_2.csv` from `Processed_datasets/Processed_Linux_dataset` as one explicit process-telemetry client;
* Linux disk and memory tables are excluded;
* expected event-time semantic field: `ts` or release-equivalent timestamp.

ToN-IoT Network Client:

* use the numbered `Network_dataset_*.csv` files from `Processed_datasets/Processed_Network_dataset` as one explicit network-flow client;
* expected event-time semantic field: `ts` or release-equivalent timestamp.

A dataset adapter may accept a release-equivalent renamed timestamp column only when:

1. exactly one column has the documented timestamp semantics;
2. the alias is recorded in the raw-data manifest;
3. the retained-row parse-success fraction is at least `scientific.datasets.timestamp_alias_acceptance.retained_row_parse_success_minimum`;
4. no outcome data are used to select the alias.

Otherwise the dataset is Invalid Data.

Naive timestamps are interpreted as UTC solely to create a deterministic order. A constant timezone shift does not alter within-client chronological ordering.

### Dataset-adapter schema resolution

Official dataset documentation defines the expected semantic roles; the immutable raw files available to the implementation define the observed schema. Literature-reported row counts, feature counts, and class counts are validation references only and are never substituted for observed values.

The expected label fields are:

| Component | Multiclass field | Binary field |
| --- | --- | --- |
| Edge-IIoTset network | `Attack_type` | `Attack_label` |
| ToN-IoT Windows 10 host | `type` | `label` |
| ToN-IoT Linux process host | `type` | `label` |
| ToN-IoT network | `type` | `label` |

A release-equivalent renamed label field is accepted only when exactly one observed column has the documented semantic role, its alias is recorded in the raw-data manifest, and the mapping can be established without examining model outcomes. Otherwise the component is Invalid Data.

Local multiclass labels are canonicalized before any support decision by:

1. Unicode NFC normalization;
2. stripping surrounding whitespace;
3. Unicode case-folding;
4. replacing every maximal run of non-alphanumeric characters with one underscore;
5. collapsing repeated underscores and stripping leading/trailing underscores.

The canonical normal label is `normal`. The binary field is never a model feature. When a binary field is present, its observed values must partition records consistently into normal versus non-normal according to the multiclass field; otherwise the component is Invalid Data.

The adapter feature order is deterministic:

1. use the official `Description_stats_datasets`/published feature order when that metadata is present in the immutable dataset release;
2. otherwise use the header order of the lexicographically smallest canonical relative-path file belonging to the selected component;
3. every additional file for the same component must contain the same feature-name set after accepted aliases; it is reordered to the canonical order before row processing;
4. duplicate column names after normalization are Invalid Data.

Feature type is resolved before missing-token normalization. Official dataset type metadata takes precedence. A documented numeric, integer, floating, boolean, duration, counter, rate, size, or timestamp-derived measurement is numeric unless separately excluded by this roadmap. A documented string, label, protocol/state symbol, categorical code, byte/string sequence, or textual field is categorical unless separately excluded. If official type metadata are absent for an otherwise permissible behavioral feature, the field is numeric only when every non-missing observed value across the immutable component parses losslessly as IEEE-754 float64; otherwise it is categorical. Type inference may use raw values but may not use labels, split membership, model outcomes, or TEST performance.

Before model-feature construction, each raw field receives exactly one adapter role:

```text
timestamp
multiclass_label
binary_label
behavioral_numeric
behavioral_categorical
forbidden_identity
forbidden_payload
forbidden_provenance
```

Fields documented or named as IP/MAC addresses, host/device identifiers, process/thread instance identifiers, flow/session identifiers, row/index identifiers, capture/source filenames, or equivalent record identities are `forbidden_identity`. Raw payload/free-text content and byte-sequence payload fields are `forbidden_payload`. Source-file, capture, acquisition, or row-provenance fields are `forbidden_provenance`. Label and timestamp fields have their dedicated roles. Every remaining field must resolve deterministically to `behavioral_numeric` or `behavioral_categorical`.

The published Edge-IIoTset processing description identifies the following fields as identifier/payload-style exclusions when present, in addition to the seven benchmark-integrity exclusions in Section 4.5:

```text
frame.time
ip.src_host
ip.dst_host
arp.src.proto_ipv4
arp.dst.proto_ipv4
icmp.transmit_timestamp
http.file_data
http.request.full_uri
http.request.uri.query
tcp.options
tcp.payload
tcp.srcport
tcp.dstport
udp.port
mqtt.msg
```

A field already excluded by another rule is removed once; exclusion reasons are all recorded in the manifest.

For ToN-IoT components, the official dataset feature-description files are the primary type/role metadata. Network identifiers such as `src_ip` and `dst_ip`, host/process instance identifiers, and any release-equivalent identity fields are excluded according to the universal identity rule. Network ports are retained only when the official description defines them as behavioral connection attributes rather than endpoint identity; process IDs and equivalent instance keys are always excluded.

Observed release differences are handled as follows:

* different row counts, class counts, timestamp ranges, feature counts, or file counts are recorded as observed facts and do not by themselves invalidate the component;
* a documented field missing because the selected official component/release does not contain it is recorded as absent;
* an additional field is admitted only when its role and type are deterministically resolvable by the rules above and it is not forbidden;
* a required timestamp or multiclass label whose semantics cannot be uniquely resolved makes the component Invalid Data;
* a field whose behavioral-versus-identity/payload/provenance role remains ambiguous after official metadata and the rules above makes the component Invalid Data rather than allowing an implementation guess;
* any adapter adaptation is recorded in the raw-data manifest and becomes part of the preprocessing dependency fingerprint before any model outcome is inspected.

## 4.5 Edge-IIoTset leakage safeguard

The following fields are excluded from principal Edge model input before categorical encoding:

```text
http.request.method
http.referer
http.request.version
dns.qry.name.len
mqtt.conack.flags
mqtt.protoname
mqtt.topic
```

A 2026 benchmark-integrity analysis reports a serialization/provenance artifact in the common Edge-IIoTset categorical preprocessing path, including `0` versus `0.0` absence spellings in affected fields, capable of encoding file provenance strongly enough to leak the binary label. The conservative principal protocol therefore removes the seven implicated fields and canonicalizes missing spellings before encoding.

This safeguard is a benchmark-integrity rule, not a FedORBIT contribution.

## 4.6 Duplicate and row-canonicalization protocol

After labels and timestamp are parsed:

1. remove all forbidden identity/provenance fields;
2. convert every numeric feature to IEEE-754 float64 for canonical hashing;
3. canonical missing numeric value is one quiet NaN representation;
4. categorical values are UTF-8 normalized with Unicode NFC;
5. canonical row serialization uses Arrow IPC scalar representation in dataset-adapter feature order;
6. exact duplicate hash is SHA-256 of those canonical feature bytes;
7. duplicate rows differing only in label are conflicting duplicates.

If a duplicate feature group contains more than one local prediction label, the dataset is Invalid Data.

Duplicate groups are never split across partitions.

## 4.7 Model architecture specification

### Network/flow classifier

```text
input
→ Linear(256)
→ LayerNorm
→ GELU
→ Dropout
→ Linear(128)
→ LayerNorm
→ GELU
→ Dropout
→ Linear(64)
→ GELU
→ Linear(number_of_local_prediction_classes)
```

Locked layer semantics:

* `LayerNorm.eps = 1e-5`;
* `LayerNorm.elementwise_affine = true`;
* GELU approximation mode = exact / `"none"`;
* Dropout probability is the client-selected value from the registered pilot grid;
* all Linear weights use Xavier-uniform initialization with gain 1;
* all Linear biases initialize to zero.

### Host classifier

```text
input
→ Linear(192)
→ ReLU
→ BatchNorm1d
→ Dropout
→ Linear(96)
→ ReLU
→ Dropout
→ Linear(48)
→ ReLU
→ Linear(number_of_local_prediction_classes)
```

Locked semantics:

* ReLU `inplace=false`;
* `BatchNorm1d.eps = 1e-5`;
* `BatchNorm1d.momentum = 0.1`;
* `BatchNorm1d.affine = true`;
* `BatchNorm1d.track_running_stats = true`;
* evaluation uses accumulated running statistics;
* Dropout probability is the selected pilot value;
* all Linear weights use Kaiming-uniform initialization with `a=0`, `mode="fan_in"`, `nonlinearity="relu"`;
* all Linear biases initialize to zero.

### Numeric precision

* model parameters and forward/backward training: float32;
* preprocessing statistics: float64;
* response aggregation: float64;
* metric aggregation: float64;
* optimization-model coefficients: float64.

## 4.8 Base training specification

The numerical training parameters retained in YAML are configuration data. AdamW selection, scheduler absence, batching/shuffle semantics, early-stopping metric, checkpoint selection and contents, loss reduction, memory-loader behavior, and other procedural choices below are fixed scientific/execution rules.

| Training parameter / fixed rule | Locked value              |
| ------------------------------- | ------------------------- |
| optimizer                       | AdamW                     |
| AdamW beta1                     | 0.9                       |
| AdamW beta2                     | 0.999                     |
| AdamW epsilon                   | $10^{-8}$                 |
| AdamW AMSGrad                   | false                     |
| AdamW maximize                  | false                     |
| AdamW fused                     | false                     |
| AdamW foreach                   | false                     |
| maximum epochs                  | 50                        |
| batch size                      | 512                       |
| retain final partial batch      | yes                       |
| shuffle TRAIN each epoch        | yes                       |
| gradient clipping               | global L2 norm 1.0        |
| learning-rate scheduler         | none                      |
| early-stopping metric           | VALID macro cross-entropy |
| early-stopping patience         | 7 completed epochs        |
| minimum improvement             | $10^{-4}$                 |
| checkpoint metric tie tolerance | $10^{-6}$                 |
| label smoothing                 | 0                         |
| loss reduction                  | weighted arithmetic mean  |
| probability log floor           | `scientific.metrics.probability_log_floor` |
| DataLoader workers              | 0                         |
| pin CUDA host memory            | true for GPU training     |
| persistent workers              | false                     |

VALID is evaluated after every completed epoch.

The selected checkpoint is the epoch with minimum VALID macro-CE. Values within the checkpoint tie tolerance are tied; the earliest epoch wins.

The checkpoint contains:

* model parameters;
* optimizer state;
* completed epoch;
* RNG states;
* selected hyperparameters;
* TRAIN class weights.

## 4.9 Base-model pilot specification

The learning-rate, weight-decay, and dropout candidate lists are experiment-grid configuration data. Pilot selection order and TEST/CONFIRM inaccessibility are fixed rules.

Pilot seeds:

```text
101
202
303
```

Learning-rate grid:

```text
3e-4
1e-3
3e-3
```

Weight-decay grid:

```text
0
1e-4
```

Dropout grid:

```text
0
0.1
```

There are exactly 12 configurations per client and 36 pilot fits per client.

The pilot seed list is authoritative only at `scientific.randomness.pilot_seeds`; it is not duplicated under the pilot grid.

Selection:

1. smallest median VALID macro-CE over the three pilot seeds;
2. smallest standard deviation;
3. learning rate closest to $10^{-3}$;
4. smaller weight decay;
5. smaller dropout.

TEST and CONFIRM are inaccessible during pilot selection.

## 4.10 Local class-weight definition

For TRAIN class $c$,

$$
\omega_c=
\frac{N}{|\mathcal C|N_c}.
$$

After computing raw weights, divide all weights by their TRAIN-example weighted mean so that

$$
\frac1N\sum_i\omega_{y_i}=1.
$$

The per-example base loss is

$$
\ell_i=
-\omega_{y_i}
\log\max(p_i(y_i),10^{-12}).
$$

For minibatch \(\mathcal B\), with class-specific intervention/curriculum multiplier \(m_c\), the optimized scalar loss is exactly

$$
\mathcal L_{\mathcal B} =
\frac{1}{|\mathcal B|}
\sum_{i\in\mathcal B}
m_{y_i}\ell_i.
$$

The denominator is the minibatch example count, not the sum of class or intervention weights. For base training, \(m_c=1\). Intervention and curriculum multipliers modify the fixed class weights **without subsequent renormalization**.

## 4.11 Source-response pilot specification

Candidate magnitudes, optimizer-step horizons, replicate counts, thresholds, and pilot-score coefficients are configuration data. Half-magnitude repetition, estimator definitions, eligibility logic, and tie order are fixed rules.

Candidate intervention magnitude:

```text
0.05
0.10
0.20
```

Candidate optimizer-step horizon:

```text
25
50
100
```

Pilot paired schedules per candidate:

```text
8
```

Each candidate is also repeated at half its intervention magnitude with the same replicate schedules. For each magnitude separately, pilot replicate derivatives, \(\hat A_{ba}\), and \(SE_{ba}\) use the same paired-shadow derivative, arithmetic-mean estimator, and `ddof=1` standard-error equations as Section 4.12, with the pilot replicate count and candidate horizon.

Eligibility constants:

| Configuration                                | Value |
| -------------------------------------------- | ----: |
| relative derivative discrepancy ceiling      |  0.25 |
| sign-agreement minimum                       |  0.80 |
| useful response magnitude threshold          | 0.005 |
| minimum useful intervention columns          |     2 |
| curvature penalty coefficient in pilot score |     2 |

For a candidate, define

$$
D_{ba} =
\frac{
|\hat A_{ba}(\epsilon)-\hat A_{ba}(\epsilon/2)|
}{
\max(|\hat A_{ba}(\epsilon/2)|,0.005)
}.
$$

Define a pilot entry \((b,a)\) as useful when

$$
\max\left(
|\hat A_{ba}(\epsilon)|,
|\hat A_{ba}(\epsilon/2)|
\right)
\ge
\texttt{scientific.source＿response＿pilot.useful＿response＿magnitude＿threshold}.
$$

Linearity is evaluated only over useful entries.

For each useful entry, sign agreement is computed from the full-magnitude replicate estimates \(A_r(b,a;\epsilon)\) as the larger of the positive-sign and negative-sign replicate fractions; an exact zero replicate counts as disagreement.

The candidate is eligible only if:

1. every shadow state and loss is finite;
2. the median \(D_{ba}\) over useful entries is no greater than `scientific.source_response_pilot.relative_derivative_discrepancy_ceiling`;
3. the median sign agreement over useful entries is at least `scientific.source_response_pilot.sign_agreement_minimum`;
4. at least `scientific.source_response_pilot.minimum_useful_intervention_columns` intervention columns contain at least one useful entry.

If there are no useful entries, the candidate is ineligible.

Let \(\hat A_{ba}=\hat A_{ba}(\epsilon)\) and \(SE_{ba}=SE_{ba}(\epsilon)\) for the full candidate magnitude. The pilot score is

$$
Q =
\mathrm{median}_{(b,a)\in\mathcal U}
\left(
\frac{|\hat A_{ba}|}
{SE_{ba}+\texttt{scientific.source＿response＿pilot.numerical＿floor}}
\right) -
\texttt{scientific.source＿response＿pilot.curvature＿penalty＿coefficient}
\mathrm{median}_{(b,a)\in\mathcal U}
D_{ba},
$$

where \(\mathcal U\) is the candidate's useful-entry set.

Tie order:

1. higher $Q$;
2. smaller optimizer horizon;
3. smaller intervention magnitude.

One source-response configuration is selected per local model family/client before confirmatory packet construction.

The pilot useful-column minimum and the final-packet stability minimum are phase-specific scientific thresholds. They are configured separately even though both are currently 2; they must not be collapsed solely because the current values match.

## 4.12 Final source-response specification

Replicate counts, confidence level, bootstrap count, numerical floors, and final stability thresholds are configuration data. Paired-shadow semantics, bootstrap construction, quantile interpolation, and no-clipping behavior are fixed rules.

| Parameter / fixed response rule                     |                           Locked value |
| --------------------------------------------------- | -------------------------------------: |
| paired replicates per intervention                  |                                     24 |
| simultaneous confidence level                       |                                    95% |
| max-$|t|$ bootstrap resamples                       |                                   2,000 |
| response-risk denominator floor                     |                              $10^{-8}$ |
| response standard-error floor                       |                             $10^{-12}$ |
| useful response magnitude threshold                 |                                  0.005 |
| minimum useful intervention columns                 |                                      2 |
| maximum median band-width / median absolute mean-response ratio |                             4 |
| bootstrap critical quantile rule                    | empirical confidence-level quantile using `higher` |
| common random numbers within positive/negative pair |                               required |

All source-response and target-local-response shadows update on TRAIN minibatches only. Every \(R_b(\theta)\) used by the response pilot, final source packet, or matched target-local response diagnostic is evaluated in model evaluation mode on META only, over the fixed retained native classes belonging to transfer concept \(b\), using the equal-native-class risk definition in Section 6.4. VALID, CONFIRM, and TEST never enter a response derivative.

For source intervention $a$, positive and negative shadows start from the exact same checkpoint, optimizer state, and RNG state.

The positive shadow multiplies class $a$'s base weight by $1+\epsilon_R$.

The negative shadow multiplies it by $1-\epsilon_R$.

For outcome class $b$,

$$
A_r(b,a)=
\frac{
R_b(\theta^-_{r,a})-
R_b(\theta^+_{r,a})
}{
2\epsilon_R
\max(R_b(\theta_0),10^{-8})
}.
$$

Let $R$ be replicate count. For vectorized response entry $e$,

$$
\hat A_e=\frac1R\sum_r A_{re},
$$

$$
SE_e=
\frac{\mathrm{sd}(A_{1e},\ldots,A_{Re};ddof=1)}
{\sqrt R}.
$$

For bootstrap replicate $q$, resample paired replicate indices with replacement, calculate $\hat A_e^{*(q)}$ and $SE_e^{*(q)}$, then

$$
T_q=
\max_e
\left|
\frac{
\hat A_e^{*(q)}-\hat A_e
}{
\max(SE_e^{*(q)},10^{-12})
}
\right|.
$$

Let $q_{0.95}$ be the empirical `higher` quantile at `scientific.source_response_final.simultaneous_confidence_level`. The quantile probability is derived from that confidence level and is not separately configurable. Then

$$
L_e=\hat A_e-q_{0.95}SE_e,
$$

$$
U_e=\hat A_e+q_{0.95}SE_e.
$$

No clipping of $L$, $U$, or $\hat A$ is applied.

A final response entry \(e=(b,a)\) is useful only when both:

$$
|\hat A_e|
\ge
\texttt{scientific.source＿response＿final.useful＿response＿magnitude＿threshold},
$$

and its simultaneous interval excludes zero:

$$
L_e\gt 0
\quad\text{or}\quad
U_e\lt 0.
$$

A final intervention column is useful when it contains at least one useful entry. The final packet stability rule counts useful columns by this definition and computes the configured median band-width / median absolute mean-response ratio over this useful-entry set only. If the useful-entry set is empty, the packet fails the final response stability rule.

## 4.13 Target-local response diagnostic specification

The matched target-local response artifact uses:

| Configuration                    |              Value |
| -------------------------------- | -----------------: |
| intervention magnitude           |               0.10 |
| shadow horizon                   | 25 optimizer steps |
| paired replicates                |                  8 |
| simultaneous bootstrap resamples |              1,000 |
| confidence level                 |                95% |

Its estimator and max-$|t|$ band construction are otherwise identical to the final source-response procedure.

## 4.14 Confirmation and assimilation specification

Step counts, replicate/bootstrap counts, confidence level, and acceptance threshold are configuration data. Hierarchical resampling, lower-tail derivation, interpolation, curriculum multiplication, and no-renormalization behavior are fixed rules.

| Parameter / fixed confirmation rule           |                  Value |
| --------------------------------------------- | ---------------------: |
| confirmation optimizer steps per shadow       |                    100 |
| paired confirmation replicates                |                     10 |
| hierarchical bootstrap resamples              |                  2,000 |
| one-sided confidence level                    |                    95% |
| confirmation lower-bound acceptance threshold | 0.01 relative macro-CE |
| maximum source proposals                      | `scientific.action.maximum_source_proposals_per_target` |
| accepted live-assimilation steps              |                    500 |

For each confirmation bootstrap resample:

1. sample ten replicate indices with replacement;
2. for each selected replicate, independently resample CONFIRM examples with replacement within each fixed evaluation class;
3. compute baseline macro-CE;
4. compute curriculum macro-CE;
5. compute relative gain;
6. average across sampled replicates.

The one-sided lower confidence bound uses the empirical lower-tail probability derived as one minus `scientific.confirmation.one_sided_confidence_level`; under the registered confidence level this is the 5th percentile. Linear quantile interpolation is fixed and is not separately configurable.

Acceptance requires

$$
LCB_{0.95}
\ge
\texttt{scientific.confirmation.lower＿bound＿acceptance＿threshold＿relative＿macro＿ce}.
$$

The current configured acceptance threshold is 0.01 relative macro-CE.

The live curriculum weight multiplier for actionable target class $j$ is

$$
m_j=1+\alpha_j.
$$

No post-multiplication renormalization is performed.

A confirmation replicate's TRAIN schedule is generated from the `confirmation_schedule` namespace and follows the same infinite-pass construction as Section 7.2: fresh deterministic TRAIN-row permutation at each pass, configured batch-size chunks, retained final partial batch, and a new deterministic permutation when more steps are required. The paired baseline and curriculum shadows for that replicate consume the identical schedule.

Live assimilation uses a separate `assimilation_schedule` namespace. Its canonical coordinates are exactly:

```text
target_client
directed_pair
condition
seed
clean_pretransfer_checkpoint_artifact_id
source_packet_artifact_id
action_artifact_sha256
```

The method name is intentionally excluded so that two registered methods applying the identical source/action to the identical clean target state receive the identical live TRAIN schedule. The stream generates the same infinite-pass TRAIN schedule construction and consumes exactly `scientific.confirmation.accepted_live_assimilation_steps` optimizer steps. Confirmation shadow schedules are never reused as live-assimilation schedules.

## 4.15 Target-local optimizer-step budget

Maximum target-local optimizer steps before final TEST for one method × directed-pair × seed cell:

```text
10,000
```

Reserved budget:

| Purpose                             | Maximum |
| ----------------------------------- | ------: |
| target-local response diagnostic    |   3,200 |
| up to three confirmation candidates |   6,000 |
| live assimilation                   |     500 |
| nontransferable safety reserve      |     300 |

The target-response maximum follows from eight interventions × eight replicate pairs × two shadows × 25 steps.

Unused budget from one category cannot be transferred to another category or another method.

## 4.16 Exact-sparse solver specification

Numerical tolerances, limits, thread counts, concurrency, and deterministic solver seed are configuration data. Backend choice, algorithm, presolve/parallelism semantics, certificate rules, and deterministic LAP tie handling are fixed solver behavior.

| Configuration                              |              Value |
| ------------------------------------------ | -----------------: |
| LP primal feasibility tolerance            |          $10^{-9}$ |
| LP dual feasibility tolerance              |          $10^{-9}$ |
| LP optimality tolerance                    |          $10^{-9}$ |
| separator/cut stopping tolerance           |          $10^{-8}$ |
| exact validation tolerance                 | $10^{-9}$ absolute |
| permutation certificate residual tolerance |         $10^{-10}$ |
| action tie tolerance                       |         $10^{-10}$ |
| action tie comparison rounding precision   |         $10^{-12}$ |
| LAP objective tie tolerance                |         $10^{-12}$ |
| maximum cuts per support                   |                500 |
| LP threads per solve                       |                  1 |
| maximum concurrently executed supports     |                  4 |

LP backend:

* HiGHS;
* simplex algorithm;
* presolve enabled;
* internal parallelism disabled;
* deterministic random seed 0.

Support-level parallel execution is allowed up to the configured concurrency because each support is scientifically independent.

When multiple LAP solutions are optimal within `solvers.exact_sparse.lap_objective_tie_tolerance`, choose the lexicographically smallest assignment by target-row order and source-node pseudonymous ID. Library-specific tie behavior is not authoritative.

## 4.17 Generic exact-QAP specification

Numerical gap/tolerance, time limit, thread count, and seed are configuration data. SCIP/PySCIPOpt selection, presolve/cut/heuristic settings, binary assignment formulation, exact product linearization, method-level use, and timeout semantics are fixed solver behavior.

| Configuration           |     Value |
| ----------------------- | --------: |
| relative MIP gap        | $10^{-9}$ |
| feasibility tolerance   | $10^{-9}$ |
| wall-time cap per solve |   3,600 s |
| threads                 |         1 |
| random seed             |         0 |

Backend:

* SCIP through PySCIPOpt;
* presolve enabled;
* cuts enabled;
* primal heuristics enabled;
* binary assignment variables;
* exact RLT/McCormick product linearization.

Let \(p_{aj}=1\) mean target padded node \(j\) is mapped to source padded node \(a\). Variables exist only when source and target nodes belong to the same coarse group. For each coarse block \(g\),

$$
\sum_{a\in\mathcal G_g^{source}}p_{aj}=1
\quad
\forall j\in\mathcal G_g^{target},
$$

$$
\sum_{j\in\mathcal G_g^{target}}p_{aj}=1
\quad
\forall a\in\mathcal G_g^{source}.
$$

For fixed action \(\alpha\), introduce \(y_{abkj}=p_{ak}p_{bj}\) for every coefficient with \(w_k\alpha_jL_{ab}\ne0\). With binary \(p\), impose exactly

$$
0\le y_{abkj}\le p_{ak},
\qquad
0\le y_{abkj}\le p_{bj},
$$

$$
y_{abkj}\ge p_{ak}+p_{bj}-1.
$$

The fixed-action QAP objective is

$$
\min_{p,y}
\sum_{k,j,a,b}
w_k\alpha_jL_{ab}y_{abkj}.
$$

The curriculum cost \(-c^T\alpha\) is constant during correspondence separation and is added when reporting the full map-conditioned objective \(J(\alpha;P)\).

### Generic Exact QAP method

When `Generic Exact QAP` appears as a transfer method, it solves the **same principal sparse robust-action problem** as `FedORBIT Exact-Sparse Solver`, with the same support enumeration, action polytope, robust master cuts, zero-action candidate, action tie rules, confirmation, assimilation, and target compute budget. The only algorithmic substitution is the fixed-action correspondence separator:

```text
FedORBIT Exact-Sparse active-image + LAP separator
→ replaced by
certified Generic Exact QAP separator
```

For every master candidate action, SCIP must certify the fixed-action worst correspondence before that separator result may be used as an exact cut/stopping certificate. If any required separator call reaches the configured time/resource limit without an optimal certificate, that method cell has outcome `Time Limit`/`Resource Limit`; no incumbent is promoted as an exact robust action and no downstream confirmation/TEST result is produced for that method cell.

This definition makes `Generic Exact QAP` a solver-method comparator rather than a different scientific transfer objective.

### Point-correspondence structural QAP

For Point-Correspondence Commitment, the roadmap objective

$$
\min_{P\in\Pi}
\lVert P^TL_sP-L_t\rVert_F^2
$$

is implemented through the permutation-invariant expansion

$$
\lVert L_s\rVert_F^2
+
\lVert L_t\rVert_F^2 -
2\langle P^TL_sP,L_t\rangle_F.
$$

The first two terms are constant, so SCIP minimizes the equivalent quadratic-assignment objective \(-2\langle P^TL_sP,L_t\rangle_F\) with the same binary assignment/product linearization. The registered point-map tie tolerance and lexicographic correspondence rule remain authoritative.

A QAP run is exact only when SCIP returns an optimal certificate satisfying the configured MIP gap and feasibility tolerance. A time-limited incumbent is never relabeled exact.

Registered uses: exact small/medium comparator cases, method-level robust-action comparison where listed, point-map structural matching, exactness verification when certified, and applicability/map-recovery diagnostics.

## 4.18 Dense-CCP specification

Penalty multipliers, iteration/cut/start limits, tolerances, wall-time limit, and thread count are configuration data. The dense action master, continuous lifted correspondence relaxation, penalty formula, deterministic starts, CCP update, projection, stopping rules, and non-exactness boundary are fixed solver behavior.

The Dense-CCP fallback uses the full dense action set \(\mathcal A\); it has no \(\ell_0\) support restriction.

### Continuous lifted correspondence relaxation

Use the same assignment orientation as Section 4.17, but relax

$$
0\le p_{aj}\le1.
$$

The blockwise row/column assignment equalities remain exact. Introduce \(y_{abkj}\) for every nonzero coefficient and impose the same McCormick inequalities

$$
0\le y_{abkj}\le p_{ak},
\qquad
0\le y_{abkj}\le p_{bj},
$$

$$
y_{abkj}\ge p_{ak}+p_{bj}-1.
$$

For fixed action \(\alpha\), the unpenalized lifted LP minimizes

$$
F_\alpha(p,y) =
\sum_{k,j,a,b}
w_k\alpha_jL_{ab}y_{abkj}.
$$

Because the feasible set relaxes the permutation set, its optimum \(v_{\rm relax}(\alpha)\) is a lower bound on the true fixed-action worst-correspondence response term.

### Integrality penalty and CCP update

For assignment variables,

$$
\phi(p) =
\sum_{a,j}p_{aj}(1-p_{aj}).
$$

At CCP iterate \(p^{(t)}\), use the affine majorization of the concave term:

$$
\phi_{\rm lin}(p;p^{(t)}) =
\sum_{a,j}
\left[
(1-2p^{(t)}_{aj})p_{aj}
+
(p^{(t)}_{aj})^2
\right].
$$

At penalty \(\lambda\), the next iterate is the optimum of the LP

$$
\min_{p,y}
F_\alpha(p,y)
+
\lambda\phi_{\rm lin}(p;p^{(t)})
$$

subject to the continuous lifted constraints above.

Penalty scaling uses

$$
M=
\max\left(
1,
\max_{a,b,j,k}
|w_k\alpha_jL_{ab}|
\right),
$$

and the penalty levels are exactly

```text
0.1 M
1 M
10 M
100 M
1000 M
```

Configuration:

| Dense setting                            |     Value |
| ---------------------------------------- | --------: |
| maximum CCP iterations per penalty level |        50 |
| assignment integrality residual          | $10^{-8}$ |
| relative objective convergence tolerance | $10^{-8}$ |
| deterministic starts                     |         5 |
| outer action cuts                        |     1,000 |
| wall-time cap                            |   3,600 s |
| LP threads                               |         1 |

The assignment integrality residual is

$$
r_{\rm int}(p) =
\max_{a,j}\min(p_{aj},1-p_{aj}).
$$

Relative objective convergence between consecutive CCP iterates uses the unpenalized \(F_\alpha\) value:

$$
\frac{
|F_\alpha^{(t+1)}-F_\alpha^{(t)}|
}{
\max(1,|F_\alpha^{(t)}|)
}
\le
\texttt{solvers.dense＿ccp.relative＿objective＿convergence＿tolerance}.
$$

A penalty-level trajectory may stop before 50 iterations only when both this objective criterion and the configured integrality-residual criterion hold. Otherwise it executes the configured maximum iterations and proceeds to the next penalty level from the last iterate. The final convergence state records whether the final penalty level met both criteria.

### Deterministic starts

The start set is:

1. one blockwise uniform barycenter, \(p_{aj}=1/n_g\) inside each allowed block;
2. the lexicographically smallest admissible block permutation;
3. up to three additional unique admissible block permutations generated from the `dense_start` namespace by independently shuffling source-node order within each block and pairing it with ascending target pseudonymous order.

Thus there are at most five starts total and at most four permutation starts. If the orbit contains fewer than four unique permutations, use every unique permutation exactly once in lexicographic order after deduplication.

Each start executes the complete penalty ladder. The final iterate of one penalty level initializes the next level.

### Projection

After every complete CCP trajectory, project its final \(p\) to the nearest admissible block permutation by solving, independently in every block,

$$
\max_P
\sum_j p_{P(j),j},
$$

using the deterministic LAP tie rule from Section 4.16. Evaluate the unpenalized correspondence objective on the projected permutation. The best projected correspondence is the one with the smallest unpenalized fixed-action objective; ties use the global lexicographic correspondence rule.

### Dense robust-action outer master

Initialize the scenario set with the lexicographically smallest admissible block permutation. Repeatedly solve

$$
\max_{\alpha,z} z
$$

subject to

$$
\alpha\in\mathcal A
$$

and, for every accumulated projected permutation \(P_r\),

$$
z
\le
w^TP_r^TLP_r\alpha-c^T\alpha.
$$

For each master action \(\alpha^{(q)}\):

1. solve the unpenalized lifted correspondence LP to obtain \(v_{\rm relax}(\alpha^{(q)})\);
2. run every deterministic CCP start and project each trajectory;
3. choose the lowest-objective projected permutation \(P_{\rm proj}^{(q)}\);
4. evaluate \(J(\alpha^{(q)};P_{\rm proj}^{(q)})\);
5. if the projected permutation is new and violates the master value by more than the exact-sparse separator/cut stopping tolerance, add it as a scenario cut;
6. otherwise terminate the outer loop with heuristic convergence.

The outer loop terminates at the first of:

* heuristic convergence above;
* `solvers.dense_ccp.outer_action_cuts` scenario additions;
* the configured wall-time cap;
* a solver/resource failure.

The wall-time cap covers the complete dense method cell, including all outer-master LPs, lifted relaxation solves, CCP trajectories, and projections.

At the returned action, record:

* master action and objective;
* best admissible projected worst-correspondence objective;
* unpenalized lifted relaxation lower bound;
* `Dense Bound Gap = projected response objective - relaxation lower bound`;
* assignment integrality residual;
* outer cut count;
* CCP convergence state;
* timeout/resource state.

The projected permutation supplies a feasible correspondence objective for the returned action. The unpenalized continuous lifted LP supplies only a fixed-action relaxation lower bound. Neither quantity certifies the global dense robust optimum.

Dense-CCP is a relaxation/heuristic. It never establishes dense exactness or supplies exactness truth. Registered uses are dense-support sensitivity, fallback diagnostics, and scalability comparisons.

## 4.19 Baseline definitions

Baseline algorithms, resource access, tie semantics, and oracle restrictions are fixed definitions. Only the point-correspondence QAP tie tolerance is retained as baseline configuration data.

### Local-Only

No transfer, target-local fixed base checkpoint only.

### Local-SIR

Uses target TRAIN/META response artifact and solves

$$
\max_{\alpha\in\mathcal A^{(s)}}
w^TL_t\alpha-c^T\alpha.
$$

It receives the same action budget, support, confirmation, live-assimilation, and target compute cap as FedORBIT.

### FedORBIT Without Confirmation

This condition is identical to the principal `FedORBIT Exact-Sparse Solver` transfer pipeline through source-packet validation, target importance, robust action construction, positive-value filtering, and deterministic source ranking.

It differs only at the confirmation decision:

1. CONFIRM is not read and no confirmation shadow is executed;
2. if at least one positive proposal exists, select the first proposal in the same ranking order that the principal method would attempt;
3. return to the same clean pre-transfer target checkpoint/optimizer/RNG state;
4. apply that proposal's curriculum weights;
5. execute the same configured live-assimilation step count using the same `assimilation_schedule` derivation coordinates;
6. open TEST only after assimilation and the normal pre-TEST artifact contract passes;
7. if no positive proposal exists, remain Local-Only.

The unused confirmation-step reservation is not transferred to live assimilation or any other computation.

For the single-source `Target Confirmation and Portability` comparison, this method is the authoritative no-confirm counterfactual. Its TEST relative macro-CE gain is `noConfirmCounterfactualGain` for the corresponding principal proposal/decision.

### Matched-resource rectangular

Uses

$$
\ell_{kj} =
\min_{P\in\Pi}
(P^TLP)_{kj}
$$

and solves

$$
\max_{\alpha\in\mathcal A^{(s)}}
w^T\ell\alpha-c^T\alpha.
$$

### Point-correspondence commitment

Chooses

$$
\hat P
\in
\arg\min_{P\in\Pi}
\lVert P^TL_sP-L_t\rVert_F^2
$$

with generic exact QAP.

Ties within $10^{-10}$ choose the lexicographically smallest correspondence.

It then optimizes action under $\hat P$.

### Coarse block-mean summary

For source coarse groups $g,h$,

$$
B_{gh} =
\mathrm{mean}
\lbrace
L_{ab}:
a\in\mathcal G_g^{source,real},
b\in\mathcal G_h^{source,real}
\rbrace.
$$

Lift to the target fine space by

$$
\tilde L_{kj}=B_{g(k),g(j)}.
$$

Null target coordinates retain zero action cap.

### Coarse block-min summary

$$
B^{min}_{gh} =
\min
\lbrace
L_{ab}:
a\in\mathcal G_g^{source,real},
b\in\mathcal G_h^{source,real}
\rbrace.
$$

Lift identically to the fine target space.

For both coarse summaries, if an ordered source coarse-block pair contains no real source entry after endpoint eligibility, define its block summary as exactly zero:

$$
B_{gh}=0,
\qquad
B^{min}_{gh}=0.
$$

This is the explicit null-evidence continuation of the roadmap's zero-response null-node semantics; an empty-set mean or minimum is never evaluated.

### Orbit-mean summary

Use

$$
\bar L =
\mathbb E_{P\sim Uniform(\Pi)}[P^TLP].
$$

Compute this analytically blockwise rather than enumerating the orbit:

* cross-group terms use the mean of all entries in the corresponding source block pair;
* same-group diagonal positions use the mean source block diagonal;
* same-group off-diagonal positions use the mean source block off-diagonal.

Action is optimized against $\bar L$ without a robust minimum.

### Coupling-destroyed packet

For each ordered coarse block pair $(g,h)$:

1. vectorize the corresponding $L$ and $U$ entries in row-major order;
2. derive one deterministic permutation from the packet seed namespace;
3. apply the same permutation to $L$ and $U$;
4. reshape.

This preserves:

* block sizes;
* empirical lower-value multiset;
* empirical upper-value multiset;
* interval pairing;
* packet dimensions;

while destroying row/column procedural geometry.

### Exact-map oracle

Uses the hidden benchmark or synthetic exact correspondence and otherwise the same action, confirmation, and assimilation budgets.

Oracle information may not influence non-oracle fitting, pilot selection, or evidence-bearing method execution.

## 4.20 Target importance definition

The class-risk floor is configuration data. META-only construction and zero weights for Normal/null nodes are fixed scientific rules.

For actionable non-null target node $k$,

$$
w_k=
\frac{
\max(R_k,10^{-4})
}{
\sum_{q\in actionable}
\max(R_q,10^{-4})
},
$$

where $R_k$ is pre-transfer target META class-conditional cross-entropy.

Normal and null nodes have $w=0$.

META is the only split used to construct target importance.

## 4.21 Seed and RNG specification

The pilot and confirmatory seed lists are configuration data. The derivation hash, serialization, namespaces, and prohibition on unscoped RNG calls are fixed reproducibility rules.

### Pilot seeds

```text
101
202
303
```

### Confirmatory seeds

```text
1103
2207
3319
4421
5531
6653
7753
8861
9973
11027
```

No confirmatory seed may be added, replaced, or removed after principal execution begins.

### Statistical base seed

```text
300
```

The statistical base seed is used only to derive deterministic bootstrap/resampling streams. Every statistical stream includes the complete contrast/family coordinates in `canonicalCoordinates`, so different contrasts do not share a bootstrap stream.

### Seed derivation

Every secondary RNG stream is generated as:

$$
seed32=
\mathrm{SHA256}
(
UTF8(
"FedORBIT|"
\Vert baseSeed
\Vert "|"
\Vert namespace
\Vert "|"
\Vert canonicalCoordinates
)
)_{0:8}
\bmod2^{32}.
$$

`canonicalCoordinates` is canonical JSON with:

* keys sorted lexicographically;
* UTF-8 encoding;
* no insignificant whitespace;
* decimal numbers serialized in shortest round-trippable form.

Required namespaces include:

```text
split
model_initialization
train_epoch_shuffle
response_schedule
response_bootstrap
anonymous_node_order
confirmation_schedule
confirmation_bootstrap
assimilation_schedule
statistical_bootstrap
synthetic_instance
coupling_destruction
dense_start
```

No global unscoped RNG call is permitted in scientific code.

## 4.22 Deterministic compute contract

For principal model execution:

* CUDA is required;
* `torch.use_deterministic_algorithms(True)`;
* cuDNN benchmark disabled;
* cuDNN deterministic mode enabled;
* TF32 disabled for matrix multiplication and cuDNN;
* stochastic mixed precision disabled;
* automatic mixed precision disabled;
* model training remains float32;
* CUDA synchronization is performed around timed GPU regions.

Smoke/unit tests may use CPU.

Any operation that cannot satisfy deterministic execution is prohibited from principal execution unless this roadmap explicitly defines a repeatability tolerance for it.

## 4.23 Statistical specification

| Statistical parameter / fixed rule |             Value |
| ---------------------------------- | ----------------: |
| nominal confidence level        |                  95% |
| nominal alpha                   | `1 - scientific.statistics.confidence_level` = 0.05 |
| superiority tests               |            two-sided |
| continuous primary paired test  |      exact sign-flip |
| CI method                       | paired BCa bootstrap |
| CI bootstrap repetitions        |               10,000 |
| minimum valid paired seeds      |                    8 |
| multiplicity correction         |       Holm step-down |
| TOST alpha per one-sided test   |                 0.05 |
| Spearman minimum valid points   |                    5 |
| exact McNemar/asymptotic switch |  25 discordant pairs |

Nominal alpha is derived as one minus the configured nominal confidence level; it is not an independent configuration field. The test family, sidedness, paired statistic, BCa method, Holm procedure, and tie order below are fixed statistical rules.

The paired continuous test statistic is the arithmetic mean paired difference.

For $n_{\rm eff}$ nonzero differences, enumerate all $2^{n_{\rm eff}}$ sign patterns whenever $n_{\rm eff}\le20$. Confirmatory seed counts make enumeration mandatory in this study.

The exact two-sided p-value is

$$
p=
\frac{
\left\lvert\lbrace
|\bar d^{perm}|
\ge
|\bar d^{obs}|-10^{-15}
\rbrace\right\rvert
}{
2^{n_{\rm eff}}
}.
$$

If every paired difference is zero, $p=1$.

BCa resampling samples paired seed indices with replacement.

If all paired differences are numerically identical within $10^{-15}$, the confidence interval is exactly the point value rather than an undefined BCa interval.

Holm ties are ordered by:

1. raw p-value;
2. lexicographic contrast name.

## 4.24 Multiplicity families

The exact families are:

```text
Primary Transfer vs Local-Only
External Source vs Local-SIR
Coupling Mechanism
Point-Correspondence Safety
Mechanism Ablations
Sparsity Sensitivity
Confirmation Safety
```

No global correction is performed across these scientifically distinct families.

No unregistered confirmatory contrast may be added after the first evidence-bearing TEST outcome is opened.

## 4.25 Claim/materiality criteria

### Strict cross-telemetry utility

Supported only when:

1. at least four of the six primary directed pairs have mean relative macro-CE gain vs Local-Only at least `scientific.materiality.realized_relative_macro_ce`;
2. each successful pair has Holm-adjusted $p$ below the configured claim threshold;
3. each successful pair has BCa lower bound strictly greater than the configured zero boundary;
4. no primary pair has mean gain at or below `scientific.materiality.harmful_transfer_relative_macro_ce_gain`;
5. the equal-pair mean of the six pair means is at least `scientific.materiality.realized_relative_macro_ce`;
6. all contributing runs satisfy strict-resource validation.

The configured required successful-pair count, Holm threshold, and BCa boundary are under `scientific.evaluation_criteria.strict_cross_telemetry_utility`. The materiality and harm thresholds are shared scientific materiality values rather than duplicated claim-specific configuration.

If any primary pair is removed from scope before principal outcome inspection by the Dataset, Client, and Strict-Resource Validation rules, full-scope `Supported` is unavailable. No reduced-scope positive state is permitted.

### External source value vs Local-SIR

Uses the same materiality, no-material-harm, Holm, CI, equal-pair, and one-pair pre-outcome reduced-scope structure versus Local-SIR, with its configured required successful-pair count, Holm threshold, and BCa boundary under `scientific.evaluation_criteria.external_source_value_vs_local_sir`.

### Coupling mechanism

Requires all of:

1. the configured theorem zero/strict classification accuracy on the controlled designed family;
2. at least the configured valid-packet fraction have coupling gap at least `scientific.materiality.coupling_objective_units`;
3. at least three of the six primary pairs have mean gap at least `scientific.materiality.coupling_objective_units` with Holm-adjusted $p$ below the configured coupling threshold;
4. coupling destruction does not satisfy the registered mechanism-retention condition below.

For primary pair \(p\), let \(\bar G^{full}_p\) and \(\bar G^{destroyed}_p\) be the seed-mean TEST relative macro-CE gains versus the identical Local-Only reference. Define

$$
Retention_p =
\frac{\bar G^{destroyed}_p}{\bar G^{full}_p}
$$

only when \(\bar G^{full}_p\gt 0\); otherwise `Retention_p = NA`. A pair is a **mechanism-retention pair** only when both:

1. the Full-vs-Coupling-Destroyed TOST contrast for that pair establishes equivalence after Holm correction in the `Mechanism Ablations` family; and
2. \(Retention_p\ge\texttt{scientific.claim＿criteria.coupling＿mechanism.destruction＿positive＿gain＿retention＿minimum}\).

The registered mechanism-retention condition is present when at least `scientific.evaluation_criteria.coupling_mechanism.primary_pairs_with_material_mean_gap_required` primary pairs are mechanism-retention pairs. `NA` retention values never count toward this condition.

### Sparse operational relevance

Requires:

1. the registered compared sparse support lies no more than the configured dense-minus-sparse gain ceiling below dense on at least the configured fraction of valid primary pair-seed units;
2. at least one of the predeclared sparse supports $s=2$ or $s=3$ has pair-mean realized gain at least `scientific.materiality.realized_relative_macro_ce` on at least the configured number of primary pairs;
3. exact-sparse solver correctness remains valid.

No TEST-driven choice between $s=2$ and $s=3$ is used to define the principal method.

### Confirmation safety

For each pair define:

$$
ARR=
harmRate_{noConfirm}-harmRate_{confirm}.
$$

Where baseline harmful rate is positive,

$$
RRR=
\frac{ARR}{harmRate_{noConfirm}}.
$$

The claim requires:

1. at least four of the six primary directed pairs satisfy $ARR\ge0.02$ or $RRR\ge0.30$;
2. those qualifying pairs have mean coverage loss $\le0.20$;
3. no primary pair has harmful-rate worsening $\gt 0.02$;
4. no primary pair loses more than 0.20 coverage;
5. equal-pair mean satisfies either $ARR\ge0.02$ or, when defined, $RRR\ge0.30$.

Pair-specific seed-level rate differences use the registered exact sign-flip procedure.

## 4.26 Resource and efficiency specification

| Configuration                                   | Value                            |
| ----------------------------------------------- | -------------------------------- |
| reference model GPU                             | NVIDIA GeForce RTX 5060 Ti 16 GB |
| solver CPU worker ceiling                       | 4                                |
| host RAM ceiling for registered efficiency runs | 16 GiB                           |
| deterministic kernel warmups                    | 3                                |
| deterministic kernel timed repetitions          | 10                               |
| primary runtime summary                         | median and p95                   |

Full training executions are timed once per scientific cell. Training is not repeated merely to manufacture timing samples.

## 4.27 Environment configuration

Dependency version identifiers are configuration data. The resolved-lockfile requirement and the prohibition on dependency upgrades once evidence-bearing confirmatory execution has begun are fixed reproducibility rules.

The implementation environment is fixed to:

| Dependency      | Required version |
| --------------- | -------------: |
| Python          |        3.13.12 |
| PyTorch         |         2.13.0 |
| NumPy           |          2.5.2 |
| SciPy           |         1.18.0 |
| scikit-learn    |          1.9.0 |
| pandas          |          3.0.5 |
| PyArrow         |         25.0.1 |
| highspy / HiGHS |         1.15.1 |
| PySCIPOpt       |          6.2.1 |
| Pydantic        |         2.13.4 |
| Typer           |         0.27.1 |
| psutil          |          7.2.2 |
| pytest          |          9.1.1 |
| pytest-cov      |          7.1.0 |

The repository must contain a fully resolved lockfile containing transitive package versions and package hashes.

No dependency upgrade is permitted after the first evidence-bearing confirmatory cell has begun execution.

## 4.28 Reporting specification

Decimal precision counts and the p-value display threshold are configuration data. The literal `<0.0001` rendering, MiB unit, integer count rendering, confidence-interval syntax, and use of unrounded figure source values are fixed reporting rules.

| Quantity                     | Display                |
| ---------------------------- | ---------------------- |
| scientific metric            | 4 decimals             |
| macro-F1 / balanced accuracy | 4 decimals             |
| p-value                      | 4 decimals             |
| p-value below 0.0001         | `<0.0001`              |
| runtime seconds              | 3 decimals             |
| memory                       | MiB, 1 decimal         |
| counts                       | integers               |
| CI                           | `estimate [low, high]` |

Figures always use unrounded source values.

## 4.29 Test and smoke execution configuration

`configs/fedorbit.yaml` is the sole YAML configuration authority. Tests and smoke execution consume its typed configuration and may restrict workload only through typed execution plans; they never redefine scientific values, registered cells, model architecture, horizons, seeds, thresholds, support, or metrics.

# Configuration YAML

Production scientific execution uses one authoritative configuration file: `configs/fedorbit.yaml`. The contents below are complete and authoritative for configuration data only. Fixed scientific and execution behavior is defined in the corresponding roadmap sections and must not be re-encoded as YAML prose, formulas, references, or procedural strings.

```yaml
scientific:
  action:
    principal_sparse_support: 2
    sparse_support_sensitivity: [1, 3]
    total_curriculum_budget: 0.50
    coordinate_cap: 0.25
    linear_cost_per_actionable_node: 0.01
    positive_source_value_threshold: 0.0
    maximum_source_proposals_per_target: 3
  materiality:
    coupling_objective_units: 0.005
    realized_relative_macro_ce: 0.01
    macro_f1_absolute: 0.005
    equivalence_relative_macro_ce:
      lower: -0.01
      upper: 0.01
    harmful_transfer_relative_macro_ce_gain: -0.01
    useful_transfer_relative_macro_ce_gain: 0.01

  transfer_support:
    source_train_minimum: 200
    source_meta_minimum: 40
    target_meta_minimum: 40
    target_confirm_minimum: 40
    target_test_minimum: 40
    local_prediction_attack_class_total_rows_minimum: 200
    minimum_actionable_target_concepts: 4
    minimum_nontrivial_block_size: 2

  datasets:
    clients:
      edge_iiotset_network:
        role: external
        source: Edge-IIoTset
        component: "network/IoT traffic"
        expected_timestamp_field: "frame.time"
      ton_iot_windows10_host:
        role: primary
        source: ToN_IoT
        component: "Windows 10 OS telemetry"
        expected_timestamp_field: "ts"
      ton_iot_linux_process_host:
        role: primary
        source: ToN_IoT
        component: "Linux process telemetry"
        expected_timestamp_field: "ts"
      ton_iot_network:
        role: primary
        source: ToN_IoT
        component: "network-flow telemetry"
        expected_timestamp_field: "ts"
    timestamp_alias_acceptance:
      retained_row_parse_success_minimum: 0.999
    primary_directed_pairs:
    - source: ton_iot_windows10_host
      target: ton_iot_linux_process_host
    - source: ton_iot_linux_process_host
      target: ton_iot_windows10_host
    - source: ton_iot_windows10_host
      target: ton_iot_network
    - source: ton_iot_network
      target: ton_iot_windows10_host
    - source: ton_iot_linux_process_host
      target: ton_iot_network
    - source: ton_iot_network
      target: ton_iot_linux_process_host
    secondary_directed_pairs: []
    local_prediction_normal_label: Normal

  split:
    duplicate_safe_chronological_intervals:
      train: [0.00, 0.55]
      meta: [0.55, 0.70]
      valid: [0.70, 0.80]
      confirm: [0.80, 0.90]
      test: [0.90, 1.00]
  preprocessing:
    missing_indicator_train_rate_threshold: 0.001
    rare_category_train_frequency_threshold: 0.001
    feature_missing_or_nonfinite_drop_threshold: 0.05
    client_invalidity_dropped_feature_fraction_threshold: 0.20
    numeric_clip:
      lower: -10.0
      upper: 10.0
    zero_iqr_replacement_scale: 1.0
  training:
    adamw:
      beta1: 0.9
      beta2: 0.999
      epsilon: 1.0e-8
    maximum_epochs: 50
    batch_size: 512
    gradient_clip_global_l2_norm: 1.0
    early_stopping:
      patience_completed_epochs: 7
      minimum_improvement: 1.0e-4
    checkpoint:
      tie_tolerance: 1.0e-6
    label_smoothing: 0.0
    dataloader_workers: 0
  base_model_pilot:
    learning_rates: [0.0003, 0.001, 0.003]
    weight_decays: [0.0, 0.0001]
    dropouts: [0.0, 0.1]
  source_response_pilot:
    intervention_magnitudes: [0.05, 0.10, 0.20]
    optimizer_step_horizons: [25, 50, 100]
    paired_schedules_per_candidate: 8
    relative_derivative_discrepancy_ceiling: 0.25
    sign_agreement_minimum: 0.80
    useful_response_magnitude_threshold: 0.005
    minimum_useful_intervention_columns: 2
    curvature_penalty_coefficient: 2.0
    numerical_floor: 1.0e-12
  source_response_final:
    paired_replicates_per_intervention: 24
    simultaneous_confidence_level: 0.95
    max_t_bootstrap_resamples: 2000
    response_risk_denominator_floor: 1.0e-8
    response_standard_error_floor: 1.0e-12
    useful_response_magnitude_threshold: 0.005
    minimum_useful_intervention_columns: 2
    median_band_width_to_median_absolute_mean_response_maximum: 4.0
  target_response_diagnostic:
    intervention_magnitude: 0.10
    shadow_optimizer_steps: 25
    paired_replicates: 8
    simultaneous_bootstrap_resamples: 1000
    confidence_level: 0.95

  confirmation:
    optimizer_steps_per_shadow: 100
    paired_replicates: 10
    hierarchical_bootstrap_resamples: 2000
    one_sided_confidence_level: 0.95
    lower_bound_acceptance_threshold_relative_macro_ce: 0.01
    accepted_live_assimilation_steps: 500
  target_optimizer_budget:
    maximum_steps_per_method_pair_seed_before_test: 10000
    reserved:
      target_response_diagnostic: 3200
      confirmation_candidates: 6000
      live_assimilation: 500
      nontransferable_safety_reserve: 300
  baselines:
    point_correspondence_commitment:
      qap_tie_tolerance: 1.0e-10
  target_importance:
    class_risk_floor: 1.0e-4
  randomness:
    pilot_seeds: [101, 202, 303]
    confirmatory_seeds: [1103, 2207, 3319, 4421, 5531, 6653, 7753, 8861, 9973, 11027]
    statistical_seed: 300
  statistics:
    confidence_level: 0.95
    exact_sign_flip_max_nonzero_differences_for_enumeration: 20
    exact_sign_flip_comparison_tolerance: 1.0e-15
    ci_bootstrap_repetitions: 10000
    identical_difference_tolerance: 1.0e-15
    minimum_valid_paired_seeds: 8
    tost_alpha_per_one_sided_test: 0.05
    spearman_minimum_valid_points: 5
    mcnemar_exact_to_asymptotic_discordant_pair_switch: 25
  evaluation_criteria:
    strict_cross_telemetry_utility:
      successful_primary_pairs_required: 4
      holm_adjusted_p_maximum: 0.05
      bca_lower_bound_strictly_greater_than: 0.0
    external_source_value_vs_local_sir:
      successful_primary_pairs_required: 4
      holm_adjusted_p_maximum: 0.05
      bca_lower_bound_strictly_greater_than: 0.0
    coupling_mechanism:
      theorem_zero_strict_classification_accuracy_required: 1.0
      real_packet_fraction_with_material_gap_minimum: 0.25
      primary_pairs_with_material_mean_gap_required: 3
      holm_adjusted_p_maximum: 0.05
      destruction_positive_gain_retention_minimum: 0.90
    sparse_operational_relevance:
      compared_sparse_support: 3
      dense_minus_sparse_gain_maximum: 0.01
      valid_unit_fraction_required: 0.75
      primary_pairs_with_useful_gain_required: 3
    confirmation_safety:
      qualifying_primary_pairs_required: 4
      absolute_risk_reduction_minimum: 0.02
      relative_risk_reduction_minimum: 0.30
      qualifying_pair_coverage_loss_maximum: 0.20
      pair_harmful_rate_worsening_maximum: 0.02
      pair_coverage_loss_maximum: 0.20
      equal_pair_absolute_risk_reduction_minimum: 0.02
      equal_pair_relative_risk_reduction_minimum: 0.30

  metrics:
    probability_log_floor: 1.0e-12
    relative_macro_ce_denominator_floor: 1.0e-12
    relative_solver_error_denominator_floor: 1.0e-12
  multi_source_selection:
    communication_cost_coefficient_in_principal_ranking: 0.0
    confirmation_cost_coefficient_in_principal_ranking: 0.0

  simplification_rules:
    rectangularization_is_sufficient:
      valid_real_packet_fraction_below_coupling_materiality_minimum: 0.90
    generic_qap_dominates:
      intended_sparse_support_maximum: 2
      median_runtime_ratio_to_exact_sparse_maximum: 1.0
      p95_runtime_ratio_to_exact_sparse_maximum: 1.2
      peak_memory_ratio_to_exact_sparse_maximum: 1.0
    sparse_support_is_operationally_irrelevant:
      dense_gain_advantage_over_support_3_minimum: 0.02
      valid_primary_unit_fraction_minimum: 0.75
      sparse_supports_that_must_fail_useful_materiality: [1, 2, 3]
    point_matching_is_sufficient:
      harmful_rate_worsening_maximum: 0.02
      utility_advantage_over_fedorbit_minimum: 0.01
    strict_interface_removes_gain:
      primary_pair_majority_required: 3
      point_gain_maximum: 0.0
      bca_upper_bound_maximum: 0.01
    source_response_is_too_unstable:
      principal_source_packet_failure_fraction_strictly_greater_than: 0.50
solvers:
  exact_sparse:
    lp_primal_feasibility_tolerance: 1.0e-9
    lp_dual_feasibility_tolerance: 1.0e-9
    lp_optimality_tolerance: 1.0e-9
    separator_cut_stopping_tolerance: 1.0e-8
    exact_validation_absolute_tolerance: 1.0e-9
    permutation_certificate_residual_tolerance: 1.0e-10
    action_tie_tolerance: 1.0e-10
    action_tie_comparison_rounding_precision: 1.0e-12
    lap_objective_tie_tolerance: 1.0e-12
    maximum_cuts_per_support: 500
    lp_threads_per_solve: 1
    maximum_concurrent_supports: 4
    deterministic_random_seed: 0
  generic_exact_qap:
    relative_mip_gap: 1.0e-9
    feasibility_tolerance: 1.0e-9
    wall_time_seconds_per_solve: 3600
    threads: 1
    random_seed: 0
  dense_ccp:
    penalty_multipliers_relative_to_scale: [0.1, 1.0, 10.0, 100.0, 1000.0]
    maximum_iterations_per_penalty_level: 50
    assignment_integrality_residual: 1.0e-8
    relative_objective_convergence_tolerance: 1.0e-8
    deterministic_starts: 5
    outer_action_cuts: 1000
    wall_time_seconds: 3600
    lp_threads: 1
generators:
  exact_separator_theorem:
    response_uniform: [-0.20, 0.20]
    serialization_upper_band_increment_uniform: [0.0, 0.05]
    target_importance_gamma:
      shape: 2.0
      scale: 1.0
    active_action_uniform: [0.05, 0.25]
    block_patterns:
    - [2]
    - [3]
    - [4]
    - [2, 2]
    - [2, 3]
    - [3, 3]
    supports: [1, 2, 3]
    generated_instances_per_block_pattern_support_seed_cell: 100
  coupling_structure:
    unconstrained_response_uniform: [-0.10, 0.10]
    compatibility: [jointly_realizable, incompatible]
    response_heterogeneity: [0.5, 1.0, 2.0]
    directed_asymmetry: [0.0, 0.5, 1.0]
    response_sparsity: [0.25, 0.5, 1.0]
    block_patterns:
    - [2, 2]
    - [2, 3]
    - [3, 3]
    supports: [1, 2, 3]
    incompatible_fixed_action_gap_strictly_greater_than: 1.0e-6
    maximum_attempts_per_instance: 10000
  common_action_unresolved_map:
    block_pattern: [2, 2]
    block_pair_response_uniform: [0.04, 0.12]
    maximum_attempts: 1000
  robust_compromise_unresolved_map:
    block_pattern: [2, 2]
    response_uniform: [-0.10, 0.20]
    robust_pre_map_value_strictly_greater_than: 0.005
    maximum_attempts_per_fixture: 100000
  map_dependent:
    block_pattern: [2, 2]
    response_uniform: [-0.15, 0.25]
    map_value_minimum: 0.01
    maximum_attempts: 100000
  scalability:
    response_uniform: [-0.10, 0.10]
    block_patterns: [balanced, maximally_skewed_two_block]
experiments:
  mathematical_primitive_validation:
    hand_fixture_seed: 0
    fixture_error_tolerance: 1.0e-10
    invalid_permutations_allowed: 0
  exact_sparse_solver_benchmark:
    synthetic_k:
      minimum: 4
      maximum: 18
    block_patterns: [balanced, maximally_skewed]
    supports: [1, 2, 3]
    exhaustive_truth_correspondence_count_maximum: 100000
    methods:
    - "FedORBIT Exact-Sparse Solver"
    - "Generic Exact QAP"
    - "FedORBIT Dense-CCP Fallback"
  synthetic_coupling_mechanism_validation:
    methods: [exact_orbit, "Matched-Resource Rectangular", "Coupling-Destroyed FedORBIT"]
  common_action_under_unidentified_map:
    fixtures_per_seed: 50
  robust_compromise_under_unidentified_map:
    fixtures_per_seed: 50
  map_dependent_action_boundary:
    fixtures_per_seed: 50
  exact_map_value_bound_validation:
    zero_map_value_fixtures_per_seed: 25
    high_map_value_fixtures_per_seed: 25
  primary_strict_cross_telemetry_transfer:
    methods:
    - "Local-Only"
    - "Local-SIR"
    - "Matched-Resource Rectangular"
    - "Point-Correspondence Commitment"
    - "Generic Exact QAP"
    - "FedORBIT Exact-Sparse Solver"
    - "Exact-Map Oracle"
  multi_source_selection_validation:
    targets:
    - ton_iot_windows10_host
    - ton_iot_linux_process_host
    - ton_iot_network
  mechanism_ablations:
    methods:
    - "FedORBIT Exact-Sparse Solver"
    - "Matched-Resource Rectangular"
    - "Point-Correspondence Commitment"
    - "Coupling-Destroyed FedORBIT"
    - "Coarse Block-Mean"
    - "Coarse Block-Min"
    - "Orbit-Mean"
    - "Local-SIR"
  target_confirmation_and_portability:
    methods:
    - "FedORBIT Exact-Sparse Solver"
    - "FedORBIT Without Confirmation"
  secondary_cross_modality_generalization:
    methods:
    - "Local-Only"
    - "Local-SIR"
    - "Matched-Resource Rectangular"
    - "Point-Correspondence Commitment"
    - "FedORBIT Exact-Sparse Solver"
  semantic_sufficiency_frontier:
    partitions:
    - oracle_fine_singleton_groups
    - principal_three_coarse_groups
    - ["Disruption or Exploitation", "Access and Discovery"]
    - one_attack_supergroup
    methods:
    - "FedORBIT Exact-Sparse Solver"
    - "Matched-Resource Rectangular"
    - "Exact-Map Oracle"
  weak_signal_support_and_heterogeneity_boundaries:
    response_scales: [1.0, 0.75, 0.5, 0.25, 0.0]
    ci_half_width_multipliers: [1.0, 1.5, 2.0, 4.0]
    target_usable_support_fractions: [1.0, 0.5, 0.25, 0.1]
    response_heterogeneity_multipliers: [0.5, 1.0, 2.0]
    support_budgets: [1, 2, 3]
    methods:
    - "FedORBIT Exact-Sparse Solver"
    - "Matched-Resource Rectangular"
    - "Local-Only"
  map_availability_applicability_audit:
    packet_only_recovery_methods:
    - "Point-Correspondence Commitment"
    - "Generic Exact QAP"
    independent_researchers: 2
    minutes_per_researcher_per_pair: 60
  scalability_and_efficiency:
    k_values: [6, 8, 10, 12, 16, 20, 24, 32]
    block_patterns: [balanced, maximally_skewed]
    exact_qap_supports: [1, 2, 3]
runtime:
  failure_handling:
    retries_after_initial_infrastructure_failure: 2
  reference_model_gpu: "NVIDIA GeForce RTX 5060 Ti 16 GB"
  solver_cpu_worker_ceiling: 4
  host_ram_ceiling_gib_for_registered_efficiency_runs: 16
  deterministic_kernel_warmups: 3
  deterministic_kernel_timed_repetitions: 10
  full_training_timing_repetitions_per_scientific_cell: 1

  artifact_layout:
    execution_root: outputs
    manuscript_root: results
    preprocessing_subdirectories:
    - inventories
    - validation
    - prepared
    - splits
    - features
    - metadata
    reusable_artifact_subdirectories:
    - models
    - scores
    - fitted
    - baselines
    - derived
    experiment_subdirectories:
      artifacts:
      - fitted
      - predictions
      - derived
      evaluations:
      - records
      - comparisons
      - aggregates
      metrics:
      - per_seed
      - per_condition
      - aggregate
      statistics:
      - tests
      - confidence_intervals
      - effects
      - multiplicity
      checkpoints:
      - training
      - execution
      diagnostics:
      - scientific
      - numerical
      - runtime
      logs:
      - execution
      - failures
      provenance:
      - configuration
      - data
      - seeds
      - code
      - environment
      - dependencies
    cache_subdirectories:
    - preprocessing
    - models
    - evaluation
    - analysis
    - staging
    manuscript_experiment_subdirectories:
      figures:
      - main
      - supplementary
      tables:
      - main
      - supplementary
      metrics:
      - primary
      - secondary
      - summary
      statistics:
      - tests
      - confidence_intervals
      - effects
      - multiplicity
    project_summary_subdirectories:
      figures:
      - main
      - supplementary
      tables:
      - main
      - supplementary
      metrics:
      - primary
      - summary
      statistics:
      - comparisons
      - confidence_intervals
      - effects
      - multiplicity
      reproducibility:
      - configuration
      - datasets
      - seeds
      - software
      - execution

environment:
  python: "3.13.12"
  pytorch: "2.13.0"
  numpy: "2.5.2"
  scipy: "1.18.0"
  scikit_learn: "1.9.0"
  pandas: "3.0.5"
  pyarrow: "25.0.1"
  highspy_highs: "1.15.1"
  pyscipopt: "6.2.1"
  pydantic: "2.13.4"
  typer: "0.27.1"
  psutil: "7.2.2"
  pytest: "9.1.1"
  pytest_cov: "7.1.0"
reporting:
  precision:
    scientific_metric_decimals: 4
    macro_f1_decimals: 4
    balanced_accuracy_decimals: 4
    p_value_decimals: 4
    p_value_less_than_threshold: 0.0001
    runtime_seconds_decimals: 3
    memory_decimals: 1
```

# 5. Strict Information and Resource Regime

## 5.1 Source-local information

A source may use only:

* its TRAIN;
* its META;
* its VALID;
* its own local fine labels;
* local coarse-group membership;
* local preprocessing;
* local classifier;
* local optimizer/checkpoint;
* local intervention experiments.

## 5.2 Target-local information

A target may use only:

* its TRAIN;
* META;
* VALID;
* CONFIRM;
* TEST only after the transfer decision is fully finalized;
* local fine labels internally;
* shared coarse groups;
* anonymous source packet;
* local model state;
* local importance vector.

## 5.3 Permitted transmitted source packet

The packet contains only:

* anonymous fine-node IDs;
* exposed coarse-group ID;
* $L$;
* $U$;
* per-node TRAIN support;
* per-node META support;
* per-node effective replicate count;
* packet schema metadata;
* source checkpoint SHA-256;
* response-configuration SHA-256;
* packet integrity SHA-256;
* packet validity state;
* technical creation timestamp.

The technical creation timestamp is RFC 3339 UTC metadata only. It is excluded from scientific identity, dependency fingerprints, solver inputs, node ordering, and `packet integrity SHA-256`.

`packet integrity SHA-256` is computed from the canonical UTF-8 serialization of every permitted packet field **except** the `packet_integrity_sha256` field itself and the technical creation timestamp. Map keys are lexicographically sorted, numeric arrays are serialized in declared row-major float64 form, and anonymous-node/coarse-group arrays retain their explicit packet order. The reusable-artifact payload checksum separately hashes the complete serialized packet file, including the technical timestamp, so physical corruption of the timestamp remains detectable without making it scientific identity.

Fine semantic names are forbidden.

## 5.4 Forbidden cross-client information

The method must never receive:

* raw source samples;
* source feature names as an alignment bridge;
* model parameters;
* gradients;
* shared embeddings;
* prototypes;
* aligned latent coordinates;
* fine semantic names;
* exact fine correspondence;
* common entity IDs;
* aligned timestamps;
* common executable queries;
* hidden map hints in filenames;
* semantic ordering hints in node order;
* map information in caches, metadata, logs, or manifests available to the method.

## 5.5 Anonymous node order

Within every coarse group and endpoint:

1. collect padded nodes;
2. use the `anonymous_node_order` RNG namespace;
3. independently shuffle source and target order;
4. assign display IDs:

```text
node-0001
node-0002
...
```

IDs are local to the packet/target state and reveal no semantic name.

## 5.6 Oracle namespace

The following information is oracle-only:

* true benchmark fine concept;
* exact benchmark source-target mapping;
* true synthetic correspondence;
* exhaustive orbit outputs used as truth;
* fine semantic labels in oracle tables.

Non-oracle code attempting oracle-path access must fail closed.

## 5.7 Strict-resource validation

Every principal cell must pass:

1. exact packet-field whitelist;
2. independent source/target anonymous ordering;
3. absence of fine semantic names in method-readable artifacts;
4. disjoint local feature namespaces;
5. absence of cross-client entity IDs;
6. absence of cross-client timestamp pairing;
7. oracle ACL isolation;
8. TEST ACL isolation before pre-TEST decision finalization;
9. resource-manifest equality with method catalogue;
10. static leakage scan;
11. dynamic access-log scan.

Any violation makes the scientific cell Invalid.

# 6. Dataset, Client, Ontology, and Preprocessing Protocol

Dataset/component identifiers, directed-pair lists, split boundaries, preprocessing thresholds, and transfer-support thresholds are configured in `scientific.datasets`, `scientific.split`, `scientific.preprocessing`, and `scientific.transfer_support` in `configs/fedorbit.yaml`. The transfer ontology, component-selection rules, split semantics, preprocessing procedure, and eligibility logic are fixed scientific definitions in this section.

## 6.1 Real benchmark clients

Primary benchmark client identities are telemetry-domain proxies, not independent real organizations:

* ToN-IoT Windows 10 Host Client;
* ToN-IoT Linux Process Host Client;
* ToN-IoT Network Client.

External-only client:

* Edge-IIoTset Network Client, retained solely for a separately configured robustness extension after a valid event-time release is supplied.

## 6.2 Primary directed pairs

Exactly:

```text
ToN-IoT Windows 10 Host → ToN-IoT Linux Process Host
ToN-IoT Linux Process Host → ToN-IoT Windows 10 Host
ToN-IoT Windows 10 Host → ToN-IoT Network
ToN-IoT Network → ToN-IoT Windows 10 Host
ToN-IoT Linux Process Host → ToN-IoT Network
ToN-IoT Network → ToN-IoT Linux Process Host
```

## 6.3 Secondary directed pairs

No secondary directed pairs are preregistered. Edge-IIoTset is not part of the
confirmatory campaign because its selected table does not provide a uniquely resolvable
event time. A future external extension requires an independently valid timestamped
release and a separately documented analysis plan.

## 6.4 Candidate hidden transfer ontology

| Exposed coarse group | Oracle fine concept | Edge candidate mapping  | ToN candidate mapping          |
| -------------------- | ------------------- | ----------------------- | ------------------------------ |
| Disruption         | DDoS                | aggregate DDoS variants | DDoS                           |
| Disruption         | Ransomware          | Ransomware              | ransomware if present/eligible |
| Exploitation       | Backdoor            | Backdoor                | backdoor if present/eligible   |
| Exploitation       | Injection           | SQL/injection family    | injection                      |
| Exploitation       | XSS                 | XSS                     | xss                            |
| Access and Discovery   | Password attack     | Password                | password                       |
| Access and Discovery   | Scanning            | aggregate scan variants | scanning                       |
| Access and Discovery   | MITM                | MITM                    | mitm if present/eligible       |

After the label canonicalization rule in Section 4.4, transfer membership is exactly:

| Oracle transfer concept | Edge canonical local labels | ToN canonical local labels |
| --- | --- | --- |
| DDoS | `ddos_udp`, `ddos_icmp`, `ddos_tcp`, `ddos_http` | `ddos` |
| Ransomware | `ransomware` | `ransomware` |
| Backdoor | `backdoor` | `backdoor` |
| Injection | `sql_injection` | `injection` |
| XSS | `xss` | `xss` |
| Password attack | `password` | `password` |
| Scanning | `port_scanning`, `fingerprinting`, `vulnerability_scanner` | `scanning` |
| MITM | `mitm` | `mitm` |

Release-equivalent spelling differences are handled only by the canonicalization rule; semantic aliases outside this table are not guessed. Edge `uploading` and ToN `dos` remain eligible local prediction classes when they meet the local support rule but are not transfer concepts in the principal ontology.

A transfer concept may contain one or more retained dataset-native prediction classes. Let \(\mathcal C_a\) be the native-class set belonging to transfer concept \(a\).

For intervention concept \(a\), the positive or negative intervention multiplier is applied identically to the fixed base class weight of every \(c\in\mathcal C_a\). Classes outside \(\mathcal C_a\) retain multiplier 1.

For outcome concept \(b\), its risk is the equal-native-class mean

$$
R_b(\theta) =
\frac{1}{|\mathcal C_b|}
\sum_{c\in\mathcal C_b}
CE_c(\theta).
$$

This equal-class aggregation prevents a multi-label transfer concept such as Edge DDoS or Scanning from being dominated by its largest native subclass. Transfer support counts are the sum of row counts over the constituent native classes, while the risk itself remains the equal-native-class mean above.

Every retained native class may belong to at most one transfer concept. A canonical label matching more than one concept or an undocumented label that would require semantic inference is not assigned to the transfer ontology.

Normal is a local prediction class only. It never enters the correspondence orbit and never receives curriculum action.

The ToN_IoT Windows and Linux sources do not guarantee support for every candidate attack concept; endpoint-specific eligibility and null padding are therefore mandatory rather than forcing a false complete ontology.

## 6.5 Local prediction label set

For each client:

1. retain Normal;
2. retain every dataset-native attack class meeting the configured total-row threshold;
3. exclude lower-support attack classes;
4. record each exclusion before model fitting.

The transfer ontology is a subset of these retained classes.

## 6.6 Raw-data identity

For every raw file record:

* dataset/release;
* component;
* canonical relative path;
* byte size;
* SHA-256;
* acquisition source descriptor;
* acquisition timestamp;
* license/use note.

After successful raw inventory, the raw dataset tree becomes read-only.

A checksum change produces a different raw-data lineage and invalidates dependent preprocessing.

## 6.7 Universal cleaning order

Execute exactly:

1. identify dataset component and resolve the adapter schema/feature order;
2. parse timestamp;
3. parse local binary/multiclass labels in builder namespace;
4. apply categorical missing-token normalization according to the resolved adapter types;
5. remove all label fields from features;
6. remove absolute timestamp from model features;
7. remove IP, MAC, host, process-instance, flow/entity identifiers where they identify individual entities rather than behavioral measurements;
8. remove raw payload strings;
9. remove source filenames, capture filenames, row numbers, and provenance fields;
10. apply the Edge-specific excluded-field list where applicable;
11. normalize numeric infinities to missing;
12. canonicalize feature rows;
13. form duplicate groups;
14. reject conflicting duplicate-label groups;
15. determine chronological split;
16. compute pre-imputation TRAIN feature-quality statistics on the raw candidate-feature set;
17. drop features exceeding the configured quality threshold and invalidate the client when the configured dropped-feature fraction is exceeded;
18. fit all remaining preprocessing on TRAIN only;
19. impute retained numeric features with TRAIN median;
20. construct configured missingness indicators from the pre-imputation TRAIN missing rates;
21. robust-scale numeric values;
22. remove constant TRAIN features;
23. clip numeric values;
24. construct TRAIN categorical vocabularies;
25. apply rare/unknown mapping;
26. one-hot encode;
27. derive local class manifest;
28. derive transfer eligibility;
29. create null padding;
30. materialize immutable preprocessing artifacts.

No resampling, SMOTE, or TEST-informed feature selection is permitted.

## 6.8 Chronological duplicate-safe split

Within each retained local class:

1. group exact duplicate feature rows;
2. order groups by earliest valid event timestamp;
3. break equal timestamps by duplicate-group SHA-256;
4. compute group midpoint fraction
$$
   f_g=\frac{r_{\rm before}+0.5n_g}{N_c};
$$
5. assign using the authoritative split intervals.

A duplicate group is indivisible.

If any retained row lacks a usable timestamp, the client is Invalid Data.

There is no principal random-split fallback.

## 6.9 Pair eligibility

A pair-seed is valid only if:

* configured source/target support rules pass;
* at least scientific.transfer_support.minimum_actionable_target_concepts non-null actionable target concepts remain;
* at least one target coarse block contains at least scientific.transfer_support.minimum_nontrivial_block_size real nodes;
* at least one source response block contains at least that many real nodes;
* all strict-resource checks pass.

A primary directed pair requires at least the configured minimum of valid paired seeds for confirmatory inference.

## 6.10 Dataset-observed values

The following are observed, never hardcoded from papers:

* raw row count;
* per-class row count;
* exact timestamp range;
* duplicate count;
* conflicting-duplicate count;
* exact feature count;
* exact class count;
* exact null pattern;
* exact padded block sizes;
* exact source packet count after eligibility;
* exact dataset checksum.

`fedorbit preprocess` records these values.

### Observed repository release, 2026-08-30

The immutable tables selected from the repository release were inspected directly. The
Edge-IIoTset network table is
`Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv`; it has
2,219,201 data rows, 63 columns, `frame.time`, `Attack_label`, and `Attack_type`.
Its observed local-class counts are Normal 1,615,643; Backdoor 24,862; DDoS_HTTP
49,911; DDoS_ICMP 116,436; DDoS_TCP 50,062; DDoS_UDP 121,568; Fingerprinting 1,001;
MITM 1,214; Password 50,153; Port_Scanning 22,564; Ransomware 10,925;
SQL_injection 51,203; Uploading 37,634; Vulnerability_scanner 50,110; and XSS 15,915.
The observed `frame.time` cells are additionally invalid for chronology: 2,096,419 have
the form `YYYY HH:MM:SS.fraction`, which omits calendar month and day, while 122,782
are not timestamp-shaped after CSV parsing. A datetime library can coerce the former by
inventing a date, but that is not a uniquely resolvable event time. Therefore this
selected Edge table is also Invalid Data for the chronological protocol in Section 6.8;
preprocessing records that validation result and does not use file order, row order, or
an inferred date.

The former `Train_Test_*` selections are not used: their lack of `ts` makes them
unsuitable for chronological splitting. The primary inputs are the ToN-IoT Processed
Windows 10, Linux process, and numbered network tables named in Section 4.4. They are
not present in this checkout as of 2026-08-30, so no row count, timestamp range, or
chronology result is asserted here. On acquisition, preprocessing must verify `ts` for
every selected file, require a single consistent schema within each logical client, and
record the observed parse result. It must not fall back to file order, row order, or an
inferred timestamp.

# 7. Local Models and Procedural-Response Estimation

## 7.1 Base models

Instantiate the architecture corresponding to the client's modality exactly as specified in Section 4.7. Numerical training and pilot parameters are read from the Configuration YAML; the architecture itself is a fixed scientific definition and is not encoded in YAML.

Input dimension is derived from the fitted local preprocessor.

Output dimension is derived from the retained local prediction class manifest.

Neither dimension is independently configured.

## 7.2 Response shadow schedules

Each replicate receives a deterministic `response_schedule` seed.

A replicate schedule defines an infinite sequence of TRAIN minibatches:

1. generate a fresh permutation of TRAIN rows at each pass;
2. slice in configured batch-size chunks;
3. retain the final partial chunk;
4. begin a new deterministic permutation when the required optimizer-step horizon exceeds one pass.

Positive and negative shadows consume exactly the same minibatch sequence.

## 7.3 Source response packet

For each real eligible source intervention node:

1. execute the selected response configuration;
2. compute paired derivative replicates;
3. construct simultaneous bands;
4. pad absent transfer nodes with zero rows and columns;
5. independently anonymize node order;
6. serialize packet.

A packet is valid only for the exact:

* source checkpoint;
* preprocessing state;
* transfer-node manifest;
* selected response configuration;
* coarse groups;
* seed.

There is no offline staleness grace period.

# 8. FedORBIT Exact-Sparse Solver

## 8.1 Fixed-action exact separator

For support

$$
S=\lbrace j:\alpha_j\gt 0\rbrace,
$$

enumerate every block-compatible injective image

$$
\sigma:S\rightarrow[K].
$$

For fixed active images,

$$
C_0(\sigma) =
\sum_{k\in S}
w_k
\sum_{j\in S}
\alpha_jL_{\sigma(k),\sigma(j)}.
$$

For each remaining target outcome $k\notin S$ and unused source node $b$ in the same block,

$$
C^\sigma_{kb} =
w_k
\sum_{j\in S}
\alpha_jL_{b,\sigma(j)}.
$$

Solve one minimum-cost LAP for every coarse block with remaining nodes.

Combine:

* fixed active images;
* each blockwise LAP completion.

This produces one admissible full correspondence.

Choose the correspondence minimizing the fixed-action objective.

The exact work counters for one fixed-action separator call are

$$
ActiveImageCandidates=N_S,
$$

and

$$
LAPCalls =
N_S
\sum_g
\mathbf 1[n_g-s_g\gt 0].
$$

A size-one completion still counts as one LAP call because the implementation invokes the same deterministic assignment primitive.

The work-count target is

$$
O\left(
N_S\sum_gn_g^3
\right).
$$

## 8.2 Deterministic separator ties

If several correspondences have objective values within the configured LAP/action tie tolerance:

1. compare source images of target nodes in ascending target pseudonymous-ID order;
2. select the lexicographically smallest image sequence.

## 8.3 Robust master

For each allowed coordinate set $S$, solve

$$
\max_{\alpha,z}z
$$

subject to:

* $\alpha\in\mathcal A$;
* $\alpha_j=0$ for $j\notin S$;
* every accumulated scenario cut
$$
  z
  \le
  w^TP_r^TLP_r\alpha-c^T\alpha.
$$

Initial scenario:

* lexicographically smallest block permutation after sorting current pseudonymous IDs.

Iteration:

1. solve LP;
2. call exact separator;
3. compute separator objective $v_{\rm sep}$;
4. if $z-v_{\rm sep}$ is within configured cut tolerance, certify;
5. otherwise add the returned correspondence and continue;
6. hitting the configured support cut cap produces `Sparse Master Non-Convergence`; that support has no certified robust optimum.

Because the principal method requires comparison across every enumerated support up to the configured support budget, `Sparse Master Non-Convergence` on any required support prevents certification of the method cell. No action from the unconverged support, incumbent master, or other supports may be promoted as the principal exact action.

## 8.4 Support enumeration

Enumerate every actionable coordinate set with size from 1 through the configured support budget.

The zero action is an additional explicit candidate.

After optimization, actual zero-valued coordinates are removed before applying deterministic action tie rules.

Final candidate tie order:

1. larger certified robust value;
2. smaller realized support;
3. lexicographically smaller target pseudonymous-node sequence;
4. lexicographically smaller action vector after configured comparison rounding.

# 9. Multiple-Source Selection

For each target state:

1. evaluate every valid candidate source packet;
2. discard nonpositive certified robust values;
3. sort descending by certified robust value;
4. break ties by stable source client name;
5. attempt confirmation sequentially;
6. stop at first accepted candidate;
7. consider no more than the configured maximum;
8. if no candidate is accepted, remain local-only.

Communication and confirmation cost coefficients are zero in the principal scientific ranking objective.

No source-packet averaging is part of the principal method.

# 10. Target Confirmation and Assimilation

## 10.1 Proposal confirmation

For each source proposal:

1. clone the exact target pre-confirm model and optimizer twice;
2. execute paired baseline and curriculum shadows;
3. use identical minibatch schedules;
4. evaluate on CONFIRM only;
5. construct the configured hierarchical one-sided lower bound;
6. accept only when the configured lower bound reaches the configured materiality threshold.

## 10.2 Rejected proposal

A rejected proposal leaves the clean target state unchanged.

No confirmation shadow becomes the live target state.

## 10.3 Accepted proposal

For an accepted proposal:

1. return to the clean pre-confirm state;
2. apply curriculum weights;
3. perform the configured live-assimilation optimizer steps;
4. recompute target META risk and importance before any later source decision.

## 10.4 TEST opening rule

TEST may be read only after:

* source selection is finalized;
* action is finalized;
* confirmation decision is finalized;
* live assimilation is complete or explicitly rejected;
* all method-specific pre-TEST artifacts are committed.

Any earlier TEST read invalidates the cell.

# 11. Baseline Fairness Contract

The registered method set is represented by descriptive names in the roadmap and in the experiment configuration. No opaque baseline codes are used.

Registered methods are:

```text
Local-Only
Local-SIR
Coarse Block-Mean
Coarse Block-Min
Orbit-Mean
Matched-Resource Rectangular
Point-Correspondence Commitment
Generic Exact QAP
FedORBIT Exact-Sparse Solver
FedORBIT Dense-CCP Fallback
Exact-Map Oracle
FedORBIT Without Confirmation
Coupling-Destroyed FedORBIT
```

All relevant packet-based principal methods receive:

* the same source packet;
* the same target checkpoint;
* the same target importance;
* the same target-local diagnostic artifact availability;
* the same action budget;
* the same support budget when sparse;
* the same seed;
* the same confirmation opportunity;
* the same live-assimilation allowance.

No comparator may receive:

* additional TEST access;
* more target labels;
* additional tuning seeds;
* a larger target compute budget;
* a more favorable local base checkpoint.

# 12. Metric Definitions

Final metrics must come from one registered metric library. Table and figure code may not reimplement them.

## 12.1 Class-conditional cross-entropy

$$
CE_c=
\frac1{n_c}
\sum_{i:y_i=c}
-\log\max(p_i(c),10^{-12}).
$$

## 12.2 Macro cross-entropy

$$
CE_{\rm macro} =
\frac1{|\mathcal C|}
\sum_cCE_c.
$$

A fixed evaluation class with zero evaluation examples makes the cell Invalid Data.

## 12.3 Relative macro-CE gain

For method $m$ and reference $b$,

$$
G_{CE}(m,b) =
\frac{
CE_b-CE_m
}{
\max(CE_b,10^{-12})
}.
$$

If $CE_b\lt 10^{-12}$, the relative metric is NA; absolute CE difference is reported and the cell cannot establish a relative-gain claim.

## 12.4 Precision, recall, F1

$$
P_c=
\frac{TP_c}{TP_c+FP_c},
$$

$$
R_c=
\frac{TP_c}{TP_c+FN_c},
$$

$$
F1_c=
\frac{2P_cR_c}{P_c+R_c}.
$$

Any zero denominator returns 0 by explicit project rule.

Macro-F1 is the arithmetic mean over the fixed evaluation class set.

## 12.5 Balanced accuracy

$$
BA=
\frac1{|\mathcal C|}
\sum_cR_c.
$$

## 12.6 Mechanism metrics

### Certified robust predicted value

The principal certified robust action objective.

### Fixed-action rectangularization gap

$$
\Gamma(\alpha).
$$

### Robust coupling value gap

$$
G_{\rm coupling}.
$$

### Coupling upper-bound diagnostic

$$
\max_{\alpha\in\mathcal B}
w^T(\bar L-\ell)\alpha.
$$

### Exact-map action value

$$
\Delta_{\rm map}.
$$

### Orbit-radius map bound

For every experiment-specific action set \(\mathcal B\),

$$
0
\le
\Delta_{\rm map}(\mathcal B)
\le
2\rho_2
\lVert w\rVert_2
R_\alpha,
$$

where

$$
\rho_2=
\max_{P\in\Pi}
\left\lVert
P^TLP-\bar L
\right\rVert_2
$$

uses the matrix spectral norm, and

$$
R_\alpha=
\sup_{\alpha\in\mathcal B}
\lVert\alpha\rVert_2.
$$

For tractable controlled experiments, \(\rho_2\) and the exact map value are computed by complete orbit enumeration. The bound is evaluated in float64 and may exceed the exact value only by the configured exact-validation tolerance.

### Predicted-realized Spearman correlation

Compute only with the configured minimum valid point count.

Report:

* rho;
* n;
* pair.

It is descriptive, not a primary significance claim.

## 12.7 Solver metrics

```text
Absolute Objective Error
Relative Objective Error
Correspondence Certificate Validity
Active-Image Candidates
LAP Calls
Scenario-Cut Count
Master Iterations
Dense Relaxation Bound
Dense Projected Objective
Dense Bound Gap
Dense Integrality Residual
```

Relative objective error is

$$
\frac{|v-v^\star|}
{\max(|v^\star|,10^{-12})}.
$$

## 12.8 Confirmation metrics

A **proposal-eligible target decision** is a target decision for which at least one source proposal has positive certified robust value after the registered ranking/filtering rules. Decisions with no positive proposal are reported separately and do not enter confirmation-coverage or confirmation-safety denominators.

For proposal-level diagnostics:

$$
Proposal\ Acceptance\ Rate =
\frac{N_{accepted}}{N_{proposed}}.
$$

$$
Harmful\ Accepted\ Rate =
\frac{
N_{(accepted\land TESTGain\le-0.01)}
}{
N_{proposed}
}.
$$

$$
Useful\ Accepted\ Rate =
\frac{
N_{(accepted\land TESTGain\ge0.01)}
}{
N_{proposed}
}.
$$

If there are no proposals, all proposal-denominator rates are NA, not zero.

For target-decision coverage:

$$
Coverage_{confirm} =
\frac{
N_{(\text{proposal-eligible decisions ending in live transfer})}
}{
N_{(\text{proposal-eligible decisions})}
}.
$$

Under `FedORBIT Without Confirmation`, every proposal-eligible decision directly assimilates its first-ranked proposal, so

$$
Coverage_{noConfirm}=1
$$

whenever the denominator is nonzero. Define

$$
CoverageLoss =
Coverage_{noConfirm}-Coverage_{confirm}.
$$

For confirmation-safety harm, define the decision-level indicators

$$
H^{confirm} =
\mathbf 1[
TESTGain_{confirm}\le-0.01
],
$$

$$
H^{noConfirm} =
\mathbf 1[
TESTGain_{noConfirm}\le-0.01
]
$$

over the same proposal-eligible target decisions. A rejected confirmed decision remains at Local-Only and therefore has \(TESTGain_{confirm}=0\) by construction. For each pair and seed, `harmRate_confirm` and `harmRate_noConfirm` are the arithmetic means of these indicators over the valid proposal-eligible decisions represented by that pair-seed artifact. The primary pair design normally contains one such decision, yielding a rate of 0 or 1. Pair summaries are arithmetic means of the seed-level rates over valid paired seeds.

Then

$$
ARR =
harmRate_{noConfirm} -
harmRate_{confirm},
$$

and, when \(harmRate_{noConfirm}\gt 0\),

$$
RRR =
\frac{ARR}{harmRate_{noConfirm}}.
$$

`RRR` is NA when the no-confirm harmful rate is zero.

Pair-level `Coverage_confirm`, `Coverage_noConfirm`, `CoverageLoss`, `harmRate_confirm`, and `harmRate_noConfirm` are arithmetic means of their valid seed-level values. Project-level equal-pair harm rates and coverage losses are arithmetic means of the six primary pair means; a primary pair without the configured minimum valid paired seeds makes the full-scope confirmation result unavailable rather than changing the denominator. Equal-pair `ARR` is the difference of the equal-pair no-confirm and confirm harmful rates. Equal-pair `RRR` is computed from those equal-pair harmful rates, not by averaging pair-specific RRR values, and is `NA` when the equal-pair no-confirm harmful rate is zero.

`Beneficial Rejected Rate` is registered only for the single-source `Target Confirmation and Portability` experiment, where each rejected principal proposal has an exact paired `FedORBIT Without Confirmation` cell using that same source/action:

$$
Beneficial\ Rejected\ Rate =
\frac{
N_{(
rejected
\land
noConfirmCounterfactualGain\ge0.01
)}
}{
N_{proposed}
}.
$$

For multi-source diagnostics with later alternative proposals, `Beneficial Rejected Rate` is NA rather than requiring unregistered counterfactual live-assimilation runs.

## 12.9 Efficiency


Record:

* wall time;
* peak host RSS;
* peak CUDA allocated bytes;
* packet serialized byte count;
* source response optimizer steps;
* target confirmation optimizer steps;
* live assimilation optimizer steps;
* timeout indicator;
* resource-limit indicator.

# 13. Statistical Analysis Protocol

## 13.1 Experimental unit

The confirmatory unit is a seed within one fixed directed pair.

The four directed pairs are fixed benchmark conditions, not independent draws from a population of organizations.

No inferential test may treat 40 pair-seed cells as 40 exchangeable organizational samples.

## 13.2 Pairing requirements

A paired contrast is valid only when both methods share:

* raw dataset lineage;
* pair;
* seed;
* split;
* target pre-transfer checkpoint;
* target importance;
* source packet when required;
* action/support budget when matched;
* confirmation budget;
* environment lineage.

The comparison engine rejects mismatched cells.

## 13.3 Continuous outcomes

Use the configured exact sign-flip test.

Point summaries:

* arithmetic mean paired difference;
* median paired difference.

For BCa confidence intervals, use the pinned SciPy implementation `scipy.stats.bootstrap` from the configured SciPy version with exactly:

```text
data = (method_values, reference_values)
statistic = arithmetic mean of (method_values - reference_values)
paired = true
vectorized = false
method = "BCa"
alternative = "two-sided"
confidence_level = scientific.statistics.confidence_level
n_resamples = scientific.statistics.ci_bootstrap_repetitions
rng = numpy.random.Generator(PCG64(derived_statistical_seed))
```

`derived_statistical_seed` is derived from `scientific.randomness.statistical_seed` through the `statistical_bootstrap` namespace with coordinates containing the exact contrast name, family, pair, metric, and CI purpose.

The statistic is always the paired arithmetic mean difference; BCa jackknife and bias/acceleration calculations are those of the pinned SciPy implementation. If all paired differences are identical within `scientific.statistics.identical_difference_tolerance`, return the exact point interval already defined in Section 4.23 instead of invoking BCa. If BCa otherwise returns a nonfinite endpoint or a degenerate-distribution warning with a nonfinite interval, record `BCa CI = NA / Degenerate` and any claim requiring that interval cannot pass. No percentile/basic fallback is permitted.

## 13.4 Binary outcomes

For genuinely binary paired observations:

* exact McNemar when discordant count does not exceed the configured switch;
* continuity-corrected asymptotic McNemar otherwise.

Confirmation safety rates with potentially multiple proposals per seed are analyzed as seed-level continuous rate differences with the exact sign-flip test, not McNemar.

## 13.5 Equivalence

Use TOST with the registered margins \(\delta_L=-0.01\) and \(\delta_U=+0.01\) for relative macro-CE differences.

Let \(d_i\) be the paired method-minus-reference difference for seed \(i\).

Lower-bound test:

$$
H_0:\mu_d\le\delta_L
\qquad\text{vs}\qquad
H_1:\mu_d\gt \delta_L.
$$

Define \(x_i=d_i-\delta_L=d_i+0.01\). Remove exact-zero \(x_i\) values using the same zero/tolerance semantics as the registered sign-flip test, enumerate every sign pattern of the remaining values, and calculate

$$
p_L =
\frac{
\left\lvert\lbrace\bar x^{perm}\ge \bar x^{obs}-10^{-15}\rbrace\right\rvert
}{
2^{n_{\rm eff,L}}
}.
$$

If every \(x_i\) is zero, \(p_L=1\).

Upper-bound test:

$$
H_0:\mu_d\ge\delta_U
\qquad\text{vs}\qquad
H_1:\mu_d\lt \delta_U.
$$

Define \(y_i=d_i-\delta_U=d_i-0.01\). Using the same exact sign enumeration,

$$
p_U =
\frac{
\left\lvert\lbrace\bar y^{perm}\le \bar y^{obs}+10^{-15}\rbrace\right\rvert
}{
2^{n_{\rm eff,U}}
}.
$$

If every \(y_i\) is zero, \(p_U=1\).

The pair-specific equivalence p-value is

$$
p_{equiv}=\max(p_L,p_U).
$$

Holm correction is applied to \(p_{equiv}\) within the exact registered family membership in Section 13.7. Equivalence is established only when the Holm-adjusted \(p_{equiv}\le\texttt{scientific.statistics.tost＿alpha＿per＿one＿sided＿test}\). No asymptotic TOST fallback is used.

## 13.6 Missingness and failed runs

No metric imputation.

No seed replacement.

A pair-specific confirmatory contrast requires the configured minimum valid paired seeds.

A FedORBIT scientific algorithmic failure on more than one confirmatory seed in a primary pair prevents that pair from satisfying a positive claim.

Infrastructure failures receive only the retry behavior defined in the failure contract.

## 13.7 Exact multiplicity-family membership

Multiplicity is applied only to the following predeclared pair-specific contrasts. A contrast absent because its pair lacks the configured minimum valid paired seeds is recorded as unavailable and is not replaced by another contrast.

### Primary Transfer vs Local-Only

Exactly six potential contrasts, one per primary directed pair:

```text
FedORBIT Exact-Sparse Solver vs Local-Only — TEST relative macro-CE gain
```

The tested seed-level quantity is \(G_{CE}(\mathrm{FedORBIT},\mathrm{Local\text{-}Only})\).

### External Source vs Local-SIR

Exactly twelve potential contrasts, two per primary directed pair:

```text
FedORBIT Exact-Sparse Solver vs Local-SIR — TEST relative macro-CE gain superiority
FedORBIT Exact-Sparse Solver vs Local-SIR — TEST relative macro-CE gain TOST equivalence
```

The superiority contrast uses the registered two-sided exact sign-flip p-value for the seed-level gain difference. The equivalence contrast uses the single pair-specific `p_equiv` from Section 13.5. Both are separate Holm inputs in this family.

### Coupling Mechanism

Exactly six potential contrasts, one per primary directed pair:

```text
Exact correspondence orbit vs Matched-Resource Rectangular — robust coupling value gap
```

The seed-level tested quantity is \(G_{\rm coupling}\) and the superiority null boundary is zero.

### Point-Correspondence Safety

Exactly twelve potential contrasts, two per primary directed pair:

```text
FedORBIT Exact-Sparse Solver vs Point-Correspondence Commitment — TEST relative macro-CE difference
FedORBIT Exact-Sparse Solver vs Point-Correspondence Commitment — TEST relative macro-CE TOST equivalence
```

The first uses the registered two-sided exact sign-flip p-value. The second uses the pair-specific `p_equiv` from Section 13.5. Both are separate Holm inputs in this family.

### Mechanism Ablations

Exactly twelve evidence-bearing multiplicity-controlled contrasts, two per primary directed pair:

```text
FedORBIT Exact-Sparse Solver vs Coupling-Destroyed FedORBIT — TEST relative macro-CE difference
FedORBIT Exact-Sparse Solver vs Coupling-Destroyed FedORBIT — TEST relative macro-CE TOST equivalence
```

The difference contrast uses the registered two-sided exact sign-flip p-value. The equivalence contrast uses the pair-specific `p_equiv` from Section 13.5. Both are separate Holm inputs in this family. Other registered ablation conditions remain required descriptive/materiality comparisons but do not receive additional confirmatory p-values unless already present in another family above.

### Sparsity Sensitivity

Exactly eighteen potential contrasts: for each primary directed pair,

```text
exact sparse s=1 vs exact sparse s=2
exact sparse s=3 vs exact sparse s=2
dense CCP vs exact sparse s=2
```

The seed-level quantity is TEST relative macro-CE difference under the common Local-Only reference.

### Confirmation Safety

Exactly six potential contrasts, one per primary directed pair:

```text
FedORBIT Without Confirmation vs FedORBIT Exact-Sparse Solver with confirmation — harmful-transfer rate difference
```

The seed-level statistic is `harmRate_noConfirm - harmRate_confirm`; positive values favor confirmation.

For every family, Holm family size is the number of the explicitly enumerated p-value inputs above that have enough valid paired seeds to be tested. A superiority p-value and a TOST `p_equiv` listed for the same pair are two distinct Holm inputs. Missing inputs are recorded with their reason and are not assigned p-values. Holm ties follow Section 4.23. No unlisted contrast may be inserted into a family after evidence-bearing TEST access begins.

# 14. Synthetic and Controlled Generator Contract

Every generator distribution, factor level, support grid, rejection threshold, and attempt limit is controlled by the `generators` fields in `configs/fedorbit.yaml`. Numerical values displayed here are readable renderings of those fields.

Every synthetic experiment is executable from this section and the Configuration YAML without inventing distributions.

## 14.1 Common RNG

All generators use NumPy `Generator(PCG64)` seeded by the `synthetic_instance` namespace.

All generated matrix and weight computations use float64.

## 14.2 Exact-separator theorem generator

For each configured block-pattern/support/seed/instance cell:

### Response matrix

$$
L_{ab}
\overset{iid}{\sim}
Uniform(-0.20,0.20).
$$

For serialization-only upper bands,

$$
U_{ab} =
L_{ab}
+
H_{ab},
\qquad
H_{ab}\sim Uniform(0,0.05).
$$

$U$ is not used as separator truth.

### Target importance

Draw

$$
z_k\sim Gamma(shape=2,scale=1)
$$

and normalize

$$
w_k=\frac{z_k}{\sum_qz_q}.
$$

### Support

Select uniformly from all supports of the required cardinality.

### Action

For $j\in S$,

$$
r_j\sim Uniform(0.05,0.25).
$$

If $\sum_jr_j\le0.50$,

$$
\alpha_j=r_j.
$$

Otherwise

$$
\alpha_j=
0.50
\frac{r_j}{\sum_qr_q}.
$$

Coordinates outside $S$ are zero.

### Cost

Use the project action cost.

### Truth

Exhaustively enumerate every block permutation for the registered small theorem block patterns.

No generic solver is used to define truth when exhaustive enumeration is available.

## 14.3 Coupling-structure generator

Registered factors:

```text
compatibility:
  jointly_realizable
  incompatible

response_heterogeneity:
  0.5
  1
  2

directed_asymmetry:
  0
  0.5
  1

response_sparsity:
  0.25
  0.5
  1

block_pattern:
  (2,2)
  (2,3)
  (3,3)

support:
  1
  2
  3
```

For every candidate:

1. draw an unconstrained matrix
$$
   R_{ab}\sim Uniform(-0.10,0.10);
$$
2. compute symmetric component
$$
   S=(R+R^T)/2;
$$
3. apply asymmetry level $a$:
$$
   Q=(1-a)S+aR;
$$
4. for each ordered coarse block pair, center entries around their block-pair mean and multiply deviations by configured heterogeneity;
5. apply exact deterministic sparsity by retaining
$$
   \max(1,\mathrm{round}(qK^2))
$$
   entries chosen by the smallest deterministic hash ranks and setting others to zero;
6. draw $w$ and $\alpha$ by the theorem-generator rules.

For sparsity ranking, every matrix coordinate \((a,b)\) receives

$$
h_{ab} =
\mathrm{SHA256}
\left(
UTF8(
"FedORBIT|synthetic-sparsity|"
\Vert baseSeed
\Vert "|"
\Vert canonicalCoordinates
\Vert "|"
\Vert a
\Vert "|"
\Vert b
)
\right),
$$

where `canonicalCoordinates` contains generator name, compatibility, heterogeneity, asymmetry, sparsity, block pattern, support, seed, and instance index using the canonical JSON rule in Section 4.21. Rank by the full 256-bit digest interpreted as an unsigned big-endian integer; break the practically impossible digest tie by ascending \((a,b)\). No RNG draw or library hash function is used for this selection.

For each positive-weight active term, compute its exact set of orbit permutations attaining its marginal minimum.

Define:

* `jointly_realizable`: intersection of every positive-weight active-term minimizer set is nonempty;
* `incompatible`: that intersection is empty and fixed-action rectangularization gap exceeds $10^{-6}$.

Use rejection sampling until the requested structural class is obtained.

Maximum attempts per generated instance:

```text
10,000
```

Failure to construct the requested class is Generator Failure and blocks the corresponding designed-family experiment rather than silently weakening the criterion.

## 14.4 Common-action unresolved-map generator

Block pattern:

```text
(2,2)
```

Every coarse block-pair response is constant:

$$
L_{ab}=\beta_{g(a),g(b)},
$$

where

$$
\beta_{gh}\sim Uniform(0.04,0.12).
$$

Target weights use the configured Gamma-normalized generator.

All four blockwise maps remain admissible, but $P^TLP=L$ for all $P$.

The common action must have positive robust value. With the configured positive response range and project action cost, a generated fixture failing positive value is rejected.

Maximum attempts:

```text
1,000
```

## 14.5 Robust-compromise unresolved-map generator

Block pattern:

```text
(2,2)
```

Draw candidate response matrices

$$
L_{ab}\sim Uniform(-0.10,0.20).
$$

Draw $w$ as configured.

Use the principal sparse action set.

Accept a fixture only when:

1. $|\Pi|\gt 1$;
2. the intersection of map-conditioned optimal-action sets is empty within action tie tolerance;
3. robust pre-map value exceeds 0.005.

Maximum candidate attempts per required fixture:

```text
100,000
```

## 14.6 Map-dependent generator

Block pattern:

```text
(2,2)
```

Draw

$$
L_{ab}\sim Uniform(-0.15,0.25).
$$

Accept only when:

$$
\Delta_{\rm map}\ge0.01.
$$

The robust method must either:

* select a lower-valued compromise;
* or abstain.

It must never output semantic map certainty.

Maximum attempts:

```text
100,000
```

## 14.7 Map-bound fixture families

Zero-map-value family uses the common-action generator.

High-map-value family uses the map-dependent generator.

Both use the principal sparse action set.

## 14.8 Scalability generator

For $K$ in the registered scalability grid:

Balanced blocks:

$$
(\lfloor K/2\rfloor,\lceil K/2\rceil).
$$

Maximally skewed two-block pattern:

$$
(K-1,1).
$$

Generate

$$
L_{ab}\sim Uniform(-0.10,0.10).
$$

Use

$$
w_k=1/K.
$$

For fixed-action separator timing with support $s$, choose the first $s$ pseudonymous nodes in the largest block.

Action values are:

$$
\alpha_j=
\min\left(0.25,\frac{0.50}{s}\right)
$$

on the selected support.

Each synthetic scalability cell records both:

* fixed-action separator metrics;
* full registered robust-action solver metrics where applicable.

# 15. Experiment Catalogue

Every experiment below is named descriptively. Numerical experiment grids, experiment-specific thresholds, limits, and categorical method/condition lists retained as configuration data are authoritative in `configs/fedorbit.yaml`. Fixed experiment membership, classifications, dependency semantics, use of the registered global seed sets, and procedural rules are authoritative in this catalogue. Derived counts are calculations and are not separately configurable.

Every descriptive experiment name below is accepted directly as the quoted experiment argument by:

```text
fedorbit run "EXPERIMENT NAME"
```

The operator supplies no scientific parameters.

## Mathematical Primitive Validation

**Classification:** Validation.

Validates:

* null padding;
* block permutations;
* objective evaluation;
* action feasibility;
* LAP construction;
* rectangular minima;
* orbit mean;
* deterministic tie handling;
* serialization.

Inputs:

* deterministic hand fixtures;
* seed 0;
* confirmatory seeds for property checks.

Pass:

* 100% registered tests pass;
* scalar/matrix fixture error $\le10^{-10}$;
* zero invalid permutations accepted.

Failure blocks every downstream experiment.

## Exact Sparse Theorem Exhaustive Validation

**Classification:** Validation.

Block patterns:

```text
(2)
(3)
(4)
(2,2)
(2,3)
(3,3)
```

Support values use every registered feasible $s\in\lbrace1,2,3\rbrace$, producing 17 block-pattern/support cells.

Per cell:

* ten confirmatory seeds;
* 100 generated instances.

Derived total:

```text
17,000 exact validation instances
```

Ground truth:

* exhaustive orbit enumeration;
* generic QAP is an independent secondary comparator.

Pass:

* zero wrong minima;
* maximum absolute objective error within exactness tolerance;
* every returned correspondence passes certificate validation.

## Coupling and Map-Bound Validation

**Classification:** Validation.

Uses the complete coupling generator factorial.

Derived instance count:

$$
2\times3\times3\times3\times3\times3\times10 =
4,860.
$$

Pass:

* structural zero/strict classification correct for every instance;
* every designed incompatible instance satisfies required positive strict gap;
* every map-value fixture respects the orbit-radius bound within exactness tolerance.

## Dataset, Client, and Strict-Resource Validation

**Classification:** Validation.

Covers:

* three primary real clients and one external-only client;
* six primary pairs;
* no secondary pairs;
* all confirmatory seeds.

Pass requirements:

* each primary pair has at least the configured minimum valid seeds;
* every valid pair-seed satisfies support/ontology requirements;
* Edge excluded fields absent from principal model matrices;
* zero forbidden resource accesses.

If a primary pair has insufficient valid seeds, its claim scope is removed **before** principal outcome inspection.

## Base-Model Hyperparameter Pilot

**Classification:** Exploratory.

Derived fits:

$$
4\ clients\times12\ configurations\times3\ pilot\ seeds=144.
$$

After deterministic selection:

$$
4\times10=40
$$

confirmatory base checkpoints are trained.

No TEST data are accessed.

## Source-Response Estimator Pilot

**Classification:** Exploratory.

Per client/model family:

* three pilot checkpoints;
* nine $(\epsilon_R,K_R)$ candidates.

Derived candidate cells:

$$
4\times3\times9=108.
$$

Each candidate uses configured paired schedules at $\epsilon_R$ and $\epsilon_R/2$.

Exact shadow-step totals are derived after source transfer-node eligibility is observed.

Every primary source domain must obtain one eligible candidate.

## Final Source-Response Band Validation

**Classification:** Validation.

Planned before eligibility exclusions:

$$
4\ source\ clients\times10\ seeds=40
$$

packets.

Each real source transfer intervention uses configured final paired repetitions and simultaneous bands.

A principal source domain requires at least the inferential minimum valid seed packets.

Response stability requirement:

* at least `scientific.source_response_final.minimum_useful_intervention_columns` useful intervention columns;
* median simultaneous band width divided by median absolute mean response over useful entries no greater than `scientific.source_response_final.median_band_width_to_median_absolute_mean_response_maximum`.

## Baseline and Oracle Correctness Validation

**Classification:** Validation.

Validation seeds:

```text
1103
5531
```

for every primary pair.

Checks:

* resource matrices;
* oracle ACL;
* rectangular baseline against analytical computation;
* point-map QAP correctness on tractable fixtures;
* generic QAP vs exhaustive truth;
* deterministic replay.

## Exact-Sparse Solver Benchmark

**Classification:** Confirmatory for solver exactness/structure.

Synthetic K:

```text
4 through 18 inclusive
```

Block patterns for each K:

* balanced;
* maximally skewed.

Supports:

```text
1
2
3
```

Seeds:

* all ten confirmatory seeds.

Exhaustive truth is computed whenever

$$
|\Pi|\le100,000.
$$

For larger correspondence sets, certified generic-QAP optimum may supply truth. If QAP fails to certify within its limit, truth is unavailable for that cell.

Methods:

* exact-sparse;
* generic exact QAP;
* dense CCP.

All valid primary real response packets are additionally benchmarked.

## Synthetic Coupling-Mechanism Validation

**Classification:** Confirmatory mechanism.

Uses the same 4,860 designed coupling instances.

Methods:

* exact orbit;
* matched rectangular;
* coupling-destroyed.

Pass interpretation follows the central mechanism criteria.

## Real-Packet Coupling-Mechanism Validation

**Classification:** Confirmatory mechanism.

Planned units:

$$
4\ primary\ pairs\times10\ seeds=40.
$$

Compare:

* exact correspondence orbit;
* matched rectangularization.

For the required fixed-action gap, use the pre-confirm principal action \(\alpha^\star_{\rm FedORBIT}\) returned by `FedORBIT Exact-Sparse Solver` for that exact pair-seed source packet, target-importance vector, and principal support. The action is computed without CONFIRM or TEST access. If the principal solver abstains with the zero action, the fixed-action gap is exactly zero and is retained as a valid mechanism observation. If the principal solver has a scientific algorithmic failure, the real-packet coupling unit is unavailable rather than replaced by another action.

The robust coupling value gap uses the principal sparse action set and the definitions in Section 3.

Use the registered coupling multiplicity family.

## Common Action Under Unidentified Map

**Classification:** Validation / DIAGNOSTIC.

Fixtures:

$$
50\times10=500.
$$

Required:

* multiple admissible maps;
* positive common optimal action;
* exact-map value within exactness tolerance of zero.

## Robust Compromise Under Unidentified Map

**Classification:** Validation / DIAGNOSTIC.

Fixtures:

$$
50\times10=500.
$$

Required:

* no common map-conditioned optimum;
* robust value $\gt 0.005$.

## Map-Dependent Action Boundary

**Classification:** Failure Boundary.

Fixtures:

$$
50\times10=500.
$$

Required:

$$
\Delta_{\rm map}\ge0.01.
$$

FedORBIT must compromise or abstain without claiming recovered semantics.

## Exact Map-Value Bound Validation

**Classification:** Validation.

Per seed:

* 25 zero-map-value fixtures;
* 25 high-map-value fixtures.

Derived total:

```text
500 fixtures
```

Every fixture must respect the registered map bound.

## Primary Strict Cross-Telemetry Transfer

**Classification:** Confirmatory.

Pairs:

* all six primary directed pairs.

Seeds:

* all ten confirmatory seeds.

Methods:

```text
Local-Only
Local-SIR
Matched-Resource Rectangular
Point-Correspondence Commitment
Generic Exact QAP
FedORBIT Exact-Sparse Solver
Exact-Map Oracle
```

Principal support:

* scientific.action.principal_sparse_support.

Planned cells before deterministic data invalidations:

$$
6\times10\times7=420.
$$

Primary outcome:

* TEST relative macro-CE gain.

TEST remains sealed until the pre-TEST decision contract passes.

## Multi-Source Selection Validation

**Classification:** Diagnostic.

Targets:

```text
ToN-IoT Windows 10 Host Client
ToN-IoT Linux Process Host Client
ToN-IoT Network Client
```

Candidate sources:

* every other primary client;
* no more than the configured maximum.

Seeds:

* ten.

Derived target decisions:

```text
30
```

Validates:

* ranking;
* tie resolution;
* sequential confirmation;
* first-accepted stop;
* no-source fallback.

## Mechanism Ablations

**Classification:** Ablation.

Primary pairs × confirmatory seeds.

Conditions:

```text
FedORBIT Exact-Sparse Solver
Matched-Resource Rectangular
Point-Correspondence Commitment
Coupling-Destroyed FedORBIT
Coarse Block-Mean
Coarse Block-Min
Orbit-Mean
Local-SIR
```

Derived condition cells before identity reuse:

$$
6\times10\times8=480.
$$

Already valid semantic cells are reused rather than duplicated.

## Sparsity and Dense Fallback

**Classification:** Robustness.

Conditions:

```text
exact sparse s=1
exact sparse s=2
exact sparse s=3
dense CCP
```

Primary pair-seed units:

$$
6\times10.
$$

Derived cells:

```text
240
```

Principal $s=2$ cells are reused when already complete.

## Target Confirmation and Portability

**Classification:** Confirmatory safety.

Pairs:

* six primary.

Methods:

```text
FedORBIT Exact-Sparse Solver
FedORBIT Without Confirmation
```

Derived planned cells:

$$
6\times10\times2=120.
$$

The confirmatory analysis uses the six primary pairs only.

## Secondary Cross-Modality Generalization

**Classification:** Generalization.

Pairs:

* no pairs are preregistered while Edge-IIoTset remains external-only.

Methods:

```text
Local-Only
Local-SIR
Matched-Resource Rectangular
Point-Correspondence Commitment
FedORBIT Exact-Sparse Solver
```

Derived planned cells:

```text
0
```

This is secondary within-ToN-suite evidence and cannot independently establish cross-dataset generalization.

## Semantic Sufficiency Frontier

**Classification:** Failure Boundary.

Partitions:

1. oracle fine singleton groups;
2. principal three coarse groups;
3. two groups:

   * Disruption or Exploitation
   * Access and Discovery;
4. one attack supergroup.

Methods:

```text
FedORBIT Exact-Sparse Solver
Matched-Resource Rectangular
Exact-Map Oracle
```

Primary pairs × confirmatory seeds.

Derived cells:

$$
4\times3\times4\times10=480.
$$

The singleton condition is oracle/diagnostic and not strict-interface evidence.

## Weak-Signal, Support, and Heterogeneity Boundaries

**Classification:** Failure Boundary.

One factor at a time only.

Response scale:

```text
1
0.75
0.5
0.25
0
```

CI half-width multiplier:

```text
1
1.5
2
4
```

Target usable support fraction:

```text
1
0.5
0.25
0.1
```

Response heterogeneity multiplier:

```text
0.5
1
2
```

Support budget:

```text
1
2
3
```

The shared baseline condition is counted once, giving 15 distinct conditions.

Methods:

```text
FedORBIT Exact-Sparse Solver
Matched-Resource Rectangular
Local-Only
```

Derived cells:

$$
15\times3\times4\times10=1,800.
$$

### Perturbation semantics

For source packet midpoint and half-width,

$$
A=(L+U)/2,
\qquad
H=(U-L)/2.
$$

Response-scale condition $r$:

$$
L'=rA-H,\qquad U'=rA+H.
$$

CI-width condition $q$:

$$
L'=A-qH,\qquad U'=A+qH.
$$

Heterogeneity condition $h$, for each ordered coarse block pair:

$$
A'_{ab} =
\bar A_{gh}
+
h(A_{ab}-\bar A_{gh}),
$$

with original half-width $H$.

Target support fractions use deterministic smallest-hash subsampling independently within each retained class of:

* TRAIN;
* META;
* CONFIRM.

VALID and TEST are unchanged.

The target base model and all downstream target artifacts are rebuilt using the subsampled TRAIN/META/CONFIRM condition.

When a perturbed support condition violates transfer eligibility, the boundary outcome is an expected `INELIGIBLE/ABSTAIN` scientific result, not infrastructure failure.

## Map-Availability Applicability Audit

**Classification:** Diagnostic.

Packet-only recovery:

* Point-Correspondence Commitment;
* generic structural QAP;
* every primary pair-seed.

Derived recovery attempts:

$$
2\times40=80.
$$

Human public-resource audit:

* two independent researchers;
* six primary directed pairs;
* exactly 60 minutes per researcher per pair;
* access to public dataset documentation, published label descriptions, exposed coarse groups, and the strict-interface resource list;
* no oracle mapping artifact, oracle comparison output, or precomputed source-target fine-map table during the timed assessment.

`fedorbit run "Map-Availability Applicability Audit"` handles the human prerequisite deterministically. Each required submission is canonical JSON with exactly:

```text
researcher_id
directed_pair
session_start_utc
session_end_utc
resources_consulted
proposed_mapping
unresolved_alternatives
rationale
```

`resources_consulted` is an ordered list of public URLs/titles actually consulted. `proposed_mapping` is an ordered list of records `{source_public_label, target_public_label, exposed_coarse_group}`. `unresolved_alternatives` is an ordered list of records identifying the affected source/target public label and the alternatives considered; it is empty only when the researcher reports no ambiguity. Timestamps are RFC 3339 UTC. JSON keys are lexicographically sorted for hashing; list order is preserved as entered except `proposed_mapping`, which is canonicalized by exposed coarse group, then source public label, then target public label before hashing.

Execution then follows:

1. if a required researcher × pair submission is absent, materialize a schema-valid blank template under the experiment's `artifacts/fitted/human_audit/` area and leave the audit cell `Blocked / Human Input Required`;
2. each researcher completes the template during one timed session and records start/end UTC timestamps, public resources consulted, the proposed complete fine-concept correspondence, unresolved alternatives if any, and free-text rationale;
3. the executor validates that elapsed time is no greater than 60 minutes, both researchers have distinct researcher IDs, every proposed source/target concept comes from the documented public component labels, and no oracle artifact appears in the recorded access trace;
4. after both submissions for a pair validate, record their SHA-256 values and only then permit oracle comparison;
5. templates/submissions are human inputs to this diagnostic only and are never method-readable resources for any transfer experiment.

A pair's public fine map is `Trivial To Reconstruct` only when **both** researchers independently, within their respective 60-minute limits:

* submit one complete one-to-one mapping covering every real non-null transfer concept present at both endpoints;
* submit no unresolved alternative for any mapped concept; and
* achieve exact oracle correspondence accuracy of 1.0 after comparison.

Every other valid outcome is `Not Demonstrated Trivial`. Researcher agreement alone without exact oracle accuracy is insufficient.

If every primary pair that is eligible for the applicability audit is `Trivial To Reconstruct`, the practical-motivation kill rule in Section 17 fires. If only a subset is trivial, practical unresolved-map wording is restricted to the nontrivial subset; the benchmark-wide natural-unavailability claim remains forbidden.

## Scalability and Efficiency

**Classification:** Robustness / EFFICIENCY.

K grid:

```text
6
8
10
12
16
20
24
32
```

Block patterns:

* balanced;
* maximally skewed.

Exact/QAP support:

```text
1
2
3
```

Dense:

* no support cap.

Seeds:

* ten.

Derived synthetic exact/QAP cells:

$$
8\times2\times3\times10\times2=960.
$$

Derived dense cells:

$$
8\times2\times10=160.
$$

Total synthetic solver cells:

```text
1,120
```

Real primary timing cells:

$$
40\times3=120
$$

before deterministic data exclusions.

Timing uses the authoritative warmup/repetition protocol.

For the `Sparse Solver Work-Structure Agreement` claim, define the predicted work coordinate for each fixed-action exact-sparse scalability cell as

$$
X =
N_S\sum_g n_g^3.
$$

Within each block-pattern × support stratum having at least `scientific.statistics.spearman_minimum_valid_points` distinct non-timeout \(K\) values, compute Spearman correlation between \(\log X\) and \(\log\) median exact-sparse runtime. The efficiency-trend component passes only when every eligible stratum has \(\rho\gt 0\). The correlation is descriptive and receives no p-value threshold. A stratum with fewer than the configured minimum points is `Insufficient Trend Evidence` and cannot establish the runtime-trend component.

The work-structure claim is Supported only when:

1. every truth-valid exact-sparse cell has `Active-Image Candidates` equal to \(N_S\);
2. every such cell has `LAP Calls` equal to the exact formula in Section 8.1;
3. no approximation replaces a required exact separator;
4. the runtime-trend component above passes.

Exact counters passing while the runtime-trend component is insufficient or nonpositive yields at most `Partially Supported`; any counter mismatch or required approximation yields `Not Supported`.

## Statistical Synthesis

**Classification:** Confirmatory ANALYSIS.

Inputs:

* completed registered artifacts only.

Performs:

* registered pair-specific contrasts;
* BCa CIs;
* exact randomization tests;
* Holm adjustment;
* equivalence;
* materiality;
* completeness checks.

The synthesis consumes the currently valid evidence-bearing metric artifacts under their dependency fingerprints. Re-running synthesis with unchanged inputs reuses the existing statistical artifacts; changed metric/statistical dependencies invalidate only the affected statistical descendants.

## Evidence Classification

**Classification:** FINAL EVIDENCE.

Inputs:

* completed valid registered results;
* completed verified statistical synthesis;
* evidence criteria.

Outputs:

* one evidence status;
* permitted manuscript scope;
* forbidden extrapolations;
* supporting artifact references.

It never changes thresholds based on observed results.

## Experiment dependency and artifact map

The Experiment Catalogue is executed through the shared artifact stages defined in Sections 20 and 21. Experiments consume immutable artifacts by dependency fingerprint; they do not acquire ownership of shared upstream artifacts merely because they are the first experiment to request them.

| Experiment | Required dependencies and reusable inputs | Primary produced artifacts | Main downstream consumers / reuse |
| --- | --- | --- | --- |
| Mathematical Primitive Validation | registered hand fixtures, primitive implementation, deterministic seeds | primitive-validation results, fixture serialization checks, pass/fail completion manifest | blocks every experiment; reused as the primitive prerequisite until its own dependencies change |
| Exact Sparse Theorem Exhaustive Validation | primitive-validation pass, exact-separator generator contract, block/support coordinates, confirmatory seeds | generated instances, exhaustive-orbit truth, generic-QAP comparison, correspondence certificates, exactness metrics | exact-sparse solver benchmark, synthetic/real mechanism experiments, principal solver use |
| Coupling and Map-Bound Validation | primitive-validation pass, coupling generator contract, map-bound fixtures, confirmatory seeds | coupling instances, structural zero/strict labels, exact coupling values, map-bound validation results | synthetic coupling mechanism, map-value diagnostics, evidence classification |
| Dataset, Client, and Strict-Resource Validation | raw manifests, prepared splits, preprocessors, processed splits, local-class manifests, transfer-eligibility manifests, strict-resource rules | pair-seed validity manifest, exclusions, resource-access validation, dataset/client validation results | every real-data pilot, packet, transfer, confirmation, generalization, ablation, and real timing experiment |
| Base-Model Hyperparameter Pilot | valid prepared client data, TRAIN/VALID processed splits, model family, pilot grid, pilot seeds | pilot checkpoints and VALID metrics, deterministic hyperparameter-selection artifact, confirmatory base checkpoints with optimizer/RNG state | source-response pilot/final packets, target importance, confirmations, local references, all real-data methods |
| Source-Response Estimator Pilot | selected base-model pilot artifacts, source TRAIN/META data, transfer eligibility, response candidate grid and paired schedules | candidate response diagnostics, eligibility results, selected response-estimator configuration per source client/model family | final source-response packet construction |
| Final Source-Response Band Validation | confirmatory source checkpoints, selected response-estimator artifact, source TRAIN/META data, transfer-node manifest, response schedules | paired replicate artifacts, simultaneous-band artifacts, final anonymous source-response packets, stability metrics | real-packet coupling, principal transfer, multi-source selection, ablations, sparsity/dense, secondary generalization, boundaries, real solver timing |
| Baseline and Oracle Correctness Validation | primitive prerequisite, validation fixtures, strict-resource matrix, tractable packets/instances, registered baseline and oracle implementations | baseline correctness certificates, oracle ACL validation, deterministic replay results | evidence-bearing baseline comparisons and principal transfer |
| Exact-Sparse Solver Benchmark | theorem-validation artifacts, synthetic instances, registered solvers; valid real response packets for real-packet timing rows | solver result/certificate artifacts, truth-availability records, runtime/memory/counter metrics | exactness/structure claims, scalability synthesis, solver result tables/figures |
| Synthetic Coupling-Mechanism Validation | validated coupling instances, exact orbit solver, rectangular baseline, coupling-destroyed construction | per-instance coupling metrics and mechanism decisions | coupling mechanism statistics, figures, evidence classification |
| Real-Packet Coupling-Mechanism Validation | valid real response packets, corresponding target-importance artifacts, pair eligibility, exact/rectangular solvers | per-pair-seed fixed-action/robust coupling metrics | coupling mechanism statistics, mechanism figures, evidence classification |
| Common Action Under Unidentified Map | common-action generator fixtures, exact solver, map-value routines | fixture actions, admissible-map diagnostics, robust/exact-map values | map-identifiability/action-certifiability evidence |
| Robust Compromise Under Unidentified Map | robust-compromise generator fixtures, exact solver, map-value routines | robust-compromise actions and values, map-conditioned optimum diagnostics | map-identifiability/action-certifiability evidence |
| Map-Dependent Action Boundary | map-dependent fixtures, exact solver, map-value routines | map-dependent actions, abstention/compromise outcomes, map-value metrics | failure-boundary evidence and claim restrictions |
| Exact Map-Value Bound Validation | zero/high map-value fixtures, exact solver, orbit-radius implementation | exact map-action values, orbit-radius bounds, validation metrics | map-value-bound figure and evidence classification |
| Primary Strict Cross-Telemetry Transfer | valid pair-seed manifest, source packet, source/target checkpoints, target importance, registered baseline/oracle validation, action solver, confirmation rules, processed TRAIN/META/CONFIRM/TEST splits | method action/proposal artifacts, confirmation decisions, live-assimilation checkpoints where applicable, canonical TEST prediction artifacts, metric rows, cell completion manifests | statistical synthesis, ablations, confirmation analysis, reporting, evidence classification |
| Multi-Source Selection Validation | all valid candidate source packets for each target, target checkpoint/importance, proposal solver artifacts, confirmation machinery | ranked proposal lists, sequential confirmation traces, selected-source/no-source decisions, resulting state artifacts | multi-source diagnostics and manuscript evidence |
| Mechanism Ablations | shared principal pair-seed inputs, response packets, target artifacts, and any already-valid identical method cells from principal transfer | only ablation-specific method/condition actions, confirmations, predictions, metrics, plus references to reused identical cells | ablation statistics/table and mechanism claims |
| Sparsity and Dense Fallback | shared response packets, target checkpoints/importance, confirmation/evaluation inputs, sparse/dense solver configs; completed principal `s=2` cells when identical | `s=1`, `s=3`, dense-specific actions/results and reused references to `s=2`; runtime/memory and realized-utility metrics | sparsity/dense statistics, table, utility-efficiency figure |
| Target Confirmation and Portability | valid primary/secondary pair artifacts, FedORBIT proposals, clean target pre-confirm checkpoints, TRAIN/CONFIRM/TEST splits, confirmation schedules | paired confirmation/no-confirm decisions, accepted/rejected states, live-assimilation checkpoints, TEST predictions and safety metrics | confirmation statistics, safety/coverage figure, primary safety claim |
| Secondary Cross-Modality Generalization | valid secondary pair-seed manifest, source packets, target checkpoints/importance, registered methods, confirmation/evaluation inputs | secondary method actions, confirmations, TEST predictions and metrics | secondary generalization statistics/table; no primary cross-dataset claim |
| Semantic Sufficiency Frontier | primary pair artifacts, coarse-group condition definition, transfer manifests, exact-map oracle, registered methods | condition-specific correspondence/action artifacts, TEST predictions where model state changes, frontier metrics | semantic sufficiency statistics, table/figure, boundary claims |
| Weak-Signal, Support, and Heterogeneity Boundaries | primary response packets/checkpoints plus perturbation definitions; for target-support conditions, deterministically subsampled TRAIN/META/CONFIRM data and rebuilt target training/downstream artifacts | perturbation-specific packets or derived packet views, rebuilt target checkpoints only for support-fraction conditions, actions, confirmations, predictions, metrics | failure-boundary synthesis/table/figure |
| Map-Availability Applicability Audit | primary packets, Point-Correspondence Commitment and Generic Exact QAP recovery implementations, strict-interface resource list, fixed human-audit protocol | packet-only recovery attempts, checksum-recorded human audit submissions, oracle comparison after both timed submissions are completed | applicability wording and claim boundaries |
| Scalability and Efficiency | scalability generator instances, registered solvers, timing protocol; valid real response packets for real timing cells | timing repetitions, runtime/memory/counter artifacts, timeout/resource-limit states, exactness status | scalability statistics/table/figure and operational-structure claim |
| Statistical Synthesis | only completed verified metric/comparison inputs from the registered experiments | paired contrasts, BCa intervals, randomization tests, Holm results, equivalence/materiality results, completeness state | evidence classification and manuscript tables/figures |
| Evidence Classification | completed valid registered results, verified statistical synthesis, evidence criteria, kill/simplification rules | final evidence-classification artifact, permitted scope, forbidden extrapolations, evidence references | `fedorbit report` and manuscript evidence only |

Repeated experiment membership does not duplicate a scientific artifact. If two experiment cells require the same prepared split, checkpoint, prediction, response packet, target-importance vector, solver result, confirmation input, or other immutable artifact under the same dependency fingerprint, both cells reference that artifact. Experiment-specific manifests record the reference rather than copying or recomputing the payload.

# 16. Evidence Criteria

The allowed evidence statuss are fixed by this section and are not configurable.

Allowed final states:

```text
Supported
Partially Supported
Mechanism Only
Conditional
Null Result
Not Supported
Not Tested
```

| Claim                                            | Support rule                                                                                | Failure rule                                                         | Valid scope                                                   |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------- |
| Exact Sparse Separator Exactness                | zero reproducible mismatch above exactness tolerance in every truth-available required case | any reproducible mismatch or invalid certificate                     | theorem assumptions and tested sparse supports                |
| Joint Correspondence Avoids Rectangular Pessimism          | authoritative coupling criteria passes                                                          | real/synthetic mechanism criteria fails or destruction retains mechanism | declared orbit/packet regimes                                 |
| Action Certification Without Fine-Map Identification | controlled common-action and robust-compromise worlds exist; map bound valid                | map recovery always required or bound violation                      | controlled/theorem regime                                     |
| Strict Cross-Telemetry Transfer Utility        | authoritative primary 4-of-6 rule                                                           | utility or strict-resource criteria fails                                | six primary benchmark directions under masked fine semantics |
| Value of External Procedural Evidence        | authoritative 4-of-6 rule vs Local-SIR                                                      | Local-SIR equivalent/superior under the registered criteria                             | six primary directions                                       |
| Operational Relevance of Sparse Support       | authoritative sparse criteria                                                                   | dense sparse-irrelevance rule                                        | tested support/action ranges                                  |
| Target Confirmation Safety               | authoritative confirmation criteria                                                             | safety/coverage criteria fails                                           | tested pairs and confirmation budget                          |
| Sparse Solver Work-Structure Agreement | exact counters plus efficiency trend                                                        | counter mismatch or approximation required                           | measured problem sizes/hardware                               |

## 16.1 Mechanical evidence-status adjudication

Evidence status is assigned from completed registered evidence only. The following precedence is mandatory:

1. `Not Tested` when required experiments have no scientifically interpretable completed evidence because they were not executed or were blocked before producing the endpoints needed by the claim.
2. `Not Supported` when an explicit claim failure rule, falsification condition, or claim-negating kill/simplification rule fires.
3. `Supported` when the full support rule passes over the full declared claim scope.
4. `Conditional` only where the rule below explicitly permits a pre-outcome data/eligibility reduction of scope and the positive criteria pass on that reduced scope.
5. `Mechanism Only` only where the claim-specific rule below permits controlled/theorem mechanism evidence without the required real-packet evidence.
6. `Partially Supported` only where at least one claim-specific positive component reaches its registered materiality/statistical requirement but the full support rule does not pass and no explicit failure rule fires.
7. `Null Result` when the required evidence is complete, no positive component reaches its registered materiality requirement, and no explicit contradictory/harm/falsification condition fires.

No state may be selected from narrative judgment.

| Claim | Deterministic non-Supported states |
| --- | --- |
| Exact Sparse Separator Exactness | `Supported` is determined only over the required truth-available cases defined by the validation/benchmark contracts; lack of independent truth in an explicitly truth-unavailable larger benchmark cell does not downgrade the claim. `Not Tested` applies only if the required exhaustive/theorem truth evidence is unavailable. Any reproducible exact mismatch or invalid certificate is `Not Supported`. |
| Joint Correspondence Avoids Rectangular Pessimism | `Mechanism Only` when the complete designed synthetic/theorem mechanism passes but the real-packet materiality criterion is unavailable or not reached and coupling destruction does not contradict the mechanism. Synthetic/theory failure or coupling-destruction retention is `Not Supported`. |
| Action Certification Without Fine-Map Identification | `Partially Supported` when exactly one of the common-action or robust-compromise controlled families satisfies its required positive construction and the orbit-radius bound remains valid. Any bound violation or evidence that exact map recovery is always required is `Not Supported`. |
| Strict Cross-Telemetry Transfer Utility | `Conditional` only when one primary pair was removed before principal outcome inspection for deterministic Invalid Data/eligibility reasons, leaving exactly three analyzable primary pairs, and all three individually satisfy the pair-level materiality, Holm, BCa, no-harm, and strict-resource criteria with their equal-pair mean meeting materiality. `Partially Supported` when one or two full-scope primary pairs satisfy all positive pair criteria, no pair is materially harmful, and strict-resource validation passes. `Null Result` when no pair is materially beneficial and no pair is materially harmful. Any material harmful pair or strict-resource failure is `Not Supported`. |
| Value of External Procedural Evidence | Uses the same state logic as Strict Cross-Telemetry Transfer Utility with Local-SIR as reference. Registered Local-SIR equivalence/superiority that fires the simplification rule is `Not Supported`. |
| Operational Relevance of Sparse Support | `Partially Supported` when at least one registered sparse support has material realized benefit on at least one primary pair but the full sparse-operational rule does not pass. `Null Result` when no sparse support reaches useful-transfer materiality and the dense sparse-irrelevance rule does not fire. The sparse-irrelevance kill rule is `Not Supported`. |
| Target Confirmation Safety | `Partially Supported` when at least one but fewer than three primary pairs meet the registered ARR/RRR criterion, with no pair exceeding the harmful-worsening or coverage-loss ceilings. `Null Result` when no pair meets either harm-reduction threshold and no worsening/coverage failure occurs. Any registered safety/coverage failure or `Confirmation Has No Safety Value` trigger is `Not Supported`. |
| Sparse Solver Work-Structure Agreement | `Partially Supported` when all exact counter formulas match and exactness is retained but one or more eligible runtime-trend strata are nonpositive or have insufficient trend evidence. Any counter mismatch or required approximation is `Not Supported`. |

A pre-outcome pair exclusion never changes a threshold or creates a replacement pair. `Conditional` wording must name the reduced empirical scope explicitly.

Final manuscript wording may not exceed these scopes.

# 17. Immutable Kill and Simplification Rules

All numerical and categorical trigger values in this section are configured under `scientific.simplification_rules` or reuse the shared materiality and claim-criteria fields in `configs/fedorbit.yaml`. The prose below defines the scientific consequence of each rule.

## Exactness Failure

**Trigger:** any reproducible exact-sparse objective error above exactness tolerance or invalid correspondence certificate.

**Consequence:** stop every exact-sparse-dependent claim; debug implementation; a mathematical counterexample requires redesign rather than tolerance inflation.

## Rectangularization Is Sufficient

**Trigger:** at least 90% of valid real packet units have absolute coupling gap below the configured materiality threshold and no primary pair satisfies the coupling mechanism criteria.

**Consequence:** remove or sharply narrow the coupling-value claim.

## Theory Classification Failure

**Trigger:** designed zero/strict classification or map bound disagrees with exhaustive truth beyond exactness tolerance.

**Consequence:** hard theory stop.

## Generic QAP Dominates

For this rule, **scientific implementation complexity** is the count of method-specific scientific primitives outside shared preprocessing, evaluation, artifact, and reporting infrastructure.

The exact-sparse primitive set is exactly:

```text
support enumeration
robust action master
active-image enumeration
blockwise LAP completion
exact separator/certificate logic
```

The Generic Exact QAP method primitive set is exactly:

```text
support enumeration
robust action master
binary block-assignment QAP construction
RLT/McCormick product linearization
MIP exactness/certificate logic
```

Both therefore have roadmap complexity count 5; no subjective code-size, line-count, developer-effort, or library-preference judgment enters the trigger.

**Trigger on intended $s\le2$ exact cases:** QAP is exact on every required case; median runtime $\le1.0\times$ exact-sparse; p95 runtime $\le1.2\times$; peak memory $\le1.0\times$; no more timeouts; and its roadmap complexity count is no greater than exact-sparse.

**Consequence:** generic QAP becomes preferred implementation and solver-novelty framing is simplified.

## Sparse Support Is Operationally Irrelevant

**Trigger:** dense improves TEST relative macro-CE by at least 0.02 over $s=3$ on at least 75% of valid primary units and every $s\le3$ condition fails useful-transfer materiality in those units.

**Consequence:** sparse theorem becomes peripheral to the empirical contribution.

## Local-SIR Is Sufficient

For each primary pair, let the seed-level comparison quantity be `G_CE(FedORBIT Exact-Sparse Solver, Local-SIR)`, so positive values favor FedORBIT. `Local-SIR Equivalent` means the registered FedORBIT-vs-Local-SIR TOST contrast establishes equivalence after Holm correction in the `External Source vs Local-SIR` family. `Local-SIR Superior` means the pair-mean comparison quantity is no greater than `-scientific.materiality.realized_relative_macro_ce`, the paired two-sided superiority contrast is Holm-significant in that family, and the paired BCa upper bound is strictly below zero.

**Trigger:** the external-procedural-evidence support rule fails and at least `scientific.evaluation_criteria.external_source_value_vs_local_sir.successful_primary_pairs_required` primary pairs are `Local-SIR Equivalent` or `Local-SIR Superior`, with no remaining valid primary pair showing a material, Holm-significant, BCa-supported FedORBIT advantage over Local-SIR.

**Consequence:** remove the external-source-value claim.

## Point Matching Is Sufficient

**Trigger:** for every primary directed pair with the configured minimum valid paired seeds, Point-Correspondence Commitment harmful-transfer rate is not worse than FedORBIT by more than 0.02 and its mean TEST relative macro-CE gain versus Local-Only is at least 0.01 greater than the corresponding FedORBIT mean gain.

No additional ambiguity stratum is implied or generated for this rule; it uses only the already registered primary pair conditions.

**Consequence:** simplify robust-correspondence motivation.

## Coupling Destruction Retains Gain

**Trigger:** the mechanism-retention condition defined in Section 4.25 is present.

**Consequence:** abandon causal coupling attribution.

## Strict Interface Removes Gain

The explicitly relaxed diagnostic is the already registered `Exact-Map Oracle` in `Primary Strict Cross-Telemetry Transfer`; no additional experiment is introduced. It relaxes only fine-correspondence uncertainty while retaining the same target checkpoint, action budget, support budget, confirmation opportunity, assimilation budget, seed, and TEST protocol.

For a primary pair, the relaxed diagnostic **succeeds** when Exact-Map Oracle has mean TEST relative macro-CE gain versus Local-Only at least `scientific.materiality.realized_relative_macro_ce` and its paired BCa lower bound is strictly greater than zero.

**Trigger:** at least `scientific.simplification_rules.strict_interface_removes_gain.primary_pair_majority_required` primary pairs simultaneously have FedORBIT mean gain no greater than zero with BCa upper bound below `scientific.simplification_rules.strict_interface_removes_gain.bca_upper_bound_maximum`, while the Exact-Map Oracle relaxed diagnostic succeeds on those same pairs.

**Consequence:** strict-interface utility claim is not supported.

## Confirmation Has No Safety Value

**Trigger:** confirmation fails both harm-reduction thresholds or coverage loss exceeds the authoritative ceiling.

**Consequence:** remove confirmation safety claim.

## Source Response Is Too Unstable

A final source packet fails the response signal criteria when either:

* fewer than `scientific.source_response_final.minimum_useful_intervention_columns` useful columns remain under Section 4.12; or
* the median simultaneous band-width / median absolute mean-response ratio over useful entries exceeds `scientific.source_response_final.median_band_width_to_median_absolute_mean_response_maximum`.

**Trigger:** more than the configured principal source-packet failure fraction fail these response signal criteria, or any required primary source domain has no eligible pilot setting.

**Consequence:** narrow to stable source domains or stop empirical transfer claim before examining favorable downstream outcomes.

## Unresolved-Map Regime Lacks Practical Motivation

**Trigger:** applicability audit demonstrates that an exact fine map is trivial to obtain under the resources intended to motivate the real deployment setting.

**Consequence:** retain only conditional algorithmic/benchmark claims and explicitly avoid claiming practical necessity of unresolved mapping.

No kill rule may be neutralized by:

* deleting seeds;
* deleting pairs after outcome inspection;
* changing support;
* changing the primary metric;
* changing confidence level;
* changing a baseline;
* widening equivalence margins;
* changing a claim threshold;
* reclassifying a scientific null as technical failure.

# 18. Failure Semantics

The infrastructure retry count is configured under `runtime.failure_handling` in `configs/fedorbit.yaml`. Terminal execution states and the prohibition on higher-resource retries are fixed execution semantics defined in this section.

## 18.1 Infrastructure failure

Examples include process crash, filesystem failure, CUDA runtime error, or malformed temporary serialization caused by infrastructure.

Retry exactly twice with identical semantic identity and configuration.

After three total failed attempts:

```text
Failed / Infrastructure Failure
```

Dependent cells are blocked.

## 18.2 Validation failure

Leakage, split overlap, conflicting duplicates, strict-resource violation, configuration mismatch, theorem primitive mismatch, or schema failure yields:

```text
Invalid
```

or

```text
Failed / Validation Failure
```

according to whether the defect is data/scientific validity or executable implementation.

## 18.3 Scientific null

Correct execution with an unfavorable scientific outcome remains:

```text
Completed
```

Evidence classification determines the scientific state.

## 18.4 Scientific boundary

A correctly executed failure-boundary condition remains:

```text
Completed
```

even when FedORBIT abstains, loses utility, or becomes ineligible.

## 18.5 Solver time/resource limit

If a solver reaches the configured limit and returns a valid state/bound artifact:

```text
Completed
```

with method outcome:

```text
Time Limit
```

or

```text
Resource Limit
```

No higher-resource retry is allowed in the same scientific experiment.

## 18.6 Scientific algorithmic failure

A correctly invoked scientific algorithm that cannot produce the certificate required by its own method contract, for reasons other than the separately defined time/resource limit, has terminal state:

```text
Failed / Scientific Algorithmic Failure
```

`Sparse Master Non-Convergence` is such a failure for `FedORBIT Exact-Sparse Solver`.

The cell produces its numerical diagnostics, failure reason, completed-support records, cut/master counters, and provenance, but it produces no certified action and does not proceed to confirmation, assimilation, or TEST scoring for that method. There is no automatic retry with a larger cut cap, changed tolerance, different support, or alternate solver. Paired statistical contrasts treat the cell as missing and apply Section 13.6.

# 19. Repository and CLI Contract

## 19.1 Repository implementation contract

FedORBIT uses one non-compatibility architecture. `configs/fedorbit.yaml` is the sole YAML configuration authority. `outputs/` holds generated scientific artifacts; `results/` holds terminal exports only.

```text
src/fedorbit/
├── __init__.py, types.py, interface.py, oracle.py, reporting.py, cli.py
├── config/{__init__,models,loading,validation}.py
├── datasets/{__init__,common,preprocessing,splitting,ontology}.py
├── datasets/edge_iiotset/{__init__,loader,schema,validation}.py
├── datasets/ton_iot/{__init__,loader,components,validation}.py
├── learning/{__init__,models,training,pilot,scoring}.py
├── response/{__init__,pilot,estimation,uncertainty,packet}.py
├── optimization/{__init__,correspondence,objective,diagnostics,assignment,exact_sparse,exact_qap,dense_ccp,certificates}.py
├── methods/{__init__,target,confirmation,assimilation,baselines}.py
├── experiments/{__init__,catalogue,cells,synthetic}.py
├── analysis/{__init__,records,metrics,comparisons,statistics}.py
└── infrastructure/{__init__,workspace,manifests,provenance,planner,execution,reuse,runtime,environment,failures}.py
```

Dataset adapters remain separate under `datasets/edge_iiotset/` and `datasets/ton_iot/`. There are no legacy package redirects or compatibility imports. `run` owns all applicable computational outputs; `report` consumes completed verified artifacts only and exports terminal evidence under `results/`.
## 19.2 Public CLI

Executable:

```text
fedorbit
```

Public commands:

```text
fedorbit doctor

fedorbit preprocess
fedorbit preprocess "DATASET NAME"
fedorbit preprocess --overwrite
fedorbit preprocess "DATASET NAME" --overwrite

fedorbit plan

fedorbit smoke
fedorbit smoke --overwrite

fedorbit run "EXPERIMENT NAME"
fedorbit run "EXPERIMENT NAME" --overwrite

fedorbit status
fedorbit status "EXPERIMENT NAME"

fedorbit report
fedorbit report "EXPERIMENT NAME"
fedorbit report "EXPERIMENT NAME" --overwrite
```

The optional `DATASET NAME` argument accepts exactly one of the configured dataset identifiers:

```text
edge_iiotset_network
ton_iot_windows10_host
ton_iot_linux_process_host
ton_iot_network
```

Matching is exact and case-sensitive. Display names, filesystem names, aliases, and source-dataset names such as `Edge-IIoTset` or `ToN-IoT` are not accepted as CLI identifiers.

No public option may override a dataset scientific role, method, seed, support, action budget, threshold, model architecture, hyperparameter grid, experiment condition, statistical procedure, or claim criterion.

Every mutating command uses the same execution sequence:

```text
validate existing artifacts
→ reuse compatible artifacts
→ identify stale descendants
→ remove stale descendants from the active namespace
→ recompute only missing/invalidated artifacts
→ validate produced artifacts
→ promote completed artifacts atomically
→ continue execution
```

Command semantics:

* `doctor` is read-only. It validates the current environment, raw-data readiness, dependency availability, deserialization/solver compatibility, and project state. It reports incompatibilities but does not invalidate already completed artifacts merely because the current working tree or current environment differs from the environment that produced them.
* `preprocess` deterministically constructs dataset-level raw manifests, cleaned data, duplicate-safe splits, fitted preprocessors, processed splits, local-class manifests, transfer-eligibility manifests, and null-padding artifacts from immutable raw data. With no dataset name, all registered datasets are resolved independently. Matching valid artifacts are reused. Only the selected dataset's stale preparation descendants are reconstructed.
* `plan` is read-only. It derives the experiment dependency graph, semantic cells, reusable artifacts, stale or blocked descendants, data-qualified exclusions, and next valid resume boundary directly from the Experiment Catalogue and artifact manifests. It performs no scientific computation.
* `smoke` runs primitive, tiny-model, packet, firewall, semantic-idempotency, crash-recovery, and non-evidence end-to-end checks without inspecting principal TEST outcomes. Its artifacts are isolated from evidence-bearing execution.
* `run EXPERIMENT` resolves the requested experiment and all prerequisites, validates existing artifacts by dependency fingerprint, reuses compatible shared artifacts, reconstructs only stale or missing dependencies, executes remaining experiment-owned cells, computes registered cell-level predictions/metrics and any experiment-local registered analysis, validates schemas/resources/provenance, and writes completion state last. Cross-experiment confirmatory synthesis remains owned by Statistical Synthesis.
* `status` is read-only. It exposes Missing, Running, Completed, Failed, Invalid, Stale, and Blocked states; for stale artifacts it reports the first changed dependency and the nearest reusable ancestor.
* `report` performs no training, scoring, optimization, confirmation, metric recomputation, or statistical recomputation. It consumes only completed verified analysis/evidence artifacts and materializes manuscript-facing tables, figures, and evidence exports.

Plain execution of a fully complete experiment performs no compute.

`--overwrite` is producer-local rather than recursively destructive:

* `preprocess ... --overwrite` reconstructs the selected preparation artifacts under the same scientific contract but does not force downstream recomputation unless the promoted preparation artifact identity changes;
* `smoke --overwrite` reconstructs smoke-owned artifacts only;
* `run EXPERIMENT --overwrite` reconstructs artifacts produced by that experiment while still reusing compatible upstream shared artifacts owned by other stages/experiments;
* `report ... --overwrite` reconstructs manuscript exports only.

`--overwrite` never creates a new scientific identity. It never means "delete all upstream work" or "rerun every ancestor." If a forced reconstruction produces the same dependency fingerprint and validated content identity as the active artifact, descendants remain valid.

# 20. Semantic Execution, Idempotency, Selective Invalidation, and Recovery Contract

## 20.1 Scientific cell identity

Scientific identity is semantic. An internal cell is identified by the applicable coordinates:

```text
experiment
dataset
source_client
target_client
directed_pair
method
condition
support
seed
```

Only dimensions scientifically relevant to the experiment are present.

UUIDs, timestamps, incrementing run numbers, random IDs, and hash-only IDs may not define scientific identity. Hashes and timestamps are provenance and compatibility information only.

A rerun of the same valid semantic cell is the same scientific result.

Shared artifacts may be referenced by multiple semantic cells. Their reusable identity is determined by artifact type, semantic producer coordinates, and dependency fingerprint, not by the first experiment that happened to request them.

## 20.2 Dependency fingerprints

Every reusable artifact has one `dependency_fingerprint_sha256` computed from only the dependencies that can materially change that artifact.

A dependency fingerprint is the canonical hash of:

1. the exact upstream artifact identities consumed;
2. the applicable scientific configuration subset;
3. the applicable semantic coordinates, including seed where stochastic behavior depends on it;
4. the registered implementation fingerprint for the code path that produces the artifact;
5. the model/generator/preprocessing definition where applicable;
6. the stage-specific numerically material runtime/ABI fingerprint where changes can affect produced values.

The implementation fingerprint is stage-local. It covers the producer function/module and registered transitive scientific code used by that producer. It is not the whole repository commit.

The numerically material runtime fingerprint is also stage-local. For example:

* preprocessing fingerprints include the exact parsing/serialization/numeric libraries that affect preprocessing output;
* training fingerprints include the framework/CUDA/library components that affect training numerics;
* solver fingerprints include the solver backend/version and relevant numerical libraries;
* pure reporting artifacts do not invalidate scientific artifacts when a plotting or document-export dependency changes.

The complete repository commit, dependency lock, operating system, hardware, driver, and environment remain recorded as provenance. They invalidate an existing artifact only when a stage-specific dependency rule explicitly includes the changed component because it can alter that artifact's computation or interpretation.

Changing any dependency included in a fingerprint creates a different compatible-artifact key. Changing anything outside that fingerprint does not invalidate that artifact.

## 20.3 Scientific execution graph

The reusable computational spine is:

```text
immutable raw inputs
→ raw manifest / parsing
→ cleaning / duplicate groups
→ chronological split
→ fitted TRAIN-only preprocessing
→ processed TRAIN / META / VALID / CONFIRM / TEST
→ local class and transfer eligibility / null padding
→ base-model pilot selection
→ confirmatory base checkpoints
→ canonical checkpoint scoring / risk artifacts
→ source-response pilot selection
→ final source-response packets
→ target importance and target-local diagnostic artifacts
→ correspondence / robust-action construction
→ proposal selection
→ target confirmation
→ live assimilation when accepted
→ canonical TEST predictions
→ registered evaluation metrics
→ registered statistical analysis
→ tables / figures / compact claim evidence / reproducibility exports
```

Synthetic experiments enter at the corresponding generator artifact and then use the same solver, metric, statistical, and reporting boundaries where applicable.

FedORBIT has no separate threshold-calibration stage. No threshold artifact or threshold cache may be introduced into this roadmap. The analogous post-scoring decision stages are response-packet construction, correspondence/action optimization, and target confirmation.

## 20.4 Reuse rules

Before any compute, the executor recursively validates the requested cell's artifact dependencies from the nearest upstream reusable boundary.

A reusable artifact must satisfy all of the following:

* its completion manifest exists and has terminal state Completed;
* every mandatory payload exists;
* payload checksums and schemas validate;
* its stored dependency fingerprint recomputes exactly from the recorded upstream artifact identities and applicable producer contract;
* all upstream artifacts referenced by the manifest remain valid;
* strict-resource and access-trace validation pass where applicable;
* any stage-specific deserialization/ABI compatibility requirement passes.

An incomplete, failed, corrupt, stale, incompatible, or provenance-invalid artifact is never promoted as a second scientific result. Reconstruction occurs under the same semantic identity. Technical failure and recovery history may be retained only in the experiment's `outputs/experiments/<descriptive-experiment-name>/logs/failures/` records and is never a scientific input.

A valid reusable artifact is not regenerated merely because:

* another experiment requests it;
* a downstream experiment previously failed;
* the repository commit changed outside its producer code path;
* comments, tests, documentation, CI files, logging, console formatting, directory naming, or reporting presentation changed;
* unrelated scientific code changed in a stage the artifact does not depend on;
* the current environment differs in components not included in that artifact's numerically material runtime fingerprint.

Examples of required reuse include:

* one prepared dataset/split/preprocessor lineage reused by every compatible experiment;
* one selected base checkpoint per client/seed reused across packet construction, baselines, confirmation, and later experiments;
* one canonical prediction/score artifact for an unchanged checkpoint × processed split × scoring definition reused by every compatible metric or experiment;
* one final source-response packet reused by every compatible pair/method/experiment;
* one target-importance artifact reused across methods sharing the same target checkpoint/META state;
* principal `s=2` FedORBIT cells reused in Mechanism Ablations and Sparsity and Dense Fallback when semantic coordinates and dependencies are identical;
* local-only reference predictions/metrics reused wherever the same base target checkpoint and TEST split define the same reference;
* statistical outputs reused by reporting until their metric inputs or statistical implementation/configuration change.

The existence of a reusable TEST prediction or metric artifact never bypasses Section 10.4. A consuming semantic cell may reference TEST-derived artifacts only after that cell's own pre-TEST source selection, action, confirmation, and live-assimilation state satisfies the TEST opening rule.

## 20.5 Selective invalidation boundaries

Invalidation propagates only from the changed artifact or contract to its descendants. Siblings and unrelated completed artifacts remain valid.

There is no FedORBIT calibration/threshold artifact. Accordingly, a generic "calibration/threshold change" invalidation boundary is not applicable; any future implementation must not invent one. FedORBIT's decision-stage invalidation is governed by response-packet, target-importance, solver/action, confirmation, and assimilation dependencies below.

| Changed dependency | Must recompute | Must not automatically recompute |
| --- | --- | --- |
| raw file bytes, raw adapter semantics, timestamp/label parsing that changes retained data | affected raw parse/cleaning, splits, preprocessors, processed splits, class/eligibility artifacts, affected checkpoints and every descendant | other datasets and clients whose raw/preparation fingerprints are unchanged |
| duplicate canonicalization, split rule, TRAIN-only preprocessing definition, leakage exclusions, feature-quality rules | affected cleaned/split/preprocessor lineage and every descendant using it | synthetic artifacts and unrelated datasets/clients |
| local class/transfer-eligibility/null-padding definition or observed support manifest | affected eligibility/null-padding, packet/target-importance/action and pair descendants; retraining only when the local prediction class set or processed model inputs change | unrelated clients; checkpoints whose model input/output and training data remain identical |
| model architecture, initialization, training algorithm, loss/class weighting, selected hyperparameters, stopping/checkpoint rule, training seed, training code path, materially relevant training runtime | affected pilot/selection artifact where selection dependencies changed, affected checkpoint(s), scores/risks, response packets, target importance, confirmation/evaluation and later descendants | prepared data/preprocessors and checkpoints for unaffected clients/seeds |
| pilot-selection logic or pilot metric implementation | selected hyperparameters and confirmatory checkpoints for the affected client, then descendants | raw/preprocessed data and unaffected client pilots |
| scoring/inference definition, model evaluation-mode semantics, probability/loss calculation code | affected canonical score/prediction/risk artifacts, response or evaluation artifacts that consume them, metrics/statistics/reporting | training checkpoints when checkpoint construction is unchanged |
| source-response candidate grid, response schedule definition, response estimator code, simultaneous-band procedure, selected response configuration | affected response pilot/final packets and every action/confirmation/evaluation descendant that consumes them | base checkpoints, preprocessing, unrelated source packets |
| target-importance definition or target META-risk code | affected target-importance artifacts and action/selection/confirmation/evaluation descendants | source packets and base checkpoints |
| correspondence rules, baseline definition, exact-sparse/QAP/dense solver implementation or applicable solver settings | affected solver/action/certificate artifacts and confirmation/evaluation descendants that depend on those actions | source packets, target checkpoints/importance, prepared data |
| confirmation rule, confirmation schedule, acceptance bound, live-assimilation implementation | affected confirmation decisions, multi-source selected-source decisions where confirmation participates, accepted live checkpoints, TEST predictions/metrics/statistics/reporting | already-valid proposal/action artifacts whose solver dependencies are unchanged |
| TEST scoring/evaluation implementation | affected TEST prediction artifacts when scoring changed; affected metrics and later descendants | pre-TEST checkpoints, packets, actions, confirmation decisions |
| metric definition/code or evaluation class-set definition | affected metric rows, paired comparisons/statistics, tables/figures/claim evidence | predictions/scores, checkpoints, packets, actions |
| statistical test, BCa/randomization/equivalence/materiality implementation or multiplicity family | affected statistical synthesis, evidence classification, statistical tables/figures | metrics and every upstream scientific artifact |
| evidence criteria, kill/simplification rule, evidence classification code | evidence-classification artifacts and report surfaces that consume them | statistical results and upstream computation |
| table/figure/report layout, labels, formatting, rendering code | only affected reporting exports | any scientific artifact, metric, or statistical result |
| README, prose documentation outside the scientific-contract snapshot, comments, tests, CI, logging, developer tooling, unused code | nothing scientific unless the change also alters a registered producer dependency | all completed scientific artifacts |

A change to the authoritative scientific contract invalidates exactly the artifact families whose applicable configuration subset changed and their descendants. The existence of a new Git commit does not itself cause invalidation.

## 20.6 Parent replacement and stale descendants

Every active artifact records its upstream artifact identities. The executor therefore derives reverse dependency edges without requiring a separate workflow engine.

When an active parent is replaced by a different valid artifact identity:

1. descendants that reference the old parent are marked Stale;
2. stale descendants are removed from the active namespace before further execution;
3. manuscript exports derived from those descendants are removed from active `results/`;
4. stale payloads are removed from the active namespace; technical stale/failure history, when retained, is limited to the experiment's failure-log records;
5. unaffected siblings remain active;
6. recomputation begins at the earliest stale descendant, not at the raw-data root.

A regenerated parent with the same validated artifact identity does not stale descendants.

No active completed descendant may silently reference a superseded parent.

## 20.7 Atomic completion and crash safety

All mutable computation occurs in a staging area that is not reusable.

For every artifact:

1. acquire the semantic-artifact write lock;
2. write payloads to a unique directory under `outputs/cache/staging/`;
3. validate schema, checksums, strict-resource rules, and dependency fingerprint;
4. write the completion manifest last;
5. atomically promote the validated directory to the active artifact location;
6. release the lock.

A crash before promotion leaves only data under `outputs/cache/staging/`. Staging data are never reusable and are discarded or cleaned on the next mutating command.

Directory existence, a checkpoint file, a metrics file, or a zero exit code alone never establishes completion.

## 20.8 Recovery

Infrastructure retry semantics remain those in Section 18.1. Retrying a failed experiment never deletes valid upstream artifacts.

Recovery follows:

```text
validate existing artifacts
→ reuse compatible artifacts
→ locate the nearest incomplete or stale descendant
→ resume from the nearest validated recovery boundary
→ recompute only that boundary and its affected descendants
→ continue
```

Registered recovery boundaries are:

* completed prepared dataset/split/preprocessor artifacts;
* completed base-model epoch with model, optimizer, and RNG state;
* completed selected base checkpoint;
* completed canonical score/prediction artifact;
* completed source-response replicate;
* completed final source-response packet;
* completed target-importance artifact;
* completed exact-sparse support;
* completed proposal/action artifact;
* completed confirmation replicate;
* completed confirmation decision and accepted live-assimilation checkpoint;
* completed independent subcell of matrix/boundary experiments;
* completed metric artifact;
* completed statistical family when every family input is unchanged.

Recovery requires equality of the recovery artifact's dependency fingerprint, semantic coordinates, upstream identities, and RNG-state schema where stochastic continuation is involved.

A partially solved LP/MIP/CCP trajectory is restarted unless a registered recovery implementation serializes and validates the backend state. No solver checkpoint is required by this roadmap.

If an implementation fix changes only a downstream producer code fingerprint, already valid upstream artifacts remain usable. For example, if experiment A is complete, experiment B crashes, and the fix changes only B's analysis or solver code, A and all shared upstream data/checkpoint/packet artifacts whose fingerprints are unchanged remain valid.

Any mismatch at a recovery boundary forces clean recomputation from the nearest earlier valid boundary of the same semantic cell; it does not force recomputation from raw inputs unless the mismatch originates there.

# 21. Artifact and Provenance Contract

## 21.1 Execution and final-evidence surfaces

`outputs/` is the complete generated computational workspace. Canonical active project-wide preparation artifacts live under `outputs/preprocessing/`; canonical reusable cross-experiment artifacts live under `outputs/artifacts/`; registered experiment-owned computation, evaluation, metrics, statistics, checkpoints, diagnostics, logs, and provenance live under `outputs/experiments/<descriptive-experiment-name>/`. `outputs/cache/` is disposable and can never establish scientific completion or manuscript evidence.

Required logical layout:

```text
outputs/
  preprocessing/
    inventories/
    validation/
    prepared/
    splits/
    features/
    metadata/
  artifacts/
    models/
    scores/
    fitted/
    baselines/
    derived/
  experiments/
    <descriptive-experiment-name>/
      artifacts/
        fitted/
        predictions/
        derived/
      evaluations/
        records/
        comparisons/
        aggregates/
      metrics/
        per_seed/
        per_condition/
        aggregate/
      statistics/
        tests/
        confidence_intervals/
        effects/
        multiplicity/
      checkpoints/
        training/
        execution/
      diagnostics/
        scientific/
        numerical/
        runtime/
      logs/
        execution/
        failures/
      provenance/
        configuration/
        data/
        seeds/
        code/
        environment/
        dependencies/
  cache/
    preprocessing/
    models/
    evaluation/
    analysis/
    staging/

results/
  experiments/
    <descriptive-experiment-name>/
      figures/
        main/
        supplementary/
      tables/
        main/
        supplementary/
      metrics/
        primary/
        secondary/
        summary/
      statistics/
        tests/
        confidence_intervals/
        effects/
        multiplicity/
  project_summary/
    figures/
      main/
      supplementary/
    tables/
      main/
      supplementary/
    metrics/
      primary/
      summary/
    statistics/
      comparisons/
      confidence_intervals/
      effects/
      multiplicity/
    reproducibility/
      configuration/
      datasets/
      seeds/
      software/
      execution/
```

The exact leaf path may include human-readable semantic coordinates and a dependency fingerprint, but a hash-only path may not replace semantic identity.

`outputs/preprocessing/` contains project-wide dataset preparation products that are scientifically reusable across compatible experiments.

`outputs/artifacts/` contains project-wide reusable artifacts with one canonical producer, including model/checkpoint state, canonical scores, reusable fitted scientific objects, reusable baseline artifacts, and deterministic derived artifacts.

`outputs/experiments/<descriptive-experiment-name>/` contains experiment-owned artifacts and records. Shared project-wide payloads are referenced rather than duplicated.

`outputs/cache/staging/` contains incomplete uncommitted work and is never a valid input. All of `outputs/cache/` is disposable and cannot establish completion or evidence.

`results/experiments/<descriptive-experiment-name>/` contains compact manuscript-facing evidence for one completed and verified registered experiment.

`results/project_summary/` contains cross-experiment manuscript evidence and reproducibility summaries assembled only after verified synthesis and evidence classification.

`results/` may contain only completed, verified compact evidence. Failed, invalid, stale, debug, cache, temporary, or development-only artifacts are forbidden there.

`results/` is never an input to scientific computation.

## 21.2 Artifact ownership and lifecycle

Shared artifact ownership is by canonical producer stage, not by consuming experiment. Project-wide preparation artifacts are materialized under `outputs/preprocessing/`; other cross-experiment reusable artifacts are materialized under `outputs/artifacts/`; experiment-owned artifacts are materialized under the owning `outputs/experiments/<descriptive-experiment-name>/` subtree. Consuming experiments reference shared artifacts by identity rather than copying them.

| Artifact family | Canonical producer | Typical consumers |
| --- | --- | --- |
| raw-data manifest / parsed component | dataset preparation | cleaning, validation |
| cleaned canonical rows / duplicate groups | dataset preparation | split construction |
| split manifest | dataset preparation | preprocessing, training, scoring, eligibility |
| fitted TRAIN-only preprocessor / processed splits | dataset preparation | model training, source response, confirmation, evaluation |
| local-class / transfer-eligibility / null-padding manifest | dataset preparation | model dimensions, packet construction, pair validation, solver |
| pilot checkpoint and pilot VALID metrics | base-model pilot | deterministic pilot selection |
| selected hyperparameter artifact | base-model pilot | confirmatory training |
| confirmatory base checkpoint with optimizer/RNG state | base-model training stage within Base-Model Hyperparameter Pilot | scoring, source response, target importance, baselines, confirmation |
| canonical score/prediction/risk artifact | canonical scoring stage | response estimation, target importance, metrics, reference reuse |
| selected response-estimator artifact | Source-Response Estimator Pilot | final packet construction |
| paired response replicate / simultaneous bands / final source-response packet | Final Source-Response Band Validation | every compatible pair/method experiment |
| target-importance / target-local diagnostic artifact | target-state derivation stage | solvers, source ranking, mechanism analysis |
| correspondence/action/certificate artifact | registered solver stage in the owning method cell | confirmation, diagnostics, evaluation |
| proposal ranking / source-selection decision | Multi-Source Selection Validation or the registered multi-source method stage | confirmation/assimilation evidence |
| confirmation replicate / decision / live-assimilation checkpoint | confirmation stage | TEST scoring and safety analysis |
| TEST prediction artifact | canonical scoring stage after final method state | metric computation |
| metric artifact | registered metric library | statistical synthesis, descriptive tables |
| paired/statistical result | Statistical Synthesis | evidence classification, tables, figures |
| evidence-classification artifact | Evidence Classification | evidence export |
| table/figure/metric/statistical/reproducibility export | `fedorbit report` | manuscript only |

An artifact lifecycle is:

```text
outputs/cache/staging/ → Completed and Active → Stale/Failed/Invalid and removed from the active namespace
```

Only Completed and Active artifacts are reusable. Failure/recovery log records may remain under the owning experiment's `logs/failures/` directory for technical audit, but they are never reusable scientific artifacts.

## 21.3 Stage-specific cache keys

The deterministic cache key is the artifact dependency fingerprint defined in Section 20.2. At minimum:

| Artifact/cache | Required dependency material |
| --- | --- |
| raw parsing | raw SHA-256 + dataset adapter/parser contract + parser implementation fingerprint + relevant parsing runtime |
| cleaning | raw-manifest identity + cleaning/duplicate/leakage contract + cleaning implementation fingerprint |
| split | cleaned-data identity + chronological split contract + seed where the registered split artifact is seed-scoped |
| preprocessor | TRAIN split identity + preprocessing contract + preprocessing implementation fingerprint |
| processed split | fitted preprocessor identity + split identity + serialization contract |
| transfer eligibility / null padding | split/class-count identities + transfer-support/ontology contract + eligibility implementation fingerprint |
| synthetic instance | generator contract + experiment condition coordinates + seed + generator implementation fingerprint |
| pilot fit | processed TRAIN/VALID identities + model/training contract + candidate hyperparameters + seed + training implementation/runtime fingerprint |
| selected hyperparameters | complete pilot-metric identities + deterministic selection rule implementation |
| confirmatory checkpoint | processed TRAIN/VALID identities + selected-hyperparameter artifact + model/training contract + seed + training implementation/runtime fingerprint |
| canonical score/prediction | checkpoint identity + processed split identity + scoring/evaluation-mode definition + scoring implementation/runtime fingerprint |
| response pilot candidate | source checkpoint identity + TRAIN/META identities + transfer-node manifest + candidate response configuration + paired schedule seed + response implementation fingerprint |
| final response packet | source checkpoint identity + selected response-estimator artifact + TRAIN/META identities + transfer-node/coarse-group manifest + final response contract + seed + response implementation fingerprint |
| rectangular minima | response-packet identity + coarse-group manifest + rectangular implementation fingerprint |
| exhaustive orbit | response packet or synthetic instance identity + block-correspondence manifest + exhaustive implementation fingerprint |
| target importance | target checkpoint identity + META score/data identity + target eligibility manifest + importance contract + implementation fingerprint |
| solver action/certificate | packet/instance identity + target-importance identity + action/cost/support/coarse-group contract + method/solver contract + solver implementation/runtime fingerprint |
| confirmation | pre-confirm checkpoint/optimizer/RNG identity + action/proposal identity + processed TRAIN/CONFIRM identities + confirmation contract + schedule seed + implementation/runtime fingerprint |
| live assimilation | clean pre-confirm state identity + accepted action identity + processed TRAIN identity + assimilation contract + RNG/schedule identity + implementation/runtime fingerprint |
| TEST prediction | final method checkpoint/state identity + TEST processed split identity + evaluation-mode/scoring contract + implementation/runtime fingerprint |
| metric | prediction identity or other registered endpoint inputs + metric definition + evaluation class-set identity + metric implementation fingerprint |
| statistical result | ordered completed metric/comparison input identities + statistical specification subset + multiplicity family + statistical implementation fingerprint |
| table/figure/report | exact verified metric/statistical/claim input identities + reporting definition + rendering implementation fingerprint |

A cache hit is valid only when the dependency fingerprint, payload checksums, schema, completion state, and upstream identities all validate.

## 21.4 Provenance versus validity

Every evidence-bearing result remains traceable to:

* semantic experiment coordinates;
* clean Git commit that produced each artifact;
* exact dependency lock recorded at production;
* full environment digest;
* OS/build information;
* CPU, GPU, RAM, CUDA runtime, and driver;
* solver versions;
* raw-data SHA-256 manifest;
* split SHA-256;
* preprocessing SHA-256;
* eligibility manifest;
* base checkpoint SHA-256;
* response packet SHA-256;
* target importance SHA-256;
* scientific configuration SHA-256;
* dependency fingerprint for every consumed reusable artifact;
* relevant producer-code fingerprint;
* material runtime fingerprint;
* seed;
* access trace;
* metric code SHA-256;
* statistical code SHA-256;
* source-data SHA-256;
* completion manifest.

These fields preserve forensic reproducibility. They are not all global invalidation triggers. Validity follows Sections 20.2–20.5.

Environment variables capable of changing scientific behavior are forbidden unless explicitly whitelisted and serialized.

## 21.5 Clean/reference and cross-experiment reuse

The following reuse is required when dependency fingerprints are identical:

* prepared data, duplicate groups, splits, preprocessors, processed splits, class manifests, eligibility, and null padding are materialized once per true lineage;
* base-model pilot fits are materialized once per client × candidate × pilot seed;
* selected hyperparameters are materialized once per client;
* confirmatory base checkpoints are materialized once per client × confirmatory seed;
* source-response pilot candidate outputs are materialized once per client/model family × candidate × pilot seed/schedule;
* final source-response packets are materialized once per source client × confirmatory seed;
* target checkpoint META predictions and target-importance vectors are materialized once per target state;
* local-only reference TEST predictions and metrics are materialized once per target checkpoint × TEST split × evaluation definition;
* identical FedORBIT/baseline semantic method cells referenced by more than one experiment are computed once;
* principal `s=2` outputs are reused by sensitivity/ablation experiments when no condition changes their actual dependencies;
* clean/reference solver and confirmation artifacts are reused across comparisons when the compared method does not alter them;
* analysis and report regeneration never retriggers training, scoring, response estimation, or solver execution.

## 21.6 Canonical artifact serialization and filenames

Every canonical artifact directory contains `manifest.json` and `completion.json`. Additional payload filenames are determined by payload type, not by producer preference. Whenever a registered experiment, table, figure, or other descriptive roadmap name is used as a filesystem slug, derive it by Unicode NFC normalization, case-folding, replacing every maximal run of characters outside `[a-z0-9]` with one hyphen, and trimming leading/trailing hyphens. Thus a registered display name has exactly one path slug while the public CLI continues to accept the exact descriptive experiment name from Section 15.

| Payload type | Canonical filename / format |
| --- | --- |
| typed manifest, provenance, solver certificate, small structured decision | `data.json` |
| row-oriented/tabular prepared data, predictions, metrics, comparisons, statistics | `data.parquet` |
| dense numeric matrix/vector payload | `arrays/<array-name>.npy`, one array per file |
| PyTorch model + optimizer + RNG checkpoint | `checkpoint.pt` |
| structured execution/failure log | `events.jsonl` |
| manuscript table export | `<table-slug>.csv` |
| manuscript figure export | `<figure-slug>.svg` |

JSON uses UTF-8, NFC strings, lexicographically sorted object keys, no insignificant whitespace, shortest round-trippable decimal rendering, and one trailing LF. JSONL applies the same canonical JSON rule independently to each line.

Parquet is written through the pinned PyArrow version with the declared typed schema and `zstd` compression. Row order is scientifically defined by the producing schema/semantic coordinates and must be explicitly sorted before serialization; Parquet physical row-group order may not be used to define scientific order.

NumPy array files use the configured numeric precision and C-order/row-major layout. Array names are descriptive schema names and are sorted lexicographically when hashed as a multi-array artifact.

`checkpoint.pt` is an implementation payload, not scientific identity; its physical SHA-256 verifies integrity while its dependency fingerprint and checkpoint semantic coordinates determine reusability.

CSV uses UTF-8, comma delimiter, RFC 4180 quoting, LF line endings, one header row, and the reporting precision rules only for manuscript-facing values. Computational tables remain Parquet and preserve unrounded values.

SVG is the canonical active figure format. The renderer must suppress nondeterministic creation-time metadata and must render from unrounded verified source values. A user may convert SVG externally for manuscript submission, but such conversion is not a scientific artifact and does not enter `results/`.

A canonical artifact may contain more than one payload category only when its manifest lists every mandatory path and checksum. Producer-specific alternate filenames/formats are not permitted for active canonical artifacts.

# 22. Machine-Readable Result Schemas

## 22.1 Dataset manifest

Required:

```text
dataset
component
raw_files
raw_sha256
raw_counts
schema
adapter_feature_order
adapter_feature_roles
accepted_schema_aliases
adapter_adaptations
timestamp_field
timestamp_range
duplicate_counts
conflicting_duplicate_counts
local_class_counts
transfer_candidate_counts
feature_quality
preprocessing_state
dependency_fingerprint_sha256
producer_code_sha256
```

## 22.2 Transfer eligibility

One row per endpoint × seed × candidate transfer concept:

```text
client
seed
coarse_group
anonymous_node_id
native_local_class_ids
present
train_count
meta_count
confirm_count
test_count
source_eligible
target_eligible
null_reason
```

`native_local_class_ids` contains opaque local class IDs only and is retained in the builder/provenance copy needed to reproduce aggregate transfer-node construction. It is omitted from the method-readable transfer-eligibility artifact. The oracle copy additionally contains the fine concept. The method-readable copy contains neither native class IDs nor the fine concept.

## 22.3 Semantic cell manifest

```text
experiment
dataset
source_client
target_client
directed_pair
method
condition
support
seed
scientific_configuration_sha256
dependency_fingerprint_sha256
producer_stage
upstream_artifact_ids
dataset_manifest_sha256
split_sha256
preprocessing_sha256
source_checkpoint_sha256
response_packet_sha256
target_checkpoint_sha256
importance_vector_sha256
resource_manifest_sha256
relevant_code_sha256
material_runtime_sha256
git_commit
git_dirty
environment_sha256
state
state_reason
```

Hashes validate compatibility and provenance; they are not scientific experiment identity.

## 22.4 Prediction schema

```text
experiment
pair
method
condition
seed
row_hash
split
true_local_class_id
predicted_local_class_id
probabilities
loss
checkpoint_artifact_id
processed_split_artifact_id
dependency_fingerprint_sha256
```

## 22.5 Metric schema

```text
experiment
pair
method
condition
seed
metric_name
metric_value
metric_unit
direction
evaluation_class_set_sha256
input_artifact_ids
dependency_fingerprint_sha256
valid
invalid_reason
```

## 22.6 Paired comparison schema

```text
contrast_name
family
pair
method_a
method_b
metric
paired_seed_count
mean_difference
median_difference
bca_ci_low
bca_ci_high
raw_p
holm_p
materiality_threshold
equivalence_margin_low
equivalence_margin_high
input_metric_artifact_ids
dependency_fingerprint_sha256
decision
```

## 22.7 Statistical metadata

```text
test_name
exact_or_asymptotic
alternative
zero_difference_count
bootstrap_resamples
bootstrap_seed
holm_rank
family_size
statistical_code_sha256
```

## 22.8 Completion manifest

Required:

```text
semantic_experiment_coordinates
producer_stage
terminal_state
dependency_fingerprint_sha256
upstream_artifact_ids
mandatory_artifact_paths
mandatory_artifact_sha256
scientific_configuration_sha256
relevant_code_sha256
material_runtime_sha256
upstream_lineage
completion_validation_state
completion_written_last
```

## 22.9 Reusable artifact manifest

Every reusable project-wide payload in `outputs/preprocessing/` or `outputs/artifacts/` has exactly one manifest containing:

```text
artifact_id
artifact_type
semantic_producer_coordinates
producer_stage
dependency_fingerprint_sha256
upstream_artifact_ids
applicable_configuration_sha256
relevant_code_sha256
material_runtime_sha256
payload_paths
payload_sha256
schema_version
created_git_commit
created_environment_sha256
state
completion_manifest_sha256
```

`artifact_id` is a stable content/provenance reference. It does not replace semantic experiment identity.

# 23. Reporting and Final Evidence Contract

`fedorbit report` is the only public operation that writes active manuscript evidence. It performs no scientific computation, consumes only completed verified artifacts, preserves lineage, reuses matching exports, and applies explicit `--overwrite` under the Semantic Execution Contract. `results/` is never a scientific input. Figures use unrounded source values; numeric display follows the Configuration YAML.

A report export has its own dependency fingerprint derived from the exact metric/statistical/claim artifacts it consumes plus the applicable table/figure/report definition. Reporting changes invalidate only reporting exports.

If any scientific input referenced by an active table, figure, metric summary, statistical export, or reproducibility export becomes stale, that export is removed from active `results/` before a replacement is written.

## 23.1 Required tables

* Dataset and Client Protocol Table — rows: Edge network, Windows10 host, Linux process, ToN network. Columns: dataset component, modality, observed raw rows, retained rows, timestamp range, local prediction classes, feature count, transfer candidates, exclusions, scientific role, raw-manifest hash.
* Transfer Ontology and Null-Padding Table — rows: candidate concept × primary pair. Columns: coarse group, source real/null, target real/null, support counts, action eligibility, null reason. Fine names are manuscript/oracle-only.
* Model and Training Protocol Table — rows: network model, host model. Columns: architecture, normalization, activation, initialization, optimizer, batch, selected learning rate, selected weight decay, selected dropout, stopping rule.
* Information Resource Matrix — rows: registered methods. Columns: target raw data, anonymous source nodes, coarse groups, source response, target-local response, fine names, exact map, confirmation, predecision TEST access, strict compatibility.
* Numerical Constants and Seeds Table — generated directly from the Configuration YAML; no manually duplicated constants.
* Experiment Matrix — rows: every registered experiment. Columns: classification, datasets/pairs, methods, seeds, conditions, derived planned cells, prerequisites, claim relationship.
* Primary Strict-Transfer Results Table — rows: primary pair × method. Columns: valid seeds, TEST macro CE, macro-F1, balanced accuracy, gain vs local, BCa CI, raw p, Holm p, strict validity, confirmation coverage. Method order: Local-Only, Local-SIR, Matched-Resource Rectangular, Point-Correspondence Commitment, Generic Exact QAP, FedORBIT Exact-Sparse Solver, Exact-Map Oracle.
* Coupling-Mechanism Results Table — condition/pair, valid units, fixed-action gap, robust coupling gap, fraction above materiality, CI, Holm p, coupling-destruction retained-gain fraction.
* Exact-Solver Results Table — K, block pattern, support, truth availability, exact mismatches, maximum absolute error, runtime median/p95, QAP runtime, dense runtime, timeouts, memory, active images, LAP calls.
* Ablation Results Table — ablation, pair, realized gain, difference vs full, equivalence, retained gain, confirmation safety.
* Sparsity and Dense Results Table — rows: support/dense condition × pair. Columns: realized gain, certified value where applicable, runtime, memory, confirmation coverage, dense-minus-sparse difference.
* Confirmation Results Table — pair, proposals, accepted, harmful accepted rate, useful accepted rate, beneficial rejected rate, coverage, no-confirm harmful rate, ARR, RRR, CI, p.
* Generalization Results Table — rows: secondary pair × method; same predictive metrics as the primary table and explicitly labeled secondary.
* Failure-Boundary Results Table — boundary dimension, setting, pair, method, certified value, realized gain, abstention, null-node count, confirmation coverage, state.
* Scalability Results Table — K, block, support, method, $N_S$, LAP calls, cuts, runtime median/p95, RSS, CUDA memory, timeout, exactness status.
* Claim Support Table — claim, final state, materiality result, statistical result, evidence completeness, scope, supporting table, supporting figure, forbidden wording.

## 23.2 Required figures

* Real Transfer-Gain Forest Plot — Y: six primary directed pairs. X: paired mean relative macro-CE gain vs local. Show BCa interval, zero, and +0.01 materiality references.
* Baseline Paired-Difference Plot — comparators: Local-SIR, Matched-Resource Rectangular, Point-Correspondence Commitment. Show seed-level paired differences by pair; no pooled cross-pair inferential annotation.
* Coupling-Gap Phase Figure — factors: compatibility, response heterogeneity, asymmetry, response sparsity, support; overlay predicted structural zero/strict state.
* Predicted vs Realized Transfer Figure — X: certified robust predicted value. Y: TEST relative macro-CE gain. Facet by pair; annotate descriptive Spearman rho and n where eligible.
* Sparsity Utility-Efficiency Figure — X: runtime. Y: realized gain. Marker size: peak memory. Conditions: $s=1,2,3$, dense.
* Confirmation Safety-Coverage Figure — X: confirmation coverage. Y: harmful accepted rate. Show paired no-confirm → confirm arrows by primary pair.
* Semantic Sufficiency Frontier Figure — X: $\log|\Pi|$. Y: realized gain. Lines: FedORBIT, rectangular, oracle.
* Failure-Boundary Figure — panels: response scale, CI width, target support, response heterogeneity, support budget. Show certified value, realized gain, abstention/ineligibility boundary.
* Scalability Figure — X: $N_S\sum_gn_g^3$, log scale. Y: runtime, log scale. Show methods, timeout markers, descriptive trend only.
* Map-Value Bound Figure — X: orbit-radius bound. Y: exact map action value. Show diagonal bound and fixture family.

## 23.3 Reporting dependency map

* dataset/protocol tables consume only valid dataset/preprocessing/eligibility manifests;
* model/training tables consume the authoritative configuration plus selected-hyperparameter and checkpoint manifests, never TEST outcomes;
* primary/generalization/ablation/confirmation/boundary tables consume completed metric and statistical artifacts for their exact registered cells;
* solver/scalability tables consume completed solver/timing/certificate artifacts;
* claim support consumes only verified statistical synthesis plus final evidence classification;
* figures consume the same verified source artifacts as their corresponding tables and never recompute a metric or statistical test.

A change to a table or figure renderer cannot make any scientific ancestor stale.

# 24. Ordered Scientific Execution Pipeline

The scientific execution layers are:

```text
inputs
→ preprocessing / splits
→ training / checkpoint selection
→ scoring and source/target risk derivation
→ response-packet construction
→ correspondence / action optimization
→ target confirmation and live assimilation
→ TEST evaluation
→ statistical analysis
→ reporting
```

For every requested experiment, each layer first applies the Section 20 validation/reuse/invalidation sequence. The ordered programme remains:

```text
environment diagnosis
→ raw-data identity
→ preprocessing
→ smoke validation
→ mathematical primitive validation
→ exact-sparse theorem validation
→ dataset/client/resource validation
→ base-model pilot
→ base-model checkpoint selection
→ source-response pilot
→ final source-response bands
→ baseline/oracle validation
→ exact-sparse solver benchmark
→ synthetic coupling mechanism
→ real-packet coupling mechanism
→ unresolved-map action diagnostics
→ principal strict transfer
→ multi-source diagnostic
→ mechanism ablations
→ sparsity/dense sensitivity
→ confirmation/portability
→ secondary generalization
→ semantic sufficiency boundary
→ weak-signal/support/heterogeneity boundaries
→ map applicability audit
→ scalability/efficiency
→ statistical synthesis
→ evidence classification
→ manuscript evidence export
```

This order defines scientific prerequisites, not a requirement to recompute every earlier stage each time a later experiment is invoked.

The executor derives dependencies from this pipeline, the Experiment Catalogue, and artifact manifests. A later experiment may begin from any already-valid upstream boundary. If a later experiment fails, earlier successful experiments and shared ancestors remain active unless a genuine dependency fingerprint changes.

# 25. Reproducibility Contract

Reproducibility is established by the semantic execution, artifact, provenance, deterministic RNG, environment, and statistical contracts in this roadmap. No separate duplicate execution phase is required to make a completed evidence-bearing artifact scientifically valid.

Every evidence-bearing result must be traceable through the complete lineage required by Sections 20 and 21, including:

* authoritative scientific configuration identity;
* immutable raw-data identity and actual observed adapter/schema manifest;
* split/preprocessing/eligibility lineage;
* selected model and response-estimator artifacts;
* source packet, target importance, solver/action, confirmation, and assimilation lineage where applicable;
* exact seed and derived RNG/schedule identities;
* relevant producer-code and material-runtime fingerprints;
* metric/statistical implementation identities;
* completion manifests and payload checksums.

A rerun of the same semantic cell with the same valid dependency fingerprint is the same scientific cell and reuses its valid artifact. A changed scientific dependency creates a stale descendant relation and requires recomputation only from the nearest changed dependency boundary under Section 20.

After evidence-bearing TEST outcomes have been opened, no scientific definition, numerical threshold, seed set, baseline, experiment membership, metric, statistical test, multiplicity family, or claim criterion may be changed **because of** the observed outcome. A legitimate correction discovered independently of favorable/unfavorable results is made in the authoritative roadmap/implementation contract, changes the applicable dependency fingerprint, and invalidates all affected descendants for deterministic recomputation. Unaffected artifacts remain valid.

Statistical synthesis consumes only currently valid completed metric/comparison artifacts. Evidence classification consumes only the corresponding verified statistical synthesis and registered claim rules. Reporting consumes only those verified outputs and never becomes a scientific input.

A result is reproducible only when its complete lineage validates. An incompatible implementation, environment component that is materially fingerprinted for that stage, raw dataset, selected pre-TEST artifact, experiment catalogue, statistical specification, or claim criterion cannot silently replace the artifact under the same active identity.

# 26. Implementation Readiness

Implementation may begin only when the typed scientific contract matches this roadmap, all scientific decisions are bound, experiment expansion is deterministic, strict-resource enforcement is testable, dependency fingerprints and selective invalidation are implemented, atomic completion prevents partial-artifact reuse, semantic execution and nearest-valid-boundary recovery are implemented, stale descendants cannot remain active after parent replacement, and no implementation agent must invent a scientific value or rule.

**Implementation readiness: Pass.**
