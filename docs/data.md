# Data: contracts, formats, truth schema

## 1. Layout on disk

```
data/synthetic/                 # in .gitignore, generated as a whole
├── static_cloud/
│   └── torus_N900_R2_r0.8_sig0.03_sampling-rejection_coeff_field2_seed20260201/
│       ├── points.npy          # (n_points, dim) float64
│       └── ground_truth.json   # the network is NOT stored: it is derived from the points
├── dynamic_graph/
│   └── cycle_lifecycle_N24_T60_w1_thr0.5_sched12-27-41-48-55_seed20260201/
│       ├── snapshots.npy       # (T, n_nodes, n_nodes) float64, symmetric
│       └── ground_truth.json
├── series/
│   └── kuramoto_N30_T3000_K6_omega_scale1_regime_length1000_seed20260201/
│       ├── series.npy          # (n_channels, n_timepoints) float64
│       ├── coupling.npy        # (T_schedule, n_channels, n_channels)
│       └── ground_truth.json
├── noisy/                      # only the perturbed array, see below
│   ├── point_jitter_<base>_level0.05_repeat0_seed.../points.npy
│   └── weight_jitter_<base>_level0.05_repeat0_seed.../weights.npy
├── null/
├── results/
│   └── stability.json          # reported numbers of Result 6; produced by the build
└── manifest.json               # registry of every generated dataset
```

The directory name of a dataset is a deterministic function of the generator and its
parameters, seed included. The same config must give the same name and the same contents.

**Family 4: only the perturbed array goes to disk.** The source is recovered from
`noise_level["base"]`, which holds the `dataset_id` of the clean dataset. There is one rule
for the whole family, and it works both ways:

| Mechanism | What is stored | Why not a pair |
|---|---|---|
| `point_jitter` | `points.npy` | a network would belong to the old points |
| `weight_jitter` | `weights.npy` | the points did not change, the weights are corrupted on purpose |
| `spurious_edges`, `missing_edges` | `snapshots.npy` | — |

A pair in which the derived array does not match the perturbed source must not land in one
directory **in either direction**: a consumer that rebuilds the network from the points gets
something other than what is on disk and sees no error. Measured: after point jitter at level
0.10 the attached network differed from the network of the recorded points by up to 0.36.

A directory holding only weights is read as a `NetworkDataset` — the `weights.npy` marker is a
**fallback**, not a primary one: a point cloud is entitled to carry a network next to
`points.npy`, and a primary marker would break the "exactly one marker" invariant that catches
a corrupted directory.

### 1.1. Naming rule

The name contains **all** generator parameters, not a selection. The rule is not obvious, so
explicitly: if a parameter does not make it into the name, two datasets with different
geometry (say, tori with different `R`) get the same name and silently overwrite each other.
`naming.py` refuses to build a name if the declared parameter order does not cover the
`params` dictionary exactly — this check is what keeps a parameter from dropping out of the
name.

- The parameter order is declared explicitly next to the generator (`register_name_order`),
  not taken from the key order of `params`: JSON is written with sorted keys, so insertion
  order does not survive a round trip.
- Frequent parameters are abbreviated for readability: `n_points`/`n_nodes`/`n_channels` →
  `N`, `n_snapshots`/`n_timepoints` → `T`, `noise_sigma` → `sig`. The rest keep their names.
- Numbers are formatted as `format(v, f".{naming.float_precision}g")`; string values are
  separated by a hyphen (`sampling-rejection`), numeric ones are written together (`N400`).
- The seed always comes last.

**Key grouping.** For generators with a schedule (family 2), listing every parameter yields an
unreadable name, and none can be dropped — two scenarios with different schedules must differ
by name. So `register_name_order` accepts not only single keys but also **a group of keys
collapsed into one token**: the group's values are formatted by the common rule and joined by
hyphens under one prefix.

```python
register_name_order(
    "cycle_lifecycle",
    ["n_nodes", "n_snapshots", "edge_weight", "threshold",
     ("sched", ["path_closes_at", "chord_added_at", "edge_removed_at",
                "component_splits_at", "component_merges_at"])],
)
# cycle_lifecycle_N24_T60_w1_thr0.5_sched12-27-41-48-55_seed20260201
```

The exact-coverage requirement on `params` is not relaxed: a key in a group counts as covered,
a key in no group still makes the name impossible to build. A parameter hash is deliberately
not used instead of grouping: it would also guarantee distinctness but would lose the second
property of the name — that the directory tells you what it holds.

## 2. Axis order (fixed, must not change)

| Object | Shape | Meaning of the axes |
|---|---|---|
| Point cloud | `(n_points, dim)` | points × coordinates |
| Weight / distance matrix | `(n_nodes, n_nodes)` | symmetric, zero diagonal for distances |
| Dynamic graph | `(T, n_nodes, n_nodes)` | time × node × node |
| Multivariate series | `(n_channels, n_timepoints)` | channel × time |
| Coupling schedule | `(T_schedule, n_channels, n_channels)` | time × node × node |

A series is stored as `(n_channels, n_timepoints)` — channel first. This differs from the
convention of the fMRI module, where series arrive as `(n_timepoints, n_rois)`; the conversion
is done at the module boundary by an explicit transpose, not "in place".

## 3. API contracts

Every generator is a function that takes parameters and an `rng` and returns a dataclass. No
generator writes to disk itself: saving is a separate function.

```python
@dataclass(frozen=True, eq=False)
class GroundTruth:
    """Machine-readable truth for a generated dataset."""
    format_version: int
    family: str                  # LAYOUT ON DISK, not the kind of truth:
                                 # static_cloud | dynamic_graph | series | noisy | null
    truth_kind: str              # KIND OF TRUTH, determines the mandatory fields:
                                 # expected_betti | betti_curve | partition |
                                 # noise_level | none
    generator: str               # torus, kuramoto, cycle_lifecycle, ...
    params: dict                 # the actual arguments of the generator call
    seed: int
    coeff_field: int | None = None
    complex: str | None = None                        # "flag" for thresholded graphs
    edge_rule: str | None = None                      # weight_above | distance_within
    edge_threshold: float | None = None               # the value of t or eps
    expected_betti: dict[int, int] | None = None      # truth_kind: expected_betti
    betti_curve: dict[int, list[int]] | None = None   # truth_kind: betti_curve
    events: list["Event"] | None = None               # truth_kind: betti_curve
    community_labels: np.ndarray | None = None        # truth_kind: partition, int64
    noise_level: dict | None = None                   # truth_kind: noise_level
    notes: str = ""

@dataclass(frozen=True)
class Event:
    """A single homological event in a dynamic dataset."""
    snapshot: int      # snapshot index
    dim: int           # homology dimension: 0, 1, 2
    type: str          # ONLY "birth" or "death"
    feature_id: str    # stable feature identifier
    narrative: str     # human-readable: "two components merge"

@dataclass(frozen=True)
class PointCloudDataset:
    points: np.ndarray            # (n_points, dim)
    ground_truth: GroundTruth
    weights: np.ndarray | None = None

@dataclass(frozen=True)
class DynamicGraphDataset:
    snapshots: np.ndarray         # (T, n, n)
    ground_truth: GroundTruth

@dataclass(frozen=True)
class NetworkDataset:            # family 4: a perturbed network without points
    weights: np.ndarray           # (n_nodes, n_nodes), symmetric
    ground_truth: GroundTruth

@dataclass(frozen=True)
class SeriesDataset:
    series: np.ndarray            # (n_channels, n_timepoints)
    coupling: np.ndarray          # (T_schedule, n, n)
    ground_truth: GroundTruth
```

Generator signature:

```python
def make_torus(n_points: int, R: float, r: float, noise_sigma: float,
               rng: np.random.Generator) -> PointCloudDataset:
```

Rules: `rng` is the last positional argument and is always required; numpy's global state is
not used; functions are pure (the same input and rng → the same output).

The dataclasses are declared with `eq=False`. A generated `__eq__` would compare fields with
`==`, which on an `np.ndarray` raises `ValueError: ambiguous truth value`; with `eq=False`
comparison degenerates to identity, and a test that compared two different objects with `==`
honestly fails. Comparison by content is `dataset_equal` / `ground_truth_equal`, bitwise on
arrays.

## 4. Schema of ground_truth.json

```json
{
  "format_version": 2,
  "dataset_id": "torus_N400_R2_r0.8_sig0.03_sampling-rejection_seed20260201",
  "family": "static_cloud",
  "truth_kind": "expected_betti",
  "generator": "torus",
  "params": {"n_points": 400, "R": 2.0, "r": 0.8, "noise_sigma": 0.03,
             "sampling": "rejection"},
  "seed": 20260201,
  "coeff_field": 2,
  "expected_betti": {"0": 1, "1": 2, "2": 1},
  "notes": "Rejection sampling по элементу площади (R + r cos φ)."
}
```

The `notes` field is part of the recorded truth and is kept verbatim in the language it was
written in: changing it would change `truth_sha256` of every dataset (§5).

### 4.1. `family` is the layout, `truth_kind` is the kind of truth

Up to family 3 the two coincided, and one field served both. The dynamic SBM is the first
object where they diverge: on disk it has `snapshots.npy`, i.e. the layout of a dynamic graph,
while its truth is a community partition and a schedule, and there is no barcode in it at all
(`docs/method.md`, §3.1). Requiring `betti_curve` of it merely because snapshots lie on disk
would encode the false "dynamic graph ⇒ it has topological truth".

So which fields are mandatory is determined by **`truth_kind`**, while `family` only says which
subdirectory holds the dataset and which array file is expected in it.

| `truth_kind` | Mandatory fields | Who |
|---|---|---|
| `expected_betti` | `coeff_field`, `expected_betti` | family 1 |
| `betti_curve` | `coeff_field`, `complex`, `edge_rule`, `edge_threshold`, `betti_curve`, `events` | `cycle_lifecycle`, `geometric_loop` |
| `partition` | `community_labels` | dynamic SBM, oscillator series |
| `noise_level` | `noise_level` | noisy datasets of family 4 |
| `none` | — | null models of family 4 |

Oscillator series sit under `partition`, not `none`: the truth of a series is which channels
are coupled with which, i.e. a partition and a schedule, and it does have `community_labels`
(`src/tdadyn/synthetic/series.py`). Only the null models fall under `none`: their truth is the
absence of a schedule, not some value of it (`docs/method.md`, §4.2).

The field is mandatory and **is not inferred from `family` by default**: an implicit default is
exactly the mechanism of silent errors that `edge_rule` below is introduced to remove.

### 4.2. `edge_rule` and `edge_threshold` — the threshold direction

`docs/method.md` §2.3 explains why there are two edge rules and why they point in opposite
directions: for a similarity dataset an edge appears from above ($w_{ij} > t$, strict), for a
metric one from below ($d_{ij} \le \varepsilon$, non-strict). This cannot be inferred from the
family: `cycle_lifecycle` and `geometric_loop` both live in `dynamic_graph/`.

`edge_threshold` deliberately duplicates a value already present in `params`. This is not
redundancy but a separation of purposes: `params` records the **generator call**, the truth
fields define **the truth itself**. Linking them through knowledge of which `params` key holds
the threshold for which generator is the same implicit convention we are moving away from.

`format_version` was raised from 1 to 2: `truth_kind` is mandatory, and the byte-stability
promise of §6 covers only optional fields.

`dataset_id` is present in the file but is not a dataclass field: it is derived from
`(generator, params, seed)`. Storing it in the object too would let the name and the parameters
drift apart. On reading it is checked against the directory name, and if `float_precision` is
given, against a freshly recomputed one as well, which catches a hand edit of `params`.

Fields whose value is `None` are not written to JSON at all (rather than as `null`): that way
adding a new optional field does not change already generated files byte for byte.

For a dynamic graph:

```json
{
  "format_version": 2,
  "dataset_id": "cycle_lifecycle_N24_T60_w1_thr0.5_sched12-27-41-48-55_seed20260201",
  "family": "dynamic_graph",
  "truth_kind": "betti_curve",
  "generator": "cycle_lifecycle",
  "params": {"n_nodes": 24, "n_snapshots": 60, "edge_weight": 1.0, "threshold": 0.5,
             "path_closes_at": 12, "chord_added_at": 27, "edge_removed_at": 41,
             "component_splits_at": 48, "component_merges_at": 55},
  "seed": 20260201,
  "coeff_field": 2,
  "complex": "flag",
  "edge_rule": "weight_above",
  "edge_threshold": 0.5,
  "betti_curve": {"0": [1, 1, "..."], "1": [0, 0, "..."], "2": ["..."]},
  "events": [
    {"snapshot": 12, "dim": 1, "type": "birth", "feature_id": "h1_a",
     "narrative": "путь замыкается в цикл"},
    {"snapshot": 27, "dim": 1, "type": "birth", "feature_id": "h1_b",
     "narrative": "хорда делит цикл на два"},
    {"snapshot": 41, "dim": 1, "type": "death", "feature_id": "h1_a",
     "narrative": "удалено ребро дуги, отрезанной хордой"},
    {"snapshot": 48, "dim": 0, "type": "birth", "feature_id": "h0_b",
     "narrative": "от висячего пути отделяется компонента"},
    {"snapshot": 55, "dim": 0, "type": "death", "feature_id": "h0_b",
     "narrative": "компоненты сливаются обратно"}
  ]
}
```

The `narrative` strings are recorded truth and stay as generated (in Russian): "the path closes
into a cycle", "a chord splits the cycle in two", "an edge of the arc cut off by the chord is
removed", "a component detaches from the dangling path", "the components merge back".

Three places in this example look arbitrary but have a reason, and all are worked out in
`method.md`:

- **`edge_rule: weight_above` is written explicitly**, although for this dataset it is
  "obvious". It is obvious only until `geometric_loop` appears, which lives in the same family
  and requires `distance_within`. Derivation — `method.md`, 2.3.
- **The death at 41 is paired with the birth at 12, not at 27** (`h1_a`, not `h1_b`). This is
  not a convention: the decomposition of a zigzag module into intervals is unique, and the class
  that dies is the one born when the path closed. Derivation — `method.md`, 2.2a. The intuitive
  "the chord made two cycles, break one and the younger dies" is wrong here.
- **`snapshot: k` means that $G_k$ already contains the change** and $G_{k-1}$ does not. The
  jump of `betti_curve` falls on the transition $k-1 \to k$. Convention — `method.md`, 2.1.

**The key consistency requirement:** `events` and `betti_curve` are two independent sources.
`events` reflects the scheduler's intent, `betti_curve` is computed by the independent
reference implementation in `reference/`. A test must verify that the jumps of `betti_curve`
correspond exactly to the event list. A discrepancy means a bug in the generator, not a reason
to adjust the log.

## 5. manifest.json

A registry of every dataset: `dataset_id`, family, generator, seed, path relative to
`output_root`, `params`, sha256 of the arrays, sha256 of the truth, package version, generation
date. It serves reproducibility and the quick detection that data were rebuilt with different
parameters. Structure: `{"format_version": 2, "datasets": [ ... ]}`, entries sorted by
`dataset_id`.

- sha256 is computed **over the bytes of the `.npy` file**, not over `ndarray.tobytes()`: the
  result can be checked with a plain `shasum -a 256`, and the header goes into the hash, so a
  changed dtype or shape cannot hide behind an identical payload.
- **`truth_sha256` is computed over a canonical serialisation of the parsed record**, not over
  the bytes of `ground_truth.json`. A byte hash would absorb indentation, key order and float
  `repr`, and cosmetic reformatting would read as a change of data. Without this field the
  "same hashes" check would not cover the expected Betti numbers, the event log and `notes` —
  that is, the whole value of the datasets: arrays do not change when the truth is edited.
  `entry_payload` does **not** drop it, unlike `generated_at` and `package_version`: it is
  reproducible and therefore takes part in the rebuild comparison rather than standing beside
  it.
- The manifest itself is not byte-reproducible: it records the generation time. The
  requirement "a rebuild gives the same hashes" is checked on the entry without `generated_at`
  and `package_version` (`entry_payload`), **within one environment**: Rössler at $c = 5.7$ is
  chaotic, and a change of scipy version, method or tolerances moves its series entirely.
- **Sets** of entries are compared in both directions: a one-sided check would miss a vanished
  dataset. A full rebuild overwrites the manifest as a whole (`write_manifest`), because
  `update_manifest` does not remove stale rows.

**The manifest is kept in git, and that is not an exception to the rule "do not touch `data/`
by hand".** The manifest is not data but **a record of a measurement**, of the same kind as
`results/stability.json`: the 38 datasets can be regenerated from the config and seed, while the
statement "the report and the paper were written against these hashes" cannot, and no rebuild
can restore it. The rule about `data/` protects against hand edits of data and binary litter in
history; the manifest is a text JSON of 38 entries, made precisely for diffing.

Moreover, the canonical serialisation of `truth_sha256` (§5) pays off **only** under version
control: its point is that cosmetic reformatting should not read as a change of data, and
without history there is nothing to compare with.

It is committed **by a full CLI run**, not by copying the current state of the directory, and
before the commit it is checked by a two-sided comparison of `entry_payload` sets against a
fresh rebuild. The individual `ground_truth.json` files of the datasets do not go into git:
their contents are already covered by the `truth_sha256` field, i.e. one entry instead of 38.

## 6. Serialisation rules

- Arrays — `.npy` (not pickle), float64.
- JSON — UTF-8, `ensure_ascii=False`, sorted keys, indent 2.
- Homology dimension keys in JSON are strings ("0", "1", "2") and are converted to int on
  reading. This is a JSON limitation, not a design choice; the conversion happens in one place.
- `expected_betti` and `betti_curve` list **every** dimension from 0 to `max_homology_dim`,
  zeros included: a missing key is indistinguishable from a forgotten one.
- `community_labels` is stored inside `ground_truth.json` (there is no separate `.npy` for it)
  and fixed as `int64`, otherwise a round trip would silently change the dtype.
- Nothing is appended to an existing dataset directory: a rebuild is always whole.
- The dataset type on reading is determined by the name of the array file (`points.npy`,
  `snapshots.npy`, `series.npy`), not by `family`: a noisy cloud and a noisy dynamic graph have
  the same family — `noisy`.
