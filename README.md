# tdadyn

[![tests](https://github.com/Semyon-ksa/tdadyn/actions/workflows/tests.yml/badge.svg)](https://github.com/Semyon-ksa/tdadyn/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Topological analysis of dynamic graphs, with answers you can check.**

`tdadyn` is a Python package for persistent homology on static and time-varying data. It has
three parts:

- **generators of data with known topology** — point clouds, networks, dynamic graphs and time
  series for which the correct answer (Betti numbers, a log of topological events, a coupling
  schedule) is known in advance and stored next to the data;
- **zigzag persistence** — the barcode of a sequence of graphs, its translation into a log of
  events ("a cycle is born / dies, a component splits off / merges") and its comparison with the
  truth by recall and precision;
- **an independent reference implementation** — homology, ranks and the zigzag barcode over
  $\mathbb{Z}/2$ without any persistence library, so that results are checked by a different
  method rather than by the same code.

The package answers the question "does my pipeline find the topology correctly" where the answer
is known — before it is applied to data where it is not.

## Features

- 17 generators in four families: manifolds ($S^1$, torus, sphere, Klein bottle, …), dynamic
  graphs with scheduled events, series of coupled oscillators (Kuramoto, Rössler), noise and null
  models.
- Machine-readable truth for every dataset (`ground_truth.json`) with a schema and validation.
- One command builds all 38 reference datasets; deterministic directory names, a `manifest.json`
  with sha256 of the arrays and of the truth.
- A bar significance criterion (a gap in the spectrum of persistences) calibrated on clouds with
  known topology.
- From cloud to network: Gaussian kernel, $k$-NN graph with a geodesic metric; a sliding window
  "series → dynamic graph".
- A zigzag diagram from graph snapshots,
  $K_0 \hookrightarrow K_0 \cup K_1 \hookleftarrow K_1 \hookrightarrow \cdots$;
  a stream of elementary operations, an adapter to Dionysus 2.
- An independent oracle for the zigzag barcode via the generalised rank and Möbius inversion.
- An event log from the barcode, matching against the truth, a false positive rate with a Wilson
  confidence interval.

## Installation

Python 3.12 is required.

```bash
git clone https://github.com/Semyon-ksa/tdadyn.git
cd tdadyn
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Optional dependencies:

| Extra | Adds | Needed for |
|---|---|---|
| `zigzag` | Dionysus 2 | the fast zigzag backend; without it only the reference oracle is available |
| `notebook` | matplotlib, nbclient, nbconvert | building the report notebooks |
| `dev` | pytest | tests |

```bash
.venv/bin/pip install -e ".[dev,zigzag]"
```

The base installation works without the zigzag backend; functions that need it import it inside
themselves and say so if it is missing.

## Quick start

The examples below are run from the repository root: generator parameters come from
`config.yaml`.

### A dataset with a known answer

```python
from tdadyn.core import load_config
from tdadyn.synthetic import build_static_cloud

config = load_config("config.yaml")
torus = build_static_cloud("torus", config)

torus.points.shape                   # (900, 3)
torus.ground_truth.expected_betti    # {0: 1, 1: 2, 2: 1}
```

Generators can also be called directly. Randomness is passed explicitly; numpy's global state is
never used:

```python
from tdadyn.core import rng_for
from tdadyn.synthetic import make_circle

circle = make_circle(n_points=400, radius=1.0, noise_sigma=0.03, coeff_field=2,
                     rng=rng_for(20260201, "circle"))
circle.ground_truth.expected_betti   # {0: 1, 1: 1, 2: 0}
```

### Check whether a pipeline recovers the topology

```python
from tdadyn.synthetic.persistence import pairwise_distances, significant_betti

check = config.topology_check
significant_betti(pairwise_distances(circle.points), max_homology_dim=2, coeff_field=2,
                  criterion=check.criterion,
                  gap_noise_floor_ratio=check.gap_noise_floor_ratio,
                  gap_min_ratio=check.gap_min_ratio)
# {0: 1, 1: 1, 2: 0}  — matches the truth
```

### A dynamic graph: from snapshots to events

```python
from tdadyn.synthetic import build_dynamic_graph
from tdadyn.zigzag import analyse

lifecycle = build_dynamic_graph("cycle_lifecycle", config)   # 60 snapshots on 24 vertices
for event in lifecycle.ground_truth.events:
    print(event.snapshot, event.dim, event.type)
# 12 1 birth   ← the path closes into a cycle
# 27 1 birth   ← a chord splits the cycle in two
# 41 1 death   ← an edge of the arc cut off by the chord is removed
# 48 0 birth   ← a component detaches from the dangling path
# 55 0 death   ← the components merge back

report = analyse(lifecycle, config=config, truth=lifecycle.ground_truth.events)
report.metrics.recall, report.metrics.precision   # (1.0, 1.0)
```

Each event also carries a `narrative` string with the description above; in the generated
datasets it is recorded in Russian.

`analyse` builds the zigzag diagram, computes the barcode, translates it into events and matches
them against the truth. The same step by step:

```python
from tdadyn.zigzag import zigzag_complexes, barcode_from

complexes = zigzag_complexes(lifecycle, max_homology_dim=1, max_nodes=30)  # 119 complexes
barcode_from(list(complexes.values()), backend="dionysus", max_homology_dim=1)
# {0: ((0, 118), (96, 108)), 1: ((23, 81), (53, 118))}
```

Bar ends are positions in the diagram: position $p$ corresponds to index $p/2$, and odd positions
are unions of neighbouring snapshots. The bar `(23, 81)` is $[11\tfrac12,\,40\tfrac12]$: a cycle
born at snapshot 12 that dies at snapshot 41.

### Reference check

```python
import numpy as np
from tdadyn.reference import flag_complex, betti_numbers

octahedron = 1 - np.kron(np.eye(3, dtype=int), np.ones((2, 2), dtype=int))
betti_numbers(flag_complex(octahedron, max_homology_dim=2, max_nodes=30), 2)
# {0: 1, 1: 0, 2: 1}  — a cavity without 3-simplices
```

## Package layout

| Module | Purpose |
|---|---|
| `tdadyn.core` | data contracts (`GroundTruth`, `Event`, `*Dataset`), config parsing and validation, derived random streams `rng_for`, the edge rule `adjacency_from_snapshot`, reading and writing datasets, the manifest |
| `tdadyn.synthetic` | generators, building datasets from the config, noise, cloud → network, sliding window, persistence and the significance criterion |
| `tdadyn.zigzag` | zigzag diagram, operation stream, Dionysus backend, barcode → events, metrics, null models |
| `tdadyn.reference` | the reference over $\mathbb{Z}/2$: boundary matrices and ranks, flag complex, zigzag oracle |
| `tdadyn.results` | the command that collects barcodes and metrics over all datasets into one artefact |

Dependencies between layers go one way. `core` and `reference` import nothing from the package.
`synthetic` and `zigzag` rely on them — the generators compute the Betti curve with the reference,
the zigzag is checked against the oracle — but do not know about each other: the generators that
produce the truth do not see the algorithm that checks it, and the algorithm reads data only from
disk. The rule is enforced by `tests/test_layering.py`. The only module that needs both the
generators and the algorithm is `tdadyn.results`, and it sits above the layers.

## Datasets

| Family | Generators | Truth | What it tests |
|---|---|---|---|
| Static clouds | `circle`, `two_circles`, `figure_eight`, `torus`, `sphere`, `klein_bottle`, `hopf_link` | Betti numbers | recovering the topology of a manifold, $H_0$–$H_2$, dependence on the coefficient field |
| Dynamic graphs | `cycle_lifecycle`, `geometric_loop` | Betti curve over snapshots and an event log | detecting and localising births and deaths of classes in time |
| SBM and oscillators | `dynamic_sbm`, `kuramoto`, `roessler` | community partition and coupling schedule | the chain "time series → network → persistence" |
| Noise and null models | `point_jitter`, `weight_jitter`, `spurious_edges`, `missing_edges`; Erdős–Rényi, uniform cloud | noise level; absence of events | degradation under noise and the false positive rate |

Several datasets are built to catch specific errors:

- the **Klein bottle** gives $(1,2,1)$ over $\mathbb{Z}/2$ and $(1,1,0)$ over $\mathbb{Z}/3$ — if the
  answers coincide, the coefficient field does not reach the computation;
- the **Hopf link** has $\beta_0 = 2$ — components passing right next to each other must not stick
  together;
- in **`cycle_lifecycle`** the death of a cycle at snapshot 41 is paired with the birth at 12, not
  at 27, although intuition suggests the opposite; the Betti curve does not see this, the barcode
  does;
- in **`geometric_loop`** the graph changes at snapshots 4, 43 and 56 while the topology does not:
  there must be no events there.

## Command line

```bash
.venv/bin/python -m tdadyn.synthetic.build --config config.yaml
```

Builds 38 datasets (about 13 MB) in `data/synthetic/` in a few seconds and writes
`manifest.json`.

| Flag | Effect |
|---|---|
| `--family NAME` | build only one family; can be repeated |
| `--output-root DIR` | write to another directory |
| `--results` | also compute the stability curve of diagrams under noise into `results/stability.json` (about nine minutes) |

```bash
.venv/bin/python -m tdadyn.results --config config.yaml --results
```

Runs the zigzag over every dynamic dataset and writes `data/synthetic/results/zigzag_events.json`:
barcodes, metrics, counts of censored, invisible and ephemeral bars, the false positive rate on the
null model and the oracle–Dionysus comparison. Needs the `zigzag` extra; takes about five minutes,
almost all of it spent in the oracle. Without `--results` it prints tables and writes nothing.

The same actions are available through `make`:

| Target | Effect |
|---|---|
| `make fast` | fast tests, about 20 seconds, without the zigzag backend |
| `make slow` | topological checks, about 11 minutes |
| `make zigzag` | tests of the zigzag layer, about 6 minutes, needs `[zigzag]` |
| `make results` | the zigzag analysis artefact |
| `make report-zigzag` | build, execute and export to HTML the report notebook |

## Data format

```
data/synthetic/
├── static_cloud/<dataset_id>/   points.npy, ground_truth.json
├── dynamic_graph/<dataset_id>/  snapshots.npy, ground_truth.json
├── series/<dataset_id>/         series.npy, coupling.npy, ground_truth.json
├── noisy/ …                     only the perturbed array
├── null/ …
├── results/                     stability.json, zigzag_events.json
└── manifest.json
```

| Object | Shape |
|---|---|
| point cloud | `(n_points, dim)` |
| weight or distance matrix | `(n_nodes, n_nodes)`, symmetric |
| dynamic graph | `(T, n_nodes, n_nodes)` |
| time series | `(n_channels, n_timepoints)` |

The directory name is a deterministic function of the generator, all its parameters and the seed:
`cycle_lifecycle_N24_T60_w1_thr0.5_sched12-27-41-48-55_seed20260201`.

```python
from tdadyn.core import load_dataset

dataset = load_dataset("data/synthetic/dynamic_graph/cycle_lifecycle_N24_T60_w1_thr0.5_sched12-27-41-48-55_seed20260201")
dataset.snapshots.shape          # (60, 24, 24)
dataset.ground_truth.edge_rule   # 'weight_above'
```

`ground_truth.json` holds the kind of truth (`truth_kind`), the generator call parameters, the seed
and the fields mandatory for that kind: Betti numbers, a Betti curve and events, community labels or
a noise level. Dynamic graphs also record the **edge rule**: for similarity networks an edge exists
when $w_{ij} > t$, for metric datasets when $d_{ij} \le \varepsilon$. Applied the wrong way round, it
silently yields the complement of the graph, so the code implements it exactly once —
`core.adjacency_from_snapshot`. The full schema is in [`docs/data.md`](docs/data.md).

## How the zigzag works

- **Complexes.** Each snapshot becomes a graph by the edge rule, the graph becomes a flag complex
  truncated at dimension $k_{\max}+1$.
- **Diagram.** Between neighbouring snapshots stands their **simplexwise union**
  $K_i \cup K_{i+1}$, not the flag complex of the union of graphs: an intermediate object contains no
  simplex that is absent from every snapshot. Indices are $0, \tfrac12, 1, \dots, T-1$.
- **Barcode.** A module over a quiver of type $A_N$ decomposes into intervals uniquely (Gabriel's
  theorem), so the barcode does not depend on the algorithm and can be checked by an independent
  computation.
- **Events.** A birth is placed at the first snapshot where the class is already visible; a death at
  the first snapshot where it is gone. A bar that runs from the start or to the end of the observation
  is marked as censored and is excluded from the metrics. A bar that covers no snapshot at all is
  counted separately as invisible.
- **Metrics.** Matching within a "dimension, type" pair with a time tolerance (0 by default); recall,
  precision and localisation error, separately by dimension.

The event log, like the Betti curve, is **blind to pairing**: two different arrangements of bars can
give the same log. So recall and precision do not check which birth is paired with which death — the
bar-for-bar barcode comparison does.

## Independent verification

`tdadyn.reference` computes the same things by a different method:

- Betti numbers — from ranks of boundary matrices over $\mathbb{Z}/2$;
- the zigzag barcode — from the generalised rank $r(b,d)$ (the rank of the map from the limit to the
  colimit) and Möbius inversion, without implementing a zigzag algorithm.

The reference is slow and limited to 30 vertices — which is why the dynamic datasets are designed on
24. On all 18 dynamic datasets, noisy ones included, the barcodes of the oracle and of Dionysus agree
bar for bar: 3192 bars, zero discrepancies.

## Reproducibility

- Every function takes `rng: np.random.Generator`; streams are derived from one base seed by name
  (`rng_for(seed, "torus")`), so adding a generator does not shift the others.
- `manifest.json` stores the sha256 of every array and the sha256 of a canonical serialisation of the
  truth. A rebuild in the same environment gives the same hashes; the zigzag artefact gives the same
  `content_sha256`.
- Caveat: the Rössler system is chaotic, and a change of scipy version, integration method or
  tolerances changes its series entirely. Reproducibility is promised within one environment.

## Limitations

- **Synthetic data do not replace real data.** The oscillator series resemble fMRI in their coupling
  structure, not in spectrum or noise level.
- **The significance criterion parameters** were calibrated on point clouds and do not transfer to
  other data types (for example, connectomes).
- **Bar significance in the zigzag is not defined.** Without it the precision on noise drops to
  fractions of a percent, and on the Erdős–Rényi null model an event is found on almost every snapshot
  (0.98, 95% interval [0.979, 0.987]). This is a measurement, not a defect: a random graph at that
  density really does keep creating and losing cycles.
- **The reference** is limited to small complexes; on large datasets it is suitable for spot checks,
  not for regression testing.

## Tests

```bash
make fast    # must pass without optional dependencies; this is what CI runs
make slow
make zigzag
```

Expected Betti numbers and barcodes in the tests are **inputs, not outputs**: they are derived from
the construction of the datasets, not copied from the output of the code. Where the answer rests on a
derivation, the test computes it itself.

## Documentation

| Document | Contents |
|---|---|
| [`docs/method.md`](docs/method.md) | constructions, formulas, reference values and every measurement |
| [`docs/data.md`](docs/data.md) | layout on disk, API contracts, truth schema, manifest |
| [`docs/validation.md`](docs/validation.md) | correctness criteria and mandatory tests |
| [`docs/zigzag.md`](docs/zigzag.md) | zigzag: construction, invariant, translation into events, metrics |
| [`notebooks/zigzag_report.html`](notebooks/zigzag_report.html) | illustrated report on the zigzag analysis |
| [`notebooks/report.html`](notebooks/report.html) | report on the generators and their verification |

## License

MIT — see [`LICENSE`](LICENSE).
