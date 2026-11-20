# Order of work by session

One session is one coherent piece of work with a commit at the end. Do not start the next one
until the previous one has passed its tests. Each session begins with a plan, without code
changes.

## S1. Skeleton and contracts

- Package structure `src/tdadyn/synthetic/`, `pyproject.toml`, `.venv`, dependencies.
- A `config.yaml` loader with validation (mandatory keys, types, ranges).
- The dataclasses of `docs/data.md`: `GroundTruth`, `Event`, `*Dataset`.
- Serialisation: writing/reading a dataset, `manifest.json`, a deterministic directory name.
- Derived random streams: from the base seed, an independent `np.random.Generator` per
  generator and per repeat.
- Tests: determinism, round trip, schema, independence from the global seed.

**Done when:** there are no generators yet, but saving a dummy dataset is reproducible and
passes schema validation.

## S2. Reference implementation of Betti numbers (reference/)

Done BEFORE the generators: without it there is nothing to check them with.

Split into two layers — this removes the question of where to cut off the dimension:

- **Layer 1, the homology engine:** takes an explicit list of simplices; boundary matrices over
  Z/2, ranks, $\beta_k = \dim\ker\partial_k - \operatorname{rank}\partial_{k+1}$.
  **$\mathbb{Z}/2$ only**, no generalisation to arbitrary $p$: row addition stays XOR, no
  division is needed. A call with $p \ne 2$ is an explicit error, not a silent computation over
  $\mathbb{Z}/2$. State the limitation in the docstring.
- **Layer 2, the flag complex builder:** takes a graph, enumerates **all** cliques of every size
  (not only maximal ones — `networkx.find_cliques` does not fit), builds up to dimension
  $k_{\max}+1$ (at `max_homology_dim: 2`, up to 3). No cut-off parameter.
- A size limit (`reference_max_nodes`), an explicit error when exceeded.
- Layer 1 tests: a point, two points, a segment, a triangle as a 1-skeleton ($\beta_1=1$), a
  filled triangle ($\beta_1=0$), a hollow tetrahedron as an explicit 2-skeleton ($\beta_2=1$).
- Layer 2 tests: $K_3 \to (1,0,0)$, $K_4 \to (1,0,0)$ (catches a cut-off at dimension 2), the
  octahedron $K_{2,2,2} \to (1,0,1)$ (an honest cavity without 3-simplices), the face count of
  $K_4$ by dimension = (4, 6, 4, 1) (catches enumeration of maximal cliques only).

**Done when:** every hand-worked example agrees. This is the foundation of trust in everything
else.

## S3. Static clouds, part 1 (planar)

- Circle, two circles, figure eight.
- Verification harness: cloud → distances → ripser → count of significant bars.
- Cross-check A vs B (see `docs/validation.md`, section 1.1): a subsample of
  `cross_check_n_points` points, scales are midpoints between neighbouring distinct distances,
  $\beta_k(\varepsilon)$ from the barcode as the number of bars with $b \le \varepsilon < d$,
  ripser with `thresh=inf`. Agreement of the implementations is checked, not a match with the
  table.
- Betti tests at $\sigma=0$ and at the working $\sigma$.

## S4. Static clouds, part 2 (surfaces)

- Torus with rejection sampling (geometry fixed: R=2.0, r=0.8, no need to change it — the
  question is solved by the criterion, not by tuning the object), sphere by Muller's method,
  Klein bottle — the tubular embedding in R⁴ with rejection sampling by the area element
  (formulas in `docs/method.md`, 1.1), Hopf link.
- **Calibrating the significance criterion — before the topological tests, not after.** Run the
  table of cases from `docs/validation.md`, section 2, on three seeds; adopt the form of the
  criterion and its parameters; only then fix them in the config and remove the "NOT CALIBRATED"
  mark. The open question on entry is the case $\beta_k = 0$ and telling the sphere in $H_1$
  (expected 0) from the Klein bottle over $\mathbb{Z}/3$ in $H_1$ (expected 1): at 400–600 points
  their gaps of 1.30x and 1.32x are almost indistinguishable; larger samples were not checked.
- Klein bottle sampling uniformity test (KS test), as for the torus.
- Sampling uniformity tests (KS test for the torus, no poles for the sphere).
- Klein bottle test parametrised by the coefficient field: $p=2 \to (1,2,1)$,
  $p=3 \to (1,1,0)$. Rational coefficients are unavailable in ripser/gudhi, so $\mathbb{Q}$ is
  checked via $p=3$.
- Hopf link test: $\beta_0 = 2$.

## S5. Cloud → weighted network

- Gaussian kernel and k-NN. For k-NN the distances are **shortest paths** in the graph
  (geodesics), symmetrisation by union, an explicit connectivity check.
- **Choosing the window in $k$:** run an increasing series ($k$ = 10, 20, 40, 80, 160) on the
  torus and the Klein bottle at the working number of points. From below the window is bounded by
  connectivity, from above by short-circuiting. Guard: $\mathrm{diam}_{geo}/\mathrm{diam}_{euclid}$
  not below `min_geodesic_diameter_ratio`. The phrase "the larger $k$, the better" is wrong — see
  `docs/method.md`, 1.3.
- **Separate calibration of the criterion for geodesics:** the parameters 0.07 and 1.55 were
  obtained on Euclidean clouds and are not inherited by another metric. Rerun the table of
  `docs/validation.md`, section 2; if other values turn out to be needed, store them separately
  rather than overwriting the Euclidean ones.
- The inverse transform to distances, a monotonicity test.
- **Kernel width as a fraction of the median pairwise distance**, per generator, not as an
  absolute number: measured, an absolute 0.5 at a torus diameter of 5.71 collapses 60% of the
  weights and breaks `one_minus` (`docs/method.md`, 1.3).
- **A guard test against a degenerate kernel:** the fraction of weights below $10^{-6}$ does not
  exceed `max_collapsed_weight_fraction`. This is the negative control of the step — not
  "`one_minus` breaks the torus" but "a degenerate kernel breaks everything".
- Both transforms (`sqrt_neg_log` and `one_minus`) are checked as equals; do not forbid
  `one_minus` — it goes to Task 3 as $d = 1-|r|$.
- Test: the topology of the network matches the topology of the source cloud — **check by the
  structure of the diagram (which classes, in which order), not by equality of persistences**: a
  monotone transform preserves the former and distorts the latter.

## S6. Dynamic graphs with events

- A scheduler for the `cycle_lifecycle` scenario from the config, keeping an event log.
- Independent computation of `betti_curve` through `reference/` — from the recorded snapshots,
  not from the scheduler's internal state: otherwise the curve describes the intent, not what was
  written to disk.
- Consistency test: Betti curve jumps ↔ the event list, no more and no less, **with positions
  checked individually** (index convention — `method.md`, 2.1).
- Pairing test: the vector of the 24-cycle over $\mathbb{Z}/2$ does not lie in $Z_1(G_{41})$
  while $c_2$ does — i.e. the death at 41 is paired with the birth at 12 (`method.md`, 2.2a).
- Node relabelling test: `betti_curve` and the events do not change.
- Triangle test: $\beta_1$ of the flag complex ≠ the cyclomatic number.
- Key grouping of the schedule in `register_name_order` (`data.md`, 1.1).

## S7. Dynamic SBM and oscillators

- SBM with a schedule of community merges/splits.
- Kuramoto and Rössler through `solve_ivp` with a fixed `t_eval`.
- The geometric variant of family 2 (`geometric_loop`), carried over from S6. The specification
  is completed in `method.md`, 2.1a: $\varepsilon = 0.8$ as the middle of the window on a
  logarithmic scale, piecewise-linear interpolation of `radius_schedule`, $n = 24$ under
  `reference_max_nodes`, both regime thresholds in the config cross-checks. The closed state of
  the arc is $\delta = 2\pi/n$, not zero.
- Before the generators: the kind of truth (`truth_kind`) and the edge rule (`edge_rule`,
  `edge_threshold`) as separate fields, `format_version` 2 (`data.md`, 4.1 and 4.2). Without the
  first the SBM cannot be built, without the second the metric dataset is read backwards.
- Sliding window → dynamic graph.
- Tests: zero coupling → zero correlations; recovered coupling correlates with the true schedule;
  a check of the relation between window length and regime duration.

## S8. Noise and null models

- Weight jitter, spurious and missing edges, point jitter — at the levels from the config.
- ER and random geometric graphs with density matching.
- Tests: monotonicity of the bottleneck distance in the noise level, accuracy of the fractions
  $p^\pm$, no events on the null models.

## S9. CLI and building every dataset

- The command `python -m tdadyn.synthetic.build --config config.yaml [--family ...]`, plus
  `--output-root` and `--results`. Builds 38 datasets (13 MB) in seconds.
- The "config → generator" layer moved from `tests/conftest.py` to `builders.py`: before S9
  nothing in the package could build a dataset, and the CLI needs exactly that.
- Family 4 layout: **only the perturbed array** (`data.md`, §1). One rule for the family and in
  both directions — otherwise a directory ends up with a pair whose derived array does not match
  the perturbed source.
- A full rebuild, `manifest.json` with sha256 of the arrays **and sha256 of the truth**: without
  the latter the check would not cover the expected Betti numbers, the event log and `notes`.
- A README describing the datasets, with an example and a section "what these data do not do".
- Check: a rebuild **within one environment** reproduces the same entries. More cannot be
  promised — Rössler is chaotic (`method.md`, §3.3).
- The stability curve of Result 6 is produced by the build, not by the tests, and is taken on the
  circle, torus and sphere (`method.md`, §4.2a).

## What is NOT part of these sessions

- Zigzag algorithms and temporal diagrams (Task 2) — a separate module.
- Any work with real fMRI data.
- Performance optimisation beyond what is necessary: the priority is correctness and readability,
  speed-ups only where the tests do not fit in a reasonable time.


---

## Task 2. Zigzag persistence and temporal diagrams

Subpackage `src/tdadyn/zigzag/`. Specification — `docs/zigzag.md`. The source of truth is
family 2 (`dynamic_graph/`); Task 2 is a consumer of `method.md`, not a co-author.

| Session | Contents | Closing criterion |
|---|---|---|
| Z0 | Tag; moving the shared layer to `tdadyn.core` and `tdadyn.reference`; documents; `[zigzag]` as optional dependencies; markers and three local targets | 484 tests green after the move and before any new code; `pip install -e .` and the `fast` target without `[zigzag]` |
| Z1 | Snapshots → complexes → zigzag diagram → stream of elementary operations; the index layer | Replaying the stream gives the same $K_i$ and $K_{i+1/2}$; every prefix is a complex |
| Z2 | Oracle in `reference/`: L1 Betti, L2 arrow ranks, L3 generalised rank + Möbius; toys T1–T4 | Hand-worked barcodes reproduced; the 12↔41 pair computed; the T4 pair distinguished |
| Z3 | Dionysus 2 adapter; index translation | Two-sided agreement oracle ↔ Dionysus, bar for bar, on `cycle_lifecycle` and `geometric_loop` |
| Z4 | Barcode → event log; recall/precision/localisation metrics; negative controls | **Result 2**: metrics measured, the 12↔41 pair reproduced by the backend |
| Z5 | Null models (per-snapshot FPR with an interval) and degradation under noise | Tables with the number of trials; not a single "zero" without an interval |
| Z6 | CLI `--results`, artefact, report, tag | **Closed.** The artefact is produced by the build, not by pytest; the report reads the artefact and is executed by a command |

**The order is inviolable in one place:** the oracle (Z2) before the backend (Z3). Otherwise the
backend becomes its own reference, and the first index-translation error gets fixed as truth.

**What is left outside Task 2, and why.**

**Vineyards** track the movement of diagram points under a **continuous** change of the
filtration: points crawl, cross and swap, and the whole apparatus exists to describe this
motion. The zigzag of family 2 is discrete by construction — the index set $I$ consists of
integers and half-integers, and nothing happens between neighbouring indices that could be
tracked continuously. This is not postponed work but **a different object**, needed where the
filtration is continuous: in Task 3 (the correlation threshold sweeps an interval) and in the
bifiltration of Result 3. It moves there, not "to later".

**Bar significance in the zigzag**: `topology_check` is not inherited, and no replacement is
introduced in this branch. The Z5 measurement shows that the question is not academic — without a
criterion the precision on noise is 0.17%, and on the null model 98% of snapshots fire — but it is
not solved by tuning a threshold to precision.

**Multiparameter persistence** (Result 3) is a separate branch that shares with the zigzag only
the section on the invariant.

**Open item: the budget.** The slow run grew during Task 2 from 10:10 to 10:48 against a guideline
of 15 minutes. Z6 adds nothing to it, but four minutes of headroom are left, and the next session
will exceed it: the cure will be to split targets, not to weaken the checks.

---

## Config keys added by Task 2

The validation schema is extended with the following keys. The S6 boundary holds: **what goes
into the config is what can be changed without rewriting the proof.**

```yaml
zigzag:
  backend: reference          # reference | dionysus | fzz — an implementation, not the truth
  max_homology_dim: 1         # the complex is built up to max_homology_dim + 1. Lower than
                              # topology_check's, deliberately, for cost; higher is
                              # forbidden by a cross-check: above the generators' ceiling
                              # there is no truth, and the bars could not be refuted
  coeff_field: 2              # p != 2 is an explicit error, as in reference/. Named as in
                              # topology_check: two spellings of one concept in one file
                              # invite reading the wrong key
  event_matching:
    tolerance: 0              # tolerance in snapshot index when matching logs
    match_dimension: true
    match_type: true
  oracle:
    level: L3                 # L1 | L2 | L3 — depth of the independent check
    max_index_count: 200      # above this, L3 only as spot checks
  report:
    invisible_bars: true
    ephemeral_bars: true
    half_index_events: true
    censored_bars: true
```

**What is not in the config, and why.**

- `union_rule` — the invariant depends on it (`docs/zigzag.md` §2.2). A module constant
  `UNION_RULE`; the second construction exists only inside the test that measures it.
- `index_convention` — the translation of a bar into an event depends on it. A constant; a key
  would invite shifting it by one and getting a systematic localisation error indistinguishable
  from an error of the algorithm.
- `significance` in any form — a significance criterion is not defined for the zigzag, and a key
  would give the impression that it is.
- `edge_rule` — a property of the dataset, read from the truth, not configured by the consumer.
