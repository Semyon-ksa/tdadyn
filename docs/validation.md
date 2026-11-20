# Correctness criteria

The main risk in this code is not a crash but a plausible-looking dataset with wrong truth.
Such an error does not show up here: it shows up in Task 2, where the algorithm will be
validated against a corrupted reference, and at best it is discovered in peer review.

## 1. The principle of independent verification

Expected Betti numbers in the tests are taken **from the table in `docs/method.md`**, not from
the output of the code. Verification runs along two independent paths:

- **path A (main pipeline):** ripser/gudhi on the distance matrix;
- **path B (reference):** an independent implementation in `src/tdadyn/reference/` that
  computes Betti numbers directly from ranks of boundary matrices over Z/2 (numpy), without
  any persistence library.

Path B is deliberately slow and applies only to small complexes (N ≤ 30 nodes) — that is why
the dynamic scenarios are designed on few nodes. A test requires agreement of A, B and the
table of expectations. Agreement of two implementations out of three sources is what tells a
check apart from self-confirmation.

### 1.1. How path B applies to family 1

Family 1 consists of clouds of hundreds of points, while path B is limited by
`reference_max_nodes`. A direct comparison is impossible, so verification splits into two
**different** tests that must not be mixed.

**Test 1 — recovering the topology of the manifold.** The full cloud (400 points), path A,
comparison with the table of expectations in `docs/method.md` by the number of significant
bars. Path B does not take part in principle: on a small sample the gaps between neighbouring
points are random, and a scale giving the table answer may not exist at all. Requiring the
table from a small cloud means forcing yourself to tune the scale until it fits — i.e. fitting
the truth.

**Test 2 — agreement of the two implementations.** A subsample of the same cloud of size
`cross_check_n_points` (around 18–20 points), several fixed scales, both paths compute the Betti
numbers **of one and the same complex**. The expected value here is not the table but mutual
agreement: whatever the implementations produce, they must produce the same thing. The table
answer will not appear at most scales, and that is not an error.

Mechanics of the comparison (otherwise incomparable things get compared):

- ripser returns **persistent** homology over all scales, the reference returns the Betti
  numbers of **one** complex at a fixed threshold. To bring them to a common form, the barcode
  gives $\beta_k(\varepsilon) = \#\{$bars with $b \le \varepsilon < d\}$. The interval is
  half-open: at $\varepsilon = d$ the bar is already dead.
- ripser is called with `thresh=inf`, otherwise bars are clipped by the threshold and the count
  diverges at large $\varepsilon$.
- Scales are taken as **midpoints between neighbouring distinct pairwise distances**, not the
  distances themselves: at the value of a distance the simplex enters the complex, and the
  comparison comes down to equality of floating-point numbers. Midpoints are free of this.
- The subsample size is bounded above not only by `reference_max_nodes` but also by the cost of
  $\partial_3$: up to $\binom{n}{4}$ 3-simplices.

**Limitation of path B: $\mathbb{Z}/2$ only.** The reference is deliberately not generalised to
arbitrary $p$ (see `docs/method.md`, section 1.2). So the double check holds for the project's
working field $\mathbb{Z}/2$ and not at $p = 3$: only path A is available there. Tests at
$p = 3$ are stated as consistency checks, not as confirmation of Betti numbers, and this
distinction must be visible in the test name and its docstring.

## 2. Definition of a significant bar

Persistent Betti numbers at $\sigma > 0$ are not exact, so a criterion is needed to separate
real features from noise bars.

**Status: the criterion is calibrated and adopted.** The earlier rule — persistence at least
`significance_ratio` × $\mathrm{diam}(X)$ — was found unfit by measurement: it gives false
negatives on anisotropic objects (torus, Klein bottle: $H_2$ is lost) and false positives on
compact ones (sphere: extra loops in $H_1$), and the signal and noise ranges in fractions of the
diameter overlap, so tuning the constant cannot fix it. Justification with numbers —
`docs/method.md`, section 1.2a.

**Scope.** The parameters were calibrated on family-1 point clouds and hold only for it. They
are not inherited by other data types — above all by the functional connectomes of Task 3:
there the distance distribution differs, and above all there is no known $\beta_k$ against which
the cut could be calibrated. More — `docs/method.md`, section 1.2a.

**The adopted rule — a gap in the spectrum with a step into the dust.** Within each dimension:

1. persistences are sorted in decreasing order, infinite bars are counted separately and are
   always significant;
2. bars below `gap_noise_floor_ratio` × $\mathrm{diam}(X)$ are set aside as dust;
3. the largest ratio of neighbouring values is sought, **including the step from the last
   surviving bar to the first set-aside one** — without this step an object that has only real
   features above the floor (two circles: 1.473 and 1.453) gives zero instead of two;
4. if the largest ratio is below `gap_min_ratio`, there are no features — this is how
   $\beta_k = 0$ arises, for which the step into the dust was introduced;
5. otherwise $\beta_k$ equals the position of that gap.

**Calibration result.** The rule is correct on all 54 combinations of the table below
(9 cases × 3 seeds × 2 noise levels). The parameters are taken from the **centre** of the
zero-miss region, not from its edge: zero holds for floor from 0.05 to 0.10 and gap from 1.45 to
1.62; 0.07 and 1.55 were adopted. The old rule gives 9 misses on the same table.

| Case | Expectation | What makes it hard |
|---|---|---|
| Circle, two circles, figure eight | (1,1,0), (2,2,0), (1,2,0) | two circles catch the rule without the step into the dust |
| Torus $R{=}2.0$, $r{=}0.8$ | (1,2,1) | $H_2$ at scale $r$; not separated at 400 points |
| Sphere | (1,0,1) | $\beta_1 = 0$: a check for false positives |
| Klein bottle, $p{=}2$ | (1,2,1) | weak $H_2$, like the torus |
| Klein bottle, $p{=}3$ | (1,1,0) | at 600 points indistinguishable from sphere noise; separated at 900 |
| Hopf link | (2,2,0) | components must not stick together |
| Null model (uniform cube) | (1,0,0) | control of false positives |

The criterion is accepted only if it is correct on **every** row. The rule is read off the
position of the largest gap, not off a number tuned to the dataset.

**Note on the null model.** The null-model row is checked on a minimal substitute — uniform
points in a cube — inside the calibration harness, because family 4 appears only in S8. The
expectation is written as (1,0,0), not "zeros in every dimension": the infinite bar in $H_0$ is
always significant, so "no features" means one component and nothing beyond it. Proper ER and
random geometric graphs go through the same table in S8.

The infinite bar in $H_0$ is always significant and is counted in $\beta_0$.

Test: the number of significant bars in dimension $k$ equals `expected_betti[k]`.

**Forbidden:** tuning the criterion parameters per dataset to make a test pass. The parameters
are common to the whole family. If a test fails with common parameters, the cause is one of:
too few points, too much noise, non-uniform sampling, an error in the parametrisation — or the
criterion itself is unfit. The first four are fixed in the generator; the last only by revising
this section with a justification in `method.md` and rechecking the whole table above, but never
by tuning to one object.

## 3. Mandatory tests by family

### Common (for every generator)

| Test | What it checks |
|---|---|
| Determinism | Two calls with an identically seeded `rng` give bitwise equal arrays |
| Independence from the global seed | `np.random.seed(...)` outside does not change the result |
| Shapes and dtype | Conformance to the axis table of `docs/data.md` |
| Symmetry | Weight matrices are symmetric, the diagonal is correct |
| Schema validity | `ground_truth.json` passes schema validation, `format_version` is current |
| Round trip | Saving and loading return an identical object |

### Family 1 — static clouds

- **Test 1:** the Betti numbers of the full cloud at $\sigma = 0$ and sufficient $N$ match the
  table (path A only, see section 1.1). The number of points is set per generator: 400 for
  planar objects, **at least 600 for the torus and the Klein bottle** — measured: at 400 the gap
  in $H_1$ of the torus lands on the wrong position, at 600 and above on the right one.
- The Betti numbers stay the same at the small $\sigma$ from the config.
- **Test 2:** on a subsample of `cross_check_n_points` points paths A and B give the same
  $\beta_0, \beta_1, \beta_2$ at all scales from `cross_check_n_scales`. The expectation is
  agreement, not the table; a mismatch with the table at an individual scale is not an error.
- **Klein bottle sampling uniformity test:** the distribution of $v$ agrees with the density
  $\propto \sqrt{(a + b\cos v)^2 + b^2\sin^2 v/4}$ (Kolmogorov–Smirnov test). The same error as
  on the torus, and just as invisible.
- **Torus sampling uniformity test:** the distribution of $\varphi$ agrees with the density
  $\propto (R + r\cos\varphi)$ (Kolmogorov–Smirnov test). A separate test, because this is
  exactly where a naive implementation looks plausible and silently breaks $\beta_2$.
- **Sphere uniformity test:** no clustering at the poles.
- **Klein bottle:** the test is parametrised by field and run at $p = 2$ and $p = 3$.
  Expectations: $\mathbb{Z}/2 \to (1,2,1)$, $\mathbb{Z}/3 \to (1,1,0)$. The status of the checks
  differs: at $p=2$ both paths (A and B) work, at $p=3$ only path A. If both answers coincide,
  the field parameter does not reach the computation — a typical bug in which `coeff_field` is
  read from the config but not passed to ripser. Rational coefficients are unavailable in ripser
  and gudhi, so $\mathbb{Q}$ is checked via $p = 3$ (see `docs/method.md`, section 1.2).
- **Hopf link:** $\beta_0 = 2$. A test that the components did not stick together.
- Monotonicity of the similarity → distance transform is checked explicitly.

### Family 2 — dynamic graphs

- The jumps of `betti_curve` correspond exactly to the `events` list (matching position,
  dimension and direction). **Multisets** of "position, direction" pairs are compared, not
  counts: otherwise a shift of all events by one goes unnoticed. Index convention —
  `docs/method.md`, 2.1.
- `betti_curve` is computed by reference implementation B, not by the main pipeline, and **from
  the recorded snapshots**, not from the scheduler's internal state.
- The `type` field takes only `birth`/`death` — a schema test.
- $\beta_1$ of the flag complex is **not** equal to the cyclomatic number where there are
  triangles: a separate test on the triangle graph ($\beta_1 = 0$, cyclomatic number 1).
- No more and no fewer events than scheduled: no unscheduled changes of β.
- **Pairing of birth and death** (`docs/method.md`, 2.2a). The canonical remainder of the claim
  is checked: the vector of the 24-cycle over $\mathbb{Z}/2$ does not lie in $Z_1(G_{41})$, $c_2$
  does, and $\dim Z_1(G_{41}) = 1$. There must be no test on the identity of a class — the basis
  of $H_1$ of the theta graph is not canonical.

  This test is mandatory precisely because `betti_curve` does not see pairing: measured,
  inverting the pair in the log breaks no other family-2 check. Without it the error would
  survive into Task 2 and would look like a defect of the zigzag algorithm.
- **No triangles on any snapshot** of `cycle_lifecycle`. A requirement of the construction
  (`docs/method.md`, 2.1): a triangle would fill a cycle, and the recorded event would not match
  what happened. It also turns the coincidence of $\beta_1$ with the cyclomatic number on this
  dataset into a consequence of the construction rather than an accident.
- **Refusal at an inadmissible $N$:** the generator does not build the dataset if the chord
  leaves an arc shorter than three edges (`docs/method.md`, 2.1). The test is negative, at
  $N = 13$ and $N = 14$: there $\beta_1$ stays 1 while a birth is declared, so the refusal must be
  explicit, not a silent output of a dataset with wrong truth.
- **The edge rule is recorded and checkable:** applying the opposite `edge_rule` to the same
  snapshots gives a different `betti_curve`. Without this test the field could be written wrong
  and nobody inside the package would notice — the generator computes the curve its own way
  (`docs/method.md`, 2.3).
- **The metric scenario `geometric_loop`** (`docs/method.md`, 2.1a):
  - the closed state of the arc coincides with the uniform circle — checked by **the minimum
    pairwise distance**, not by Betti numbers: at $\delta = 0$ points 0 and $n-1$ coincide, and
    $\beta$ does not see it;
  - every snapshot stays between both thresholds of the regime, and the check uses the actual
    radii, not the schedule nodes: the radius is interpolated;
  - radius breathing changes the graph and does **not** change $\beta$ — a negative control on
    snapshots named individually, not on a range: a range check would also pass on a dataset
    where the complex never changed.
- **Invariance under node relabelling:** `betti_curve` and the events do not change. Labels are
  not randomised in the data (snapshots must stay readable by eye), so the check is a test.
- **The repeated state is preserved in the data:** snapshots after the edge is restored are
  byte-for-byte equal to snapshots before the split; there are five distinct graphs over 60
  snapshots. Memoisation inside the generator speeds up the curve computation but has no right
  to collapse the snapshots themselves.

### Family 3 — series and SBM

- With coupling "switched off" ($\varepsilon = 0$, $K = 0$) the channel correlations are
  statistically indistinguishable from zero — a test for the absence of spurious coupling in the
  construction itself.
- With coupling on according to the schedule, the matrix recovered by the sliding window
  correlates with the true $A(t)$ above `sliding_window.min_coupling_agreement`. The statistic is
  the $|r|$ matrix **averaged over the windows of a regime** against the $A$ of that regime: per
  single window the agreement is too noisy to threshold (measured: the minimum over windows ranges
  from $-0.02$ to $0.23$ across seeds). Checked on all three calibration seeds.
- **The null control is a permutation of channel labels**, not a comparison with another
  regime. Another regime is not a null control: the partitions are nested (`method.md`, 3.1a),
  neighbouring regimes share most of their structure, and agreement with another regime reaches
  0.440 against a true minimum of 0.383 — the classes overlap. Label permutation gives
  $-0.039\ldots0.068$, and the gap is clean.
- **The window is bounded on both sides** (`method.md`, 3.4): above by the regime duration,
  below by the oscillation period. Both bounds are config checks, not data checks. The lower one
  was added after a measurement showed that at 0.16 periods per window the within-block (0.946)
  and between-block (0.936) correlations are indistinguishable and coupling is not recovered at
  all.
- **Coupling above the synchronisation threshold:** the effective within-block coupling
  $(K/N)(|{\rm block}|-1)$ exceeds $K_c \approx 1.6\,\sigma_\omega$. A config check.
- **Dynamic SBM: the truth is a partition, not a barcode** (`method.md`, 3.1). Checked:
  `truth_kind: partition`, absence of `betti_curve` and `events`, labels of shape
  `(n_snapshots, n_nodes)` in int64, and that neighbouring partitions are related by
  refinement.
- **Labels are tied to the data:** edge density within a community is higher than between —
  otherwise `community_labels` could be any array of the right shape.
- **A community merge is not a homological event** — a test on a robust statement: after removing
  isolated vertices the graph is connected on **every** snapshot. The formulation "$\beta_0$ does
  not change at the merge" is unfit: over a series of 60 snapshots $\beta_0$ takes both 1 and 2 on
  every seed, because a node occasionally draws no edge at all (3.5 such vertices expected per
  dataset, 2, 3 and 2 measured). The invariant must be checked, not the absence of a random vertex
  on two particular snapshots.
- **The `reference_max_nodes` limit does not apply to SBM** — deliberately: it exists for the
  reference check of the Betti curve, and SBM has no curve. The requirement follows the kind of
  truth, as in the schema.
- The integrator is reproducible: the same initial conditions and `t_eval` → the same series.
  For Rössler this is reproducibility of the computation, not robustness: at $c = 5.7$ the system
  is chaotic, and a change of scipy version, method or tolerances moves the series entirely.

### Family 4 — noise and null models

- Monotonicity: the bottleneck distance between the clean and noisy diagrams does not decrease
  as the noise level grows (on average over repeats).
- **Stability is checked against $2\,d_H$, not $d_H$** (`method.md`, 4.1). The factor follows
  from parametrising Rips by diameter. Measured: without it nine of twelve combinations violate
  the bound with ratios up to 1.33. The check must use $H_1$: in $H_0$ both forms hold, and a
  test on zero-dimensional bars would pass with a wrong theorem in the document.
- At zero noise level the dataset coincides with the source **bitwise**, and the draw still
  happens — otherwise the random stream shifts between levels and the "clean" dataset stops being
  the same object the noisy ones are compared with.
- **Zero-level check through the manifest:** the sha256 of the perturbed array equals the sha256
  of the corresponding base. Both numbers already lie in `manifest.json`, the comparison costs
  nothing, and it catches at once a wrongly connected base, a shift of the random stream between
  levels and an unconditionally applied `clip` — what would otherwise surface as an unexplained
  step at the start of the stability curve. For `weight_jitter` the base is a cloud whose network
  is derived rather than stored, so that pair is skipped; the test requires the number of
  compared pairs to be above zero, otherwise the skip would hollow out the check.
- **Only the perturbed array goes to disk** (`docs/data.md`, §1): `point_jitter` has no network,
  `weight_jitter` has no points. Checked on the files actually written.
- The fractions $p^+$ and $p^-$ match the number of edges actually changed (counted over the
  upper triangle). Checked on **both** family-2 scenarios: their edge rules are opposite, and for
  the metric one "add an edge" means decrease a distance.
- **Null models: an empty event log, not the absence of bars.** The earlier formulation ("no
  significant long-lived bars beyond what is expected for a random graph") demanded the
  impossible: a random graph honestly has cycles at every working density, and $(1,0,0)$ is
  reached only at $p \gtrsim 0.8$, i.e. on an almost complete graph (`method.md`, 4.2). What is
  checked is a statement about the construction: the event list is empty, there is no schedule,
  the dataset is reproducible. The false positive rate is measured by the Task 2 algorithm on
  these data, not claimed by the generator.
- **The stability curve is taken on the circle, torus and sphere**, not on one object
  (`method.md`, 4.2a): the sphere's $H_1$ barely moves, the circle's $H_2$ is flat, and a set of
  one object is informative by half. The artefact is produced by the build (`--results`), not by
  the tests: the tests **check** monotonicity and the $2d_H$ bound, the pipeline **produces** the
  reported numbers of Result 6.
- **Uniform cloud as a null model: $(1,0,0)$ on every realisation**, not on average. It is
  calibrated on `n_realizations` × calibration seeds, not on three seeds: zero misses out of three
  gives, by the rule of three, a 95% upper bound of about 63%, which for a dataset whose whole
  value lies in the absence of false positives is no proof.

## 4. Test organisation

```
tests/
├── test_config.py             # config.yaml loader: keys, types, ranges
├── test_rng.py                # derived random streams
├── test_io_roundtrip.py       # saving/reading a dataset and manifest.json
├── test_determinism.py        # common, parametrised over every generator
├── test_schema.py             # ground truth and manifest
├── test_static_betti.py       # slow: topological checks of family 1
├── test_sampling_uniformity.py
├── test_dynamic_events.py     # slow: consistency of events and betti_curve
├── test_series.py
├── test_noise_models.py
└── reference/
    └── test_reference_impl.py # the reference implementation against known small complexes
```

Slow tests are marked `@pytest.mark.slow` and run separately. The fast run
(`pytest -q -m "not slow"`) must fit into tens of seconds.

The reference implementation in `reference/` is itself checked on complexes whose homology is
known by hand. The check is split by layer.

**Layer 1 — the homology engine** (takes an explicit list of simplices, so the question of
dimension cut-off does not arise): a point, two disconnected points ($\beta_0=2$), a segment, a
triangle as a 1-skeleton ($\beta_1=1$), a filled triangle ($\beta_1=0$), a hollow tetrahedron as
an explicitly given 2-skeleton ($\beta_2=1$).

**Layer 2 — the flag complex builder** (takes a graph, builds simplices up to dimension
$k_{\max}+1$):

| Graph | Expectation | What it catches |
|---|---|---|
| Triangle $K_3$ | $\beta=(1,0,0)$ | $\beta_1$ ≠ cyclomatic number |
| Complete graph $K_4$ | $\beta=(1,0,0)$ | Cut-off at dimension 2: it would give a spurious $\beta_2=1$ |
| Octahedron $K_{2,2,2}$ | $\beta=(1,0,1)$ | A real cavity: the maximal clique is 3, no 3-simplices |

Plus a structural test, not about Betti numbers: **the face count of $K_4$ by dimension must be
(4, 6, 4, 1)**. It catches enumeration of maximal cliques only, which would give (4, 0, 0, 1) — a
complex not closed under faces. The general rule of the test: for any simplex in the complex, all
its subsets must be present in the complex.

The hollow tetrahedron is **not** used as a flag complex test: it is not the flag complex of any
graph (the flag complex of $K_4$ is the solid tetrahedron). Worse, an implementation cutting off
at dimension 2 would give the same $(1,0,1)$ on $K_4$, and such a test would pass with wrong code.
The octahedron plays the role of an honest $H_2$ test.

Without this, the "independent" path B does not deserve trust.

## 5. What the tests do not verify

The tests do not confirm that a dataset is **scientifically meaningful** for the task — only that
it matches its declared truth. Meaningfulness (whether the dynamics is realistic enough, whether
the synthetic series analogue resembles fMRI in spectrum and noise level) remains a matter of
expert judgement and is discussed separately.
