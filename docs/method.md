# Method: constructions, formulas, reference values

## 0. Notation

### Topology

| Symbol | Meaning | Explanation |
|---|---|---|
| $S^n$ | $n$-dimensional sphere | The index is the dimension of the **surface**, not of the ambient space. $S^1$ is a circle in $\mathbb{R}^2$; $S^2$ is a spherical shell in $\mathbb{R}^3$ (without the interior, hence $\beta_2=1$) |
| $T^2$ | Torus | The surface of a doughnut, two-dimensional |
| $H_k$ | $k$-th homology group | $H_k = \ker\partial_k / \operatorname{im}\partial_{k+1}$ |
| $\beta_k$ | $k$-th Betti number | $\beta_k = \dim H_k$ — the number of independent holes of dimension $k$: $\beta_0$ components, $\beta_1$ loops, $\beta_2$ cavities |
| $\partial_k$ | Boundary operator | Maps a $k$-simplex to its boundary. $H_2$ requires $\partial_3$, i.e. 3-simplices |
| $\mathrm{VR}_\varepsilon(X)$ | Vietoris–Rips complex | The flag complex of the proximity graph at scale $\varepsilon$ |
| $X$, $d$, $\mathrm{diam}(X)$ | Point cloud, Euclidean distance, diameter | $\mathrm{diam}(X)=\max_{i,j} d(x_i,x_j)$ |

### Graphs

| Symbol | Meaning |
|---|---|
| $K_n$ | Complete graph on $n$ vertices (everyone connected to everyone). $K_4$ — 4 vertices, 6 edges |
| $K_{n_1,n_2,\dots}$ | Complete multipartite graph: edges between parts, none within a part. $K_{2,2,2}$ = octahedron |
| $G(t)$, $G_i$ | Snapshot of a dynamic graph at time $t$ (index $i$) |
| $m$, $n$, $c$ | Number of edges, vertices, connected components |

### Complexes and filtrations

| Symbol | Meaning |
|---|---|
| $K$, $K_i$ | Simplicial complex (from German *Komplex*). $K_i$ is the complex built from snapshot $G_i$ |
| $K_i \hookrightarrow K_i \cup K_{i+1} \hookleftarrow K_{i+1}$ | Union zigzag: inclusions with alternating direction |
| $w$, $\tau$ | Bifiltration parameters: coupling strength and temporal stability |

### Warning: the letter K is taken three times

In the project it means the **complete graph** ($K_4$), a **simplicial complex** ($K_i$) and the
**coupling constant of the Kuramoto model** (`K` in `config.yaml`). In formulas context separates
these meanings; in code it does not. So the letter K is not used in identifiers:

| Meaning | What to call it in code |
|---|---|
| Complete graph | `complete_graph(n)` |
| Complete multipartite | `complete_multipartite_graph([2, 2, 2])` |
| Simplicial complex of a snapshot | `snapshot_complex`, `complex_at(t)` |
| Kuramoto coupling constant | `coupling_strength` |
| Homology dimension | `hom_dim`, `k` is allowed only as a loop index |

---

## 1. Family 1: static clouds with known topology

The scheme common to all generators:

1. choose $N$ points on a manifold $M$ (as uniformly by area/length as possible);
2. add Gaussian noise $\mathcal{N}(0,\sigma^2 I)$ in the ambient space;
3. pairwise distance matrix → Vietoris–Rips filtration → persistence diagram;
4. compare the number of significant bars in each dimension with the reference table.

### 1.1. Parametrisations

**Circle $S^1$** $\subset\mathbb{R}^2$: $\theta\sim U(0,2\pi)$, $(r\cos\theta, r\sin\theta)$.
Uniform in $\theta$ = uniform in arc length.

**Two disjoint circles:** two copies of $S^1$ of radius $r$ with centres at distance $s > 2r$.
The parameter $s$ sets the "lifetime" of the second component in $H_0$.

**Figure eight (wedge of two circles):** the lemniscate, $x = \cos t$, $y = \sin t\cos t$,
$t\sim U(0,2\pi)$; or `tadasets.infty_sign`. Note: uniformity in $t$ does not give uniformity in
arc length — the density drops near the self-intersection point. Acceptable, but must be noted in
the metadata.

**Torus $T^2$** $\subset\mathbb{R}^3$:
$$((R + r\cos\varphi)\cos\theta,\ (R + r\cos\varphi)\sin\theta,\ r\sin\varphi),\quad R > r.$$
Naive sampling $\theta,\varphi\sim U(0,2\pi)$ is **not** uniform by area: the area element is
proportional to $(R + r\cos\varphi)$, so the inner rim is oversampled. Rejection sampling is
needed: accept $\varphi$ with probability $(R + r\cos\varphi)/(R + r)$. Non-uniformity distorts
the filtration and can eat the $\beta_2$ bar.

**Sphere $S^2$:** Muller's method — take $g\sim\mathcal{N}(0,I_3)$ and normalise $g/\|g\|$. Naive
sampling of uniform spherical angles clusters points at the poles.

**Klein bottle.** An embedding in $\mathbb{R}^4$ (in $\mathbb{R}^3$ there is only an immersion with
self-intersection, which spoils the ground truth). The "tubular" parametrisation is adopted,
$u, v \in [0, 2\pi)$, parameters $a > b > 0$:
$$x = (a + b\cos v)\cos u,\quad y = (a + b\cos v)\sin u,$$
$$z = b\sin v\,\cos\tfrac{u}{2},\quad w = b\sin v\,\sin\tfrac{u}{2}.$$

The map factors correctly through the identification $(u,v) \sim (u + 2\pi,\,-v)$, i.e. it is
indeed a Klein bottle and not a torus: checked numerically (discrepancy of order $10^{-15}$), no
self-intersections. The condition $a > b$ is mandatory, otherwise the factor $(a + b\cos v)$
vanishes and the surface degenerates.

**Sampling — rejection, as for the torus.** Here $|\partial_v| = b$ is constant and
$\partial_u \perp \partial_v$, so the area element equals
$$b\,\sqrt{(a + b\cos v)^2 + \tfrac{b^2\sin^2 v}{4}}.$$
At $a=1.8$, $b=1.0$ it varies in $v$ from 0.8 to 2.8 — a factor of 3.5, so naive sampling
$v \sim U(0,2\pi)$ is not uniform by area. Measured: rejection increases the gap in $H_2$ from
4.19x to 5.31x (600 points).

**Hopf link:** circle 1 in the $xy$ plane centred at the origin, circle 2 in the $xz$ plane centred
at $(r,0,0)$ — they are linked but do not intersect.

### 1.2. Reference Betti numbers (Z/2 coefficients)

| Object | β₀ | β₁ | β₂ | Note |
|---|---|---|---|---|
| Circle $S^1$ | 1 | 1 | 0 | basic test |
| Two disjoint circles | 2 | 2 | 0 | test of $H_0$ |
| Figure eight (wedge of two circles) | 1 | 2 | 0 | density near the self-intersection |
| Torus $T^2$ | 1 | 2 | 1 | needs rejection sampling |
| Sphere $S^2$ | 1 | 0 | 1 | test of $H_2$ without $H_1$ |
| Hopf link | **2** | 2 | 0 | see the warning below |
| Klein bottle ($\mathbb{Z}/2$) | 1 | 2 | 1 | **only** over $\mathbb{Z}/2$ |
| Klein bottle ($\mathbb{Q}$ or $\mathbb{Z}/3$) | 1 | 1 | 0 | in characteristic $\ne 2$ torsion is invisible |

**Warning about the Hopf link.** Two linked circles have exactly the same homology as two unlinked
ones: $\beta = (2,2,0)$. Linking is a property of the embedding, not of the homotopy type, and
persistent homology of the space does not distinguish it. So the dataset is used as a **geometric
stress test** (the pipeline must not prematurely glue together two components passing close to
each other), not as a source of a separate topological signature. Expecting $\beta_0 = 1$ here is
an error.

**Warning about the coefficient field.** The Klein bottle over $\mathbb{Z}/2$ gives $(1,2,1)$, over
$\mathbb{Q}$ $(1,1,0)$. The reason is torsion in $H_1$: the Klein bottle has a loop that is not
itself a boundary but becomes one when traversed twice. Over $\mathbb{Z}/2$ the coefficient 2 is
zero, and this loop is visible as a separate class; in a field of characteristic other than 2 it is
not detected. Neither answer is "more correct" — they answer different questions.

**Important for the implementation:** ripser and gudhi compute only over $\mathbb{Z}/p$ for prime
$p$; they have no rational coefficients. The "$\mathbb{Q}$" row cannot be checked directly. Instead
$p = 3$ is used: in characteristic $\ne 2$ torsion does not show, and the answer coincides with the
rational one.

**Decision on the reference (adopted, not to be revisited without discussion).** The reference
implementation of layer 1 works **only over $\mathbb{Z}/2$**: row addition is XOR, no division is
needed, the code stays short and transparent, and that is its only purpose. Generalising to an
arbitrary $p$ would require Gaussian elimination with inverses modulo $p$ and would complicate
exactly the component whose simplicity makes it trustworthy.

Consequence: **the case $p = 3$ is checked only by the main pipeline (ripser), without independent
confirmation.** This is a deliberate loss. It is acceptable because $p=3$ plays an auxiliary role
here — to show that the answer depends on the field — while the project's working field
$\mathbb{Z}/2$ is fully covered by the double check. The test at $p=3$ is treated as a check of
*consistency* (the field parameter does reach the computation), not as an independent confirmation
of the Betti numbers. The reference's docstring states this limitation explicitly; an attempt to
call it with $p \ne 2$ is an explicit error, not a silent computation over $\mathbb{Z}/2$.

The field is fixed in `config.yaml` (`topology_check.coeff_field`), and the expected values are
stored with the field indicated. The Klein bottle dataset must be validated against the table for
its field.

### 1.2a. Significance criterion: why normalising by diam is wrong

The rule "a bar is significant if $d - b \ge \texttt{significance\_ratio} \times \mathrm{diam}(X)$"
turned out to be systematically biased, with errors **in both directions**. This was established by
measurement, not by argument; the numbers below are reproducible.

**False negatives on anisotropic objects.** For the torus $R{=}2.0$, $r{=}0.8$ the $H_2$ cavity
lives at the scale of the small radius $r$, while the threshold is normalised by
$\mathrm{diam}(X) = 2(R+r)$. At 400 points the $H_2$ bar is 0.257 against a threshold of 0.851 —
it fails, although the feature is there. More points help slowly: 0.421 at 600, 0.738 at 1500. The
same on the Klein bottle over $\mathbb{Z}/2$ at 900 points: a bar of 0.853 against a threshold of
0.857 — it fails by half a percent.

**False positives on compact objects.** For the sphere $S^2$ (400 points) the diameter is small
(2.12), the threshold is 0.32, and noise bars in $H_1$ of length 0.260, 0.256 and 0.254 sit right
against it; on some seeds they pass, giving $(1,1,1)$ or $(1,2,1)$ instead of $(1,0,1)$ — loops
where there are none.

**The diameter does not separate signal and noise in principle.** A noise bar of the sphere is
0.123 of the diameter, while the useful $H_2$ bar of the torus at 400 points is 0.045 of the
diameter, i.e. **three times smaller than the noise bar**. The ranges overlap, so **no** common
threshold normalised by the diameter will give the right answer on both objects at once. This is
not a matter of tuning a constant.

**A gap in the spectrum separates them confidently.** If the persistences within one dimension are
sorted in decreasing order, a real feature is followed by a sharp drop, while noise goes in an even
staircase. Spectra measured at the **working** number of points — torus $R{=}2$, $r{=}0.8$, 900
points; sphere 400 (values from `n_points_by_generator`):

| Dimension | Spectrum | Ratios of neighbours |
|---|---|---|
| Torus $H_1$ (exp. 2) | 1.813 1.034 0.477 0.431 | 1.75x **2.17x** 1.11x |
| Torus $H_2$ (exp. 1) | 0.641 0.107 0.095 0.095 | **6.01x** 1.12x 1.00x |
| Sphere $H_1$ (exp. 0) | 0.260 0.256 0.254 0.206 | 1.02x 1.01x 1.24x |

The table was deliberately taken at the working size rather than at 600 points as in the first
edition: at 600 the largest gap in the torus's $H_1$ stands at position 1 (1.76x against 1.54x),
i.e. at that size the rule would give $\beta_1 = 1$. This is the very failure for which 900 points
are adopted below — it could not be used to illustrate the criterion working.

The largest gap stands exactly at the position equal to the expected $\beta_k$; for the sphere, which
has no features, there is no pronounced gap at all. As the number of points grows the gap gets
stronger (in the torus's $H_2$: 3.99x at 600, 6.01x at 900, 7.18x at 1500).

**Why a floor is needed: the naive rule fails in the tail.** The rule "cut at the largest gap in the
whole spectrum", without cutting off the dust, is unusable. Among hundreds of noise bars of length
$10^{-3}$ and smaller, the ratio of neighbours easily reaches fivefold by chance alone, and the
largest gap of the whole spectrum ends up deep in the tail. The rule honestly cuts there and reports
hundreds of features: 398 measured against one real one. The floor removes the dust from the
competition, leaving gaps only between meaningful candidates.

**The case $\beta_k = 0$: solved by the step into the dust.** The rule "cut at the largest gap"
always returns at least one feature, so a mechanism is needed that gives zero on an object without
features. It is obtained not by adding a separate threshold but by extending the spectrum itself:
bars below the noise floor are set aside, and **the step from the last surviving bar down to the
first set-aside one also counts as a gap**. Then a spectrum of $k$ features followed by dust has its
largest gap exactly at position $k$; a spectrum consisting of an even staircase of noise has no
pronounced gap anywhere, and the rule returns zero. The floor is a fraction of the diameter, but this
is not a return to the old rule: at 0.07 of the diameter it removes only obvious dust, while the
torus's cavity and the sphere's noise loops survive it, so the scale-invariant part still does the
discriminating.

Without this step the rule fails on **two disjoint circles**: exactly two nearly equal bars remain
above the floor (1.473 and 1.453, ratio 1.01), there is no gap between them, and a rule looking only
at gaps between survivors returns zero loops instead of two. Measured: 18 misses out of 54 against
zero with the step into the dust.

**Stability of the gap requires 900 points, not 600.** At 600 points the position of the largest gap
in $H_1$ is unstable across seeds: for the torus the ratios at positions 1 and 2 are 1.73x and 1.62x,
i.e. the false one is largest, and $\beta_1$ comes out 1 instead of 2; for the Klein bottle over
$\mathbb{Z}/2$ the ratios 1.34x, 1.36x, 1.56x give $\beta_1 = 3$. At 900 points the true cut gives
2.0–2.35x against random 1.3–1.65x on all three calibration seeds. Telling the sphere in $H_1$
(expected 0) from the Klein bottle over $\mathbb{Z}/3$ (expected 1), which was almost impossible at
400–600 points, is resolved at 900 by the same mechanism.

**Calibration result.** The rule is correct on **all 54 combinations** of the table in
`docs/validation.md`, section 2 (9 cases × 3 seeds × 2 noise levels). The adopted parameters are
taken from the centre of the zero-miss region, not from its edge: zero holds for
`gap_noise_floor_ratio` from 0.05 to 0.10 and `gap_min_ratio` from 1.45 to 1.62; 0.07 and 1.55 were
adopted. The old rule gives 9 misses on the same table.

The plateau in `gap_min_ratio` is narrow — about $\pm 5\%$ around the adopted value. So when new
objects are added to family 1 the criterion must be rechecked: the margin is small, and a new object
can shrink the region to nothing. **It is not recorded which cases set each boundary of the plateau**
— this should be added here at the next calibration run. Knowing which object pushes from below and
which from above is the most useful thing to have when extending the set: it shows at once where it
will break.

**The $\beta_k = 0$ branch has a non-zero false positive rate.** This is the first real test of the
"step into the dust" — the part of the rule introduced for the no-feature case, which before S8 had
not been checked on anything. Measured on uniform clouds in $\mathbb{R}^3$, **on the generator's own
random stream**, 200 trials (10 seeds × `n_realizations`), expectation $(1,0,0)$:

| $n$ | Misses | Rate | 95% upper (Wilson) |
|---|---|---|---|
| 200 | 12/200 | 6.0% | 10.2% |
| 400 | 3/200 | **1.5%** | 4.3% |

**A caution about sample size in calibration.** The first estimate was taken on 60 trials and on a
foreign stream and gave "a plateau around 2%, going from 200 to 400 gives nothing". Both turned out
to be noise: at 60 trials 1/60 and 4/60 are statistically indistinguishable, and on the generator's
stream the same 200 points give 6%, i.e. four times more than 400. The rule is the same as for the
criterion itself: **a rate estimate taken on tens of trials shows nothing**, and the stream it is
taken on must coincide with the one that then builds the datasets — otherwise a different sample is
measured.

Roughly, from an exploratory run (60 trials, foreign stream), the rate at $n=60$ is of order 40% in
$\mathbb{R}^3$ and 58% in $\mathbb{R}^2$; in the plane the same rate needs about twice as many points
as in space. These numbers were not refined, because the working size lies far from them.

**The mechanism of the residual failure differs from the failures at small $n$.** At $n=60$ these are
extra loops $(1,k,0)$ — ordinary noise, pushed down by the number of points. The case surviving at
$n=200$ is an extra cavity $(1,0,1)$, and it works differently: the only $H_2$ bar has persistence
0.110 against a floor of 0.108, i.e. it passes the floor by 2%, and **it has no competitor in its
dimension**. The gap rule compares neighbouring values; when exactly one bar remains above the floor
there is nothing to compare with, and the lone survivor is significant by construction. For
comparison, in $H_1$ on the same cloud four bars stand above the floor with ratios 1.09, 1.00, 1.01 —
all below `gap_min_ratio`, and this dimension honestly gives zero.

That is, the failure arises not where there is a lot of noise but where there is exactly enough of it
for one single bar to survive. This is the price of the "step into the dust" construction, not an
implementation defect: requiring the lone survivor to pass a gap of `gap_min_ratio` too would remove
this case, but would change the calibrated criterion and require rerunning the whole table of
`docs/validation.md` §2 — at a risk to the torus's $H_2$ bar, which is the tightest there already.

**Practical consequence.** A null dataset cannot claim "zero misses"; it allows the rate to be
measured. The threshold in `config.yaml` (`null_models.max_false_positive_rate`) is set above the
95% bound of the true rate (4.3%), not above the zero observed on the calibration seeds: at 60 trials
zero misses is compatible with a rate of several percent, and a threshold set by it would be a claim
the measurement does not support.

**Scope of the parameters: family 1 only.** The values 0.07 and 1.55 were calibrated on point clouds
in $\mathbb{R}^3$ and $\mathbb{R}^4$ with known topology. They must not be transferred to other data
types without recalibration, and this concerns above all Task 3, where the same apparatus is applied
to functional connectomes:

- the distribution of pairwise distances for a connectivity matrix is fundamentally different from
  that of a sample from a smooth manifold, so the fraction of the diameter that separates dust will
  be different too;
- the calibration technique itself is unavailable there: a real connectome has no known $\beta_k$
  against which to check that the cut landed at the right position.

Practical consequence: the values in `topology_check` are parameters of the **synthetic validation
harness**, not a universal pipeline setting. In Task 3 significance is determined in its own way
(null models, permutation tests, comparison with an ensemble of random graphs), and inheriting these
two numbers silently is an error.

### 1.3. From a cloud to a weighted network

Two ways, both needed:

- **Gaussian similarity kernel:** $w_{ij} = \exp(-d_{ij}^2 / 2\sigma^2)$ — gives a fully connected
  weighted network, i.e. the same type of object as a connectome (Task 3).
- **Weighted $k$-NN graph with a geodesic metric:** distances are shortest paths in the $k$-NN graph
  with Euclidean edge weights (the Isomap construction). This approximates the **intrinsic metric of
  the manifold**, not the Euclidean one.

**$k$-NN is not a means of controlling complexity.** The earlier wording ("a sparse alternative for
controlling complexity") is wrong and has been replaced. Shortest paths give a **dense** $n \times n$
distance matrix, so the Rips complex over it is exactly as large as on Euclidean distances — there is
no saving. Complexity is controlled by other means described in 1.2: edge collapse, sparse Rips,
limiting the homology dimension. The value of geodesics is elsewhere: on curved manifolds the
intrinsic metric reflects their structure more faithfully than the ambient one.

**The parameter $k$ works in a window, not "the larger the better".** From below the graph falls apart
into components, and some geodesic distances become infinite. From above **short-circuiting** sets in
(a known failure of Isomap): chord edges cutting across the holes appear, and the intrinsic metric
degenerates into the ambient one. Measured on the torus (400 points, Euclidean diameter 5.71):

| $k$ | 8 | 15 | 30 | 60 | 120 | 250 |
|---|---|---|---|---|---|---|
| $\mathrm{diam}_{geo}$ | 7.72 | 7.26 | 6.97 | 6.52 | 5.99 | 5.72 |

The geodesic diameter slides monotonically towards the Euclidean one and at $k=250$ matches it to
within 0.2%: the graph has become practically complete. Hence a cheap guard that needs no persistent
computations — the ratio $\mathrm{diam}_{geo}/\mathrm{diam}_{euclid}$: approaching one means the
metric has degenerated.

**Symmetrisation.** A $k$-NN graph is directed by nature ($j$ is among the $k$ nearest to $i$, but not
necessarily vice versa). **Union** is adopted: an edge exists if at least one of the two considers the
other a neighbour. It gives connectivity at a smaller $k$ than intersection, and so widens the working
window from below. The rule is fixed in the config.

**Connectivity is mandatory.** With a disconnected graph some distances are infinite, and persistent
computations get garbage. The connectivity check is explicit, with a clear error, not a silent
replacement of infinities by a large number.

**The geodesic metric over k-NN does not apply to disconnected spaces.** Requiring a connected graph
forces bridges between components, and $m$ bridges create $(m-1)$ extra cycles: $\beta_1 = 2 + (m-1)$
for two cycles. Checked on the flag complex by the reference implementation — two disjoint 6-cycles,
bridges between non-adjacent pairs:

| Bridges $m$ | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| $\beta$ | (2,2,0) | (1,2,0) | (1,3,0) | (1,4,0) | (1,5,0) |

The longevity of these artefacts is determined by how close the components come: for well-separated
objects they are short-lived and cut off by the floor, for objects passing close they compete with the
real features. The Hopf link is the limiting case: the components pass right next to each other, so
**no setting of $k$ gives the right answer**.

Measured confirmation of both poles. The ratio $\mathrm{diam}_{geo}/\mathrm{diam}$ over the three
calibration seeds, both noise levels; "disconnected" means the graph is disconnected, "!" means the
1.10 guard fired:

| $k$ | 60 | 80 | 100 | 120 |
|---|---|---|---|---|
| Two circles, seed 20260201 | 1.35 / 1.26 | 1.26 / 1.18 | 1.21 / 1.15 | 1.17 / 1.12 |
| Two circles, seed 7 | disconnected | 1.27 / 1.19 | 1.21 / 1.15 | 1.14 / **1.09!** |
| Two circles, seed 99 | disconnected | 1.28 / 1.20 | 1.22 / 1.15 | 1.15 / **1.09!** |
| Hopf, seed 20260201 | 1.47 / 1.15 | **1.01!** | **1.00!** | **1.00!** |
| Hopf, seed 7 | 1.39 / 1.24 | **1.00!** | **1.00!** | **1.00!** |
| Hopf, seed 99 | 1.38 / 1.23 | **1.02!** | **1.01!** | **1.01!** |

The two circles are 1.0 apart at radius 1.0, so their bridge cycles are short-lived: a window
$k = 80\ldots100$ exists, 100 is adopted. In the Hopf link the circles pass right next to each other,
the bridge cycles live long, and a third bar of 0.44–0.54 appears in $H_1$ next to the first two of
1.3–1.8 — an extra class the criterion cannot tell from the real ones. At no $k$ does the answer
become right, so the dataset is excluded from the geodesic table. On the Euclidean metric it passes:
the limitation belongs to the construction, not to the object.

**The window in $k$ at the working number of points.** Measured on the torus and the Klein bottle
(900 points, `union`, criterion with Euclidean parameters):

| $k$ | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|
| $\mathrm{diam}_{geo}/\mathrm{diam}$, torus | 1.315 | 1.257 | 1.227 | 1.202 | 1.114 |
| $\beta$, torus | (1,1,1) | (1,2,1) | (1,2,1) | (1,2,1) | (1,2,1) |
| $\mathrm{diam}_{geo}/\mathrm{diam}$, Klein | 1.341 | 1.283 | 1.245 | 1.218 | 1.122 |
| $\beta$, Klein | (1,0,1) | (1,2,1) | (1,2,1) | (1,2,1) | (1,2,1) |

From below the window closes at $k = 10$: the graph is still connected, but geodesic paths go the
long way round, and one $H_1$ class is lost. From above the guard fires around $k \approx 180$.
$k = 40$ is adopted — the middle of the window on a logarithmic scale; $k = 20$ is correct but sits
right at the edge, and the neighbouring value half as large is already wrong.

**The significance criterion parameters are not inherited by the geodesic metric — but a run showed
that they fit.** The values 0.07 and 1.55 were calibrated on Euclidean clouds, and the geodesic metric
is not a monotone reparametrisation of the same distances but a different metric. So the table of
section 2 of `docs/validation.md` was rerun on geodesic distances, three seeds, two noise levels.

Result: on the domain of applicability of the construction (all objects except the Hopf link — see
below) a zero-miss region exists, it is wide — floor from 0.04 to 0.12, gap from 1.45 to 1.8 — and
**the Euclidean values 0.07 and 1.55 lie inside it**. No separate parameters are introduced for
geodesics. It matters that this was established by a run: a coincidence was a possible outcome, not
grounds for skipping the check.

The Hopf link is not part of this run: it is excluded from the geodesic table for the reason worked
out above — bridge cycles of components passing right next to each other are long-lived and
indistinguishable from real features at any $k$.

The inverse transform into distances for the filtration must be a **monotonically decreasing**
function of similarity: $d = \sqrt{-2\sigma^2\log w}$ (the inverse of the Gaussian kernel) or
$d = 1 - w$. A non-monotone transform permutes the order in which edges enter and silently changes the
answer.

**But monotonicity is not enough.** A monotone $\varphi$ preserves the *structure* of the diagram: the
classes are the same, the order of births and deaths is the same, bars map to
$(\varphi(b), \varphi(d))$. But persistence is distorted non-linearly, since
$\varphi(d) - \varphi(b) \ne \varphi(d - b)$, and the significance criterion compares exactly
persistences and their ratios. **The criterion is not invariant under reparametrisation of the
filtration.** Practical consequence for Task 3: the choice between $d = 1-|r|$ and
$d = \sqrt{2(1-r)}$ for a correlation matrix is also a monotone transform, and it too will shift
significance; the criterion's parameters are tied not only to the data type but also to the chosen
parametrisation (see "Scope" in 1.2a).

**A degenerate kernel: the main source of errors here.** The width $\sigma$ of the Gaussian kernel must
be commensurate with the scale of the data. At $\sigma = 0.5$ and a torus diameter of 5.71 the
fraction of weights below $10^{-6}$ is **60%** — almost all geometry collapses to zero. The transform
$\sqrt{-2\sigma^2\log w}$ does not notice, because it is exactly inverse to the kernel and unrolls the
saturation back; $1-w$ has nothing to unroll with and works with already destroyed data. Measured on
the torus (600 points, ratios in the $H_1$ spectrum, the right cut is at position 2):

| Kernel width | $\sqrt{-2\sigma^2\log w}$ | $1 - w$ |
|---|---|---|
| $\sigma = 0.5$ | 1.80x **1.87x** 1.17x — right | 1.02x 1.34x 1.11x — **broken** |
| $\sigma = 2.0$ | 1.80x **1.87x** 1.17x — right | 2.24x **2.32x** 1.20x — right |

Conclusion: $1 - w$ works properly and at an adequate width separates even more cleanly. The kernel is
to blame, not the transform. So $\sigma$ is set per generator as a fraction of the median pairwise
distance, not as an absolute number for the whole family, and is accompanied by a guard test on the
fraction of collapsed weights.

**Neither transform is forbidden.** $1 - w$ is the standard choice for connectomes ($d = 1 - |r|$),
and it is the one that goes to Task 3; removing it from the harness would mean not testing the very
function that will run on real data. $\sqrt{-2\sigma^2\log w}$, conversely, makes sense only for a
Gaussian kernel: a correlation matrix is not a Gaussian kernel.

---

## 2. Family 2: dynamic graphs with scheduled events

### 2.1. Construction

A scheduler builds a sequence of snapshots $G_0,\dots,G_{T-1}$ on $N$ nodes and at the same time keeps
a truth log. The basic scenario of the life cycle of a cycle:

1. a path without cycles ($\beta_1 = 0$);
2. closing the path by an edge → **birth** of a class in $H_1$;
3. adding a chord that splits the cycle → **birth of a second** class in $H_1$;
4. removing an edge → **death** of a class in $H_1$;
5. splitting the graph into two components → **birth** of a class in $H_0$;
6. restoring an edge → **merge**, i.e. **death** of a class in $H_0$.

**The concrete realisation on $N = 24$.** The path $0-1-\dots-23$; closing by the edge $(23,0)$; the
chord $(0,12)$, splitting the 24-cycle into two cycles of length 13; removing $(6,7)$ on the arc
$0..12$; removing $(3,4)$ on the dangling path (it detaches $\{4,5,6\}$, $\beta_1$ does not change);
restoring $(3,4)$. No triangles arise on any snapshot — otherwise the flag complex would fill a cycle
and the event would differ from the one recorded in the log. No isolated nodes arise either.

**Admissibility condition: both arcs at least three edges long.** The construction is not tied to
$N = 24$, but it does not work for every $N$ either. The chord $(a,b)$ splits the $N$-cycle into arcs
of $b - a$ and $N - b + a$ edges; for both generated cycles to be cycles of the flag complex, each arc
must carry **at least three edges**. With the chord fixed at $(0,12)$ the arcs are 12 and $N - 12$, so
the condition reduces to $N \ge 15$.

Both violations produce not a failure but silently wrong truth:

| $N$ | Long arc | What happens | $\beta_1$ after the chord |
|---|---|---|---|
| 13 | 1 edge | the chord $(0,12)$ coincides with the closing edge $(0, N-1)$, adding it is a no-op | 1 |
| 14 | 2 edges | the chord closes the triangle $\{0,12,13\}$, the flag complex fills it | 1 |
| $\ge 15$ | $\ge 3$ edges | both cycles live | 2 |

In both degenerate cases $\beta_1$ stays 1, while the log still declares the birth of a second class.
This is exactly the failure `docs/validation.md` begins with: not a crash but a plausible-looking
dataset with wrong truth, which would surface only in Task 2. Measured: at $N = 13$ and $N = 14$ the
curve gives $\beta_1 = 1$ at snapshot 27.

Hence the requirement on the generator: check the arc lengths and **refuse to build the dataset**
rather than output it silently. The check is stated in terms of arcs, not as "$N \ge 15$" — the bound
is a consequence of the chord's position and must move with it.

**Event index convention.** The record `snapshot: k` means that $G_k$ **already contains** the change
and $G_{k-1}$ does not yet. The jump of `betti_curve` falls on the transition $k-1 \to k$. The
alternative convention ("the change happened after $k$") gives an off-by-one shift, invisible inside
family 2 and surfacing only in Task 2, where it looks like a systematic error of the event localisation
algorithm. So the consistency test checks the positions of the jumps by name, not only their number.

**A repeated state is an intentional property.** Restoring $(3,4)$ returns the graph exactly to the
state it was in between removing $(6,7)$ and the split, so there are five distinct graphs in the
scenario and 60 snapshots. Coinciding snapshots at different times are not a defect and not a reason
to "fix" deduplication: they check that the zigzag pipeline does not collapse states by graph identity
and honestly reports the separate birth and death of $H_0$.

**Randomness.** The scenario is deterministic: `rng` is accepted by the common generator contract but
not used. A permutation of node labels is not applied to the dataset itself — snapshots must stay
readable by eye when debugging. Instead a **test** is set up: a random permutation of labels changes
neither `betti_curve` nor the event list. It catches the same class of bugs (a silent dependence on the
path running $0,1,\dots,23$ in order) without spoiling the data.

### 2.1a. The geometric variant (`geometric_loop`)

Points on a circle with a radius changing in time and a controlled gap in the arc; a Rips complex on
each frame. This is the only family-2 dataset with a **metric** rather than purely combinatorial
filtration, i.e. the only one where the chain matches a real dynamic connectome. The specification that
was missing in S6 is below; all numbers were measured by the reference implementation, not derived from
formulas.

**Placement of the points.** $\theta_j = j\,(2\pi - \delta)/(n-1)$, $j = 0,\dots,n-1$, on a circle of
radius $R(t)$. The gap in the arc at a fixed number of nodes is exactly $\delta$: the points are laid
out along an arc of $2\pi - \delta$, not around the whole circle.

**The closed state is $\delta = 2\pi/n$, not zero.** At $\delta = 0$ the formula gives
$\theta_{n-1} = 2\pi \equiv \theta_0$, i.e. two coinciding points. Substituting $\delta = 2\pi/n$ gives
a step of $(2\pi - 2\pi/n)/(n-1) = 2\pi/n$ **exactly**, i.e. exactly a uniform circle. The formula does
not change — what changes is the value called "closed".

The difference is not cosmetic. At $n=24$, $\varepsilon=0.8$, $R=1$:

| $\delta$ | Faces of the complex | $\beta$ | min distance |
|---|---|---|---|
| $0$ | (24, 76, 84, 36) | (1,1,0) | $2.4\cdot10^{-16}$ |
| $2\pi/n$ | (24, 72, 72, 24) | (1,1,0) | 0.261 |

**$\beta$ coincides**, so the event consistency test would pass and the discrepancy would stay in the
dataset silently — the same class of failure as an inadmissible $N$ in 2.1. It is caught not by $\beta$
but by the degenerate geometry: at $\varepsilon = 0.20$ the construction with a doubled vertex gives
$\beta_0 = 23$ (the duplicate is always connected) where the uniform circle gives 24. Hence the
requirement on the test: check the minimum pairwise distance, not only the Betti numbers.

**$\varepsilon$ is not a forgotten knob but a parameter that sets a qualitative regime.** Two
circumstances.

First, $\mathrm{VR}(S^1; r)$ does not become contractible as the relative scale grows: by
Adamaszek–Adams it is homotopy equivalent to $S^1$ at geodesic $r < 2\pi/3$ (in chords:
$\varepsilon < R\sqrt{3}$), then to $S^3$, then $S^5$ and so on. At `max_homology_dim: 2` the death of
$\beta_1$ will be visible, but the born $\beta_3$ will not, and the event will look like "the loop just
disappeared" for no reason.

Second, from below the inter-point step $2\pi R/n$ grows with the radius, and at $\varepsilon$ below it
a growing radius gives not a gap in the arc but a disintegration of the graph into $n$ components.

Hence the window: $2\pi R_{\max}/n < \varepsilon < R_{\min}\sqrt{3}$. At $n=24$ and $R \in [1.0, 1.4]$
this is $(0.367,\ 1.732)$. Both bounds are confirmed by measurement:

| $\varepsilon$ | 0.20 | 0.30 | 0.40 | 0.80 | 1.70 | 1.90 |
|---|---|---|---|---|---|---|
| $R = 1.0$ | (24,0,0) | (1,1,0) | (1,1,0) | (1,1,0) | (1,1,0) | **(1,0,0)** |
| $R = 1.4$ | (24,0,0) | **(24,0,0)** | (1,1,0) | (1,1,0) | (1,1,0) | (1,1,0) |

$\varepsilon = 0.8$ is adopted — the middle of the window on a logarithmic scale
($\sqrt{0.367\cdot1.732} = 0.797$), by the same rule by which $k$ was chosen in 1.3.

**Gap width.** For there to be no edge across the gap, the chord must exceed $\varepsilon$ at the
**smallest** radius: $2R_{\min}\sin(\delta/2) > \varepsilon$, whence at the adopted values
$\delta > 0.823$. The bound is close: at $\delta = 0.8$ the loop does not break at the small radius and
no event happens. $\delta = 1.2$ is adopted.

**Interpolation of `radius_schedule`** — piecewise linear in the snapshot index, nodes evenly spread
over the whole length of the sequence.

**Schedule.** The radius breathes and generates no events; only the gap generates events.

| Snapshots | $\delta$ | $\beta$ |
|---|---|---|
| 0–19 | $2\pi/n$ | (1, 1, 0) |
| 20–44 | 1.2 | (1, 0, 0) |
| 45–59 | $2\pi/n$ | (1, 1, 0) |

**Radius breathing changes the graph without changing $\beta$ — and this is the main negative control
of the dataset.** The statement "the radius changes every frame but the graph does not" is false: at the
adopted parameters the graph changes at snapshots 4 ($72 \to 48$ edges), 43 ($45 \to 66$) and 56
($48 \to 72$), and no event happens at any of them. There are four distinct graphs in the dataset. The
test for the absence of unscheduled events must name these three snapshots individually: a check "over
the whole range" would also pass on a dataset where the complex never changed, i.e. it would check
nothing.

**Margin at the boundary.** The minimum $|\text{chord} - \varepsilon|$ over all snapshots and point
pairs is $9.9\cdot10^{-4}$ (snapshot 42) — thirteen orders of magnitude above machine zero. Byte-level
reproducibility of the snapshots across platforms is not at risk; this is checked before writing the
sha256 into the manifest, not after.

**What the dataset gives beyond `cycle_lifecycle`.** The faces (24, 72, 72, 24) — 72 triangles, so
$\beta_1 = 1$ while the cyclomatic number is $72 - 24 + 1 = 49$. In `cycle_lifecycle` there are no
triangles by construction, and the coincidence of $\beta_1$ with the cyclomatic number there is a
consequence of the construction; here the discrepancy is 49-fold on real data.

### 2.2. Semantics of events: how a "merge" translates into homology

This is where mistakes are most frequent. Intuitive words and homological events are not the same, and
the log must keep both levels.

| Event on the graph | Homological event | What happens to β |
|---|---|---|
| Components merge | Death of an $H_0$ class (by the seniority rule the younger one dies) | β₀ − 1 |
| A component splits | Birth of an $H_0$ class | β₀ + 1 |
| A path closes into a cycle | Birth of an $H_1$ class | β₁ + 1 |
| A chord is added to a cycle | Birth of a new $H_1$ class ("splitting" the cycle) | β₁ + 1 |
| A cycle is filled with triangles | Death of an $H_1$ class | β₁ − 1 |
| An edge of a cycle is removed | Death of an $H_1$ class | β₁ − 1 |

Note the row about the chord: in the language of graphs it is a "split" of one cycle into two, while in
the language of homology it is a **birth**, not a split. A barcode has no operation of splitting a
class; there are only births and deaths. So in JSON the `type` field takes only the values `birth` /
`death`, and the intuitive wording goes into `narrative`.

### 2.2a. Which $H_1$ class dies when an edge is removed

Table 2.2 says that $\beta_1$ decreases by one but not which birth this death is paired with. The
answer looks like a free convention and is not one: the decomposition of a zigzag module into interval
modules is unique (Krull–Schmidt for representations of a quiver of type $A_n$), so the pairing is
determined and cannot be assigned.

**Analysis for the `cycle_lifecycle` scenario.** Over $\mathbb{Z}/2$ denote: $c_1$ — the cycle
$0-\dots-12$ together with the chord, $c_2$ — the cycle $12-\dots-23-0$ together with the chord,
$c = c_1 + c_2$ — the original 24-cycle (the chord enters the sum twice and cancels).

| Snapshots | Graph | $H_1$ |
|---|---|---|
| 12–26 | 24-cycle | $\langle c \rangle$ |
| 27–40 | theta graph | $\langle c_1, c_2 \rangle$ |
| 41– | $(6,7)$ removed from the arc of $c_1$ | $\langle c_2 \rangle$ |

All snapshot inclusions are injective on $H_1$, so the union zigzag around 27–41 has the shape
$1 \to 2 \leftarrow 1$ with images $L_{\text{left}} = \langle c_1 + c_2 \rangle$ on the left and
$L_{\text{right}} = \langle c_2 \rangle$ on the right. These are **different lines** in a
two-dimensional space, hence $V_{27} = L_{\text{left}} \oplus L_{\text{right}}$, and the intervals come
out as $[12, 40]$ and $[27, \infty)$.

**The class that dies is the one born when the path closed (`h1_a`), not when the chord was added.**

That this is not a convention is shown by a counterexample: if not $(6,7)$ but the chord itself is
removed, $c = c_1 + c_2 = L_{\text{left}}$ survives, and then the class born at 27 is the one that dies.
The answer depends on which edge is removed — so it is determined by geometry, not by a choice of basis.
Conversely, removing an edge from either of the two arcs gives the same answer:
$\langle c_1 + c_2 \rangle$ coincides neither with $\langle c_1 \rangle$ nor with $\langle c_2 \rangle$.

**Why intuition fails here.** The tempting argument is "the chord split the cycle in two; break one of
them and it dies, i.e. the young class". It rests on the feeling that the elder class should outlive the
younger. The seniority rule of 2.2 is stated for the death of $H_0$ when components merge and does not
apply to the death of $H_1$ by edge removal at all: there are no two classes competing for one death.

**How it is checked.** A test on the identity of a class need not and must not be written — the basis of
$H_1$ of the theta graph is not canonical. What is checkable and sufficient: the vector of the 24-cycle
over $\mathbb{Z}/2$ **does not lie** in $Z_1(G_{41})$, while $c_2$ does. Three lines on the reference
implementation; exactly the part of the claim that is canonical.

**The cost of the error.** A truth log with an inverted pairing breaks no test inside family 2:
`betti_curve` does not depend on pairing. The error would surface in Task 2, where a correct zigzag
algorithm would systematically report the "wrong" birth–death pair, and debugging would go into the
algorithm, not the log.

### 2.3. The complex the truth refers to

The truth is defined for the **flag complex (clique complex)** of the thresholded graph, because that is
what the pipeline uses. The construction rule: every clique of the graph is declared a simplex. No
information beyond the graph is introduced.

**Threshold convention:** an edge exists when $w_{ij} > t$, strictly. A weight equal to the threshold
does not exceed it; at $t = 0$ a non-strict comparison would make the graph complete, turning absent
edges (zero weights) into existing ones.

**There are two edge rules, and they point in opposite directions.** The convention above is about
**similarity**: the larger the weight, the stronger the connection, an edge appears from above. For the
metric dataset (2.1a) the disk holds **distances**, and an edge appears from below:
$d_{ij} \le \varepsilon$. The comparison is non-strict here, because $\varepsilon$ is an inclusion
radius, not an exceedance threshold.

| Rule | Edge condition | What a snapshot holds |
|---|---|---|
| `weight_above` | $w_{ij} > t$, strict | similarities (`cycle_lifecycle`, connectome) |
| `distance_within` | $d_{ij} \le \varepsilon$, non-strict | distances (`geometric_loop`) |

The direction **is not inferred from the family and is not a default convention**: it is written into
`ground_truth.json` in the `edge_rule` field, together with the threshold value in `edge_threshold`
(`docs/data.md`, §4). A consumer that applies the similarity rule to a metric dataset gets the
**complement** of the graph — and gets it silently, because the complement of a sparse graph is also
connected and also gives plausible Betti numbers. Inside the package such an error does not surface at
all: the generator computes `betti_curve` its own way. It would surface in Task 2, in someone else's
code.

Practical consequence: the rule is applied by **one function** for the whole project, not rewritten as
a comparison at every place of use.

**Consequence 1: $\beta_1 \ne m - n + c$.** The cyclomatic number of the graph is not equal to $\beta_1$
of the flag complex, since triangles are filled and kill the corresponding cycles. Computing $\beta_1$
as the cyclomatic number is a typical and silent error. A triangle of three edges gives $\beta_1 = 0$,
not 1.

**Consequence 2: the complex is built up to dimension $k_{\max}+1$.** $\beta_2$ needs 3-simplices:
$\beta_2 = \dim\ker\partial_2 - \operatorname{rank}\partial_3$, and with a cut-off at dimension 2 the
term $\operatorname{rank}\partial_3$ vanishes, and every closed 2-chain is wrongly counted as a cavity.
At `max_homology_dim: 2` the complex is built up to dimension 3. There must be no "cut off at 2"
parameter in the API — its only purpose would be to enable a wrong mode.

**Consequence 3: the complex is closed under faces.** If a clique of size $k+1$ is in the complex, all
its subsets are too. **All** cliques of each size must be enumerated, not only maximal ones. The flag
complex of $K_4$ contains 4 vertices, 6 edges, 4 triangles and 1 tetrahedron — the triangles are not
"absorbed" by the tetrahedron but live in their own dimension and form the rows of the matrix
$\partial_3$.

`networkx.find_cliques` returns only **maximal** cliques (the Bron–Kerbosch algorithm) and does not fit
this task: on $K_4$ it gives one tetrahedron without edges and triangles, the boundary matrices do not
match in dimensions, and $\beta_1$ grows from 0 to 3. The right way is to iterate
`itertools.combinations` over all subsets, checking pairwise adjacency. Slower, but the reference does
not need speed.

Control examples (checked against ripser):

| Graph | Flag complex | $\beta$ |
|---|---|---|
| $K_3$ (triangle) | A filled face | $(1,0,0)$ |
| $K_4$ | A solid tetrahedron | $(1,0,0)$ |
| $K_{2,2,2}$ (octahedron) | The boundary of $S^2$, no 3-simplices | $(1,0,1)$ |

ripser with `maxdim=2` internally enumerates simplices up to dimension 3 and gives $\beta_2 = 0$ on
$K_4$ — the reference implementation must agree with this behaviour.

---

## 3. Family 3: dynamic SBM and coupled oscillators

### 3.1. Dynamic SBM

A stochastic block model with a time-varying block matrix $B(t)$: communities merge and split according
to a schedule. Generation — `networkx.stochastic_block_model` in a loop over time with a shared `rng`.
Truth: **community labels over time and the schedule of merges/splits**, and nothing beyond that.

**Correction: a community merge is not a homological event.** An earlier edition of this section
promised that the dataset "checks $\beta_0$ and its relation to modularity". This is false, and
measurably so. At `p_in: 0.35`, `p_out: 0.02` and blocks of 15 nodes there are 7 to 41 inter-block
edges per snapshot (three calibration seeds) — the blocks are stitched together by dozens of edges,
they are never separate components, and merging two communities changes neither the number of
components nor anything in homology at all.

That $\beta_0$ does move nonetheless does not help but hinders. Over a series of 60 snapshots it takes
the values 1 and 2 on **every** calibration seed, and the cause is not communities but **isolated
vertices**: the probability that a node draws no edge at all is
$(1-p_{in})^{14}(1-p_{out})^{45} \approx 9.7\cdot10^{-4}$, which on a $60\times60$ dataset gives 3.5
such vertices expected; 2, 3 and 2 measured. That is, $\beta_0$ here tracks a random shortfall of edges
at one node, not community structure, and its excursions are not tied to the schedule.

The robust statement looks different: **after removing isolated vertices the graph is connected on
every snapshot**, and this does not change at the merge or the split. That is what the test asserts,
not equality of $\beta_0$ before and after the merge: the latter holds on the measured seeds but rests
on the absence of an isolated vertex on exactly those two snapshots.

**The two regimes do not combine in one dataset.** $\beta_0$ becomes meaningful truth only at
$p_{out} \to 0$, when the blocks really fall apart into components; a relation to modularity, on the
contrary, requires $p_{out} > 0$, otherwise there is nothing to relate. These are two different
objects, and trying to get both from one is a source of confusion, not economy.

**Rejected ways of giving SBM topological truth** (recorded so that the question is not reopened):

- *Record the Betti curve measured by the reference.* This is truth as pipeline output. That it is
  computed by an independent implementation does not change matters: for family 2 the curve is **also
  predicted by the construction**, and the test compares two sources. SBM has no prediction, there is
  nothing to compare with, and `ground_truth.json` would receive a regression fixture under the name of
  truth. As a fixture it is legitimate, but then it must also be stored separately.
- *Lower $p_{out}$ to the disintegration regime.* The disintegration of the blocks stays random, and
  "the answer is known in advance" degenerates into "known with high probability": a bad seed silently
  gives a dataset with wrong truth. A deterministic variant exists ($p_{out} = 0$ plus a spanning tree
  wired into each block, then $\beta_0$ equals the number of blocks exactly), but that is a different
  dataset, not this one.

So the SBM has `truth_kind: partition` (`docs/data.md`, §4): the truth is a partition and a schedule,
fully known in advance, and the fields `betti_curve` and `events` are not written.

### 3.1a. What a coupling regime is

A section that was missing: the schedule $A(t)$ was mentioned in all the formulas below, but nowhere was
it said what exactly defines one regime.

**A regime is a partition of the channels into blocks.** The coupling matrix of a regime is block
diagonal: $A_{ij} = 1$ if $i$ and $j$ lie in the same block and $i \ne j$, otherwise 0. Symmetric, with
a zero diagonal. Stored in `coupling.npy` of shape `(T_schedule, n_channels, n_channels)`, one matrix
per regime.

**The partitions must be related by refinement.** This is the content of the word "merge/split" in the
roadmap, and it does not hold automatically: the transition $[15,15] \to [10,10,10]$ is neither a merge
nor a split, because neither partition refines the other; the blocks are simply recut. The earlier
regime names of the form `block_2` and `block_3` hid this, naming by the number of blocks what must be
defined by their composition.

The adopted schedule at 30 channels:

$$[10,10,10] \;\longrightarrow\; [20,10] \;\longrightarrow\; [10,10,10]$$

The first transition is a **merge** (two blocks of 10 become one of 20), the second a **split** (back
again). Each pair of neighbouring partitions is related by refinement, so the event on the graph is
described by one word, not a "recut".

**The transient is discarded, and this is physics, not bookkeeping.** Rössler starts off the attractor,
and the first hundreds of steps are a descent onto it, not the dynamics of the coupled system. A window
falling on this stretch gives correlations unrelated to $A(t)$ and fails the agreement check of
`docs/validation.md`. So `transient_length` steps are integrated under the matrix of the **first**
regime, discarded, and only then `regime_length` steps are written for each regime.

Hence a hard relation, checked in the config:

$$n_{\text{timepoints}} = \texttt{regime\_length} \times |\texttt{regimes}|$$

Without it `coupling.npy` and the series drift apart: a window computed by time index would be matched
with the wrong regime, and the error would look like poor recovery of connectivity rather than a
desynchronisation.

### 3.2. Kuramoto model

$$\dot\theta_i = \omega_i + \frac{K}{N}\sum_{j} A_{ij}(t)\,\sin(\theta_j - \theta_i),$$
where $\omega_i$ are the natural frequencies (e.g. $\mathcal{N}(0,1)$) and $A(t)$ is a known coupling
schedule. Observable: $x_i(t) = \sin\theta_i(t)$.

### 3.3. Coupled Rössler systems

$$\dot x_i = -y_i - z_i + \varepsilon\sum_j A_{ij}(t)(x_j - x_i),\quad
\dot y_i = x_i + a y_i,\quad \dot z_i = b + z_i(x_i - c)$$
with the standard $a=0.2$, $b=0.2$, $c=5.7$. Integration — `scipy.integrate.solve_ivp` with a fixed
output grid (`t_eval`), method RK45 or LSODA, tolerances in `config.yaml`.

**Integrate one regime per call, not the whole series in one.** $A(t)$ is piecewise constant, i.e. the
right-hand side is discontinuous at regime boundaries. The adaptive RK45 step does not know this: a step
that straddles a boundary averages two different right-hand sides, error control on it is computed from
a wrong estimate, and the result depends on where exactly the step fell. Both things suffer at once —
accuracy and reproducibility, for whose sake this whole section is written.

So `solve_ivp` is called once per regime: the final state of the previous call becomes the initial state
of the next. Then each call integrates a smooth system, and the discontinuity falls exactly at the
junction, where it belongs.

**The coupling $\varepsilon$ works in a window, not "the larger the better".** Diffusive coupling on a
complete block of size $m$ gives a transverse coupling of order $\varepsilon m$, and above a certain
value **complete synchronisation** sets in: the channels within a block become bitwise identical, $|r|$
within the block equals exactly one, and no variability remains within the block. As an fMRI analogue
such a dataset is meaningless — there is nothing to recover.

From below the coupling simply does not show. Measured at 30 channels and blocks of 10 (gap between the
mean $|r|$ within a block and between blocks):

| $\varepsilon$ | 0.002 | 0.005 | 0.01 | 0.02 | 0.04 | 0.15 |
|---|---|---|---|---|---|---|
| Gap in $\lvert r\rvert$ | 0.090 | 0.440 | **0.542** | 0.445 | 0.334 | 0.348 |
| Within-block spread | 19.0 | 8.80 | 6.67 | 5.89 | $2.7\cdot10^{-9}$ | 0 |

At $\varepsilon \ge 0.04$ the within-block spread drops to machine zero — this is complete
synchronisation; the gap meanwhile stays non-zero only thanks to between-block correlations and is
misleading. $\varepsilon = 0.01$ is adopted, the middle of the window $[0.005,\ 0.02]$ on a logarithmic
scale — by the same rule by which $k$ in 1.3 and $\varepsilon$ in 2.1a were chosen.

**On Rössler reproducibility separately.** At $c = 5.7$ the system is chaotic, and trajectories diverge
exponentially from any perturbation. The requirement "the same initial conditions and `t_eval` → the
same series" holds because the computation is deterministic, but the dataset is **not** robust to a
change of scipy version, method or tolerances: under them the series moves entirely, not slightly. This
limitation is recorded in the dataset's `notes`, not considered a defect.

### 3.4. Sliding window

A window of length $L$ with step $s$; in each window a Pearson correlation matrix. The result is a
sequence of weighted fully connected graphs, i.e. a dynamic graph.

**The sign of correlation: similarity is $|r|$.** Distances are $d = 1 - |r|$ with a zeroed diagonal.
The absolute value is not cosmetic here: $1 - r$ at $r < 0$ exceeds one, and the distance matrix check
requires non-negativity and zeros on the diagonal. In substance, anticorrelation is also a connection,
and a strong one in absolute value; discarding it by sign would declare strongly anticorrelated channels
unconnected.

**The Fisher transform is not on the way to distances.** $z = \operatorname{arctanh} r$ is needed where
correlations are added, averaged or compared statistically: it has an approximately constant variance,
which $r$ does not. But $z$ is unbounded above, $\operatorname{arctanh}(1) = \infty$ on the diagonal, and
$1 - z$ does not lie in $[0,1]$ and is not a metric. So correlation is used here **in two different
ways**, and they must not be mixed:

| Use | Quantity | Why |
|---|---|---|
| Edge weight, then distance | $|r|$, then $1 - |r|$ | input of the filtration |
| Statistic of agreement with $A(t)$ | $z = \operatorname{arctanh} r$ | comparing recovered connectivity with the true one |

The `fisher_z` flag in the config refers to the second and only to it.

**The window is bounded on both sides, not one.** An earlier edition named only the upper bound; there
was no lower one, and the config violated it sixfold.

$$\text{oscillation period} \;\ll\; L\,\Delta t \;\ll\; \text{regime duration}$$

From above: if the window is longer than the interval between regime changes, the event is smeared and
cannot be localised. From below: if the window is shorter than several periods, the correlation measures
not coupling but a common trend — over a fraction of a period any two smooth sinusoids look alike.
Measured at $K = 6$, 30 channels, blocks of 10:

| $\Delta t$ | Periods per window | $\lvert r\rvert$ within block | between blocks | Agreement |
|---|---|---|---|---|
| 0.01 | 0.16 | 0.946 | 0.936 | 0.067 |
| 0.05 | 0.80 | 0.833 | 0.764 | 0.144 |
| 0.1 | 1.59 | 0.767 | 0.559 | 0.301 |
| 0.2 | 3.18 | 0.721 | 0.483 | 0.347 |
| 0.4 | 6.37 | 0.717 | 0.408 | 0.463 |

At 0.16 periods the within-block and between-block correlations are 0.946 and 0.936 — coupling is not
recovered at all, and no agreement threshold can save this. $\Delta t = 0.2$ is adopted, i.e. about
three periods per window at 10% of the regime duration.

**The second condition is supercritical coupling.** For Kuramoto the effective coupling within a block
equals $(K/N)(\lvert\text{block}\rvert - 1)$, and it must exceed the critical
$K_c \approx 1.6\,\sigma_\omega$. At $K = 1.2$, $N = 30$ and blocks of 10 this is $0.36$ against $1.6$ —
the system is below the synchronisation threshold. $K = 6$ is adopted, i.e. $1.8$ against $1.6$. The two
conditions are independent: raising $K$ without fixing the window does not help (at $K = 60$ the
agreement is still 0.712, because the window limitation dominates).

**What the agreement statistic does not do: it does not distinguish regimes.** Agreement with *another*
regime reaches 0.440 against a true minimum of 0.383, i.e. the classes overlap, and no threshold will
separate them. This is not a defect but a consequence of the nesting of partitions in 3.1a: $[20,10]$ is
a coarsening of $[10,10,10]$, and neighbouring regimes share most of their structure. The honest null
control here is **a permutation of channel labels**: it gives $-0.039$ to $0.068$ against the true
$0.383\ldots0.878$, and the gap is clean.

Practical consequence: `min_coupling_agreement` confirms that the window recovers coupling and is
**not** a tool for localising a regime change. Localisation needs a different apparatus, and that is the
work of Task 2, not a check of the generator.

---

## 4. Family 4: noise and null models

**Controlled noise** on the reference datasets, each with its level as a parameter:

- Gaussian jitter of edge weights: $w' = \operatorname{clip}(w + \mathcal{N}(0,\sigma^2))$ into the
  admissible range;
- spurious edges: a fraction $p^+$ of the absent edges, weights drawn from the empirical distribution of
  existing ones;
- missing edges: a fraction $p^-$ of edges removed;
- jitter of point positions in a cloud: $\sigma$ in the ambient space.

**Null models:** Erdős–Rényi $G(n,p)$ and uniform point clouds. Requirement: the temporal diagram
algorithm must NOT detect events where there are none — control of the false positive rate. What this
requirement means for the dataset itself and what belongs to Task 2 is worked out in 4.2.

**How this is used in Result 6:** the bottleneck distance between the diagram of the "clean" dataset and
the diagrams of its noisy versions as a function of the noise level, compared with the stability theorem
(Cohen-Steiner, Edelsbrunner, Harer). Computed by the build with the `--results` flag and stored in
`results/stability.json`; the choice of objects and number of points — in 4.2a.

### 4.1. The stability bound for Rips: a factor of 2, not 1

The naive form $d_B \le d_H$ is **wrong** and is rejected by measurement. The right one is
$$d_B \le 2\,d_{GH} \le 2\,d_H .$$

The two is not fitted to the measurement; it follows from the parametrisation. The filtration value of a
simplex in the Rips complex is its **diameter**, not its radius. If each point is moved by at most
$\delta$, then for any simplex $\sigma$
$$\Bigl|\max_{i,j\in\sigma} d(x_i,x_j) - \max_{i,j\in\sigma} d(y_i,y_j)\Bigr|
\;\le\; 2\max_i \lVert x_i - y_i\rVert \;\le\; 2\delta,$$
because both ends of the pair realising the maximum move. That is, the two filtrations are
$2\delta$-interleaved, and the Cohen-Steiner theorem gives $d_B \le 2\delta$. The form without the two
would correspond to a filtration parametrised by **radius** (the Čech complex with balls of radius $r$),
not by diameter.

Checked on the circle, 400 points, Gaussian jitter, three calibration seeds, on $H_1$. **The random
stream is `rng_for(seed, "stability")`**, the same as in
`test_the_stability_theorem_holds_with_the_factor_two`: without specifying the stream the measurement is
not reproducible, and the first edition of this table diverged from the run for exactly this reason.

| $\sigma$ | $d_B / d_H$ | $d_B \le d_H$ | $d_B \le 2 d_H$ |
|---|---|---|---|
| 0.02 | **1.05** / 0.73 / 0.96 | **no** | yes |
| 0.05 | **1.14** / 0.97 / **1.07** | **no** | yes |
| 0.10 | **1.33** / **1.03** / **1.15** | **no** | yes |
| 0.15 | **1.24** / **1.03** / **1.29** | **no** | yes |

Nine of twelve combinations violate $d_B \le d_H$, the ratio reaches 1.33; none violates the $2 d_H$
bound (the maximum over $H_1$ is 1.33 against the admissible 2.0). The ratio grows with the noise level,
i.e. this is not an edge effect of small perturbations.

**A trap in writing the test.** In $H_0$ the bound $d_B \le d_H$ is never violated, because
zero-dimensional bars are all born at zero and bottleneck behaves more gently on them. A test written on
$H_0$ alone would pass and leave a wrong bound in the document; the failure is visible only in $H_1$.

---

### 4.2a. The stability curve is not computed on one circle

Result 6 is $d_B$ as a function of the noise level, and the object it is taken on decides what it claims
at all. A circle is not enough, and this follows from the history of the criterion in 1.2a: the old rule
lost $H_2$ on the torus and the Klein bottle and **invented** loops on the sphere, i.e. anisotropic and
compact objects erred in different directions. The statement "diagrams are stable under noise", backed
by a single circle with $\beta_1 = 1$, says nothing about $H_2$ — and $H_2$ is the most fragile part of
the pipeline.

The adopted set is **circle, torus, sphere**. Measured at 400 points, point jitter, mean over 10 repeats:

| Object | $H_1$ at $\sigma = 0 \ldots 0.10$ | $H_2$ at $\sigma = 0 \ldots 0.10$ |
|---|---|---|
| Circle | 0 → 0.357 | flat, about 0.02 |
| Sphere | almost flat, 0.036 → 0.054 | 0 → **0.299** |
| Torus | 0 → 0.231 | 0 → **0.121** |

A single circle would give a flat $H_2$ at about 0.02 and would say nothing about the second dimension;
the signal in $H_2$ is carried by the sphere and the torus. The converse is also true: the sphere has
$\beta_1 = 0$, and its $H_1$ barely moves — that is, each object is informative exactly in its own
dimension, and a set of one object is informative by half.

**The number of points here is not the same as for recovering Betti numbers.** The torus is set to 900
points so that its $H_2$ bar separates from noise; the curve measures something else — how far the
diagram moves, and this is defined for any bar, significant or not. At 900 points one diagram costs 25
seconds and the curve would take a quarter of an hour. The number of points used is written into the
artefact itself so that the numbers stay readable.

### 4.2. A null dataset measures the false positive rate rather than claiming zero

The wording "the null model has no significant long-lived bars" looks natural and is **impossible to
satisfy**. A random graph honestly has cycles; this is not an artefact of computation and not noise that
can be cut off by a threshold — it is the null distribution against which everything else is measured.

Measured on ER, $n = 24$, flag complex by the reference implementation, 5 realisations per density,
stream `rng_for(seed, "er_density", repeat)`.

**The spread over five realisations is given, not a "typical" value.** The first edition of the table
kept the first realisation of five under the name of typical, and that turned out to be a statistic that
flips with a change of stream: at $m = 180$ the first realisation gives $(1,0,0)$, while two of five have
$\beta_2 = 2$. The spread does not suffer from such a substitution and says exactly what the table is
needed for.

| $m$ edges | 12 | 24 | 36 | 60 | 100 | 180 | 220 |
|---|---|---|---|---|---|---|---|
| $p$ | 0.043 | 0.087 | 0.130 | 0.217 | 0.362 | 0.652 | 0.797 |
| $\beta_0$ | 12–14 | 4–5 | 1–3 | 1 | 1 | 1 | 1 |
| $\beta_1$ | 0–1 | 2–5 | 8–12 | 13–22 | 3–7 | 0 | 0 |
| $\beta_2$ | 0 | 0 | 0 | 0 | **3–6** | **0–2** | 0 |
| clean $(1,0,0)$ out of 5 | 0 | 0 | 0 | 0 | 0 | 3 | 5 |

From below the graph is scattered into components. Above the connectivity threshold
$p \approx \ln n / n = 0.13$ it is connected but carries 8 to 22 independent cycles; in the middle
$\beta_2$ appears as well. It becomes reliably clean only at $p \gtrsim 0.8$: at $p = 0.652$ three
realisations of five are already clean, but the other two keep $\beta_2 = 2$, i.e. the regime has not
yet left the region where features appear by themselves. The density of `cycle_lifecycle` (24 edges out
of 276) gives $\beta_0 = 4\ldots5$ and $\beta_1 = 2\ldots5$ — the working regime of the dataset lies deep
inside the region where cycles are always present.

**Hence the truth of the null dataset: there is no schedule, the event list is empty.** This is a
statement about the construction, it is checkable at the generator level and does not depend on what
persistence shows. The observed Betti curve is computed by the reference but **is not written into
`ground_truth.json`**: a measured curve under the name of truth is truth as pipeline output, exactly the
fork worked out for SBM in 3.1, and it is resolved the same way.

The false positive rate is a quantity that the null dataset allows one to **measure**, not a property it
has. It is measured by the event localisation algorithm, i.e. in Task 2, and published in Result 6.

**Consequence for the construction.** A Rips filtration on a point cloud has no density parameter at
all: the filtration goes through every threshold. Density makes sense only for a thresholded graph. So
the null models are split between the project's two pipelines:

| Null model | Pipeline | Density |
|---|---|---|
| Uniform cloud in a cube | family 1: cloud → Rips | undefined and unnecessary |
| Erdős–Rényi | family 2: snapshots → flag complex | equal to the density of `cycle_lifecycle` |

---

## 5. Known artefacts that are NOT bugs

- **The Rips complex of a circle as the scale grows** is successively homotopy equivalent to
  $S^1, S^3, S^5,\dots$ (Adamaszek–Adams). So at large scales spurious classes in odd dimensions
  naturally appear. This is expected; it does not need "fixing" — it is enough to limit the range of
  significance.
- **Betti numbers from a noisy sample are recovered only as persistent ones**, i.e. through the number
  of long-lived bars, not exactly. Requiring an exact match at $\sigma > 0$ is meaningless.
- **$H_0$ always contains one infinite bar.** When counting components it must be accounted for
  explicitly, otherwise $\beta_0$ comes out one smaller.
- **Double counting with a symmetric weight matrix** is a source of a silent error in density and in the
  fractions $p^\pm$: count over the upper triangle.

## References

- Edelsbrunner, Harer. Computational Topology: An Introduction.
- Carlsson, de Silva. Zigzag persistence.
- Cohen-Steiner, Edelsbrunner, Harer. Stability of persistence diagrams.
- Adamaszek, Adams. The Vietoris–Rips complexes of a circle.
- Otter et al. A roadmap for the computation of persistent homology.
