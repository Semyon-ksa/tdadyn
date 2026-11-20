# Zigzag persistence: formalisation and contract (Tasks 1–2)

This document fixes: the construction of the zigzag from family-2 snapshots, the invariant and
the ground for its uniqueness, the semantics of events, the translation of a barcode into an
event log, and an independent way of checking it. It does **not** describe the generators: their
truth lives in `method.md` and `data.md`, and Task 2 is their reader, not a co-author. No edits to
`method.md` are made from this branch.

A field that is absent here: multiparameter persistence (Result 3). It shares with this document
only the section on the invariant; the (w, τ) bifiltration is described separately.

---

### 0.1 Protocol for changing `method.md`

Task 2 does not edit `method.md`: the truth of the data lives there, and Task 2 is its consumer.
If the algorithm gives 27↔41 while §2.2a says 12↔41, the algorithm is fixed. The mechanics are the
same as with the table of Betti numbers: a document that the consumer may rewrite stops being a
check for it.

The membership test: **will this entry change the contents of `data/`?** No — then the entry goes
into `zigzag.md`. The union construction, the translation of a bar into an event, the oracle —
these describe the consumer, not the data.

There is one exception: a measurement **on the reference** has shown a defect in the data
themselves. Then the order is reference → generator → `method.md` → truth log → rebuild of the
dataset, **in a separate Task 5 commit**. The trigger is the reference, not the output of the
algorithm.

---

## 1. Input

### 1.1 Snapshots

The input is a family-2 dataset (`dynamic_graph/`): a sequence of snapshots
$G_0, \dots, G_{T-1}$ on a fixed vertex set $V$, $|V| = n$, plus a truth log.

The edge rule is taken from the dataset's `edge_rule` field and applied by **a single
implementation**, `core.contracts.adjacency_from_snapshot`:

| `edge_rule` | rule | strictness | datasets |
|---|---|---|---|
| `weight_above` | edge when $w_{ij} > t$ | strict | `cycle_lifecycle`, SBM, connectome |
| `distance_within` | edge when $d_{ij} \le \varepsilon$ | non-strict | `geometric_loop` |

The strictness is not sloppy notation (`method.md` §2.3): at $t = 0$ a non-strict comparison would
make the graph complete, and $\varepsilon$ is an inclusion radius, not an exceedance threshold.

Task 2 **has no right** to implement this rule a second time. The field was introduced in S7
precisely for this; a second implementation is a silent error in which the metric dataset is read
as the complement of the graph, and inside the package every test passes.

### 1.2 The complex on a snapshot

$K_i = \mathrm{Fl}_{\le k_{\max}+1}(G_i)$ — the flag complex truncated at dimension $k_{\max}+1$.
Truncating one above the requested homology dimension is mandatory: without $\partial_{k+1}$,
$k$-dimensional cycles are systematically overcounted (S2). The truncation is the same on snapshot
and intermediate indices; it is functorial with respect to inclusions, so the diagram stays
correct.

**Where the flag complex builder lives.** It stays in `reference/` and is split into two
responsibilities, as the homology engine and the complex builder were split in S2:

| Responsibility | Where | Input |
|---|---|---|
| Thresholding: snapshot → adjacency matrix | `core.contracts.adjacency_from_snapshot` | weights/distances + `edge_rule` + threshold |
| Flag complex: adjacency → list of simplices | `reference.flag_complex` | **an already built graph** |

A signature of the form `flag_complex(weights, threshold)` is to be removed, not marked
deprecated: it contains a second implementation of the edge rule, hard-wired in the similarity
direction, and on the metric dataset it builds the complex of the **complement** of the graph.

**Done in Z0.** The `threshold` parameter was removed from `flag_complex` and from
`betti_numbers_of_graph`; the input is a 0/1 matrix, and a non-binary matrix is **rejected**
rather than thresholded by default: the only way to get one here is to skip the thresholding step.
The strictness of the comparison is checked where it now lives — on `adjacency_from_snapshot`. No
hand-written binarisations remained in the project after this: a third one lived in the family-1
cross-check (`tests/test_static_betti.py`) and was also moved onto the common implementation.

---

## 2. The zigzag

### 2.1 Construction and indices

$$K_0 \hookrightarrow K_0 \cup K_1 \hookleftarrow K_1 \hookrightarrow K_1 \cup K_2
\hookleftarrow \cdots \hookleftarrow K_{T-1}$$

The index set is $I = \{0, \tfrac12, 1, \tfrac32, \dots, T-1\}$; half-integer indices are
intermediate objects: $K_{i+1/2} := K_i \cup K_{i+1}$. Below, a "snapshot index" is an integer
one, an "intermediate" index a half-integer one.

### 2.2 The union of complexes, not the flag complex of the union of graphs

This is a fork, not a detail of notation. It is always true that
$$\mathrm{Fl}(G_i) \cap \mathrm{Fl}(G_j) = \mathrm{Fl}(G_i \cap G_j),$$
but for the union only the inclusion holds,
$$\mathrm{Fl}(G_i) \cup \mathrm{Fl}(G_j) \subseteq \mathrm{Fl}(G_i \cup G_j),$$
and it is strict in general: three edges lying in different snapshots generate a triangle in
$\mathrm{Fl}(G_i \cup G_j)$ and do not generate it in the union of complexes.

**Adopted: $K_{i+1/2} = K_i \cup K_{i+1}$ (simplexwise union).** Reason: the intermediate object
must not contain a simplex that exists in no snapshot. Otherwise a triangle present in neither
$G_i$ nor $G_{i+1}$ kills a cycle at an intermediate index, and a "birth–death" pair appears in the
barcode that corresponds to no state of the data. A test on such a barcode will pass: it is
self-consistent.

The two constructions coincide on monotone transitions (additions only or removals only). Where
there is monotonicity and where there is not is determined by the geometry of the datasets, not by
general words:

- **`cycle_lifecycle`** — every transition adds or removes one edge, so all transitions are
  monotone and the constructions coincide identically.
- **`geometric_loop`, radius breathing** (snapshots 4, 43, 56) — the points scale uniformly, every
  pairwise distance is multiplied by $R(t')/R(t)$, so the edge sets are nested, and these
  transitions are monotone too, even though the graph changes substantially ($72 \to 48$,
  $45 \to 66$, $48 \to 72$ edges).
- **`geometric_loop`, change of $\delta$** (transitions $19 \to 20$ and $44 \to 45$) — the angular
  positions $\theta_j = j(2\pi-\delta)/(n-1)$ change **all at once**, so edges can appear and
  disappear simultaneously.

**Measured in Z1: exactly one of them is non-monotone, not two.** An earlier edition called both
$\delta$ transitions risky; the measurement refutes this:

| Transition | Edges | Disappeared | Appeared | Result |
|---|---|---|---|---|
| $19 \to 20$, gap opens | 48 → 45 | 3 | 0 | **nested, monotone** |
| $44 \to 45$, gap closes | 66 → 48 | 21 | 3 | non-monotone |

At $19 \to 20$ exactly three chords across the gap disappear — $(0,22)$, $(0,23)$, $(1,23)$ — and
nothing appears. The cause of the asymmetry is not the change of $\delta$ as such but on which
side of $\varepsilon$ the **third-neighbour** chord stands at the current radius: at $R = 1.258$ it
equals 0.818 and fails $\varepsilon = 0.8$ in both the open and the closed configuration, while at
$R = 1.203$ it passes in the open one and is lost on closing (21 edges of the form $(j, j{+}3)$).

Consequence for Result 2: both events of the metric dataset still fall on transitions 20 and 45,
but **the death of $H_1$ at 20 happens on a monotone transition**, where the constructions coincide
identically and the choice of union rule cannot influence it in principle. Only the localisation
of the birth at 45 remained in question.

**The difference was measured, and it is zero.** Over all 59 transitions of both datasets
$\mathrm{Fl}(G_i \cup G_{i+1}) \setminus (K_i \cup K_{i+1}) = \varnothing$ — with truncation at
dimension 2 and at dimension 3. The check is not hollow: a negative control (the three edges of a
triangle spread over two snapshots) gives exactly the missing triangle $\{(0,1,2)\}$, i.e. it can
tell the constructions apart.

This is a **measurement**, not a claim that the choice is irrelevant. `UNION_RULE` remains the
simplexwise union for the reason given above, which does not depend on the measurement: the
intermediate object must not contain a simplex absent from every snapshot. A correct conclusion
resting on the zero difference would be withdrawn together with it at the first dataset with
different geometry.

`UNION_RULE` is **a module constant, not a config key**: the invariant depends on it, and by the S6
boundary only what can be changed without rewriting the proof goes into the config. The second
construction is implemented only inside the test that measures it.

### 2.3 The elementary sequence of operations

Backends take not a diagram of complexes but a stream of insertions and removals of single
simplices. The unfolding:

- transition $K_i \to K_{i+1/2}$: insert the simplices of $K_{i+1/2} \setminus K_i$ **in increasing
  dimension**;
- transition $K_{i+1/2} \to K_{i+1}$: remove the simplices of $K_{i+1/2} \setminus K_{i+1}$ **in
  decreasing dimension**.

Otherwise the intermediate states stop being simplicial complexes. The order within one dimension
is arbitrary and does not affect the coarse barcode, but it **does affect ephemeral bars** (birth
and death within one transition). Ephemeral bars collapse in the coarse indexing into an interval
of zero length; they are discarded, but their **number is recorded** — the growth of their number
on noisy datasets is a substantive measurement, not implementation noise.

**Measured in Z3, and the number must be published decomposed.** Ephemeral bars are invisible to
the coarse path by construction — no index of the diagram catches them, so the oracle cannot output
them in principle. They are visible only on the fine path, where the backend is given one time per
operation:

| Dataset | $H_0$: prologue + outside | $H_1$: prologue + outside |
|---|---|---|
| `cycle_lifecycle` | 23 + **0** | 0 + **0** |
| `geometric_loop` | 23 + **0** | 48 + **94** |

The prologue part measures **the assembly order of $K_0$**, not the data: building a 24-cycle from
the empty complex gives birth to 24 components and kills 23 of them, and this follows from the
lexicographic order fixed above, which is arbitrary in substance. Publishing the sum as a
characteristic of the dataset would pass this order off as a property of the data; the second
column is the substantive one.

**The Z5 measurement confirms this: the prologue is constant, all the signal is in the second
half.**

| Base, mechanism | 0.0 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|
| `cycle_lifecycle`, `spurious`: prologue | 23 | 23 | 25 | 28 |
| — outside the prologue | **0** | **54** | **334** | **857** |
| `geometric_loop`, `spurious`: prologue | 71 | 75 | 80 | 87 |
| — outside the prologue | **94** | **253** | **566** | **1567** |

The prologue part drifts by units, the second grows by orders of magnitude. At zero noise the sum
consists entirely of the prologue, so without the decomposition the zero reference point would be
false. For `cycle_lifecycle` under `missing_edges` there are zero ephemeral bars outside the
prologue at every level: there are no triangles there, and nothing can be born inside a transition.

The order within a dimension is **fixed lexicographically** (Z1), not left to the set: Z5 compares
the barcode at zero noise with the base bitwise, and a non-deterministic order would make that
comparison meaningless. The arbitrariness of the choice is kept nonetheless — one value is fixed,
not derived as the only correct one.

The invariant of the layer: replaying the operation stream from scratch gives bitwise the same
$K_i$ and $K_{i+1/2}$ as direct construction. This is the test that catches unfolding errors before
they become barcode errors. Its second half is the check that **every prefix of the stream is a
simplicial complex**: it is done directly, by building the complex at every prefix, since at 448
operations it takes a fraction of a second.

**Prologue.** The assembly of $K_0$ from the empty complex also enters the stream (47 and 168
insertions) rather than being taken as an initial state: left censoring of a bar starting at index
0 (§5.1a) is visible only to a layer that saw the first complex being assembled.

### 2.4 Index convention

From `method.md` §2.2: `snapshot: k` means that $G_k$ **already contains** the change and
$G_{k-1}$ does not. A bar interval is written in the coarse indices $I$ as a closed $[b,d]$.

An off-by-one discrepancy here looks in the report like a systematic localisation error of the
algorithm and is indistinguishable from one. So the translation "operation counter → coarse index"
is a separate layer with its own tests on toy examples (§5.1), not a line in the adapter.

**Z1 measurement: the operation counter does not determine the coarse index, and this is not an
edge case.** The overwhelming majority of family-2 transitions are **empty** — the graph does not
change, no simplices are added or removed — so several neighbouring indices share one state and one
counter, and the backend does not see such a transition at all: there is no operation.

| Dataset | Empty transitions | Distinct counters over 119 checkpoints | Longest group of indices sharing a state |
|---|---|---|---|
| `cycle_lifecycle` | 113 of 119 (95%) | 6 | 30 |
| `geometric_loop` | 112 of 119 (94%) | 7 | 45 |

So the layer translates a counter not into an index but into a **closed range** of indices with
that state (`CoarseSpan`), and a consumer that needs a single index must name the end: the left one
for a birth, the right one for a death. A function returning one index would choose the end
silently, and a wrong choice shifts a bar **by up to 45 indices** — an order of magnitude further
than the off-by-one this section warns about, and just as indistinguishable from a defect of the
algorithm.

The behaviour of the counter **inside** a non-empty block is fixed separately: there the state is
intermediate and equal to no $K$, so the range ends at the last index already reached. This keeps
the mapping everywhere defined and monotone.

---

## 3. The invariant and its uniqueness

### 3.1 Decomposition

The zigzag module $V: I \to \mathrm{Vect}_{\mathbb{Z}/2}$ is a representation of a quiver of type
$A_N$, $N = 2T-1$, with arbitrary orientation of the arrows. By Gabriel's theorem the
indecomposable representations of $A_N$ are exactly the interval modules $\mathbb{I}[b,d]$; by
Krull–Schmidt the multiplicities are uniquely determined. Hence the barcode
$$V \cong \bigoplus_{[b,d]} \mathbb{I}[b,d]^{\oplus m_{[b,d]}}$$
is well defined and does not depend on the algorithm. The orientation of the arrows does not affect
the list of indecomposables — only which intervals arise.

References: Carlsson–de Silva (zigzag persistence), Carlsson–de Silva–Morozov (zigzag diamond),
Gabriel.

### 3.2 Scope of the seniority rule

The seniority rule of `method.md` §2.2 is stated for the death of $H_0$ when components merge, and
in that domain it **holds**: two classes compete for one death, the inclusions are injective, and
the decomposition of the module makes the elder survive. In `cycle_lifecycle` this is checkable —
at 55 the bar born at 48 dies, not the bar running from zero.

It **does not carry over** to the death of $H_1$ when an edge is removed: there are no two classes
competing for one death, and the pairing is determined by the decomposition, not by age. The
reference case is `cycle_lifecycle`, where the death at 41 is paired with the birth at 12, not at
27 (`method.md` §2.2a). The intuition "the chord made two cycles, break one and the younger dies"
is wrong; the answer depends on which edge is removed.

**The algorithm must reproduce the 12↔41 pair. If it gives 27↔41, the error is in it, not in the
log.**

### 3.3 Generalised rank and Möbius inversion — a computable oracle

For $b \le d$ in $I$ define the restriction $V|_{[b,d]}$ and
$$r(b,d) := \mathrm{rank}\bigl(\varinjlim V|_{[b,d]} \;\longleftarrow\; \varprojlim V|_{[b,d]}\bigr)$$
— the rank of the canonical map from the limit to the colimit (the generalised rank invariant,
Kim–Mémoli). Since every module over $A_N$ is interval-decomposable,
$$r(b,d) = \#\{\text{bars} \supseteq [b,d]\},$$
and the multiplicities are recovered by Möbius inversion on intervals:
$$m_{[b,d]} = r(b,d) - r(b^-,d) - r(b,d^+) + r(b^-,d^+),$$
where $b^-$ and $d^+$ are the neighbouring indices on the left and right ($r := 0$ beyond the
bounds of $I$).

Over $\mathbb{Z}/2$ the limit is the subspace of compatible tuples in $\bigoplus V_i$, the colimit
a quotient; both constructions come down to ranks of matrices. This gives the barcode **without a
single persistence library** and without implementing a zigzag algorithm — i.e. a genuinely
independent oracle, not a second copy of the code under test.

### 3.4 Why coverings are not enough

It is tempting to check the algorithm by the number of bars living at each index and the number of
bars crossing each arrow. Both numbers are necessary and together **insufficient**:

$$\{[1,4],\,[2,3]\} \quad\text{and}\quad \{[1,3],\,[2,4]\}$$

give the same vertex covering $(1,2,2,1)$ and the same edge covering $(1,2,1)$, but they are
different barcodes. The vertex covering fixes the number of births and deaths at each point, but
not the pairing of births with deaths.

This is exactly the same class of error as the Betti curve's blindness to pairing in S6, only one
level up. The L3 check (§5.1) is mandatory; L1 and L2 are fast filters, not the criterion.

---

## 4. Semantics of events

### 4.1 Types

`Event.type` takes only `birth` and `death` — as in family 2. Merging and splitting of classes do
not exist as operations in a barcode; the intuitive description goes into `narrative`. A merge of
two components looks like the death of one bar while the other survives, and **which one dies is
determined by the decomposition**, not by the order of appearance (§3.2).

### 4.2 Barcode → event log

For a bar $[b,d]$ in dimension $k$:

| condition | event |
|---|---|
| the support of the bar contains an integer index, $b > 0$ | `birth` at **the first snapshot index** $\ge b$ |
| the support of the bar contains an integer index, $d < T-1$ | `death` at **the first snapshot index** $> d$ |
| $d = T-1$ | right censoring, no event is generated |
| $b = 0$ | left censoring, the `birth` is marked `censored: true` |
| the support contains no integer index | invisible bar (§4.3), generates no events |

An earlier edition of the table distinguished integer and half-integer ends and sent the latter to
`birth_in_transition` / `death_in_transition`. **The Z2 measurement showed that the log is not
recovered this way**: in `cycle_lifecycle` both $H_1$ bars have half-integer ends ($[11½,40½]$,
$[26½,59]$), and the rule would have given "in transition" where the truth log records an ordinary
birth at 12 and death at 41. The rule above was checked on **all seven** events of both family-2
datasets and agrees everywhere:

| Bar | birth | death | Truth log |
|---|---|---|---|
| `cl` $H_1$ $[11½, 40½]$ | 12 | 41 | 12 / 41 |
| `cl` $H_1$ $[26½, 59]$ | 27 | censored | 27 |
| `cl` $H_0$ $[48, 54]$ | 48 | 55 | 48 / 55 |
| `gl` $H_1$ $[0, 19½]$ | censored | 20 | 20 |
| `gl` $H_1$ $[44½, 59]$ | 45 | censored | 45 |

The shift $d \mapsto d+1$ for a death at integer $d$ is a special case of "the first snapshot index
$> d$", not a separate rule. It is still the only place where the shift is introduced, and it is
tested separately.

**Half-integer ends are not rounded — and that is not the same as the rule above.** The prohibition
was introduced for bars whose **whole** support is half-integer: such a class exists in no snapshot,
and assigning it a snapshot time would invent an event. Such bars are counted and recorded (§4.3)
but generate no events. For a bar with a long integer support only the end is half-integer, and
"the first snapshot index $\ge b$" does not round it but names the snapshot in which the class is
first visible in the data.

### 4.3 Invisible bars

A bar whose support contains no integer index corresponds to no state of the data. Such bars are
**counted and recorded**, not silently discarded: in the metric of §6 they are false positives by
construction, and their number is a characteristic of the union construction that must be known
before transferring to connectomes.

**Measured in Z5.** On clean datasets there are none (Z3); under noise they appear:

| Base, mechanism | 0.0 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|
| `cycle_lifecycle`, `missing` | 0 | 0 | 19 | **36** |
| `cycle_lifecycle`, `spurious` | 0 | 6 | 8 | 4 |
| `geometric_loop`, `missing` | 0 | 0 | 0 | 3 |
| `geometric_loop`, `spurious` | 0 | 0 | 0 | 0 |

Before Z5 the quantity was known only where it equals zero, i.e. it had never been measured.

---

## 5. Verification

### 5.1 Three levels and layers

`reference/` already contains a layer that computes ranks of boundary matrices over $\mathbb{Z}/2$
without persistence libraries. The zigzag oracle is built **in that layer**, not in `zigzag/`.

| level | what is computed | status |
|---|---|---|
| L1 | $\beta_k(K_i)$ and $\beta_k(K_{i+1/2})$ at every index | necessary, insufficient |
| L2 | ranks of the induced maps on each arrow (number of bars across the arrow) | necessary, insufficient (§3.4) |
| L3 | $r(b,d)$ + Möbius inversion → the full barcode | sufficient |

Toy examples with a barcode computed by hand are a mandatory input to L3 before any run on the
datasets:

- **T1.** Two vertices, the edge present only at snapshot 1. $H_0$: covering $(2,1,1,1,2)$ over
  indices $0,\tfrac12,1,\tfrac32,2$; barcode $\{[0,2], [0,0], [2,2]\}$.
- **T2.** The boundary of a triangle appears and disappears: one bar in $H_1$ with known ends.
- **T3.** A theta graph with an edge removed — a minimal version of the fork of §3.2; the pairing
  is computed in three lines.
- **T4.** The module $\{[1,4],[2,3]\}$ assembled by hand: the oracle must tell it apart from
  $\{[1,3],[2,4]\}$. This is a test **of the oracle's discriminating power**, not of the data.

### 5.1a Predicted barcodes of family 2

Derived from `method.md` §2.1, §2.1a, §2.2a — i.e. **before** running the algorithm. The oracle must
reproduce them; if the oracle gives something else, the oracle is checked first.

**An earlier edition of this section was wrong by half a step, and this was corrected by
measurement (Z2).** Bars were written in snapshot indices, while the barcode lives on $I$, where
unions stand between the snapshots. Below are the measured supports; the Betti curve over all 119
indices was computed by the independent `reference` layer, the barcode by the oracle.

**`cycle_lifecycle`** ($T = 60$; events: birth 12 in $H_1$, birth 27 in $H_1$, death 41 in $H_1$,
birth 48 in $H_0$, death 55 in $H_0$):

| dim | bars | comment |
|---|---|---|
| $H_0$ | $[0,59]$, $[48,54]$ | the second is the component $\{4,5,6\}$ detached by removing $(3,4)$ |
| $H_1$ | $[11\tfrac12,\,40\tfrac12]$, $[26\tfrac12,\,59]$ | pairing from §2.2a: the first bar ends at $40\tfrac12$ rather than running to 59 |

**Why the $H_1$ ends are half-integer and the $H_0$ ends integer.** Closing the path at snapshot 12
is an **addition** of an edge, so $K_{11½} = K_{11}\cup K_{12} = K_{12}$, and the class already
exists at $11½$. Removing $(6,7)$ at 41 is a **removal**, so $K_{40½} = K_{40}$, and the dying class
survives to $40½$. For $H_0$ the signs are opposite: the birth comes from removing an edge
(detaching $\{4,5,6\}$ at 48), the death from adding one (restoring $(3,4)$ at 55), and in both cases
the union coincides with the *previous* snapshot, not the next. Hence the integer ends $[48,54]$.

That is, a half-integer end is neither an error nor a choice but a consequence of the direction in
which the transition is monotone. This also explains why §2.2 was worth measuring: monotonicity of
the transitions is determined there.

Five events and four bars agree: three births (12, 27, 48) are visible, the birth of the bar
$[0,59]$ is left-censored, the deaths of the bars $[26½,59]$ and $[0,59]$ are right-censored.

**`geometric_loop`** ($T = 60$; $\beta$: $(1,1,0)$ on 0–19, $(1,0,0)$ on 20–44, $(1,1,0)$ on
45–59):

| dim | bars | comment |
|---|---|---|
| $H_0$ | $[0,59]$ | the graph is connected on every snapshot |
| $H_1$ | $[0,\,19\tfrac12]$, $[44\tfrac12,\,59]$ | left-censored and right-censored respectively |

The second $H_1$ bar **cannot** be a continuation of the first: there is no class between 20 and
44, and intervals are connected. This is the substantive difference between "the loop came back" in
the language of graphs and two different classes in the language of homology.

**The Betti curve does not distinguish pairing, and on `cycle_lifecycle` this is directly visible.**
The ends of the $H_1$ bars form the sets $\{11½, 26½\}$ and $\{40½, 59\}$, and **both** layouts —
$\{[11½,40½],[26½,59]\}$ and $\{[11½,59],[26½,40½]\}$ — give the same curve and the same arrow
ranks. L3 decides. This is §3.4 not on a constructed example but on the project's data.

Both datasets have one infinite (two-sidedly censored) bar in $H_0$ (`method.md` §5). It is counted
explicitly; thrown away as "trivial", it makes $\beta_0$ one smaller.

### 5.2 Algebraic check of the pairing

The 12↔41 pair is checked by computation, not by comparison with the document: the vector of the
24-cycle **does not lie** in $Z_1(G_{41})$, while $c_2$ does. A test on the identity of a class must
not be written — the basis of $H_1$ of the theta graph is not canonical.

The expectation in the test is **computed**. Transcribing expectations from `method.md` checks that
the code matches the document, not the document itself (the S6–S9 rule).

### 5.3 Agreement of backends

Dionysus 2 (`zigzag_homology_persistence`) is compared with the oracle **bar for bar**, not by the
number of bars. A discrepancy is considered a defect of the adapter (index translation, order of
operations) until proven otherwise. The adapter is validated first on T1–T4, where the answer is
known by hand, and only then on the datasets.

**The agreement is two-sided, and no third party is needed.** The check here is the oracle: it
computes the barcode independently — through the generalised rank and Möbius inversion — i.e. by a
different method, not a different implementation of the same method. A second backend
(fast-zigzag) repeats the same class of algorithm, and when it disagrees with the first it cannot
say which is right; it adds a vote, not an independent computation. Its value is speed on 60
snapshots, and speed is not part of the criterion of Result 2.

So fzz is optional: it is not on PyPI, it is built from source only if Dionysus does not fit the
time budget, and then that is separate work with its own cost.

**Measured in Z3: the agreement is complete, zero discrepancies.** The toys T1–T3 and both family-2
datasets agreed in both dimensions, bar for bar with multiplicities. **Z6 extended the comparison to
all 18 datasets, noisy ones included: 3192 bars, 0 discrepancies.** Checking agreement only where the
barcode has four bars is the weakest version of the claim; an adapter error is more likely to show
up where a dataset gives nine hundred. T3 plays the role of T4 for the backend here: it has two
layouts with identical vertex and arrow coverings, and Dionysus picks the right one, i.e. it passes
the fork of §3.4. T4 itself is unavailable to the adapter — it is a module assembled from matrices,
with no complexes behind it.

The translation convention, checked first on T1, where the barcode is written out by hand: position
$p$ goes as time $p$; a simplex present at positions $[a,b]$ is passed as $[a,\,b{+}1)$, one interval
per run of presence; a bar comes back half-open and becomes the closed $[b,\,d-1]$, an infinite death
becomes the last position. A negative control stands next to it: a convention shifted by one gives a
discrepancy, i.e. the comparison does not compare something with itself.

**The backend returns dimension $k_{\max}+1$, and it must not be published.** On `geometric_loop`
Dionysus outputs **69 bars in $H_2$** for complexes truncated at dimension 2: there is no
$\partial_3$, and every closed 2-chain is counted as a cavity (`method.md` §2.3, consequence 2).

That these are artefacts is not a claim about the absence of homology nobody computed. The generator
built its complexes one dimension higher (`topology_check.max_homology_dim` is 2, i.e. up to
dimension 3, with $\partial_3$) and recorded $\beta_2 = 0$ in `betti_curve` on every snapshot of both
datasets. The 69 bars contradict a number that **was** computed correctly — that is what sets them
apart from a finding.

The adapter cuts them off itself rather than leaving it to the caller: `cycle_lifecycle` has no
triangles at all and produces not a single such bar, so a check on one dataset would not have caught
them.

**Cost.** 16 ms per dataset, 0.30 s for all 19 dynamic datasets — against seconds for the oracle.
Speed is not a problem with the backend, which once more confirms that a second backend is not
needed.

### 5.4 Negative controls

- `geometric_loop`, snapshots **4, 43, 56**: the graph changes, $\beta$ does not. The algorithm must
  not generate events on these transitions. The test is written against them **by name**.
- `cycle_lifecycle`, the repeated state at 55: the state coincides with snapshots 41–47. The
  pipeline must not collapse snapshots by graph identity. Do not "fix" this when refactoring
  deduplication.
- Null datasets (`null/`): not a check for zero but a measurement of the false positive rate (§6).

### 5.5 Invariance

Relabelling the vertices must give the same barcode (in family 2 the relabelling is not applied to
the data, so it is checked by a test). The barcode does not depend on the order of simplices of one
dimension within a transition — up to ephemeral bars (§2.3).

---

## 6. Metrics (the criterion of Result 2)

Matching the predicted log against the true one:

- matching only within a (`type`, `dim`) pair; different dimensions and types are not matched;
- a tolerance $\tau$ in snapshot index, $\tau = 0$ by default; matching is greedy in increasing
  $|\Delta t|$, ties broken by increasing time of the true event;
- reported quantities: recall, precision, mean and maximum localisation error $|\Delta t|$,
  separately by dimension;
- as separate rows: the number of invisible bars, the number of ephemeral bars, the number of events
  **derived from a half-integer end of a bar**, the number of censored bars.

**Measured in Z4.** On the clean datasets recall and precision equal one at $\tau = 0$:

| Dataset | Predicted | In truth | recall | precision | $H_0$ | $H_1$ |
|---|---|---|---|---|---|---|
| `cycle_lifecycle` | 5 | 5 | 1.0 | 1.0 | 2/2 | 3/3 |
| `geometric_loop` | 2 | 2 | 1.0 | 1.0 | — | 2/2 |

This is **an expected outcome, not an achievement**: the datasets are constructed, the algorithm is
exact. The value of the metric lies in the fact that under noise it stops being one — a spot
measurement gives precision 4/120 on `spurious_edges` at level 0.02 — and the report, not the reader,
must say so.

**The event log is blind to pairing, and Result 2 must say so.** The two $H_1$ layouts of
`cycle_lifecycle` give **the same** log: birth 12, birth 27, death 41. An event does not remember
which bar it was derived from, so recall and precision **do not check the 12↔41 pair**; the barcode
comparison checks it — algebraically in Z2, backend against oracle in Z3. This is the same blindness
as the Betti curve in S6 and the coverings in §3.4, one level further: curve → coverings → **event
log**.

**The localisation error at $\tau = 0$ is identically zero**: only an exact coincidence can match.
It is published because the list above requires it, but it is not an informative quantity.
Measured: on eight noisy datasets raising $\tau$ from 0 to 1 recovers one event in **one** of them,
i.e. under noise the failure is omission and a flood of false events, not displacement. The default
$\tau = 0$ stays.

**Censored events are marked and do not enter the metrics.** A censored birth does not claim that a
class was born — it says that the class already existed when the window opened. Counting it as a
false positive would penalise the algorithm for the finiteness of observation; counting it as a hit
would demand from the truth log a record that is deliberately absent there.

**The row "events at half-integer indices" was reread.** After the revision of §4.2 there are no
events at half-integer indices at all: the rule moves them to snapshot ones. What is counted is what
makes sense — how many events are **derived** from a half-integer end of a bar, and that is 3 of 5
for `cycle_lifecycle` and 2 of 2 for `geometric_loop`. A diagnostic of the union construction, not a
defect.

Measured report rows: censored ends 3 and 4, invisible bars **0 and 0**, ephemeral — see §2.3,
decomposed into the prologue and the rest.

On the null datasets the reported quantity is the false positive rate **per snapshot** with a
confidence interval (Wilson, or the rule of three at zero misses). Three seeds that gave zero mean an
upper bound of 63%, not zero — the S8 lesson.

**Measured in Z5, and the answer is not "rarely wrong".** 60 ER realisations (3 calibration seeds ×
20) at the density of `cycle_lifecycle`, $p = 0.0833$:

| Denominator | Rate | 95% Wilson | Trials |
|---|---|---|---|
| per snapshot | **0.9833** | [0.9786, 0.9870] | 3540 of 3600 |
| per realisation | 1.0000 | [0.9398, 1.0000] | 60 of 60 |
| events per snapshot | **12.9** | — | 46 323 events ($H_0$ 25 710, $H_1$ 20 613) |

**Every snapshot of the null model is a topological event.** This is not a defect of the algorithm:
`method.md` §4.2 says plainly that a random graph at this density honestly has
$\beta_0 = 4\ldots5$ and $\beta_1 = 2\ldots5$ on every snapshot, and the edges are redrawn
independently, so classes are born and die continuously.

**Both denominators** are published, because neither on its own is honest. Per snapshot is what the
list above requires, but the trials there are not independent: neighbouring complexes share edges,
and the interval is narrower than the true one. Per realisation the trials are independent, but there
are sixty of them, and the interval is wide. The number of events per snapshot goes as a third row:
the rate of 0.98 is saturated and says nothing about scale.

**The threshold `null_models.max_false_positive_rate` does not apply here.** It was calibrated for
the significance criterion on family-1 clouds (`method.md` §1.2a) — the fraction of realisations in
which the criterion invented a feature, 1.5% with a bound of 4.3%. The zigzag measures the fraction
of snapshots that fire. Comparing them would declare failure while measuring different quantities.

**Degradation under noise** (16 dynamic datasets in `noisy/`; the other eight are built on a cloud
and have no snapshots):

| Base, mechanism | 0.0 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|
| `cycle_lifecycle`, `spurious`: recall | 1.00 | 0.80 | 0.60 | 0.60 |
| — precision | 1.00 | 0.0078 | 0.0027 | **0.0017** |
| `geometric_loop`, `spurious`: recall | 1.00 | **1.00** | **1.00** | **1.00** |
| — precision | 1.00 | 0.0051 | 0.0026 | 0.0021 |

`geometric_loop` keeps recall at one at every level — both of its events are always found, only the
precision deteriorates; `cycle_lifecycle` loses events. `spurious_edges` is an order of magnitude
more harmful than `missing_edges`. Level 0.0 reproduces the clean dataset in all four combinations.

**There is no localisation error column in the table** for the reason worked out above: at $\tau = 0$
it is identically zero, and a column of zeros would read as a result.

---

## 7. Limits of applicability

- **The barcode depends on the union convention (§2.2).** Barcodes obtained under different
  conventions are not comparable; the convention is stated on publication.
- **Truncation of the complex.** Bars of dimension $k_{\max}$ are reliable; dimension $k_{\max}+1$ is
  not computed, and its absence from the output is not a claim of its absence in the data (the case
  of $\beta_3$ for $\mathrm{VR}(S^1)$ from S7).
- **The metric family.** $\mathrm{VR}(S^1; r)$ does not become contractible as the scale grows but is
  successively equivalent to $S^1, S^3, S^5, \dots$ (Adamaszek–Adams, threshold
  $\varepsilon < R\sqrt3$). The regime of $\varepsilon$ is a qualitative choice fixed in the
  generator; Task 2 inherits it and does not redefine it.
- **The `topology_check` parameters (0.07, 1.55) do not carry over here.** They were calibrated on
  family-1 clouds, and, moreover, the significance criterion is not invariant under reparametrisation
  of the filtration. In the zigzag the question of bar significance is posed anew and is not solved
  in this branch.
- **Complexity.** 60 snapshots of dense complexes on 24 nodes — the L3 oracle is quadratic in the
  number of indices; on larger datasets it remains a spot check, not a regression test.

---

## 8. References

1. G. Carlsson, V. de Silva. Zigzag persistence.
2. G. Carlsson, V. de Silva, D. Morozov. Zigzag persistent homology and real-valued functions.
3. P. Gabriel. Unzerlegbare Darstellungen I.
4. W. Kim, F. Mémoli. Generalized persistence diagrams for persistence modules over posets.
5. M. Botnan, M. Lesnick. Algebraic stability of zigzag persistence modules.
6. T. K. Dey, T. Hou. Fast computation of zigzag persistence.
7. M. Adamaszek, H. Adams. The Vietoris–Rips complexes of a circle.
8. A. Tausz, G. Carlsson. Applications of zigzag persistence to topological data analysis.
