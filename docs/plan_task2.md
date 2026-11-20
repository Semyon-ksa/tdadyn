# Plan of Task 2 by session (Z0–Z6)

Task 2 of the first stage: computational methods for building "temporal persistence diagrams".
The criterion from the proposal is correct detection of events known in advance (birth, merging,
disappearance of cycles) on synthetic graphs with controlled dynamics; the metrics are event
recall/precision and the localisation error in time.

Placement: `src/tdadyn/zigzag/` next to `src/tdadyn/synthetic/`, one package. Specification —
`docs/zigzag.md`.

**The premise on which this decision stood was refuted by measurement; the conclusion survived but
now rests on something else.** The only real argument against a single package was the cost of the
zigzag dependencies: a build from source and Boost. For Dionysus this turned out to be false — Z0
measured that 2.2.3 installs **as a prebuilt wheel**, Boost is not needed, and `pip install`
finishes in a clean venv in seconds. Only fast-zigzag (fzz) is expensive from source, and it is not
on PyPI at all.

The conclusion does not change — optional dependencies, import inside the function, a fast target
without `[zigzag]` — but it is now written in terms of fzz, not Dionysus. A correct conclusion on a
false premise is more dangerous than a wrong one: at the next review someone checks the premise,
sees that Boost is not needed, and removes a restriction that is right for a different reason.

The common property of all the forks below is the same as in S6–S9: **a wrong choice does not break
a test**, it gives a plausible barcode with wrong truth. Hence the order of the sessions: first the
independent oracle, then the backend, and never the other way round.

---

## Z0 — Foundation. Not a line of algorithm

**What to do.**

1. Tag the current state (`synthetic-v1`). The report and the paper claim reproducibility; the claim
   makes sense only as a pointer to a fixed commit, and this will also be the reference in the
   paper.
2. Lift **two different packages** to the top level, in a separate commit, with no new code:
   - `tdadyn.core` — `rng_for`, config parsing, `contracts`, reading datasets;
   - `tdadyn.reference` — the reference; it moves up but **does not become part of `core`**. `core`
     is a common utility imported by everyone; `reference` is a reference whose value lies in not
     sharing code with the pipeline. Inside `core` it would immediately grow back-links.

   Do not touch the generators, the build, the CLI or the manifest: they have one consumer, and it
   will stay one. **Do not leave re-exports in `tdadyn.synthetic`** — see below.
3. Add `docs/zigzag.md`; add a Task 2 section to `docs/roadmap.md`; add the Task 2 keys to the config
   validation schema. Do not edit `method.md`.
4. Packaging of the backends: optional dependencies `pip install tdadyn[zigzag]`, the backend
   imported inside a function, a separate CI job whose failure does not colour the main one. A
   `slow_zigzag` marker for slow runs.

**Check after the move** (one line, before any new code):

```
grep -rn "from tdadyn.synthetic import \(rng_for\|load_config\|adjacency_from_snapshot\|load_dataset\)" src tests
```

It must return nothing. This is exactly the check that re-exports break: with them a consumer's
import from `synthetic` becomes invisible, and that is precisely the forbidden direction of
dependency. Compatibility shims are for external users; the package has none yet, and internal
consumers are fixed in the same commit.

**Criterion.** 484 tests green **after** the move and **before** any new code; `pip install -e .`
works in an environment without zigzag backends; the fast target runs in such an environment and
passes.

The wording "without Dionysus and Boost" was replaced: Dionysus does not need Boost (Z0
measurement), and a criterion referring to a non-existent obstacle checks the wrong thing. What is
checked is the absence of **optional dependencies as such**, and only one of them is expensive —
fzz.

**Trap.** Mixing the move with new code. If something breaks, it will be unclear what.

**On "splitting CI" without `.github/` and a remote.** There is no CI now, so the criterion "green
CI" is unreachable in Z0 and must be replaced by what can be checked locally. The split is not a
workflow file but a property of the run: **the fast target must not depend on optional backends.**
So Z0 introduces three targets (Makefile or `scripts/`):

| Target | What it does | Environment |
|---|---|---|
| `fast` | `pytest -q -m "not slow and not slow_zigzag"` | without `[zigzag]` |
| `slow` | `pytest -m slow` | without `[zigzag]` |
| `zigzag` | `pytest -m slow_zigzag` | with `[zigzag]` |

`.github/workflows/` is added when a remote appears, and only calls these same targets. Until then
**"green CI" stays an open item of Result 4**, not of Z0: it belongs to the package readiness
criterion of the proposal on a par with pip installation and documentation. The same goes for the
repository URL in the paper (open item 3 of section 4 of `HANDOFF_S6-S9.md`): for work whose
argument is reproducibility, the absence of a public link hits the main thesis.

**Closed after Z6, partially.** A remote appeared: `github.com/Semyon-ksa/tdadyn`, and with it
`.github/workflows/tests.yml`, which calls `make fast` and only that, in an environment without
`[zigzag]`. The first run is green: 577 passed, 2 skipped explicitly (modules that need Dionysus) —
the same numbers as in a clean clone with a fresh venv before the commit. `slow` and `zigzag` are not
part of CI and remain local targets.

The second part stays open: the repository is **private**, so there is still no link for the paper.
Before publishing, the history was rewritten — the commit author had been recorded by the machine
name — the file trees on all 68 commits and three tags were compared with the previous ones and
match, but the commit hashes are different. The paper should refer to tags, not to hashes.

The tag is set locally and gains meaning on the first push. If there is no local git repository at
all, the first action of Z0 becomes `git init` and a commit of the current state — and then the
README says honestly that the S1–S9 history was not carried into the repository.

---

## Z1 — Snapshots → zigzag diagram. No persistence

**What to do.**

- Read family 2 through `core.contracts.adjacency_from_snapshot`; `edge_rule` is not implemented
  anew.
- $K_i$, truncation at $k_{\max}+1$, the index set $I$ with half-integer intermediates.
- The union construction — `UNION_RULE = COMPLEX_UNION`, a module constant (`docs/zigzag.md` §2.2).
  The alternative is implemented only inside the test that measures it.
- Unfolding into a stream of elementary operations: insertions in increasing dimension, removals in
  decreasing dimension.
- The translation layer "operation counter ↔ coarse index" — separate, with its own tests.

**Measurements of this session — done.**

- **The difference between the union constructions is zero on all 59 transitions** of both datasets,
  with truncation at dimension 2 and at dimension 3. A negative control (three edges of a triangle
  in different snapshots) gives exactly the missing triangle, i.e. the check tells the constructions
  apart and the zero is not tautological.
- **Monotonicity was checked by name, and the document was wrong.** `zigzag.md` §2.2 called both
  $\delta$ transitions non-monotone; **one** is — $44 \to 45$. At $19 \to 20$ exactly three chords
  across the gap disappear and nothing appears. The cause is not the change of $\delta$ but the
  position of the third-neighbour chord relative to $\varepsilon$ at the current radius.
  Consequence: the death of $H_1$ at 20 happens on a monotone transition, where the choice of union
  rule cannot influence it. §2.2 is corrected.
- **Sizes and the stream.** $\lvert I\rvert = 119$ for both datasets; $\lvert K_i\rvert$ 47–49 and
  91–168; 52 and 448 operations; building all complexes takes 0.07 s. Over all 19 dynamic datasets on
  disk (including `null/` and the 16 `noisy/` datasets) the stream does not exceed 4853 operations.
  The Z3 and Z5 budgets are not at risk, which separately confirms the decision on fzz: speed is not a
  problem here to be solved by a second backend.
- **The operation counter does not determine the coarse index.** 95% of transitions are empty, up to
  45 indices share one state; the layer translates the counter into a range, not an index. Analysis
  — `zigzag.md` §2.4.

**Check of the inherited defect — done in Z0, before the move.** `flag_complex` did apply
`> threshold` itself, so it had to be established by which path the `geometric_loop` numbers in
`method.md` §2.1a were measured. Snapshot 0 was rebuilt through `core.contracts`
(`distance_within`, threshold 0.8):

| Reading | Faces |
|---|---|
| `adjacency_from_snapshot` → `flag_complex` | **(24, 72, 72, 24)** |
| `method.md` §2.1a expects | (24, 72, 72, 24) |
| complement: raw distances, $d > 0.8$ | (24, 204, 728, 990) |

Exact match; 72 edges, minimum pairwise distance 0.261 — also as in §2.1a. The complement differs
threefold in edges and fortyfold in tetrahedra, so the match is not accidental. **The path was right,
the defect was only in the signature**: `method.md` is not edited, the choice $\varepsilon = 0.8$ is
not called into question, the order of work does not change.

**Criterion.** Replaying the operation stream from scratch gives exactly the same $K_i$ and
$K_{i+1/2}$ as direct construction, on every family-2 dataset. Every prefix of the stream is a
simplicial complex (checked explicitly, not assumed).

---

## Z2 — Independent oracle in `reference/`

**What to do.**

- L1: $\beta_k$ at every index of $I$.
- L2: ranks of the induced maps on each arrow.
- L3: the generalised rank $r(b,d) = \mathrm{rank}(\varprojlim \to \varinjlim)$ over $\mathbb{Z}/2$
  and Möbius inversion $m_{[b,d]} = r(b,d) - r(b^-,d) - r(b,d^+) + r(b^-,d^+)$.
- Toys T1–T4 from `docs/zigzag.md` §5.1, including T4 — a test of the oracle's discriminating power
  ($\{[1,4],[2,3]\}$ against $\{[1,3],[2,4]\}$ with identical coverings).
- Algebraic check of the 12↔41 pairing: the expectation is **computed**, not transcribed.

**Criterion — met.** The oracle reproduces the hand-worked barcodes of T1–T3; distinguishes the T4
pair; on `cycle_lifecycle` it outputs the $H_1$ bars $[11½,40½]$ and $[26½,59]$, from which the
12↔41 pair follows rather than being postulated. The run is marked `slow_zigzag` — its first
consumer.

**Z2 measurements — done.**

- **§5.1a was wrong by half a step in $H_1$** of both datasets: bars were written in snapshot
  indices, while the barcode lives on $I$. Measured and corrected; for $H_0$ both rows are right, and
  the cause of the asymmetry is the direction of monotonicity of the transition.
- **The rule translating a bar into an event was rewritten** (§4.2) and checked on all seven events
  of family 2. The old one sent half-integer ends to `*_in_transition` and did not recover the log.
- **Cost of L3.** 20 221 pairs with non-zero support, about 81 000 Gaussian eliminations, no system
  wider than 132 columns. Run time: 6.2 s and 2.2 s on `cycle_lifecycle` ($H_0$ and $H_1$), 5.8 s and
  0.3 s on `geometric_loop` — 15 s in total against an upper estimate of 80 s. There is one shortcut,
  and it is exact: when $r(b,d) = 0$ all larger $d$ are skipped by monotonicity of $r$.

**Traps.**

- The oracle must not reuse code from `zigzag/`. Otherwise it is not an oracle but a second copy.
  Solved by the oracle **taking** complexes and speaking in positions, not coarse indices; a test
  links the two.
- The cost of L3 is quadratic in $|I|$; on large datasets it remains a spot check.

---

## Z3 — Backends and adapters

**What to do.**

- The Dionysus 2 adapter (`zigzag_homology_persistence`).
- Index translation: the backend's half-open intervals → closed coarse ones. Validation **first on
  T1–T4**, where the answer is known by hand, and only then on the datasets.
- Two-sided agreement: oracle ↔ Dionysus, bar for bar.

**Criterion — met.** The barcodes coincide completely on `cycle_lifecycle` and `geometric_loop`,
both dimensions, bar for bar with multiplicities. Zero discrepancies, so the rule "a discrepancy is
considered a defect of the adapter until proven otherwise" was never needed; it remains in force.

**Z3 measurements — done.**

- **Complete agreement** on T1–T3 and both datasets. T3 plays the role of T4 for the backend: two
  layouts with identical coverings, Dionysus picks the right one. T4 itself is unavailable to the
  adapter — a module with no data behind it.
- **The 12↔41 pair is reproduced by the backend**, i.e. the clause of Result 2 that Z2 could not
  close is closed.
- **Dimension $k_{\max}+1$ is cut off by the adapter.** On `geometric_loop` Dionysus returns 69 bars
  in $H_2$ with $\partial_3$ absent; `cycle_lifecycle` has no triangles, and a check on one dataset
  would not have caught them.
- **Ephemeral bars are measured** (open item 4 closed): 23+0 and 0+0 for `cycle_lifecycle`, 23+0 and
  48+94 for `geometric_loop`, where the first term is the prologue. They are visible only on the fine
  path; the coarse path and the oracle cannot see them by construction.
- **There are no invisible bars** on any dataset.
- **Cost.** 16 ms per dataset, 0.30 s for all 19 dynamic datasets. The backend does not consume the
  Z5 budget; the oracle is the expensive part.

**fzz was demoted to optional, and not for convenience.** A second backend was never the check: the
check is **the oracle**, which computes the answer independently, through the generalised rank and
Möbius inversion. A second backend repeats the same class of algorithm and, when it disagrees with
the first, cannot say which is right — it adds a vote, not an independent computation. The value of
fzz was speed on 60 snapshots, and speed is not part of the criterion of Result 2.

Hence the order: fzz is built from source **only if Dionysus does not fit the budget**, and then that
is separate work with its own cost, not an item of Z3.

**Trap.** Comparing "the number of bars" or Betti curves. Coverings are not enough
(`docs/zigzag.md` §3.4) — exactly the same blindness as the Betti curve's to pairing in S6.

**SBM is not part of this session:** its `truth_kind` is a partition, not a barcode; it can serve
only as a self-consistency check of the backends, and this has to be said out loud rather than
obtained by accident.

---

## Z4 — Temporal diagrams and events. This is Result 2

**What to do.**

- Barcode → event log by the table of `docs/zigzag.md` §4.2, including the shift $d \mapsto d+1$ for
  a death (the only place where the shift is introduced).
- Half-integer ends, invisible bars, ephemeral bars, censoring — counted and recorded, not discarded.
- Matching against the truth log and the metrics of §6: recall, precision, localisation error,
  separately by dimension.
- Negative controls by name: `geometric_loop` 4/43/56, the repeated state at 55 in
  `cycle_lifecycle`.

**Criterion (proposal) — met.** On `cycle_lifecycle` and `geometric_loop` recall and precision at
$\tau = 0$ equal one, separately by dimension; the localisation error is zero; the 12↔41 pair was
already reproduced by the backend in Z3.

**Z4 measurements — done.**

- **The ones are an expected outcome, not an achievement**, and this is written in §6: the datasets
  are constructed, the algorithm is exact. Under noise the precision falls to 4/120 (a spot
  measurement; the number belongs to Z5).
- **The event log is blind to pairing.** Both $H_1$ layouts give the same log, so the metrics of
  Result 2 **do not check** the 12↔41 pair — the barcode comparison does. The boundary is fixed by a
  test, not left to the reader.
- **Open item 2 closed**: censored events are marked and do not enter the metrics.
- **Open item 3 closed**: $\tau = 0$ is achievable on clean data, and raising it to 1 recovers one
  event in one of eight noisy datasets. The default stays, and the localisation error at $\tau = 0$
  is identically zero and not informative.
- **Rows of §6**: censored ends 3 and 4, invisible bars 0 and 0, events from a half-integer end 3 of
  5 and 2 of 2.

**Metrics are computed separately by dimension, and this is not a formality.** The five events of
`cycle_lifecycle` split into three in $H_1$ (births 12 and 27, death 41) and two in $H_0$ (birth 48
when $\{4,5,6\}$ detaches, death 55 when $(3,4)$ is restored). A pooled recall over all dimensions
would hide an algorithm that is flawless in $H_0$ and blind in $H_1$.

---

## Z5 — Null models and noise

**What to do.**

- `null/`: the false positive rate **per snapshot**, with a confidence interval (Wilson; the rule of
  three at zero misses). This is a measurement, not a check for zero.
- `noisy/` (24 datasets, four mechanisms): degradation of recall/precision and of the localisation
  error as a function of the noise level.
- The number of invisible and ephemeral bars as a function of the noise level — a characteristic of
  the union construction, needed before transferring to connectomes.
- Level 0.0 is a free check: the barcode at zero level must coincide with the base bitwise.

**Criterion — met.** Tables with measured numbers and intervals in `zigzag.md` §6, §2.3 and §4.3.
Not a single "zero" without the number of trials: a `FalsePositiveRate` cannot be built without a
denominator, and the denominator survives printing.

**Z5 measurements — done.**

- **The null model fires almost everywhere**: 0.9833 per snapshot, 95% [0.9786, 0.9870], 3600
  trials; 12.9 events per snapshot. Both denominators are published — per snapshot (trials are
  correlated, the interval is optimistic) and per realisation (independent, but there are 60).
- **Degradation**: recall falls slowly, precision collapses to 0.0017. `geometric_loop` keeps recall
  at one at every level, `cycle_lifecycle` loses events. `spurious_edges` is an order of magnitude
  more harmful than `missing_edges`.
- **Invisible bars measured for the first time at a non-zero value**: up to 36 under noise, zero on
  the clean datasets.
- **Ephemeral: the prologue is constant**, the part outside the prologue grows by orders of magnitude
  — the open question of Z3 is closed, and the decomposition is justified.
- **Level 0.0 reproduces the clean dataset** in all four combinations.
- **The bounds are now computed, not quoted**: `wilson_interval` reproduces the 4.3% and 10.2% of
  `method.md` §1.2a, `zero_hits_upper_bound` the 63% for three seeds.
- **The threshold `null_models.max_false_positive_rate` does not apply**: it concerns the
  significance criterion on family-1 clouds, not the fraction of snapshots that fire.

---

## Z6 — Build, report, closing the result

**What to do.**

- The CLI `python -m tdadyn.zigzag.run --results` produces `results/zigzag_events.json` and tables.
  **The artefact is produced by the build, not by pytest** — otherwise the reported number of Result
  2 is a side effect of the slow test run (the S9 rule).
- A manifest entry, a hash of the canonical serialisation, a tag.
- Report: every number in the prose is the output of an executed cell. Where the full computation is
  too long, this is stated explicitly as a quotation from `docs/zigzag.md`. The lesson of the S9
  report (three places, two different numbers) applies.
- `docs/roadmap.md` is updated; vineyards are moved with an explicit reason, not silently.

**Criterion — met**, with one amendment to the wording. "Green CI" was moved to the open items of
Result 4 already in Z0: there was neither CI nor a remote then, and a criterion referring to
something with nowhere to run checks the wrong thing. After Z6 both appeared, and the `fast` target
is green in CI (§Z0). The checkable part is done:

- **a reproducible artefact** — `data/synthetic/results/zigzag_events.json`, 207 KB, produced by
  `python -m tdadyn.results --results`; a rebuild gives the same `content_sha256`, only
  `generated_at` changes;
- **the report** — `notebooks/zigzag_report.ipynb`, built and **executed** by the target
  `make report-zigzag`, and it reads the numbers from the artefact rather than recomputing them;
- **the slow run within budget** — 10:48 against a guideline of 15 minutes.

**Z6 measurements — done.**

- **The artefact costs about five minutes**, and almost all of it is the oracle–backend comparison.
  It is included deliberately: the claim of Result 2 includes the agreement of two independent
  computations. The first version compared only the two clean datasets (7 bars, 43 s), and the
  checklist rejected it: an adapter error is more likely to show up where a noisy dataset gives nine
  hundred bars. It became **3192 bars on 18 datasets, 0 discrepancies**; measured, the oracle is in
  places cheaper on a noisy dataset than on a clean one — cost was not the reason.
- **The zigzag test target takes 5:43**, of which 4:46 is the single rebuild of the artefact in the
  hash reproducibility test. The other tests read the committed file.
- **The CLI had to be lifted above the layers.** The module began as `tdadyn.zigzag.run`, and the
  layering guard rejected it: the null realisations require calling the generators, i.e. a zigzag
  making its own data. The artefact needs both layers, so it lives above both — `tdadyn.results`, the
  single composition root, and the guard enforces this.
- **`.gitignore` silently ignored the artefact** — the exceptions are per file, exactly as with
  `manifest.json`. The check uses `git check-ignore`.
- **Three defects were caught by the tests themselves**: an empty event list in the artefact, an
  upper Wilson bound of 0.9999999999999998 at $\hat p = 1$, and the `.gitignore` mentioned above.
- **Events of the noisy datasets are not serialised**: 450 KB out of 609. The barcode, `n_events` and
  the counters remain.

---

## Test time budget

Measured on `synthetic-v1`: the fast run is 429 tests in 15.5 s, the slow one 55 tests in **610 s
(10:10)** against a guideline of 15 minutes. The headroom is under five minutes, and three tests eat
half of it: 108 s the false positive rate of the null model, 73 s the geodesic Betti numbers, 51 s the
failure of the old criterion.

A zigzag on 60 snapshots will break through this headroom, and the L3 oracle is quadratic in the
number of indices. The cure is planned in Z0, not when the run stalls: markers (`slow`,
`slow_zigzag`), split run targets, full L3 on `cycle_lifecycle`, spot checks on the rest.

---

## What I already consider underspecified

Each session begins with a plan that ends with a list of places where the specification would have
to be guessed. This list is for comparison with that one; a discrepancy in either direction is
informative.

1. ~~Whether the union constructions coincide on transitions 20 and 45~~ — **measured in Z1: they
   coincide, and not only there but on all 59 transitions of both datasets.** Along the way it turned
   out that the question itself was imprecise: one of the two named transitions is non-monotone.
2. ~~Behaviour of the algorithm on censored bars~~ — **closed in Z4**: an event is generated and
   marked `censored`, but it does not enter recall and precision. A censored birth does not claim the
   class was born; it says the class already existed when the window opened. One answer for the whole
   project.
3. ~~The default value of the tolerance $\tau$~~ — **measured in Z4**: on clean data $\tau = 0$ gives
   unit recall and precision, and on eight noisy datasets raising it to 1 recovers one event in one.
   Under noise the failure is omission and a flood of false events, not displacement; $\tau$ remains a
   setting with default 0.
4. ~~The fate of ephemeral bars~~ — **measured in Z3**: 23 for `cycle_lifecycle` and 165 for
   `geometric_loop`, of which 94 lie outside the prologue. The number is published decomposed into the
   prologue and the rest: the prologue part measures the assembly order of $K_0$, not the data.
   Whether it stays constant under noise is a question for Z5.
5. ~~Whether SBM takes part in Task 2 at all~~ — **closed in Z1 by a limit, not a decision.**
   `dynamic_sbm` has 60 nodes with `reference_max_nodes: 30`, so the flag complex builder rejects it:
   there is nothing to build its complexes with. The other 19 dynamic datasets on disk do build.
6. Bar significance in the zigzag: `topology_check` does not carry over, there is no replacement, and
   none is introduced in this branch. **Z5 gave a measured ground for the question not being
   academic**: without a significance criterion the precision on noise falls to 0.17%, and on the null
   model 98% of snapshots fire at 12.9 events per snapshot. Still, it is not to be solved by tuning a
   threshold to precision.
7. How the contribution of the $k_{\max}+1$ truncation is checked on the metric dataset, where
   $\beta_3$ exists but is not computed.
