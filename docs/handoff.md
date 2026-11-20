# Handoff to Task 2: temporal persistence diagrams

This document is written for whoever starts Task 2 (zigzag algorithms and localisation of
topological events) and takes the data from `tdadyn.synthetic` as input. It does not retell
`method.md`; it says **what is guaranteed, what is measured, and where an algorithm will go
wrong in a way that looks like its own defect**.

Every number below was checked by a run on the current HEAD.

## 1. What to take

```bash
.venv/bin/python -m tdadyn.synthetic.build --config config.yaml
```

38 datasets, 13 MB, a few seconds. Task 2 needs these above all:

| Dataset | Directory | What it holds |
|---|---|---|
| `cycle_lifecycle` | `dynamic_graph/` | five scheduled events, truth predicted by the construction |
| `geometric_loop` | `dynamic_graph/` | two events, **metric** filtration |
| `dynamic_sbm` | `dynamic_graph/` | partition truth, **no** homological events |
| `erdos_renyi` | `null/` | null model: no schedule, empty log |
| `spurious_edges`, `missing_edges` | `noisy/` | the same scenarios under edge noise |

```python
from tdadyn.core import load_dataset, adjacency_from_snapshot

d = load_dataset("data/synthetic/dynamic_graph/cycle_lifecycle_N24_T60_w1_thr0.5_sched12-27-41-48-55_seed20260201")
gt = d.ground_truth
graph = adjacency_from_snapshot(d.snapshots[0], edge_rule=gt.edge_rule, edge_threshold=gt.edge_threshold)
```

## 2. What exactly is guaranteed

Which truth fields are mandatory is determined by **`truth_kind`**, not by the directory
(`data.md`, §4.1). This matters: `dynamic_sbm` is stored as a dynamic graph and has no
barcode at all.

| `truth_kind` | What is guaranteed |
|---|---|
| `betti_curve` | the Betti curve **and** the event log; consistent and checked |
| `partition` | community labels over time and the schedule; no curve |
| `none` | no schedule, `events == []` |
| `noise_level` | mechanism, level and `base` — the identifier of the clean dataset |

**Only family 2 is predicted by its construction.** For `cycle_lifecycle` and
`geometric_loop` the Betti curve is both computed by the reference implementation from the
recorded snapshots and predicted by the construction, and a test compares the two sources.
Nowhere else is this so: SBM and the null models **deliberately** carry no curve in their
truth — a measured curve under the name of truth would be truth as pipeline output
(`method.md`, §3.1 and §4.2).

## 3. Traps on which the algorithm will look broken

This is the main section. Each item is a discrepancy that will show up as a systematic error
of your algorithm, although the cause is in how the data were read.

**Index convention.** `snapshot: k` means that $G_k$ **already contains** the change and
$G_{k-1}$ does not yet; the jump of the curve falls on the transition $k-1 \to k$. The
alternative reading shifts the whole log by one and looks like a systematic localisation
error (`method.md`, §2.1).

**Pairing of the death.** In `cycle_lifecycle` the death at snapshot 41 is paired with the
birth at 12 (`h1_a`), not with the birth at 27. This is **not a convention**: the
decomposition of a zigzag module into intervals is unique, and the class that dies is the one
born when the path closed. The intuitive "the chord made two cycles, break one and the
younger dies" is wrong here (`method.md`, §2.2a).

Important: **the Betti curve does not see pairing.** Inverting the pair in the log breaks no
check inside family 2 — this was measured. If your algorithm reports a different pair,
consult §2.2a, not the curve.

**The threshold direction is opposite in two scenarios of one family.**

| Dataset | `edge_rule` | Threshold | What a snapshot holds |
|---|---|---|---|
| `cycle_lifecycle` | `weight_above` | 0.5, strict | similarities |
| `geometric_loop` | `distance_within` | 0.8, non-strict | **distances** |

Applying the similarity rule to the metric dataset yields the **complement** of the graph —
connected, plausible and wrong, without a single error. Read `edge_rule` from the truth and
apply it through `adjacency_from_snapshot`; this is the only place in the project where the
rule is applied (`method.md`, §2.3).

**The repeated state is intentional.** `cycle_lifecycle` has five distinct graphs over 60
snapshots, `geometric_loop` four. Snapshots 41–47 and 55–59 of the former are byte-for-byte
identical. The pipeline **must not** collapse states by graph identity: the birth and death
of $H_0$ at 48 and 55 must be reported separately.

**Radius breathing changes the graph without changing $\beta$.** In `geometric_loop` the
graph changes at snapshots 4, 43 and 56 (72→48, 45→66, 48→72 edges), and there are no events
there. This is the strongest negative control of the dataset: a detector that fires on a
change of complex will be caught here.

**A community merge is not a homological event.** In the SBM the blocks are stitched together
by dozens of edges (7–41 inter-block edges per snapshot), the graph is connected, and the
merge at snapshot 20 changes nothing in homology. $\beta_0$ meanwhile wanders between 1 and 2
— because of random isolated vertices, not communities. The robust statement: after removing
isolated vertices the graph is connected on **every** snapshot.

**The null model's $\beta$ fluctuates, and that is not a defect.** ER at the density of
`cycle_lifecycle` ($p = 0.087$) gives $\beta_0 = 4\ldots5$ and $\beta_1 = 2\ldots5$ over five
realisations; a clean $(1,0,0)$ is reliably reached only at $p \gtrsim 0.8$, i.e. on an almost
complete graph. The spread over realisations is the answer here: a null model has no single
"typical" value. A random graph honestly has cycles — that is the null distribution against
which you measure (`method.md`, §4.2).

## 4. What Task 2 must measure and the generator cannot

**The false positive rate of the localisation algorithm.** The null datasets exist exactly
for this: they do not claim zero, they provide a distribution on which the rate can be
measured. It is published in Result 6.

A reference point for comparison: the **significance criterion** on uniform clouds has a
false positive rate of 1.5% at the working size (3 misses out of 200, 95% upper bound 4.3%,
`method.md`, §1.2a). This is a floor — the contribution of the criterion itself, before any
zigzag.

## 5. What must not be inherited

**The `topology_check` parameters do not transfer.** `gap_noise_floor_ratio` 0.07 and
`gap_min_ratio` 1.55 were calibrated on family-1 clouds in $\mathbb{R}^3$ and $\mathbb{R}^4$.
For the connectomes of Task 3 and any other data — only after recalibration, and there is
nothing to calibrate against: no known $\beta_k$ (`method.md`, §1.2a).

**The criterion is not invariant under reparametrisation of the filtration.** A monotone
similarity → distance transform preserves the structure of the diagram but distorts the
persistences, and the criterion compares exactly those. Choosing between $d = 1-|r|$ and
$d = \sqrt{2(1-r)}$ will shift significance (`method.md`, §1.3).

**The stability bound is $d_B \le 2\,d_H$, not $d_H$.** The factor two comes from
parametrising Rips by diameter. Measured: without it 9 of 12 combinations violate the bound,
with ratios up to 1.33. In $H_0$ both forms hold, so the check must use $H_1$
(`method.md`, §4.1).

## 6. Open questions

- **Which cases set the boundaries of the criterion's plateau** — still not recorded. One
  boundary is known: the $\beta_k = 0$ branch fails when exactly one bar survives above the
  noise floor and there is nothing to compare it with (`method.md`, §1.2a).
- **There is one measure of diagram discrepancy — bottleneck.** Wasserstein is not computed in
  the project and is no longer promised. If Task 2 needs it, `persim.wasserstein` is available
  among the dependencies, but the stability bounds of §4.1 are stated for bottleneck and do not
  carry over to it.
- **Rössler reproducibility holds within one environment.** At $c = 5.7$ the system is
  chaotic, and a change of scipy version, method or tolerances moves the whole series. This
  does not concern family 2: `solve_ivp` plays no part there.

## 7. Where to look

| | |
|---|---|
| `docs/method.md` §2 | family-2 constructions, conventions, pairing |
| `docs/method.md` §1.2a | the significance criterion, its calibration and limits |
| `docs/validation.md` §3 | mandatory tests, including those that catch these traps |
| `docs/data.md` §4 | truth schema, `truth_kind`, `edge_rule` |
| `tests/test_dynamic_events.py` | how to match curve jumps against the event log |
