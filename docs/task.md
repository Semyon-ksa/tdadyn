# Problem statement

## Place in the project

Task 5 of the first stage: "Creation of synthetic and reference datasets for validating
methods". The generators serve three consumers:

| Consumer | What it must verify on these data |
|---|---|
| Task 3 — persistent invariants of fully connected weighted networks | Does the static pipeline recover known Betti numbers |
| Task 2 — "temporal persistence diagrams" (zigzag) | Does the algorithm detect topological events known in advance |
| Result 6 — stability and complexity | How diagrams degrade under controlled noise; are there false positives on null models |

The order in the stage roadmap is not accidental: the data (Task 5) are made **before**
the algorithms (Task 2); otherwise there would be nothing to validate against, and the
temptation would appear to declare as truth whatever the algorithm produced.

## What is required

A library `tdadyn.synthetic` that generates four families of datasets. Every dataset comes
with a machine-readable description of its truth (ground truth).

**Family 1 — static point clouds and networks with known Betti numbers.** Samples from
manifolds whose topology is known analytically (circle, two circles, figure eight, torus,
sphere, Klein bottle, Hopf link). A weighted network is built from the cloud. Truth: Betti
numbers β₀, β₁, β₂.

**Family 2 — dynamic graphs with scheduled topological events.** A sequence of graph
snapshots in which, at moments fixed in advance, topological features are born, merge,
split and die. Truth: the event log plus an independently computed Betti curve over the
snapshots.

**Family 3 — dynamic SBM and multivariate series of coupled oscillators.** A synthetic
analogue of fMRI: series with a KNOWN coupling schedule, from which a sliding window builds a
dynamic weighted graph. It validates the whole chain "time series → network → persistence"
where the answer is known.

**Family 4 — noise and null models.** Controlled corruption of the reference datasets and
graphs with no topological structure. Truth: the noise level as a parameter; for null
models, the absence of events.

## Scope of the task

- Real fMRI data play no part here — that is a separate module of the project. No generator
  reads neuroimaging data.
- The task is generation and its verification, not the development of zigzag algorithms
  themselves. Persistent computations are used here only to verify the generators.
- No sensitive data: the datasets are fully synthetic, so there are no restrictions on file
  contents or on the environment where the work is done.

## Already decided and not to be revisited without discussion

- The default homology coefficients are the field Z/2 (ripser works over Z/p; the choice
  affects the Klein bottle, see `docs/method.md`).
- The maximum homology dimension is 2.
- Verification uses **persistent** Betti numbers with bar significance, not the Betti
  numbers of a single complex at a fixed threshold.

## Completion criterion for the task as a whole

A seed-reproducible collection of datasets with truth metadata, covering every verification
scenario (static, dynamic events, series analogue, noise, null models), that passes the full
correctness test suite of `docs/validation.md`.
