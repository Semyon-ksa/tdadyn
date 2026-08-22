"""Null models and noise: the pipeline on a dataset, and rates with their denominators.

Two things live here, and they share one principle.

**A frequency is never reported without the number of trials.** S8 spent a session on this:
an estimate taken on sixty trials showed nothing, "zero misses on three seeds" turned out to
bound the true rate at 63% rather than at zero, and both mistakes were invisible until the
denominator was written down.  :class:`FalsePositiveRate` therefore cannot be constructed
without one, and its printed form carries it.

**The bounds are computed, not quoted.**  Until now the project's confidence bounds — 4.3%,
10.2%, 63% — were derived off-line and copied into ``method.md``.  Nothing in the package
could reproduce them, which is exactly the state to avoid: a number in prose
that is a memory of a calculation rather than its output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .backend import dionysus_barcode
from .construction import zigzag_complexes
from .events import EventLog, EventReport, event_log, event_report
from .indices import index_set
from .operations import coarse_bar_of, operation_stream

DEFAULT_CONFIDENCE = 0.95


def wilson_interval(
    hits: int, trials: int, *, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    hits, trials : int
        Successes and attempts.
    confidence : float, default 0.95
        Two-sided confidence level.

    Returns
    -------
    tuple of float
        Lower and upper bound, clipped to ``[0, 1]``.

    Notes
    -----
    Wilson rather than the normal approximation because the rates here sit near the ends of
    the range, where the normal interval leaves the unit interval and reports negative
    frequencies.  It reproduces the bounds ``method.md`` §1.2a publishes for 3/200 and
    12/200, which is what its test checks.

    The interval assumes the trials are independent.  Where they are not — snapshots inside
    one realisation share edges with their neighbours — it is narrower than the truth, and
    the caller has to say so rather than let the number stand unqualified.
    """
    _check_counts(hits, trials)
    z = _z_for(confidence)
    proportion = hits / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    lower, upper = max(0.0, centre - half), min(1.0, centre + half)
    # At the ends the interval touches 0 and 1 exactly; floating point lands a couple of
    # ulps short, and a bound that does not contain the rate it was computed from reads as
    # a defect wherever it is printed.
    if hits == 0:
        lower = 0.0
    if hits == trials:
        upper = 1.0
    return lower, upper


def zero_hits_upper_bound(trials: int, *, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Upper bound on a rate when no hit was observed at all.

    Parameters
    ----------
    trials : int
        Number of attempts, all of them misses.
    confidence : float, default 0.95

    Returns
    -------
    float
        The largest rate consistent with observing nothing: :math:`1 - \\alpha^{1/n}`.

    Notes
    -----
    The familiar rule of three, :math:`3/n`, is the large-sample form of this and agrees
    with it from a few dozen trials onwards — at ``n = 60`` both give about 0.05.  At small
    ``n`` the rule breaks down: it returns 1 for three trials, while the exact bound is
    **0.63**, which is the number ``method.md`` uses to reject "zero misses on three seeds"
    as a claim.  The exact form is used so that the same function serves both ends.
    """
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    alpha = 1 - confidence
    return 1 - alpha ** (1 / trials)


@dataclass(frozen=True)
class FalsePositiveRate:
    """A measured rate that cannot be separated from its denominator.

    Both fields are required, and the denominator survives into ``__str__``: a frequency
    printed without the number of trials is the failure S8 catalogued, and making it
    impossible to construct is cheaper than remembering not to.
    """

    hits: int
    trials: int

    def __post_init__(self) -> None:
        _check_counts(self.hits, self.trials)

    @property
    def rate(self) -> float:
        return self.hits / self.trials

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.hits, self.trials)

    def upper_bound_without_hits(self) -> float | None:
        """The bound that applies when nothing was observed; ``None`` when something was.

        Kept separate from :attr:`interval` because the two answer different questions, and
        conflating them is how "we saw nothing" becomes "the rate is zero".
        """
        if self.hits:
            return None
        return zero_hits_upper_bound(self.trials)

    def __str__(self) -> str:
        lower, upper = self.interval
        return (
            f"{self.rate:.4f} ({self.hits}/{self.trials}, 95% [{lower:.4f}, {upper:.4f}])"
        )


def _check_counts(hits: int, trials: int) -> None:
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    if hits < 0 or hits > trials:
        raise ValueError(f"hits must lie between 0 and trials, got {hits} of {trials}")


def _z_for(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence}")
    # Inverse normal CDF by bisection: no scipy here, and the reference layer's habit of
    # obvious code over fast code applies to anything whose number ends up in a report.
    target = 1 - (1 - confidence) / 2
    low, high = 0.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2
        if 0.5 * (1 + math.erf(middle / math.sqrt(2))) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


# --- the pipeline on one dataset ----------------------------------------------------


def analyse(dataset, *, config, truth, with_fine_path: bool = False) -> EventReport:
    """Run the whole zigzag on one dataset and report what §6 asks for.

    Parameters
    ----------
    dataset : DynamicGraphDataset
        Any dataset whose truth records ``edge_rule`` and ``edge_threshold``.
    config : Config
        Supplies ``zigzag.max_homology_dim``, the matching switches and the node limit.
    truth : sequence
        The true event log, passed in rather than read off the dataset: a noisy dataset
        carries none — its ``truth_kind`` is ``noise_level`` — and its events belong to the
        base named in ``noise_level["base"]``.  A null dataset's log is empty, which is the
        whole point of it.
    with_fine_path : bool, default False
        Also run the backend over the operation stream, which is the only way ephemeral bars
        are visible.  Off by default because it doubles the work for a number many callers
        do not need; left off, the report says ``None`` rather than zero.

    Returns
    -------
    EventReport
        Metrics plus the counters of §6.

    Notes
    -----
    The diagram is built **once**.  Writing ``zigzag_complexes(dataset, ...)[index]`` inside
    a comprehension over the indices rebuilds all of it per index and turns 0.08 s into 8 s;
    that mistake was made while measuring for this session and is worth naming.
    """
    ceiling = config.zigzag.max_homology_dim
    build = {"max_homology_dim": ceiling,
             "max_nodes": config.topology_check.reference_max_nodes}
    diagram = zigzag_complexes(dataset, **build)
    indices = index_set(len(dataset.snapshots))
    barcodes = dionysus_barcode([diagram[index] for index in indices],
                                max_homology_dim=ceiling)
    log = event_log(barcodes, indices=indices)

    ephemeral = in_prologue = None
    if with_fine_path:
        ephemeral, in_prologue = ephemeral_counts(diagram, max_homology_dim=ceiling)

    matching = config.zigzag.event_matching
    return event_report(
        log=log,
        truth=truth,
        tolerance=matching.tolerance,
        match_dimension=matching.match_dimension,
        match_type=matching.match_type,
        ephemeral_bars=ephemeral,
        ephemeral_in_prologue=in_prologue,
    )


def ephemeral_counts(diagram, *, max_homology_dim: int) -> tuple[int, int]:
    """Ephemeral bars of a diagram, and how many of them belong to the prologue.

    Returns
    -------
    tuple of int
        Total, and the part that lies inside the assembly of the first complex.

    Notes
    -----
    Ephemeral bars are born and die inside one transition, so no index of the diagram ever
    sees them and neither the coarse barcode nor the oracle can produce them.  They exist
    only in the fine indexing, one time per operation.

    The two numbers are returned separately because they mean different things.  The
    prologue builds :math:`K_0` from nothing, so its ephemeral bars count the order of that
    assembly — lexicographic, fixed in Z1, arbitrary in substance.  Measured across the
    noise levels the prologue part drifts by units while the rest grows by orders, so a
    total would report the assembly order as if it were the signal.
    """
    from tdadyn.reference import SimplicialComplex

    stream = operation_stream(diagram)
    present: set[tuple[int, ...]] = set()
    states = [SimplicialComplex(faces={})]
    for operation in stream.operations:
        if operation.kind == "insert":
            present.add(operation.simplex)
        else:
            present.discard(operation.simplex)
        states.append(SimplicialComplex.from_simplices(present))

    fine = dionysus_barcode(states, max_homology_dim=max_homology_dim)
    prologue = len(stream.prologue)
    ephemeral = [
        (birth, death)
        for dim in fine
        for birth, death in fine[dim]
        if coarse_bar_of(stream, birth, death) is None
    ]
    return len(ephemeral), sum(1 for _, death in ephemeral if death <= prologue)


# --- null models --------------------------------------------------------------------


@dataclass(frozen=True)
class NullMeasurement:
    """What a null model lets you measure: a rate, not an absence.

    The truth of a null dataset is that there is no schedule and the event log is empty
    (``method.md`` §4.2), so **every** uncensored event is a false positive.  What the
    dataset cannot do is establish that there are none; it establishes how often there are.
    """

    per_snapshot: FalsePositiveRate
    per_realisation: FalsePositiveRate
    events: int
    events_by_dimension: dict[int, int]

    @property
    def events_per_snapshot(self) -> float:
        """Reported next to the rate because the rate saturates.

        At 0.98 the proportion of snapshots that fire says almost nothing about magnitude;
        this says the algorithm reports about thirteen events on each of them.
        """
        return self.events / self.per_snapshot.trials


def measure_null(datasets, *, config) -> NullMeasurement:
    """Measure the false-positive rate of the event detector on null datasets.

    Parameters
    ----------
    datasets : iterable of DynamicGraphDataset
        Realisations of the null model; more than one, because a rate taken on a single
        realisation is the estimate S8 showed to be worthless.
    config : Config

    Returns
    -------
    NullMeasurement
        Rates per snapshot and per realisation, with their denominators attached.

    Notes
    -----
    Two denominators, and both are reported because neither alone is honest.

    * **Per snapshot** is what ``docs/zigzag.md`` §6 asks for, and its trials are *not*
      independent: neighbouring complexes share edges, so an event at one snapshot makes one
      at the next more likely, and the Wilson interval is narrower than the truth.
    * **Per realisation** has independent trials — each is its own random stream — but far
      fewer of them, so the interval is wide.  It is the conservative reading.

    The threshold ``null_models.max_false_positive_rate`` is **not** consulted.  It is
    calibrated for the significance criterion on family-1 point clouds (``method.md`` §1.2a)
    — the share of realisations where that criterion invented a feature — and this is the
    share of snapshots where the zigzag reports an event.  Comparing them would declare a
    failure by measuring two different things.
    """
    fired = snapshots = events = 0
    realisations = firing_realisations = 0
    by_dimension: dict[int, int] = {}
    for dataset in datasets:
        report_log = _null_log(dataset, config=config)
        uncensored = report_log.uncensored()
        events += len(uncensored)
        for event in uncensored:
            by_dimension[event.dim] = by_dimension.get(event.dim, 0) + 1
        firing = {event.snapshot for event in uncensored}
        fired += len(firing)
        snapshots += len(dataset.snapshots)
        realisations += 1
        firing_realisations += 1 if firing else 0

    return NullMeasurement(
        per_snapshot=FalsePositiveRate(hits=fired, trials=snapshots),
        per_realisation=FalsePositiveRate(hits=firing_realisations, trials=realisations),
        events=events,
        events_by_dimension=dict(sorted(by_dimension.items())),
    )


def _null_log(dataset, *, config) -> EventLog:
    ceiling = config.zigzag.max_homology_dim
    diagram = zigzag_complexes(
        dataset,
        max_homology_dim=ceiling,
        max_nodes=config.topology_check.reference_max_nodes,
    )
    indices = index_set(len(dataset.snapshots))
    barcodes = dionysus_barcode([diagram[index] for index in indices],
                                max_homology_dim=ceiling)
    return event_log(barcodes, indices=indices)


# --- degradation under noise --------------------------------------------------------


@dataclass(frozen=True)
class DegradationRow:
    """One noisy dataset, measured against the truth of the dataset it was made from."""

    base: str
    kind: str
    level: float
    bars: int
    report: EventReport

    @property
    def recall(self) -> float | None:
        return self.report.metrics.recall

    @property
    def precision(self) -> float | None:
        return self.report.metrics.precision


def degradation_row(dataset, *, config, truth, with_fine_path: bool = False) -> DegradationRow:
    """Measure one noisy dataset.

    Parameters
    ----------
    dataset : DynamicGraphDataset
        A dataset whose truth is a ``noise_level``.
    truth : sequence
        The event log of the **base** dataset: a noisy dataset carries none of its own.

    Notes
    -----
    The localisation error is not among the reported columns.  At the default tolerance of
    zero it is identically zero — only an exact hit can match — so a column of zeros would
    read as a result (``docs/zigzag.md`` §6).  It stays inside ``report.metrics`` for anyone
    who raises the tolerance.
    """
    noise = dataset.ground_truth.noise_level
    if noise is None:
        raise ValueError(
            f"dataset {dataset.ground_truth.generator!r} carries no noise_level; "
            "degradation is measured on family 4, against the truth of its base"
        )
    report = analyse(dataset, config=config, truth=truth, with_fine_path=with_fine_path)
    ceiling = config.zigzag.max_homology_dim
    diagram = zigzag_complexes(
        dataset,
        max_homology_dim=ceiling,
        max_nodes=config.topology_check.reference_max_nodes,
    )
    indices = index_set(len(dataset.snapshots))
    barcodes = dionysus_barcode([diagram[index] for index in indices],
                                max_homology_dim=ceiling)
    return DegradationRow(
        base=noise["base"],
        kind=noise["kind"],
        level=float(noise["level"]),
        bars=sum(len(bars) for bars in barcodes.values()),
        report=report,
    )
