"""Barcode into an event log, and the metrics of Result 2.

The translation is the table of ``docs/zigzag.md`` §4.2, and the whole of it comes down to
two asymmetric rules: a birth lands on the **first snapshot index at or after** the bar's
birth, a death on the **first snapshot index strictly after** its death.  The familiar
``d -> d+1`` shift is the second rule at an integer end, not a rule of its own; it is
introduced here and nowhere else.

**What the event log cannot do.**  An event does not know which bar it came from once it is
in the log, so a log cannot distinguish two barcodes with the same endpoints.  Measured on
``cycle_lifecycle``: the correct pairing and the crossed one produce the *same three
events*.  Recall and precision therefore do **not** check the 12<->41 pair — the barcode
comparison does (Z2 by algebra, Z3 against the backend).  This is the same blindness the
Betti curve had in S6 and the coverings have in §3.4, one level further out, and it belongs
next to the metrics rather than in a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass

from .indices import CoarseIndex

EVENT_TYPES = ("birth", "death")


@dataclass(frozen=True)
class PredictedEvent:
    """One event read off a bar.

    Deliberately **not** :class:`tdadyn.core.contracts.Event`.  A true event carries a
    ``feature_id`` — a stable identifier of the feature — and a narrative, and a predicted
    one has neither and could not acquire them.  What it does carry is provenance: the bar
    it came from, and whether that bar's end was censored or half-integer.  The two types
    look alike, which is a reason to keep them apart rather than to merge them: comparing a
    predicted log with a true one must go through the matcher, not through ``==``.
    """

    snapshot: int
    dim: int
    type: str
    censored: bool
    from_half_integer: bool
    bar: tuple[int, int]


@dataclass(frozen=True)
class EventLog:
    """The events of a barcode, with the counters ``docs/zigzag.md`` §6 asks to report."""

    events: tuple[PredictedEvent, ...]
    invisible_bars: int
    censored_ends: int

    def uncensored(self) -> tuple[PredictedEvent, ...]:
        """Events that are claims of detection.

        A censored birth is not one: it says the class was already present when the window
        opened.  Counting it as a false positive would penalise the algorithm for the window
        being finite, and counting it as a hit would need a truth log that deliberately has
        no such entry.  So it is recorded and left out of the metrics.
        """
        return tuple(event for event in self.events if not event.censored)

    def from_half_integer_ends(self) -> int:
        """How many events came from a half-integer bar end.

        ``docs/zigzag.md`` §6 asks for "events at half-integer indices".  After §4.2 was
        rewritten in Z2 there are none — the rule puts every event on a snapshot.  The
        quantity that still means something is this one, and on family 2 it is most of them.
        """
        return sum(1 for event in self.events if event.from_half_integer)


def events_from_barcode(
    bars, *, indices: tuple[CoarseIndex, ...], dim: int
) -> EventLog:
    """Translate one dimension of a barcode into events.

    Parameters
    ----------
    bars : iterable of (int, int)
        Closed intervals of positions along the zigzag.
    indices : tuple of CoarseIndex
        The coarse index at each position; this is what says which positions are snapshots.
    dim : int
        Homology dimension the bars belong to.

    Returns
    -------
    EventLog
        The events, plus the counts of invisible bars and censored ends.
    """
    last = len(indices) - 1
    events: list[PredictedEvent] = []
    invisible = censored = 0

    for birth, death in bars:
        if birth > death:
            raise ValueError(f"bar birth {birth} lies after death {death}")
        if birth < 0 or death > last:
            raise ValueError(
                f"bar [{birth}, {death}] lies outside the {len(indices)} positions of the "
                "index set"
            )
        visible = [position for position in range(birth, death + 1)
                   if indices[position].is_snapshot]
        if not visible:
            # No state of the data ever held this class (docs/zigzag.md 4.3).
            invisible += 1
            continue

        if birth == 0:
            censored += 1
        events.append(
            PredictedEvent(
                snapshot=indices[visible[0]].snapshot,
                dim=dim,
                type="birth",
                censored=birth == 0,
                from_half_integer=not indices[birth].is_snapshot,
                bar=(birth, death),
            )
        )

        if death == last:
            censored += 1
            continue
        after = next(
            position for position in range(death + 1, len(indices))
            if indices[position].is_snapshot
        )
        events.append(
            PredictedEvent(
                snapshot=indices[after].snapshot,
                dim=dim,
                type="death",
                censored=False,
                from_half_integer=not indices[death].is_snapshot,
                bar=(birth, death),
            )
        )
    return EventLog(tuple(events), invisible_bars=invisible, censored_ends=censored)


def event_log(barcodes: dict[int, tuple], *, indices: tuple[CoarseIndex, ...]) -> EventLog:
    """Translate a whole barcode — every dimension — into one log."""
    events: list[PredictedEvent] = []
    invisible = censored = 0
    for dim in sorted(barcodes):
        one = events_from_barcode(barcodes[dim], indices=indices, dim=dim)
        events.extend(one.events)
        invisible += one.invisible_bars
        censored += one.censored_ends
    return EventLog(tuple(events), invisible_bars=invisible, censored_ends=censored)


# --- matching and metrics (docs/zigzag.md §6) ---------------------------------------


@dataclass(frozen=True)
class Match:
    """One true event paired with the prediction that stood for it."""

    truth: object
    predicted: PredictedEvent
    delta: int


@dataclass(frozen=True)
class DimensionMetrics:
    """Metrics of one homology dimension, or of everything at once."""

    n_truth: int
    n_predicted: int
    n_matched: int
    recall: float | None
    precision: float | None
    mean_localisation_error: float
    max_localisation_error: int


@dataclass(frozen=True)
class EventMetrics(DimensionMetrics):
    """Summary metrics plus the per-dimension breakdown.

    The breakdown is not a formality.  ``cycle_lifecycle`` has three events in :math:`H_1`
    and two in :math:`H_0`; a summary recall would hide an algorithm that is perfect in one
    dimension and blind in the other, which is the failure worth catching.
    """

    by_dimension: dict[int, DimensionMetrics]


def match_events(
    predicted,
    truth,
    *,
    tolerance: int,
    match_dimension: bool = True,
    match_type: bool = True,
) -> tuple[Match, ...]:
    """Pair predicted events with true ones, greedily (``docs/zigzag.md`` §6).

    Parameters
    ----------
    predicted : sequence of PredictedEvent
        Censored events are **not** filtered here; the caller decides, and
        :meth:`EventLog.uncensored` is what it usually passes.
    truth : sequence
        True events; anything with ``snapshot``, ``dim`` and ``type``, which is what
        :class:`tdadyn.core.contracts.Event` carries.
    tolerance : int
        Largest admissible ``|delta t|``.  Zero means only an exact hit matches.
    match_dimension, match_type : bool
        Whether a pair must agree on dimension and on birth-versus-death.

    Returns
    -------
    tuple of Match
        Each true and each predicted event is used at most once.

    Notes
    -----
    Greedy over all admissible pairs at once, ordered by ``|delta t|`` and then by the time
    of the true event.  Both keys are fixed by §6: without the second, a tie would be broken
    by whatever order the events happened to arrive in, and the metric would depend on it.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")

    candidates = []
    for truth_index, want in enumerate(truth):
        for predicted_index, got in enumerate(predicted):
            if match_dimension and got.dim != want.dim:
                continue
            if match_type and got.type != want.type:
                continue
            delta = abs(got.snapshot - want.snapshot)
            if delta <= tolerance:
                candidates.append((delta, want.snapshot, truth_index, predicted_index))

    used_truth: set[int] = set()
    used_predicted: set[int] = set()
    matches: list[Match] = []
    for delta, _, truth_index, predicted_index in sorted(candidates):
        if truth_index in used_truth or predicted_index in used_predicted:
            continue
        used_truth.add(truth_index)
        used_predicted.add(predicted_index)
        matches.append(Match(truth[truth_index], predicted[predicted_index], delta))
    return tuple(matches)


def event_metrics(
    predicted,
    truth,
    *,
    tolerance: int,
    match_dimension: bool = True,
    match_type: bool = True,
) -> EventMetrics:
    """Recall, precision and localisation error, in total and per dimension.

    Returns
    -------
    EventMetrics
        ``recall`` and ``precision`` are ``None`` when their denominator is zero — which is
        a real case, not a guard: ``geometric_loop`` has no :math:`H_0` events at all, and
        reporting 0.0 there would read as a failure to find something that was never there.

    Notes
    -----
    The localisation error is reported because §6 asks for it, and at ``tolerance = 0`` it is
    identically zero: only an exact hit can match.  Measured across eight noisy datasets,
    raising the tolerance recovers one event in one of them, so under noise the failure is
    missing and spurious events rather than misplaced ones.  recall and precision are the
    numbers that carry information.
    """
    matches = match_events(
        predicted, truth, tolerance=tolerance,
        match_dimension=match_dimension, match_type=match_type,
    )
    dimensions = {event.dim for event in truth} | {event.dim for event in predicted}
    by_dimension = {
        dim: _metrics(
            [event for event in truth if event.dim == dim],
            [event for event in predicted if event.dim == dim],
            [match for match in matches if match.truth.dim == dim],
        )
        for dim in sorted(dimensions)
    }
    summary = _metrics(list(truth), list(predicted), list(matches))
    return EventMetrics(
        n_truth=summary.n_truth,
        n_predicted=summary.n_predicted,
        n_matched=summary.n_matched,
        recall=summary.recall,
        precision=summary.precision,
        mean_localisation_error=summary.mean_localisation_error,
        max_localisation_error=summary.max_localisation_error,
        by_dimension=by_dimension,
    )


def _metrics(truth, predicted, matches) -> DimensionMetrics:
    deltas = [match.delta for match in matches]
    return DimensionMetrics(
        n_truth=len(truth),
        n_predicted=len(predicted),
        n_matched=len(matches),
        recall=len(matches) / len(truth) if truth else None,
        precision=len(matches) / len(predicted) if predicted else None,
        mean_localisation_error=sum(deltas) / len(deltas) if deltas else 0.0,
        max_localisation_error=max(deltas) if deltas else 0,
    )


@dataclass(frozen=True)
class EventReport:
    """Everything ``docs/zigzag.md`` §6 asks a run to report.

    The counters are separate lines rather than part of the metrics on purpose: they
    describe the **construction**, not the algorithm's accuracy.  Their value is that they
    have to be known before any of this reaches a connectome — an invisible bar is a false
    positive by construction, and the ephemeral count says how much of the answer depends on
    the arbitrary order inside a transition.
    """

    metrics: EventMetrics
    events: tuple[PredictedEvent, ...]
    invisible_bars: int
    censored_ends: int
    events_from_half_integer_ends: int
    ephemeral_bars: int | None
    ephemeral_in_prologue: int | None

    def ephemeral_outside_prologue(self) -> int | None:
        """Ephemeral bars that are not an artefact of assembling the first complex.

        The prologue builds :math:`K_0` from nothing, so its ephemeral bars count the order
        of that assembly — lexicographic, fixed in Z1, and arbitrary in substance.
        Publishing the total as a property of the dataset would report that order as data;
        this is the half that is about the data.
        """
        if self.ephemeral_bars is None or self.ephemeral_in_prologue is None:
            return None
        return self.ephemeral_bars - self.ephemeral_in_prologue


def event_report(
    *,
    log: EventLog,
    truth,
    tolerance: int,
    match_dimension: bool = True,
    match_type: bool = True,
    ephemeral_bars: int | None = None,
    ephemeral_in_prologue: int | None = None,
) -> EventReport:
    """Assemble everything §6 asks a run to report.

    Parameters
    ----------
    log : EventLog
        The predicted log; censored events are excluded from the metrics here.
    truth : sequence
        The true event log.  Passed in rather than read off a dataset because a noisy
        dataset carries none: its ``truth_kind`` is ``noise_level`` and the events belong to
        the base dataset named in ``noise_level["base"]``.  Z5 needs that, and a signature
        that fetched the truth itself would not allow it.
    tolerance, match_dimension, match_type : see :func:`match_events`
    ephemeral_bars, ephemeral_in_prologue : int, optional
        From the fine path — one time per operation rather than per coarse index.  Ephemeral
        bars are invisible to the coarse path by construction, so they cannot be derived
        here and are supplied by whoever ran the backend over the operation stream.  Left
        as ``None`` the report says so instead of reporting zero.

    Returns
    -------
    EventReport
        Metrics and the counters.
    """
    metrics = event_metrics(
        log.uncensored(), truth, tolerance=tolerance,
        match_dimension=match_dimension, match_type=match_type,
    )
    return EventReport(
        metrics=metrics,
        events=log.events,
        invisible_bars=log.invisible_bars,
        censored_ends=log.censored_ends,
        events_from_half_integer_ends=log.from_half_integer_ends(),
        ephemeral_bars=ephemeral_bars,
        ephemeral_in_prologue=ephemeral_in_prologue,
    )
