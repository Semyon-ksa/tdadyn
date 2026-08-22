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
