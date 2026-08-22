"""Barcode into an event log, and the metrics of Result 2.

The danger here is not the arithmetic.  It is that the numbers say more than they measure:
on the built datasets recall and precision come out at one, and a report can be read as
"the algorithm is fully checked".  It is not — the event log is blind to pairing, and that
blindness is pinned by a test below rather than left to the reader.
"""

from __future__ import annotations

import pytest

from tdadyn.zigzag import index_set
from tdadyn.zigzag.events import (
    EventLog,
    PredictedEvent,
    event_log,
    events_from_barcode,
)


# --- the translation rule, on answers known by hand ---------------------------------


def test_the_death_shift_is_introduced_here_and_only_here():
    """T1, where the barcode is written out in ``docs/zigzag.md`` §5.1.

    Two vertices with an edge only on snapshot 1: the components merge at 1 and split again
    at 2.  The bar ``(0, 0)`` dies at position 0, and the death event lands on snapshot 1 —
    the first snapshot index **after** it.  That is the ``d -> d+1`` shift of §4.2, and it
    is a special case of "first snapshot index > d" rather than a separate rule.
    """
    indices = index_set(3)
    log = events_from_barcode(((0, 0), (0, 4), (4, 4)), indices=indices, dim=0)
    plain = [(event.snapshot, event.type) for event in log.events if not event.censored]
    assert plain == [(1, "death"), (2, "birth")]


def test_a_left_censored_birth_is_emitted_and_marked():
    """``docs/zigzag.md`` §4.2 asks for the mark, not for silence.

    A class already present when the observation starts did not get born at snapshot 0; the
    event records that the window began too late, which is a different statement.
    """
    indices = index_set(3)
    log = events_from_barcode(((0, 4),), indices=indices, dim=0)
    assert [(event.snapshot, event.type, event.censored) for event in log.events] == [
        (0, "birth", True)
    ]
    assert log.censored_ends == 2  # both ends of this bar touch the boundary


def test_a_right_censored_bar_produces_no_death():
    indices = index_set(3)
    log = events_from_barcode(((2, 4),), indices=indices, dim=1)
    assert [event.type for event in log.events] == ["birth"]
    assert log.censored_ends == 1


def test_an_invisible_bar_produces_nothing_and_is_counted():
    """A bar whose support holds no snapshot index describes no state of the data.

    Built by hand: neither dataset has one, which is itself a measurement (``§4.3``), and a
    rule that has never met its own edge case is not a checked rule.
    """
    indices = index_set(3)
    log = events_from_barcode(((1, 1),), indices=indices, dim=1)
    assert log.events == ()
    assert log.invisible_bars == 1
    assert log.censored_ends == 0


def test_an_event_records_whether_its_bar_end_was_half_integer():
    """Diagnostic of the union rule, not a defect.

    A half-integer end means the class first appears in a union rather than in a snapshot.
    The event is still placed on a snapshot — §4.2 — but where it came from is worth
    counting, because on family 2 most events come from such ends.
    """
    indices = index_set(3)
    log = events_from_barcode(((1, 3),), indices=indices, dim=1)
    assert [(event.snapshot, event.type, event.from_half_integer) for event in log.events] == [
        (1, "birth", True),
        (2, "death", True),
    ]


def test_the_whole_log_merges_the_dimensions():
    indices = index_set(3)
    log = event_log({0: ((0, 4),), 1: ((1, 3),)}, indices=indices)
    assert sorted({event.dim for event in log.events}) == [0, 1]
    assert log.invisible_bars == 0


def test_a_bar_outside_the_index_set_is_refused():
    indices = index_set(3)
    with pytest.raises(ValueError, match="outside"):
        events_from_barcode(((0, 9),), indices=indices, dim=0)
    with pytest.raises(ValueError, match="birth"):
        events_from_barcode(((3, 1),), indices=indices, dim=0)


def test_the_log_is_a_frozen_record_of_what_it_counted():
    indices = index_set(3)
    log = events_from_barcode(((0, 4),), indices=indices, dim=0)
    assert isinstance(log, EventLog)
    assert isinstance(log.events[0], PredictedEvent)
    with pytest.raises(Exception):
        log.events[0].snapshot = 5
