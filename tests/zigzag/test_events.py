"""Barcode into an event log, and the metrics of Result 2.

The danger here is not the arithmetic.  It is that the numbers say more than they measure:
on the built datasets recall and precision come out at one, and a report can be read as
"the algorithm is fully checked".  It is not — the event log is blind to pairing, and that
blindness is pinned by a test below rather than left to the reader.
"""

from __future__ import annotations

import pytest

from tdadyn.zigzag import index_set
from tdadyn.core import Event
from tdadyn.zigzag.events import (
    EventLog,
    PredictedEvent,
    event_log,
    event_metrics,
    events_from_barcode,
    match_events,
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


# --- matching and metrics (docs/zigzag.md §6) ---------------------------------------


def predicted(*triples):
    """Predicted events from ``(snapshot, dim, type)`` triples, for the matcher tests."""
    return tuple(
        PredictedEvent(snapshot=s, dim=d, type=t, censored=False,
                       from_half_integer=False, bar=(0, 0))
        for s, d, t in triples
    )


def truth(*triples):
    """True events, built as the real thing: the matcher must take what a dataset carries.

    ``feature_id`` and ``narrative`` are what a predicted event cannot have, and they play
    no part in matching — §6 matches on ``(type, dim)`` and time only.
    """
    return tuple(
        Event(snapshot=s, dim=d, type=t, feature_id=f"t{s}", narrative="")
        for s, d, t in triples
    )


def test_the_matcher_prefers_the_nearer_candidate():
    matches = match_events(
        predicted((12, 1, "birth"), (11, 1, "birth")),
        truth((10, 1, "birth")),
        tolerance=2,
    )
    assert [(m.truth.snapshot, m.predicted.snapshot) for m in matches] == [(10, 11)]


def test_a_tie_goes_to_the_earlier_true_event():
    """``docs/zigzag.md`` §6 fixes the tie-break, so it is not left to sort stability."""
    matches = match_events(
        predicted((12, 1, "birth")),
        truth((10, 1, "birth"), (14, 1, "birth")),
        tolerance=2,
    )
    assert [(m.truth.snapshot, m.predicted.snapshot) for m in matches] == [(10, 12)]


def test_each_event_is_used_at_most_once():
    matches = match_events(
        predicted((10, 1, "birth")),
        truth((10, 1, "birth"), (10, 1, "birth")),
        tolerance=0,
    )
    assert len(matches) == 1


def test_a_birth_never_matches_a_death_and_a_dimension_never_crosses():
    assert match_events(predicted((10, 1, "death")), truth((10, 1, "birth")),
                        tolerance=5) == ()
    assert match_events(predicted((10, 0, "birth")), truth((10, 1, "birth")),
                        tolerance=5) == ()


def test_the_matching_switches_reach_the_matcher():
    """``match_dimension`` and ``match_type`` are config keys, so they must do something."""
    across = match_events(predicted((10, 0, "birth")), truth((10, 1, "birth")),
                          tolerance=0, match_dimension=False)
    assert len(across) == 1
    across_type = match_events(predicted((10, 1, "death")), truth((10, 1, "birth")),
                               tolerance=0, match_type=False)
    assert len(across_type) == 1


def test_the_tolerance_bounds_what_can_match():
    for tolerance, expected in ((0, 0), (1, 1), (5, 1)):
        matches = match_events(predicted((11, 1, "birth")), truth((10, 1, "birth")),
                               tolerance=tolerance)
        assert len(matches) == expected


def test_metrics_are_reported_per_dimension_and_not_only_in_total():
    """A summary would hide an algorithm perfect in one dimension and blind in another.

    Here it is blind in :math:`H_1` and perfect in :math:`H_0`; the summary recall of 1/2
    says nothing about which.
    """
    metrics = event_metrics(
        predicted((48, 0, "birth")),
        truth((48, 0, "birth"), (12, 1, "birth")),
        tolerance=0,
    )
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.by_dimension[0].recall == pytest.approx(1.0)
    assert metrics.by_dimension[1].recall == pytest.approx(0.0)


def test_precision_counts_the_predictions_that_were_not_asked_for():
    metrics = event_metrics(
        predicted((12, 1, "birth"), (13, 1, "birth"), (14, 1, "birth")),
        truth((12, 1, "birth")),
        tolerance=0,
    )
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1 / 3)


def test_the_localisation_error_is_zero_by_construction_at_zero_tolerance():
    """Published because §6 asks for it, and informative only above zero tolerance.

    At :math:`\\tau = 0` only an exact hit can match, so the mean and the maximum are zero
    whatever the algorithm did.  Measured on eight noisy datasets, raising the tolerance
    recovers one event in one of them — so under noise the failure is missing events and
    false ones, not misplaced ones.  recall and precision are the informative numbers.
    """
    exact = event_metrics(predicted((12, 1, "birth")), truth((12, 1, "birth")), tolerance=0)
    assert exact.mean_localisation_error == pytest.approx(0.0)
    assert exact.max_localisation_error == 0

    loose = event_metrics(predicted((13, 1, "birth")), truth((12, 1, "birth")), tolerance=2)
    assert loose.mean_localisation_error == pytest.approx(1.0)
    assert loose.max_localisation_error == 1


def test_metrics_of_an_empty_truth_are_not_a_division_by_zero():
    """``geometric_loop`` has no :math:`H_0` events at all, so this case is real."""
    metrics = event_metrics((), (), tolerance=0)
    assert metrics.recall is None
    assert metrics.precision is None
