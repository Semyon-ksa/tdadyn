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
    event_report,
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


# --- on the datasets: this is Result 2 ----------------------------------------------


DATASET_TESTS = pytest.mark.slow_zigzag


def log_of(dataset, config, zigzag_of):
    complexes, indices = zigzag_of.complexes(dataset)
    bars = {
        dim: zigzag_of.bars(dataset, dim)
        for dim in range(config.zigzag.max_homology_dim + 1)
    }
    return event_log(bars, indices=indices), indices


@DATASET_TESTS
def test_every_event_of_the_truth_log_is_found_and_nothing_else_is(
    family_two_dataset, config, zigzag_of
):
    """recall and precision at :math:`\\tau = 0`, the headline of Result 2.

    Both come out at one, and that is the **expected** outcome rather than an achievement:
    the datasets are constructed and the algorithm is exact.  What makes the metric worth
    having is that it stops being one under noise — measured at precision 4/120 on
    ``spurious_edges`` at level 0.02, which is Z5's number, not this one.
    """
    log, _ = log_of(family_two_dataset, config, zigzag_of)
    metrics = event_metrics(
        log.uncensored(),
        family_two_dataset.ground_truth.events,
        tolerance=config.zigzag.event_matching.tolerance,
        match_dimension=config.zigzag.event_matching.match_dimension,
        match_type=config.zigzag.event_matching.match_type,
    )
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.max_localisation_error == 0

    for dim, per in metrics.by_dimension.items():
        if per.n_truth:
            assert per.recall == pytest.approx(1.0), f"H_{dim}"
        if per.n_predicted:
            assert per.precision == pytest.approx(1.0), f"H_{dim}"


@DATASET_TESTS
def test_the_event_log_cannot_tell_the_two_pairings_apart(cycle_lifecycle, config, zigzag_of):
    """The boundary of Result 2, pinned rather than left to the reader.

    An event does not remember which bar produced it, so a log cannot separate the correct
    pairing from the crossed one — both give birth 12, birth 27, death 41.  recall and
    precision therefore do **not** check the 12<->41 pair; the barcode comparison does, in
    Z2 by algebra and in Z3 against the backend.

    Same blindness as the Betti curve in S6 and the coverings in §3.4, one level further
    out: curve, then coverings, then the event log.
    """
    _, indices = zigzag_of.complexes(cycle_lifecycle)
    bars = zigzag_of.bars(cycle_lifecycle, 1)
    births = sorted(birth for birth, _ in bars)
    deaths = sorted(death for _, death in bars)
    correct = ((births[0], deaths[0]), (births[1], deaths[1]))
    crossed = ((births[0], deaths[1]), (births[1], deaths[0]))
    assert tuple(sorted(bars)) == correct

    def times(pairing):
        log = events_from_barcode(pairing, indices=indices, dim=1)
        return sorted((e.snapshot, e.type) for e in log.uncensored())

    assert times(correct) == times(crossed)


@DATASET_TESTS
def test_the_report_lines_of_section_six(family_two_dataset, config, zigzag_of):
    """Counters that describe the construction, not the accuracy.

    Written as a table so a reader sees which are zero.  Invisible bars are zero on both
    datasets, and that is a measurement about the union rule rather than an omission.
    """
    expected = {
        "cycle_lifecycle": {"censored": 3, "invisible": 0, "half": 3},
        "geometric_loop": {"censored": 4, "invisible": 0, "half": 2},
    }[family_two_dataset.ground_truth.generator]

    log, _ = log_of(family_two_dataset, config, zigzag_of)
    assert log.censored_ends == expected["censored"]
    assert log.invisible_bars == expected["invisible"]
    assert log.from_half_integer_ends() == expected["half"]


@DATASET_TESTS
def test_no_event_lands_on_a_breathing_snapshot(geometric_loop, config, zigzag_of):
    """``geometric_loop`` 4, 43, 56: the complex changes and nothing is reported.

    Named snapshot by snapshot (``docs/zigzag.md`` §5.4).  A detector that fired on any
    change of the complex would be caught exactly here.
    """
    log, _ = log_of(geometric_loop, config, zigzag_of)
    assert {event.snapshot for event in log.events}.isdisjoint({4, 43, 56})


@DATASET_TESTS
def test_the_repeated_state_yields_two_separate_events(cycle_lifecycle, config, zigzag_of):
    """Snapshots 41-47 and 55-59 are byte-identical, and 48 and 55 are still both reported.

    Deduplicating states by graph identity would collapse the birth and the death of
    :math:`H_0` into nothing (``method.md`` §2.1).
    """
    log, _ = log_of(cycle_lifecycle, config, zigzag_of)
    zero_dim = {(e.snapshot, e.type) for e in log.uncensored() if e.dim == 0}
    assert zero_dim == {(48, "birth"), (55, "death")}


@DATASET_TESTS
def test_the_whole_report_of_section_six(cycle_lifecycle, config, zigzag_of):
    """One assembled report, with the ephemeral count coming off the fine path.

    Ephemeral bars cannot be derived from the coarse barcode — no index of the diagram ever
    sees them — so they are supplied by the caller, and the report says ``None`` rather than
    zero when nobody ran the fine path.  Reporting zero for "we did not look" is the kind of
    number this project has spent two sessions removing.
    """
    from tdadyn.reference import SimplicialComplex
    from tdadyn.zigzag import (
        coarse_bar_of,
        dionysus_barcode,
        operation_stream,
        zigzag_complexes,
    )

    log, _ = log_of(cycle_lifecycle, config, zigzag_of)
    blind = event_report(
        log=log, truth=cycle_lifecycle.ground_truth.events,
        tolerance=config.zigzag.event_matching.tolerance,
    )
    assert blind.ephemeral_bars is None
    assert blind.ephemeral_outside_prologue() is None

    stream = operation_stream(
        zigzag_complexes(
            cycle_lifecycle,
            max_homology_dim=config.zigzag.max_homology_dim,
            max_nodes=config.topology_check.reference_max_nodes,
        )
    )
    present, states = set(), [SimplicialComplex(faces={})]
    for operation in stream.operations:
        if operation.kind == "insert":
            present.add(operation.simplex)
        else:
            present.discard(operation.simplex)
        states.append(SimplicialComplex.from_simplices(present))
    fine = dionysus_barcode(states, max_homology_dim=config.zigzag.max_homology_dim)
    ephemeral = [
        (birth, death)
        for dim in fine
        for birth, death in fine[dim]
        if coarse_bar_of(stream, birth, death) is None
    ]
    prologue = len(stream.prologue)

    full = event_report(
        log=log, truth=cycle_lifecycle.ground_truth.events,
        tolerance=config.zigzag.event_matching.tolerance,
        ephemeral_bars=len(ephemeral),
        ephemeral_in_prologue=sum(1 for _, death in ephemeral if death <= prologue),
    )
    assert full.metrics.recall == pytest.approx(1.0)
    assert full.metrics.precision == pytest.approx(1.0)
    assert full.ephemeral_bars == 23
    # Every one of them is an artefact of building the first complex, and none is about the
    # data: the number that matters is the second.
    assert full.ephemeral_outside_prologue() == 0
