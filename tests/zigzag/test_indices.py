"""The coarse index layer: half-integer indices and the refusal to round them.

Why this is a layer with its own tests rather than arithmetic inline in the adapter:
an off-by-one here appears in the report as a systematic localisation error of the
algorithm and is indistinguishable from one (``docs/zigzag.md`` §2.4).  So the arithmetic
is written once, in one place, and checked against the definition rather than against
whatever the adapter happens to do.
"""

from __future__ import annotations

import pytest

from tdadyn.zigzag import CoarseIndex, index_set


def test_the_index_set_has_two_t_minus_one_entries():
    # docs/zigzag.md 2.1: I = {0, 1/2, 1, ..., T-1}, snapshot indices and the
    # intermediate unions between them.
    for n_snapshots in (2, 3, 60):
        assert len(index_set(n_snapshots)) == 2 * n_snapshots - 1


def test_the_index_set_alternates_snapshot_and_intermediate():
    indices = index_set(4)
    assert [index.is_snapshot for index in indices] == [True, False, True, False, True,
                                                        False, True]
    assert [index.snapshot for index in indices if index.is_snapshot] == [0, 1, 2, 3]


def test_the_index_set_is_sorted_and_starts_at_the_first_snapshot():
    indices = index_set(5)
    assert indices == tuple(sorted(indices))
    assert indices[0] == CoarseIndex.at_snapshot(0)
    assert indices[-1] == CoarseIndex.at_snapshot(4)


def test_an_intermediate_lies_strictly_between_its_neighbours():
    left, right = CoarseIndex.at_snapshot(7), CoarseIndex.at_snapshot(8)
    middle = CoarseIndex.between(7)
    assert left < middle < right
    assert not middle.is_snapshot


def test_asking_an_intermediate_for_its_snapshot_is_an_error_not_a_rounding():
    """Half-integer ends are never rounded (``docs/zigzag.md`` §4.2).

    A birth at :math:`i + 1/2` means a class present in neither snapshot but present in
    their union — a real phenomenon of the construction.  Rounding would turn it into a
    plausible event at a wrong time, which is worse than refusing.
    """
    middle = CoarseIndex.between(20)
    with pytest.raises(ValueError, match="intermediate"):
        middle.snapshot


def test_a_snapshot_index_round_trips():
    for i in (0, 1, 41, 59):
        assert CoarseIndex.at_snapshot(i).snapshot == i


def test_the_doubling_is_the_only_representation():
    # 2i for a snapshot, 2i+1 for the union that follows it: exact arithmetic, and
    # "snapshot or intermediate" is a parity test rather than a float comparison.
    assert CoarseIndex.at_snapshot(20).doubled == 40
    assert CoarseIndex.between(20).doubled == 41


def test_indices_print_the_way_the_documents_write_them():
    assert str(CoarseIndex.at_snapshot(12)) == "12"
    assert str(CoarseIndex.between(20)) == "20½"


def test_a_negative_or_non_integer_position_is_refused():
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises((TypeError, ValueError)):
            CoarseIndex.at_snapshot(bad)
        with pytest.raises((TypeError, ValueError)):
            CoarseIndex.between(bad)


def test_index_set_needs_at_least_two_snapshots():
    # A single snapshot is not a zigzag: there is no arrow and no intermediate.
    with pytest.raises(ValueError, match="at least two"):
        index_set(1)
