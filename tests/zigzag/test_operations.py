"""The elementary operation stream: the criterion of this session.

Backends take a stream of single-simplex insertions and removals, not a diagram of
complexes.  Unfolding the diagram into that stream is where an error becomes an error of
the barcode — and an unnoticeable one, since a barcode computed from a wrong stream is
still self-consistent.  Hence the two checks below: replaying the stream from nothing must
reproduce the complexes exactly, and every prefix of it must be a simplicial complex.
"""

from __future__ import annotations

import pytest

from tdadyn.reference import SimplicialComplex
from tdadyn.zigzag import (
    CoarseIndex,
    index_set,
    operation_stream,
    replay,
    zigzag_complexes,
)


def build_args(config):
    return {
        "max_homology_dim": config.zigzag.max_homology_dim,
        "max_nodes": config.topology_check.reference_max_nodes,
    }


# --- the criterion ----------------------------------------------------------------


def test_replaying_the_stream_reproduces_every_complex(family_two_dataset, config):
    """The invariant of this layer (``docs/zigzag.md`` §2.3).

    Replay is a second, independent path from the stream back to the complexes: it knows
    nothing of the snapshots and only applies operations in order.  Agreement of the two
    paths is what makes the stream trustworthy before any persistence is computed on it.
    """
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    stream = operation_stream(diagram)
    assert replay(stream) == diagram


def test_every_prefix_of_the_stream_is_a_simplicial_complex(family_two_dataset, config):
    """Checked directly, not assumed.

    Intermediate states must be complexes, or the backend is fed something that is not one.
    ``SimplicialComplex.from_simplices`` verifies closure under faces and refuses to repair
    it, so building the complex at each prefix is the check.  At a few hundred operations
    over a couple of hundred simplices that is affordable; on a larger dataset one would
    fall back to the incremental invariant (an inserted simplex has all its facets present,
    a removed one has no cofaces), which is weaker only in that it assumes the start state.
    """
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    stream = operation_stream(diagram)

    present: set[tuple[int, ...]] = set()
    for position, operation in enumerate(stream.operations):
        if operation.kind == "insert":
            present.add(operation.simplex)
        else:
            present.discard(operation.simplex)
        try:
            SimplicialComplex.from_simplices(present)
        except ValueError as error:  # pragma: no cover - only on a broken unfolding
            pytest.fail(f"prefix of length {position + 1} is not a complex: {error}")


# --- the unfolding ----------------------------------------------------------------


def test_insertions_go_up_in_dimension_and_removals_go_down(family_two_dataset, config):
    """``docs/zigzag.md`` §2.3, and it is exactly what keeps the prefixes complexes.

    A vertex has to arrive before the edge that needs it, and a triangle has to leave
    before the edge it sits on.
    """
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    stream = operation_stream(diagram)
    for block in stream.blocks:
        dims = [len(operation.simplex) - 1 for operation in block]
        kinds = {operation.kind for operation in block}
        if kinds == {"insert"}:
            assert dims == sorted(dims)
        elif kinds == {"remove"}:
            assert dims == sorted(dims, reverse=True)
        else:
            assert not kinds or kinds <= {"insert", "remove"}, kinds


def test_a_block_never_mixes_insertions_and_removals(family_two_dataset, config):
    # Each arrow of the zigzag points one way: K_i -> K_i+1/2 only adds, K_i+1/2 -> K_i+1
    # only removes. A block holding both would mean the diagram was built wrongly.
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    for block in operation_stream(diagram).blocks:
        assert len({operation.kind for operation in block}) <= 1


def test_the_stream_is_deterministic(family_two_dataset, config):
    """The order inside one dimension is arbitrary but fixed, and that matters later.

    ``docs/zigzag.md`` §2.3: the order within a dimension does not change the coarse
    barcode but does change the ephemeral bars.  Z5 compares barcodes at noise level 0.0
    bitwise against the clean set, which a set-iteration order would make impossible.
    """
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    first = operation_stream(diagram)
    second = operation_stream(diagram)
    assert first.operations == second.operations


def test_the_prologue_builds_the_first_complex_from_nothing(cycle_lifecycle, config):
    """The assembly of :math:`K_0` is part of the stream, not a starting state.

    Left censoring of the bar :math:`[0, 59]` (``docs/zigzag.md`` §5.1a) is only visible to
    a layer that saw the first complex being built; starting from a non-empty state would
    hide it.
    """
    diagram = zigzag_complexes(cycle_lifecycle, **build_args(config))
    stream = operation_stream(diagram)
    first = diagram[CoarseIndex.at_snapshot(0)]
    assert len(stream.prologue) == sum(len(faces) for faces in first.faces.values())
    assert {operation.kind for operation in stream.prologue} == {"insert"}


def test_the_stream_is_as_short_as_the_diagram_demands(cycle_lifecycle, config):
    """Sizes, as the measurement that feeds the Z3 time budget.

    ``cycle_lifecycle`` moves one edge at a time, so between neighbours there is almost
    nothing to do: the stream is dominated by the 47 insertions that build the first
    complex.  The numbers are computed from the diagram rather than transcribed.
    """
    diagram = zigzag_complexes(cycle_lifecycle, **build_args(config))
    stream = operation_stream(diagram)
    first = diagram[CoarseIndex.at_snapshot(0)]

    expected_prologue = sum(len(faces) for faces in first.faces.values())
    indices = index_set(len(cycle_lifecycle.snapshots))
    expected_moves = sum(
        len(set(_simplices(diagram[after])) ^ set(_simplices(diagram[before])))
        for before, after in zip(indices, indices[1:])
    )
    assert len(stream.operations) == expected_prologue + expected_moves


def _simplices(complex_):
    return [simplex for dim in complex_.faces for simplex in complex_.faces[dim]]


# --- operation counter to coarse index --------------------------------------------


def test_the_span_at_a_checkpoint_holds_the_index_it_belongs_to(family_two_dataset, config):
    """A counter names a *range* of indices, and the index in question is inside it.

    Demanding equality with a single index would demand an inverse that does not exist:
    an empty transition leaves several indices at the same counter, and the stream carries
    nothing to tell them apart.
    """
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    stream = operation_stream(diagram)
    indices = index_set(len(family_two_dataset.snapshots))
    assert len(stream.checkpoints) == len(indices)
    assert stream.checkpoints[-1] == len(stream.operations)
    for index, counter in zip(indices, stream.checkpoints):
        span = stream.coarse_span_of(counter)
        assert span.first <= index <= span.last


def test_every_index_of_a_span_really_holds_the_same_complex(family_two_dataset, config):
    """The property that makes the span correct rather than merely convenient."""
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    stream = operation_stream(diagram)
    for counter in stream.checkpoints:
        span = stream.coarse_span_of(counter)
        expected = diagram[span.first]
        inside = [index for index in diagram if span.first <= index <= span.last]
        assert all(diagram[index] == expected for index in inside)


def test_most_transitions_are_empty_so_a_single_index_would_be_a_guess(
    family_two_dataset, config
):
    """The measurement behind :class:`CoarseSpan`, kept as a test.

    If the datasets ever changed so that every transition moved the graph, a single-index
    map would become defensible and this test would say so by failing.
    """
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    stream = operation_stream(diagram)
    empty = sum(1 for block in stream.blocks if not block)
    assert empty > 0.9 * len(stream.blocks)

    widest = max(
        1 + (span.last.doubled - span.first.doubled)
        for span in (stream.coarse_span_of(counter) for counter in stream.checkpoints)
    )
    assert widest >= 30, "a span this wide is why picking one end silently is not an option"


def test_the_checkpoints_never_go_backwards(family_two_dataset, config):
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    checkpoints = operation_stream(diagram).checkpoints
    assert list(checkpoints) == sorted(checkpoints)


def test_a_counter_inside_a_block_reports_the_index_already_reached(cycle_lifecycle, config):
    """The definition, spelled out because the document does not fix it.

    A counter strictly inside a block belongs to no coarse index — the state is
    mid-transition and is no :math:`K`.  The span then ends at the last index already
    reached, so the map stays total and monotone; Z3 translates half-open backend intervals
    against exactly this.
    """
    diagram = zigzag_complexes(cycle_lifecycle, **build_args(config))
    stream = operation_stream(diagram)
    indices = index_set(len(cycle_lifecycle.snapshots))
    checkpoints = stream.checkpoints
    for position in range(len(checkpoints) - 1):
        start, stop = checkpoints[position], checkpoints[position + 1]
        if stop - start < 2:
            continue
        for counter in range(start + 1, stop):
            assert stream.coarse_span_of(counter).last == indices[position]


def test_a_counter_beyond_the_stream_is_refused(cycle_lifecycle, config):
    diagram = zigzag_complexes(cycle_lifecycle, **build_args(config))
    stream = operation_stream(diagram)
    with pytest.raises(ValueError, match="beyond"):
        stream.coarse_span_of(len(stream.operations) + 1)
    with pytest.raises(ValueError, match="negative"):
        stream.coarse_span_of(-1)
