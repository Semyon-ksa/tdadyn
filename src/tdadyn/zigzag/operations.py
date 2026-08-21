"""The diagram of complexes unfolded into elementary insertions and removals.

Backends do not take a diagram; they take a stream of operations on single simplices
(``docs/zigzag.md`` §2.3).  Unfolding is where a mistake stops being visible: a barcode
computed from a wrong stream is still self-consistent, so nothing downstream objects.  The
guard is the invariant of this module — replaying the stream from nothing reproduces the
complexes exactly, and every prefix is a simplicial complex.

Order within a transition is forced by that second requirement:

* :math:`K_i \\to K_{i+1/2}` inserts by **ascending** dimension — a vertex must arrive
  before the edge that needs it;
* :math:`K_{i+1/2} \\to K_{i+1}` removes by **descending** dimension — a triangle must
  leave before the edge it sits on.

Within one dimension the order is arbitrary and does not affect the coarse barcode, but it
does affect the ephemeral bars (born and dead inside one transition).  It is therefore
**fixed** — lexicographic — rather than left to set iteration: Z5 compares barcodes at
noise level 0.0 against the clean dataset bitwise, and a non-deterministic order would make
that comparison meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

from tdadyn.reference import SimplicialComplex

from .indices import CoarseIndex, index_set

OperationKind = Literal["insert", "remove"]


class CoarseSpan(NamedTuple):
    """The closed range of coarse indices that share one state of the complex.

    Almost every transition of family 2 is **empty**: the graph does not change, so
    :math:`K_i`, :math:`K_{i+1/2}` and :math:`K_{i+1}` are the same complex and sit at the
    same operation counter.  Measured on the shipped datasets, 95% of the 119 transitions
    are empty and as many as 45 consecutive indices share a single counter.

    So an operation counter does **not** determine a coarse index, and a function returning
    one would be picking silently.  It determines this span, and a caller that needs a
    single index has to say which end it means: the left end for a birth, the right end for
    a death.  Getting that wrong displaces a bar by up to 45 indices — far past the
    off-by-one ``docs/zigzag.md`` §2.4 warns about, and equally indistinguishable in a
    report from a defect of the algorithm.
    """

    first: CoarseIndex
    last: CoarseIndex


@dataclass(frozen=True)
class Operation:
    """One elementary step: a single simplex enters or leaves the complex."""

    kind: OperationKind
    simplex: tuple[int, ...]


@dataclass(frozen=True)
class OperationStream:
    """The whole unfolding, with the map back to coarse indices.

    Attributes
    ----------
    operations : tuple of Operation
        The flat stream, in order.
    blocks : tuple of tuple of Operation
        The same operations grouped by transition.  The first block is the prologue that
        builds :math:`K_0` from the empty complex; after it there is one block per arrow.
    checkpoints : tuple of int
        One counter per coarse index: after ``checkpoints[j]`` operations the state equals
        the complex at the ``j``-th index.  ``checkpoints[0]`` is the length of the
        prologue, ``checkpoints[-1]`` the length of the whole stream.
    indices : tuple of CoarseIndex
        The coarse indices the checkpoints correspond to.
    """

    operations: tuple[Operation, ...]
    blocks: tuple[tuple[Operation, ...], ...]
    checkpoints: tuple[int, ...]
    indices: tuple[CoarseIndex, ...]

    @property
    def prologue(self) -> tuple[Operation, ...]:
        """The operations that build :math:`K_0` from nothing.

        Kept in the stream rather than treated as a starting state: the left censoring of a
        bar that runs from index 0 (``docs/zigzag.md`` §5.1a) is only visible to a layer
        that saw the first complex being assembled.
        """
        return self.blocks[0]

    def coarse_span_of(self, counter: int) -> CoarseSpan:
        """Translate an operation counter into the coarse indices it stands for.

        Parameters
        ----------
        counter : int
            Number of operations already applied, from 0 to ``len(operations)``.

        Returns
        -------
        CoarseSpan
            Every index whose complex equals the state after ``counter`` operations, as a
            closed range.  A caller needing one index picks ``first`` for a birth and
            ``last`` for a death, and thereby says so.

        Notes
        -----
        Two distinct reasons make this a range rather than an index, and neither is a corner
        case here:

        * an empty transition leaves several indices sharing one state, and the backend
          never sees the transition at all — it has no operation to report;
        * a counter falling strictly inside a non-empty block is mid-transition and is no
          :math:`K` at all; it is reported as the last index already reached, so that the
          map stays total and monotone.

        ``docs/zigzag.md`` does not fix either, so both are fixed here and tested rather
        than decided again inside an adapter, where they would be invisible.
        """
        if not isinstance(counter, int) or isinstance(counter, bool):
            raise TypeError(f"counter must be an int, got {type(counter).__name__}")
        if counter < 0:
            raise ValueError(f"counter must not be negative, got {counter}")
        if counter > len(self.operations):
            raise ValueError(
                f"counter {counter} lies beyond the stream of {len(self.operations)} "
                "operations"
            )
        reached = [
            position
            for position, checkpoint in enumerate(self.checkpoints)
            if checkpoint <= counter
        ]
        last = reached[-1] if reached else 0
        matching = [
            position
            for position in reached
            if self.checkpoints[position] == self.checkpoints[last]
        ]
        first = matching[0] if matching else last
        return CoarseSpan(first=self.indices[first], last=self.indices[last])


def operation_stream(diagram: dict[CoarseIndex, SimplicialComplex]) -> OperationStream:
    """Unfold a diagram of complexes into elementary operations.

    Parameters
    ----------
    diagram : dict of CoarseIndex to SimplicialComplex
        The output of :func:`tdadyn.zigzag.construction.zigzag_complexes`.

    Returns
    -------
    OperationStream
        The stream, its blocks and the checkpoints back to coarse indices.
    """
    indices = tuple(diagram)
    if not indices:
        raise ValueError("the diagram is empty; there is nothing to unfold")

    blocks: list[tuple[Operation, ...]] = []
    previous: set[tuple[int, ...]] = set()
    for index in indices:
        current = _simplices(diagram[index])
        blocks.append(_transition(previous, current))
        previous = current

    operations = tuple(operation for block in blocks for operation in block)
    checkpoints, running = [], 0
    for block in blocks:
        running += len(block)
        checkpoints.append(running)
    return OperationStream(
        operations=operations,
        blocks=tuple(blocks),
        checkpoints=tuple(checkpoints),
        indices=indices,
    )


def replay(stream: OperationStream) -> dict[CoarseIndex, SimplicialComplex]:
    """Rebuild the diagram from the stream alone.

    Parameters
    ----------
    stream : OperationStream
        The unfolding to replay.

    Returns
    -------
    dict of CoarseIndex to SimplicialComplex
        The complex at each coarse index, reconstructed by applying the operations in
        order and reading the state off at the checkpoints.

    Notes
    -----
    This is deliberately a second, independent path from the stream back to the complexes:
    it knows nothing about snapshots, edge rules or unions.  Agreement between it and the
    direct construction is the invariant of the layer.
    """
    rebuilt: dict[CoarseIndex, SimplicialComplex] = {}
    present: set[tuple[int, ...]] = set()
    position = 0
    for index, checkpoint in zip(stream.indices, stream.checkpoints):
        while position < checkpoint:
            operation = stream.operations[position]
            if operation.kind == "insert":
                present.add(operation.simplex)
            else:
                present.discard(operation.simplex)
            position += 1
        rebuilt[index] = SimplicialComplex.from_simplices(present)
    return rebuilt


def _simplices(complex_: SimplicialComplex) -> set[tuple[int, ...]]:
    return {simplex for dim in complex_.faces for simplex in complex_.faces[dim]}


def _transition(
    before: set[tuple[int, ...]], after: set[tuple[int, ...]]
) -> tuple[Operation, ...]:
    """Operations turning one complex into the next.

    Insertions ascend in dimension and removals descend, so that the state after each
    single operation is still closed under faces.  Both directions can appear only if the
    diagram is malformed: an arrow of the zigzag points one way.
    """
    inserted = sorted(after - before, key=_by_dimension)
    removed = sorted(before - after, key=_by_dimension, reverse=True)
    return tuple(
        [Operation("insert", simplex) for simplex in inserted]
        + [Operation("remove", simplex) for simplex in removed]
    )


def _by_dimension(simplex: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """Sort key: dimension first, then lexicographic so the order is reproducible."""
    return len(simplex), simplex
