"""The independent zigzag oracle: module, levels L1 and L2, and the toys T1-T4.

This is path B for task 2 (``docs/zigzag.md`` §5.1).  It shares no code with
:mod:`tdadyn.zigzag` — the layering guard checks that — and it arrives at the barcode by a
different route: the rank of the canonical map from the limit to the colimit on every
interval, then Mobius inversion.  A second implementation of the same algorithm would be a
vote, not a check.

The toys are built as **explicit lists of simplices** rather than through flag complexes.
The homology engine takes an explicit list (layer 1 of S2), and going through flag complexes
would drag in the arc-length constraint that has nothing to do with what is checked here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn.reference import SimplicialComplex, betti_numbers
from tdadyn.reference.oracle import (
    ZigzagModule,
    arrow_ranks,
    barcode,
    betti_curve,
    generalized_rank,
    module_from_complexes,
)


def complexes_of(*snapshots):
    """Turn snapshot simplex lists into the zigzag diagram over ``2T - 1`` indices.

    Written here rather than imported from :mod:`tdadyn.zigzag`: the oracle's tests must
    not depend on the layer the oracle checks, and for explicit simplex lists the union is
    one line.
    """
    built = [SimplicialComplex.from_simplices(simplices) for simplices in snapshots]
    diagram = [built[0]]
    for left, right in zip(built, built[1:]):
        union = {s for source in (left, right) for dim in source.faces
                 for s in source.faces[dim]}
        diagram.append(SimplicialComplex.from_simplices(union))
        diagram.append(right)
    return diagram


# --- the toys ---------------------------------------------------------------------

TWO_VERTICES = [(0,), (1,)]
ONE_EDGE = [(0,), (1,), (0, 1)]

T1 = (TWO_VERTICES, ONE_EDGE, TWO_VERTICES)
"""T1: two vertices, an edge only on snapshot 1 (``docs/zigzag.md`` §5.1)."""

THREE_VERTICES = [(0,), (1,), (2,)]
TRIANGLE_BOUNDARY = THREE_VERTICES + [(0, 1), (1, 2), (0, 2)]

T2 = (THREE_VERTICES, TRIANGLE_BOUNDARY, THREE_VERTICES)
"""T2: the boundary of a triangle appears and disappears.  Not filled: the 2-simplex is
absent from the list, which an explicit complex allows and a flag complex would not."""

# T3: poles 0 and 1 joined by paths through 2, 3 and 4.  No triangles, so beta_1 is the
# cyclomatic number: one cycle at first, two once the third path arrives, one again after
# an edge of the *first* cycle is cut.
_VERTICES = [(v,) for v in range(5)]
_PATH_2 = [(0, 2), (1, 2)]
_PATH_3 = [(0, 3), (1, 3)]
_PATH_4 = [(0, 4), (1, 4)]
ONE_CYCLE = _VERTICES + _PATH_2 + _PATH_3
THETA = ONE_CYCLE + _PATH_4
THETA_CUT = _VERTICES + [(1, 2)] + _PATH_3 + _PATH_4

T3 = (ONE_CYCLE, THETA, THETA_CUT)
"""T3: the fork of §3.2 in miniature — which of two classes dies when an edge is cut."""


def hand_module(last_arrow):
    """The T4 module ``{[1,4],[2,3]}`` or ``{[1,3],[2,4]}``, differing only in one arrow.

    Orientation is a real zigzag (forward, backward, forward), so the code path exercised
    is the one the datasets use.
    """
    forward = np.array([[True], [False]])           # V1 -> V2
    backward = np.eye(2, dtype=bool)                # V3 -> V2
    return ZigzagModule(
        dimensions=(1, 2, 2, 1),
        arrows=(
            (True, forward),
            (False, backward),
            (True, np.array(last_arrow, dtype=bool)),
        ),
    )


T4_NESTED = [[True, False]]
"""``{[1,4],[2,3]}``: the last arrow keeps the first generator, so the outer bar reaches 4."""

T4_CROSSING = [[False, True]]
"""``{[1,3],[2,4]}``: the last arrow keeps the second generator instead."""


# --- the module -------------------------------------------------------------------


def test_the_arrow_direction_is_read_off_the_inclusions():
    """Nothing about direction is supplied from outside; the inclusion decides.

    Note that directions do **not** simply alternate.  On T1 the transition adds an edge,
    so the union equals the next snapshot and three of the four inclusions point forward:
    ``K_0 < K_½ = K_1 = K_1½ > K_2``.  Expecting strict alternation would be assuming the
    transitions are all strict, which on family 2 is false for 95% of them.
    """
    module = module_from_complexes(complexes_of(*T1), 0)
    assert [forward for forward, _ in module.arrows] == [True, True, True, False]

    # T3 ends with a removal, so its last inclusion points the other way.
    reversed_end = module_from_complexes(complexes_of(*T3), 1)
    assert [forward for forward, _ in reversed_end.arrows] == [True, True, True, False]


def test_complexes_that_are_not_nested_are_refused():
    left = SimplicialComplex.from_simplices([(0,), (1,), (0, 1)])
    right = SimplicialComplex.from_simplices([(0,), (2,), (0, 2)])
    with pytest.raises(ValueError, match="nested neither way"):
        module_from_complexes([left, right], 0)


def test_equal_complexes_give_an_identity_arrow():
    same = SimplicialComplex.from_simplices(ONE_EDGE)
    module = module_from_complexes([same, same], 0)
    _, matrix = module.arrows[0]
    assert matrix.shape == (1, 1)
    assert matrix.all()


# --- L1 ---------------------------------------------------------------------------


def test_l1_reproduces_the_betti_numbers_of_every_index():
    diagram = complexes_of(*T3)
    for dim in (0, 1):
        assert betti_curve(diagram, dim) == tuple(
            betti_numbers(complex_, dim)[dim] for complex_ in diagram
        )


def test_l1_on_t1_matches_the_covering_the_document_states():
    # docs/zigzag.md 5.1: covering (2,1,1,1,2) over indices 0, 1/2, 1, 3/2, 2.
    assert betti_curve(complexes_of(*T1), 0) == (2, 1, 1, 1, 2)


def test_l1_on_t2_shows_the_loop_only_around_the_middle():
    assert betti_curve(complexes_of(*T2), 1) == (0, 1, 1, 1, 0)
    assert betti_curve(complexes_of(*T2), 0) == (3, 1, 1, 1, 3)


def test_l1_on_t3_cannot_tell_the_two_pairings_apart():
    """The Betti curve of T3 is symmetric, and that is the whole point of T3.

    Both ``{[0,1½],[½,2]}`` and ``{[0,2],[½,1½]}`` produce (1,2,2,2,1).  L1 is necessary
    and not sufficient; the pairing is decided at L3 (``docs/zigzag.md`` §3.4).
    """
    assert betti_curve(complexes_of(*T3), 1) == (1, 2, 2, 2, 1)


# --- L2 ---------------------------------------------------------------------------


def test_l2_on_t1_counts_the_components_that_survive_the_arrow():
    # The first arrow merges the two vertices into one component: rank 1 out of dim 2.
    # The middle complexes are equal, so those arrows are isomorphisms.
    assert arrow_ranks(module_from_complexes(complexes_of(*T1), 0)) == (1, 1, 1, 1)


def test_l2_on_t3_is_also_blind_to_the_pairing():
    """Arrow ranks (1,2,2,1) follow from either pairing — §3.4 again, one level up."""
    assert arrow_ranks(module_from_complexes(complexes_of(*T3), 1)) == (1, 2, 2, 1)


def test_l2_never_exceeds_the_dimensions_it_connects():
    module = module_from_complexes(complexes_of(*T3), 1)
    for (forward, matrix), rank in zip(module.arrows, arrow_ranks(module)):
        assert rank <= min(matrix.shape)


def test_the_two_t4_modules_are_indistinguishable_below_l3():
    """Built by hand so that the coverings agree and only the barcode differs.

    ``docs/zigzag.md`` §3.4: vertex covering (1,2,2,1) and arrow covering (1,2,1) are the
    same for both, so a checker that stops at L1 or L2 passes on the wrong answer.
    """
    nested, crossing = hand_module(T4_NESTED), hand_module(T4_CROSSING)
    assert nested.dimensions == crossing.dimensions
    assert arrow_ranks(nested) == arrow_ranks(crossing) == (1, 2, 1)


def test_a_module_with_mismatched_arrow_shapes_is_refused():
    with pytest.raises(ValueError, match="shape"):
        ZigzagModule(dimensions=(1, 2), arrows=((True, np.eye(3, dtype=bool)),))


def test_a_module_needs_one_arrow_fewer_than_it_has_spaces():
    with pytest.raises(ValueError, match="arrows"):
        ZigzagModule(dimensions=(1, 2, 2), arrows=((True, np.array([[True]])),))


# --- L3: generalized rank and Mobius inversion -------------------------------------


def test_l3_on_t1_reproduces_the_barcode_written_by_hand():
    """``docs/zigzag.md`` §5.1 states it: ``{[0,2], [0,0], [2,2]}`` in snapshot terms.

    In positions along the zigzag those are ``(0,4)``, ``(0,0)`` and ``(4,4)``: the two
    vertices are one component for the whole middle, and each end carries one extra
    component that exists only at that snapshot.
    """
    assert barcode(module_from_complexes(complexes_of(*T1), 0)) == ((0, 0), (0, 4), (4, 4))


def test_l3_on_t2_gives_one_loop_with_half_integer_ends():
    """The loop lives on the union before it lives on a snapshot, and after it too.

    Positions 1 and 3 are the unions ``K_½`` and ``K_1½``.  The edges arrive with snapshot
    1, so ``K_½ = K_1`` already carries the loop; they leave with snapshot 2, so
    ``K_1½ = K_1`` still carries it.  Exactly the half-step of ``docs/zigzag.md`` §5.1a.
    """
    assert barcode(module_from_complexes(complexes_of(*T2), 1)) == ((1, 3),)
    assert barcode(module_from_complexes(complexes_of(*T2), 0)) == (
        (0, 0), (0, 0), (0, 4), (4, 4), (4, 4)
    )


def test_l3_on_t3_decides_the_pairing_that_l1_and_l2_could_not():
    """T3 is the fork of §3.2 in miniature, and only L3 resolves it.

    The cut removes an edge of the **first** cycle, so the class carried in from position 0
    is the one that dies, and the class that survives to the end is the one born with the
    third path.  The alternative pairing ``{(0,4), (1,3)}`` has the same Betti curve and the
    same arrow ranks — both are checked above — so a checker stopping below L3 accepts it.
    """
    bars = barcode(module_from_complexes(complexes_of(*T3), 1))
    assert bars == ((0, 3), (1, 4))
    assert bars != ((0, 4), (1, 3))


def test_l3_tells_the_two_t4_modules_apart():
    """The test of discriminating power (``docs/zigzag.md`` §5.1, T4).

    Same dimensions, same arrow ranks, different barcodes.  Without this the oracle could
    be a covering counter and nothing would say so.
    """
    assert barcode(hand_module(T4_NESTED)) == ((0, 3), (1, 2))
    assert barcode(hand_module(T4_CROSSING)) == ((0, 2), (1, 3))


# --- properties of the oracle -----------------------------------------------------


@pytest.mark.parametrize("toy, dim", [(T1, 0), (T2, 0), (T2, 1), (T3, 0), (T3, 1)])
def test_the_bars_account_for_every_dimension(toy, dim):
    """Sum of bar lengths equals the total dimension: nothing invented, nothing lost."""
    module = module_from_complexes(complexes_of(*toy), dim)
    bars = barcode(module)
    assert sum(death - birth + 1 for birth, death in bars) == sum(module.dimensions)


@pytest.mark.parametrize("toy, dim", [(T1, 0), (T2, 1), (T3, 1)])
def test_the_bars_covering_an_index_reproduce_its_betti_number(toy, dim):
    """L3 against L1: necessary, and **not** sufficient — T3 above is the counterexample."""
    diagram = complexes_of(*toy)
    module = module_from_complexes(diagram, dim)
    for position, expected in enumerate(betti_curve(diagram, dim)):
        covering = sum(1 for birth, death in barcode(module) if birth <= position <= death)
        assert covering == expected


@pytest.mark.parametrize("toy, dim", [(T1, 0), (T3, 1)])
def test_the_bars_crossing_an_arrow_reproduce_its_rank(toy, dim):
    """L3 against L2: also necessary and also not sufficient (T4)."""
    module = module_from_complexes(complexes_of(*toy), dim)
    for position, rank in enumerate(arrow_ranks(module)):
        crossing = sum(
            1 for birth, death in barcode(module) if birth <= position and death >= position + 1
        )
        assert crossing == rank


def test_the_generalized_rank_does_not_grow_when_the_interval_does():
    """``r(b,d)`` counts the bars containing ``[b,d]``, so widening can only drop it.

    The barcode search prunes on this — once ``r`` hits zero the wider intervals are
    skipped — so the property is checked rather than assumed.
    """
    module = module_from_complexes(complexes_of(*T3), 1)
    n = len(module.dimensions)
    for birth in range(n):
        ranks = [generalized_rank(module, birth, death) for death in range(birth, n)]
        assert ranks == sorted(ranks, reverse=True)


def test_the_generalized_rank_does_not_depend_on_which_component_is_read():
    """The canonical map sends a compatible tuple to the class of *any* component.

    The implementation reads the leftmost one.  If that choice mattered, the construction
    would be wrong, and the error would look exactly like a wrong barcode.
    """
    module = module_from_complexes(complexes_of(*T3), 1)
    n = len(module.dimensions)
    for birth in range(n):
        for death in range(birth, n):
            expected = generalized_rank(module, birth, death)
            for component in range(birth, death + 1):
                assert generalized_rank(module, birth, death, component=component) == expected


def test_the_oracle_refuses_a_field_other_than_two():
    with pytest.raises(NotImplementedError, match="Z/2"):
        module_from_complexes(complexes_of(*T1), 0, coeff_field=3)


def test_an_interval_outside_the_module_is_refused():
    module = hand_module(T4_NESTED)
    with pytest.raises(ValueError, match="interval"):
        generalized_rank(module, 2, 1)
    with pytest.raises(ValueError, match="interval"):
        generalized_rank(module, 0, 9)
