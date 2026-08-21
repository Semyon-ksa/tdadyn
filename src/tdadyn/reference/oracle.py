"""The independent zigzag oracle: barcode over Z/2 without a persistence library.

This is path B for task 2.  Its whole value is that it computes the answer a **different
way** rather than a second time: the barcode comes out of the generalized rank invariant
:math:`r(b,d) = \\mathrm{rank}(\\varprojlim \\to \\varinjlim)` and Mobius inversion
(``docs/zigzag.md`` §3.3), with no persistence algorithm anywhere.  A second implementation
of the same algorithm would add a vote, not a check: when two of those disagree neither
says which is right.

**The oracle is given the complexes; it does not build them.**  Building them here would
mean a second implementation of the union rule, forbidden for the same reason as a second
implementation of the edge rule.  The independence that matters is of the *answer*: the
oracle and the backend have to run on the same complexes or there is nothing to compare.

**It speaks in positions, not in coarse indices.**  A module is a sequence
:math:`V_0, \\dots, V_{N-1}` with an arrow between neighbours; that half-integer positions
mean unions is the caller's knowledge, and keeping it there is what lets this module stay
free of any import from :mod:`tdadyn.zigzag`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .homology import (
    REFERENCE_COEFF_FIELD,
    SimplicialComplex,
    boundary_matrix,
    coordinates_mod2,
    nullspace_mod2,
    rank_mod2,
    row_reduce_mod2,
)

Arrow = tuple[bool, np.ndarray]
"""``(forward, matrix)``: ``forward`` says the map goes from the left space to the right
one, and ``matrix`` has shape ``(dim target, dim source)``."""


@dataclass(frozen=True)
class ZigzagModule:
    """A representation of an :math:`A_N` quiver over Z/2.

    Parameters
    ----------
    dimensions : tuple of int
        :math:`\\dim V_i` at each position.
    arrows : tuple of (bool, numpy.ndarray)
        One per neighbouring pair, in order.  Orientation is arbitrary and may mix
        directions — that is what makes it a zigzag rather than a filtration.

    Notes
    -----
    Built either from a diagram of complexes (:func:`module_from_complexes`) or directly
    from matrices.  The second path is not a convenience: the discriminating-power test of
    ``docs/zigzag.md`` §5.1 (T4) is a *module* with no data behind it, and an oracle that
    could only be fed complexes could not be tested against it.
    """

    dimensions: tuple[int, ...]
    arrows: tuple[Arrow, ...]

    def __post_init__(self) -> None:
        if len(self.arrows) != len(self.dimensions) - 1:
            raise ValueError(
                f"{len(self.dimensions)} spaces need {len(self.dimensions) - 1} arrows, "
                f"got {len(self.arrows)}"
            )
        for position, (forward, matrix) in enumerate(self.arrows):
            left, right = self.dimensions[position], self.dimensions[position + 1]
            source, target = (left, right) if forward else (right, left)
            if matrix.shape != (target, source):
                raise ValueError(
                    f"arrow {position} has shape {matrix.shape}, expected "
                    f"{(target, source)} for a map between spaces of dimension "
                    f"{source} and {target}"
                )

    def source_and_target(self, position: int) -> tuple[int, int]:
        """Positions the arrow maps from and to."""
        forward, _ = self.arrows[position]
        return (position, position + 1) if forward else (position + 1, position)


def betti_curve(complexes, dim: int, *, coeff_field: int = REFERENCE_COEFF_FIELD):
    """Level L1: :math:`\\beta_k` at every position.

    Necessary and **not** sufficient: the Betti curve is blind to pairing, which is the
    lesson of S6 one level up (``docs/zigzag.md`` §3.4).
    """
    _check_field(coeff_field)
    return tuple(len(_homology_basis(complex_, dim)[0]) for complex_ in complexes)


def arrow_ranks(module: ZigzagModule) -> tuple[int, ...]:
    """Level L2: the rank of the induced map on each arrow.

    For an interval-decomposable module this is the number of bars containing both ends of
    the arrow.  Necessary and still not sufficient: ``{[1,4],[2,3]}`` and ``{[1,3],[2,4]}``
    have the same arrow ranks.
    """
    return tuple(rank_mod2(matrix) for _, matrix in module.arrows)


def module_from_complexes(
    complexes, dim: int, *, coeff_field: int = REFERENCE_COEFF_FIELD
) -> ZigzagModule:
    """Build the homology module of a zigzag of complexes.

    Parameters
    ----------
    complexes : sequence of SimplicialComplex
        Neighbouring complexes must be nested one way or the other; the direction of each
        arrow is **read off the inclusion** rather than supplied, because in a zigzag
        :math:`K_i \\subseteq K_{i+1/2} \\supseteq K_{i+1}` and anything else means the
        diagram was not built as a zigzag.
    dim : int
        Homology dimension.
    coeff_field : int, default 2
        Only 2 is implemented.

    Returns
    -------
    ZigzagModule
        Spaces :math:`H_k(K_i)` and the maps induced by the inclusions.

    Notes
    -----
    When two neighbours are **equal** — which is what an empty transition looks like, and
    on family 2 that is most of them — both inclusions hold and the direction is genuinely
    arbitrary.  The map is an isomorphism either way, so neither L2 nor the barcode can
    notice; ``forward`` is recorded in that case.
    """
    _check_field(coeff_field)
    complexes = list(complexes)
    if len(complexes) < 2:
        raise ValueError(f"a zigzag needs at least two complexes, got {len(complexes)}")

    prepared = [_homology_basis(complex_, dim) for complex_ in complexes]
    dimensions = tuple(len(basis) for basis, _ in prepared)

    arrows: list[Arrow] = []
    for position, (left, right) in enumerate(zip(complexes, complexes[1:])):
        left_all = _all_simplices(left)
        right_all = _all_simplices(right)
        if left_all <= right_all:
            forward = True
            source, target = position, position + 1
        elif right_all <= left_all:
            forward = False
            source, target = position + 1, position
        else:
            raise ValueError(
                f"complexes at positions {position} and {position + 1} are nested "
                "neither way; a zigzag has K_i <= K_i+1/2 >= K_i+1, so this diagram was "
                "not built as one"
            )
        arrows.append(
            (forward, _induced_map(complexes[source], complexes[target],
                                   prepared[source], prepared[target], dim))
        )
    return ZigzagModule(dimensions=dimensions, arrows=tuple(arrows))


# --- homology bases and induced maps ----------------------------------------------


def _homology_basis(complex_: SimplicialComplex, dim: int):
    """A basis of :math:`H_k` as cycle vectors, plus the boundary matrix it lives with.

    Returns ``(basis_columns, boundaries)``: the basis as a list of boolean vectors over
    the ``dim``-simplices, and :math:`\\partial_{k+1}` whose columns span :math:`B_k`.

    The basis is obtained in one elimination rather than in a loop of rank tests: reduce
    ``[B | Z]`` and keep the columns of ``Z`` that carry a pivot.  Those extend a basis of
    :math:`B_k` to one of :math:`Z_k`, which is exactly a basis of the quotient.
    """
    cycles = nullspace_mod2(boundary_matrix(complex_, dim))
    boundaries = boundary_matrix(complex_, dim + 1)
    n_simplices = len(complex_.simplices(dim))
    if boundaries.shape[0] != n_simplices:
        boundaries = np.zeros((n_simplices, 0), dtype=bool)

    stacked = np.column_stack([boundaries, cycles]) if cycles.shape[1] else boundaries
    _, pivots = row_reduce_mod2(stacked)
    offset = boundaries.shape[1]
    basis = [cycles[:, index - offset] for index in pivots if index >= offset]
    return basis, boundaries


def _induced_map(source, target, source_prepared, target_prepared, dim: int) -> np.ndarray:
    """Matrix of :math:`H_k(\\text{source}) \\to H_k(\\text{target})` for an inclusion.

    A class is carried over by extending its cycle representative with zeros onto the
    target's simplices, then written over the target's homology basis **plus** its
    boundaries; the boundary coordinates are dropped, which is precisely "modulo
    :math:`B_k`".
    """
    source_basis, _ = source_prepared
    target_basis, target_boundaries = target_prepared
    target_simplices = {simplex: index
                        for index, simplex in enumerate(target.simplices(dim))}
    positions = [target_simplices[simplex] for simplex in source.simplices(dim)]

    spanning = _spanning(target_basis, target_boundaries, len(target_simplices))
    matrix = np.zeros((len(target_basis), len(source_basis)), dtype=bool)
    for column, cycle in enumerate(source_basis):
        carried = np.zeros(len(target_simplices), dtype=bool)
        if positions:
            carried[positions] = cycle
        solution = coordinates_mod2(spanning, carried)
        if solution is None:  # pragma: no cover - would mean a cycle is not a cycle
            raise AssertionError(
                "a cycle of the source is not a cycle of the target; the complexes are "
                "not nested or the boundary matrices are inconsistent"
            )
        matrix[:, column] = solution[: len(target_basis)]
    return matrix


def _spanning(basis, boundaries, n_simplices: int) -> np.ndarray:
    """Homology basis followed by the boundaries: together they span the cycle space."""
    columns = list(basis) + [boundaries[:, index] for index in range(boundaries.shape[1])]
    if not columns:
        return np.zeros((n_simplices, 0), dtype=bool)
    return np.column_stack(columns)


def _all_simplices(complex_: SimplicialComplex) -> set[tuple[int, ...]]:
    return {simplex for dim in complex_.faces for simplex in complex_.faces[dim]}


def _check_field(coeff_field: int) -> None:
    if coeff_field != REFERENCE_COEFF_FIELD:
        raise NotImplementedError(
            f"the oracle works only over Z/{REFERENCE_COEFF_FIELD}, got "
            f"coeff_field={coeff_field}; the config refuses p != 2 for the same reason "
            "(docs/zigzag.md 3.1)"
        )
