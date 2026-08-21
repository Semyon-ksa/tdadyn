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


# --- L3: generalized rank and Mobius inversion ------------------------------------


def generalized_rank(
    module: ZigzagModule, birth: int, death: int, *, component: int | None = None
) -> int:
    """Level L3: :math:`r(b,d)`, the rank of :math:`\\varprojlim \\to \\varinjlim`.

    Parameters
    ----------
    module : ZigzagModule
        The module.
    birth, death : int
        Closed interval of positions, ``birth <= death``.
    component : int, optional
        Which component of a compatible tuple the canonical map reads.  Any of them gives
        the same rank — that is what makes the map canonical — and the parameter exists so
        that a test can check it rather than take it on trust.  Defaults to ``birth``.

    Returns
    -------
    int
        The rank.  Because every module over :math:`A_N` is interval decomposable
        (Gabriel), this equals the number of bars containing ``[birth, death]``, which is
        what Mobius inversion then turns back into multiplicities.

    Notes
    -----
    Over Z/2 both constructions are ranks of matrices and nothing else.  The limit is the
    subspace of compatible tuples in :math:`\\bigoplus V_i` — the kernel of the
    compatibility matrix.  The colimit is the quotient of the same sum by the relations
    identifying a vector with its image.  The canonical map takes a compatible tuple to the
    class of one component; the relations make the choice immaterial.
    """
    n = len(module.dimensions)
    if not 0 <= birth <= death < n:
        raise ValueError(
            f"interval [{birth}, {death}] does not lie inside a module of {n} positions"
        )
    if component is None:
        component = birth
    if not birth <= component <= death:
        raise ValueError(
            f"component {component} lies outside the interval [{birth}, {death}]"
        )

    dimensions = module.dimensions[birth : death + 1]
    if any(dimension == 0 for dimension in dimensions):
        # A zero space anywhere inside kills every compatible tuple, so the limit is zero.
        return 0
    offsets, total = [], 0
    for dimension in dimensions:
        offsets.append(total)
        total += dimension

    def block_of(position: int) -> slice:
        start = offsets[position - birth]
        return slice(start, start + module.dimensions[position])

    constraints, relations = [], []
    for position in range(birth, death):
        _, matrix = module.arrows[position]
        source, target = module.source_and_target(position)
        source_block, target_block = block_of(source), block_of(target)
        source_dim = module.dimensions[source]
        target_dim = module.dimensions[target]

        # Compatibility: f(v_source) + v_target = 0, one row block per arrow.
        constraint = np.zeros((target_dim, total), dtype=bool)
        constraint[:, source_block] = matrix
        constraint[:, target_block] ^= np.eye(target_dim, dtype=bool)
        constraints.append(constraint)

        # The same equations read as generators of what the colimit divides out.
        relation = np.zeros((total, source_dim), dtype=bool)
        relation[source_block, :] = np.eye(source_dim, dtype=bool)
        relation[target_block, :] ^= matrix
        relations.append(relation)

    compatibility = (
        np.vstack(constraints) if constraints else np.zeros((0, total), dtype=bool)
    )
    limit = nullspace_mod2(compatibility)

    identifications = (
        np.column_stack(relations) if relations else np.zeros((total, 0), dtype=bool)
    )
    read = block_of(component)
    images = np.zeros((total, limit.shape[1]), dtype=bool)
    images[read, :] = limit[read, :]

    if images.shape[1] == 0:
        return 0
    return rank_mod2(np.column_stack([images, identifications])) - rank_mod2(identifications)


def barcode(module: ZigzagModule) -> tuple[tuple[int, int], ...]:
    """The full barcode, by Mobius inversion of the generalized rank.

    Returns
    -------
    tuple of (int, int)
        Bars as closed intervals of positions, sorted, **with repetition**: a bar of
        multiplicity two appears twice.  Comparing barcodes as sets would silently lose a
        repeated bar, and the comparisons this oracle exists for are bar for bar.

    Notes
    -----
    :math:`m_{[b,d]} = r(b,d) - r(b^-,d) - r(b,d^+) + r(b^-,d^+)`, with :math:`r := 0`
    outside the module (``docs/zigzag.md`` §3.3).

    The search prunes on the monotonicity of :math:`r`: for a fixed birth, once
    :math:`r(b,d)` reaches zero every wider interval is zero as well and is skipped.  That
    is the only shortcut taken, it is exact rather than heuristic, and it is checked by a
    test instead of being assumed.
    """
    n = len(module.dimensions)
    ranks: dict[tuple[int, int], int] = {}
    for birth in range(n):
        for death in range(birth, n):
            value = generalized_rank(module, birth, death)
            if value == 0:
                break
            ranks[(birth, death)] = value

    def rank_at(birth: int, death: int) -> int:
        if birth < 0 or death >= n or birth > death:
            return 0
        return ranks.get((birth, death), 0)

    bars: list[tuple[int, int]] = []
    for (birth, death), value in sorted(ranks.items()):
        multiplicity = (
            value
            - rank_at(birth - 1, death)
            - rank_at(birth, death + 1)
            + rank_at(birth - 1, death + 1)
        )
        if multiplicity < 0:
            raise AssertionError(
                f"Mobius inversion gave multiplicity {multiplicity} for [{birth}, {death}]; "
                "the generalized ranks are inconsistent, which means a bug in this module"
            )
        bars.extend([(birth, death)] * multiplicity)
    return tuple(sorted(bars))


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
