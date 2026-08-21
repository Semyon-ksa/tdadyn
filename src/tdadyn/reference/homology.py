"""Layer 1: homology of an explicitly given simplicial complex over Z/2.

This is path B of ``docs/validation.md`` §1 — the independent check the whole project
is validated against.  Its only virtue is that it can be trusted, so it is written to
be read rather than to be fast: dense boolean matrices, plain Gaussian elimination,
no persistence library anywhere.

**Only Z/2.** Generalising to an arbitrary prime would require modular inverses and
proper Gaussian elimination, and would complicate exactly the component whose
simplicity is what makes it worth trusting (``docs/method.md`` §1.2).  Over Z/2 the
addition of two rows is XOR and no division is needed.  A call with :math:`p \\ne 2`
raises instead of quietly computing over Z/2.

This layer takes the simplices it is given and asks no questions about where they
came from, so the question of truncating the dimension does not arise here.  It is
settled one layer up, in :mod:`tdadyn.reference.flag`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

REFERENCE_COEFF_FIELD = 2
"""The only field of coefficients this implementation supports."""


@dataclass(frozen=True)
class SimplicialComplex:
    """An abstract simplicial complex, closed under taking faces.

    Parameters
    ----------
    faces : dict of int to tuple of tuple of int
        Simplices grouped by dimension; each simplex is a sorted tuple of vertex
        labels, and the tuples within a dimension are sorted.  Build through
        :meth:`from_simplices` rather than directly.
    """

    faces: dict[int, tuple[tuple[int, ...], ...]]

    @classmethod
    def from_simplices(cls, simplices: Iterable[Sequence[int]]) -> "SimplicialComplex":
        """Build a complex from an explicit list of simplices.

        Parameters
        ----------
        simplices : iterable of sequence of int
            Every simplex of the complex, in any order.  Faces must be listed too:
            they are **verified, never added**.

        Returns
        -------
        SimplicialComplex
            The complex.

        Raises
        ------
        ValueError
            If a simplex repeats a vertex, is listed twice, or if some face of a
            listed simplex is missing.

        Notes
        -----
        Missing faces are an error rather than something to repair silently.  A
        builder that enumerates only maximal cliques produces exactly such a
        non-closed collection, and completing it here would hide the bug that
        ``docs/validation.md`` §4 devotes a structural test to catching.
        """
        by_dim: dict[int, set[tuple[int, ...]]] = {}
        for raw in simplices:
            simplex = _normalise_simplex(raw)
            bucket = by_dim.setdefault(len(simplex) - 1, set())
            if simplex in bucket:
                raise ValueError(f"simplex {simplex} is listed twice")
            bucket.add(simplex)
        faces = {dim: tuple(sorted(bucket)) for dim, bucket in sorted(by_dim.items())}
        _check_closed_under_faces(faces)
        return cls(faces=faces)

    def simplices(self, dim: int) -> tuple[tuple[int, ...], ...]:
        """Return the simplices of one dimension, empty if there are none.

        Parameters
        ----------
        dim : int
            Dimension: 0 for vertices, 1 for edges, and so on.

        Returns
        -------
        tuple of tuple of int
            The simplices, sorted.
        """
        return self.faces.get(dim, ())

    @property
    def dimension(self) -> int:
        """Highest dimension present, or ``-1`` for the empty complex."""
        return max(self.faces, default=-1)

    def face_counts(self) -> tuple[int, ...]:
        """Count simplices per dimension, from 0 up to :attr:`dimension`.

        Returns
        -------
        tuple of int
            For the flag complex of the complete graph on four nodes this is
            ``(4, 6, 4, 1)``: triangles are not absorbed by the tetrahedron, they
            live in their own dimension and form the rows of the boundary matrix.
        """
        return tuple(len(self.simplices(dim)) for dim in range(self.dimension + 1))


def betti_numbers(
    complex: SimplicialComplex,
    max_homology_dim: int,
    *,
    coeff_field: int = REFERENCE_COEFF_FIELD,
) -> dict[int, int]:
    """Compute Betti numbers over Z/2 from the ranks of the boundary matrices.

    Parameters
    ----------
    complex : SimplicialComplex
        The complex to measure.  To obtain :math:`\\beta_k` the complex must contain
        the simplices of dimension :math:`k+1`; otherwise :math:`\\operatorname{rank}
        \\partial_{k+1}` vanishes and every closed :math:`k`-chain is miscounted as a
        hole.  See ``docs/method.md`` §2.3.
    max_homology_dim : int
        Highest homology dimension to report.
    coeff_field : int, default 2
        Field of coefficients; only 2 is implemented.

    Returns
    -------
    dict of int to int
        :math:`\\beta_k` for every ``k`` from 0 to ``max_homology_dim`` inclusive,
        zeros included — a missing key would be indistinguishable from a forgotten one.

    Raises
    ------
    NotImplementedError
        If ``coeff_field`` is not 2.

    Notes
    -----
    :math:`\\beta_k = \\dim\\ker\\partial_k - \\operatorname{rank}\\partial_{k+1}`,
    and :math:`\\dim\\ker\\partial_k = n_k - \\operatorname{rank}\\partial_k` by the
    rank–nullity theorem, with :math:`\\operatorname{rank}\\partial_0 = 0`.
    """
    if coeff_field != REFERENCE_COEFF_FIELD:
        raise NotImplementedError(
            f"the reference implementation works only over Z/{REFERENCE_COEFF_FIELD}, "
            f"got coeff_field={coeff_field}. This is deliberate (docs/method.md §1.2): "
            "for p != 2 only the main pipeline (path A) is available, and a test at "
            "p = 3 is a consistency check rather than an independent confirmation."
        )
    if not isinstance(max_homology_dim, (int, np.integer)) or isinstance(max_homology_dim, bool):
        raise TypeError(f"max_homology_dim must be an int, got {type(max_homology_dim).__name__}")
    if max_homology_dim < 0:
        raise ValueError(f"max_homology_dim must be >= 0, got {max_homology_dim}")

    ranks = {
        dim: rank_mod2(boundary_matrix(complex, dim)) for dim in range(max_homology_dim + 2)
    }
    betti = {}
    for dim in range(max_homology_dim + 1):
        value = len(complex.simplices(dim)) - ranks[dim] - ranks[dim + 1]
        if value < 0:
            raise AssertionError(
                f"beta_{dim} came out negative ({value}); the boundary matrices are "
                "inconsistent, which means a bug in this module"
            )
        betti[dim] = value
    return betti


def boundary_matrix(complex: SimplicialComplex, dim: int) -> np.ndarray:
    """Build the boundary matrix :math:`\\partial_{dim}` over Z/2.

    Parameters
    ----------
    complex : SimplicialComplex
        The complex.
    dim : int
        Dimension of the domain: columns are ``dim``-simplices, rows are their
        ``(dim - 1)``-faces.

    Returns
    -------
    numpy.ndarray
        Boolean matrix of shape ``(n_{dim-1}, n_{dim})``.  Over Z/2 the orientation
        signs of the boundary operator all collapse to 1.
    """
    if dim < 0:
        raise ValueError(f"dim must be >= 0, got {dim}")
    columns = complex.simplices(dim)
    if dim == 0:
        # The boundary of a vertex is the empty chain: partial_0 maps into the zero
        # space, so it has no rows and rank 0.
        return np.zeros((0, len(columns)), dtype=bool)

    rows = complex.simplices(dim - 1)
    position = {simplex: index for index, simplex in enumerate(rows)}
    matrix = np.zeros((len(rows), len(columns)), dtype=bool)
    for column, simplex in enumerate(columns):
        for face in combinations(simplex, dim):
            matrix[position[face], column] = True
    return matrix


def row_reduce_mod2(matrix: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reduced row echelon form over Z/2, with the pivot columns.

    Parameters
    ----------
    matrix : numpy.ndarray
        Boolean matrix; it is not modified.

    Returns
    -------
    reduced : numpy.ndarray
        The reduced form: row ``i`` has its leading one in ``pivots[i]`` and that column
        is zero in every other row.
    pivots : tuple of int
        Indices of the pivot columns, increasing.  Their number is the rank.

    Notes
    -----
    Adding one row to another over Z/2 is XOR, and no division is needed — which is the
    whole reason this module is restricted to that field.  Everything else here is built
    on this one elimination rather than on a second copy of it.
    """
    work = np.array(matrix, dtype=bool, copy=True)
    if work.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {work.shape}")
    n_rows, n_columns = work.shape

    pivots: list[int] = []
    row = 0
    for column in range(n_columns):
        if row == n_rows:
            break
        candidates = np.flatnonzero(work[row:, column])
        if candidates.size == 0:
            continue
        pivot = row + int(candidates[0])
        if pivot != row:
            work[[row, pivot]] = work[[pivot, row]]
        affected = np.flatnonzero(work[:, column])
        affected = affected[affected != row]
        if affected.size:
            work[affected] ^= work[row]
        pivots.append(column)
        row += 1
    return work, tuple(pivots)


def rank_mod2(matrix: np.ndarray) -> int:
    """Rank of a boolean matrix over Z/2.

    Parameters
    ----------
    matrix : numpy.ndarray
        Boolean matrix; it is not modified.

    Returns
    -------
    int
        The rank, that is the number of pivot columns of :func:`row_reduce_mod2`.
    """
    return len(row_reduce_mod2(matrix)[1])


def nullspace_mod2(matrix: np.ndarray) -> np.ndarray:
    """Basis of the kernel of a boolean matrix over Z/2, as columns.

    Parameters
    ----------
    matrix : numpy.ndarray
        Boolean matrix of shape ``(n_rows, n_columns)``.

    Returns
    -------
    numpy.ndarray
        Boolean matrix of shape ``(n_columns, n_columns - rank)`` whose columns are a
        basis of the kernel.

    Notes
    -----
    Each free column of the reduced form contributes one basis vector: a one in that
    column, and in each pivot column the entry the reduced form records there.  The
    zigzag oracle uses this for the cycle spaces :math:`Z_k` and for the limit of a
    module restricted to an interval (``docs/zigzag.md`` §3.3).
    """
    reduced, pivots = row_reduce_mod2(matrix)
    n_columns = reduced.shape[1]
    free = [column for column in range(n_columns) if column not in set(pivots)]
    kernel = np.zeros((n_columns, len(free)), dtype=bool)
    for position, column in enumerate(free):
        kernel[column, position] = True
        for row, pivot in enumerate(pivots):
            kernel[pivot, position] = reduced[row, column]
    return kernel


def coordinates_mod2(spanning: np.ndarray, vector: np.ndarray) -> np.ndarray | None:
    """Write a vector over a spanning set: solve ``spanning @ x = vector`` over Z/2.

    Parameters
    ----------
    spanning : numpy.ndarray
        Boolean matrix whose **columns** span the subspace.
    vector : numpy.ndarray
        Boolean vector of length ``spanning.shape[0]``.

    Returns
    -------
    numpy.ndarray or None
        One solution ``x``, or ``None`` if the vector is outside the span.

    Notes
    -----
    The spanning set need not be independent, so the solution need not be unique; any
    one of them is a correct decomposition, and the caller uses only the coordinates it
    asked for.  The oracle needs this to express the image of a homology class in the
    basis chosen at the target index: the class is written over that basis **plus** the
    boundaries, and the boundary coordinates are then discarded.
    """
    spanning = np.asarray(spanning, dtype=bool)
    vector = np.asarray(vector, dtype=bool)
    if spanning.ndim != 2:
        raise ValueError(f"spanning must be 2-D, got shape {spanning.shape}")
    if vector.shape != (spanning.shape[0],):
        raise ValueError(
            f"vector of length {vector.shape} does not match spanning set of shape "
            f"{spanning.shape}"
        )
    augmented = np.column_stack([spanning, vector])
    reduced, pivots = row_reduce_mod2(augmented)
    last = spanning.shape[1]
    if last in pivots:
        # A pivot in the augmented column is the row 0 = 1: no solution.
        return None
    solution = np.zeros(last, dtype=bool)
    for row, pivot in enumerate(pivots):
        solution[pivot] = reduced[row, last]
    return solution


def _normalise_simplex(raw: Any) -> tuple[int, ...]:
    """Turn one simplex into a sorted tuple of ints, rejecting malformed input."""
    vertices = tuple(raw)
    if not vertices:
        raise ValueError("the empty simplex is not part of a simplicial complex")
    for vertex in vertices:
        if not isinstance(vertex, (int, np.integer)) or isinstance(vertex, bool):
            raise TypeError(f"vertex labels must be ints, got {vertex!r} in {vertices}")
    simplex = tuple(sorted(int(vertex) for vertex in vertices))
    if len(set(simplex)) != len(simplex):
        raise ValueError(f"simplex {vertices} repeats a vertex")
    return simplex


def _check_closed_under_faces(faces: dict[int, tuple[tuple[int, ...], ...]]) -> None:
    """Verify that every codimension-1 face of every simplex is present.

    Checking one codimension is enough: applied from the bottom dimension upwards it
    implies closure under all faces.
    """
    for dim in sorted(faces):
        if dim == 0:
            continue
        below = set(faces.get(dim - 1, ()))
        for simplex in faces[dim]:
            for face in combinations(simplex, dim):
                if face not in below:
                    raise ValueError(
                        f"the complex is not closed under faces: {simplex} is present "
                        f"but its face {face} is not. Every subset of a simplex must be "
                        "listed as well."
                    )
