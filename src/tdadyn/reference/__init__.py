"""Independent reference computation of Betti numbers — path B of ``docs/validation.md``.

Deliberately free of ripser, gudhi and persim: a check that shares code with the thing
it checks is not a check.  Only Z/2, only small complexes, only obvious algorithms.
"""

from .flag import betti_numbers_of_graph, flag_complex
from .homology import (
    REFERENCE_COEFF_FIELD,
    SimplicialComplex,
    betti_numbers,
    boundary_matrix,
    coordinates_mod2,
    nullspace_mod2,
    rank_mod2,
    row_reduce_mod2,
)

__all__ = [
    "REFERENCE_COEFF_FIELD",
    "SimplicialComplex",
    "betti_numbers",
    "betti_numbers_of_graph",
    "boundary_matrix",
    "coordinates_mod2",
    "flag_complex",
    "nullspace_mod2",
    "rank_mod2",
    "row_reduce_mod2",
]
