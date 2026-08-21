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
from .oracle import (
    ZigzagModule,
    arrow_ranks,
    barcode,
    betti_curve,
    generalized_rank,
    module_from_complexes,
)

__all__ = [
    "REFERENCE_COEFF_FIELD",
    "SimplicialComplex",
    "ZigzagModule",
    "arrow_ranks",
    "barcode",
    "betti_curve",
    "betti_numbers",
    "betti_numbers_of_graph",
    "boundary_matrix",
    "coordinates_mod2",
    "flag_complex",
    "generalized_rank",
    "module_from_complexes",
    "nullspace_mod2",
    "rank_mod2",
    "row_reduce_mod2",
]
