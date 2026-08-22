"""Zigzag persistence and time diagrams — task 2.

This layer **consumes** the truth of family 2 and never shapes it: the datasets and their
event logs come from :mod:`tdadyn.synthetic`, and a disagreement between an algorithm here
and the log there is fixed here until the reference says otherwise (``docs/zigzag.md``
§0.1).  Accordingly nothing in this package imports the generators; the input is read
through :mod:`tdadyn.core` and the complexes are built by :mod:`tdadyn.reference`.
"""

from .backend import dionysus_barcode, presence_runs
from .construction import UNION_RULE, snapshot_complex, union_complex, zigzag_complexes
from .indices import CoarseIndex, index_set
from .operations import (
    CoarseSpan,
    Operation,
    OperationStream,
    coarse_bar_of,
    operation_stream,
    replay,
)

__all__ = [
    "UNION_RULE",
    "CoarseIndex",
    "CoarseSpan",
    "Operation",
    "OperationStream",
    "coarse_bar_of",
    "dionysus_barcode",
    "index_set",
    "presence_runs",
    "operation_stream",
    "replay",
    "snapshot_complex",
    "union_complex",
    "zigzag_complexes",
]
