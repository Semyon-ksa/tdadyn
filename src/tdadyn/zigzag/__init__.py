"""Zigzag persistence and time diagrams — task 2.

This layer **consumes** the truth of family 2 and never shapes it: the datasets and their
event logs come from :mod:`tdadyn.synthetic`, and a disagreement between an algorithm here
and the log there is fixed here until the reference says otherwise (``docs/zigzag.md``
§0.1).  Accordingly nothing in this package imports the generators; the input is read
through :mod:`tdadyn.core` and the complexes are built by :mod:`tdadyn.reference`.
"""

from .analysis import (
    FalsePositiveRate,
    NullMeasurement,
    analyse,
    ephemeral_counts,
    measure_null,
    wilson_interval,
    zero_hits_upper_bound,
)
from .backend import barcode_from, dionysus_barcode, presence_runs
from .construction import UNION_RULE, snapshot_complex, union_complex, zigzag_complexes
from .events import (
    DimensionMetrics,
    EventLog,
    EventMetrics,
    EventReport,
    Match,
    PredictedEvent,
    event_log,
    event_metrics,
    event_report,
    events_from_barcode,
    match_events,
)
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
    "DimensionMetrics",
    "EventLog",
    "EventMetrics",
    "EventReport",
    "FalsePositiveRate",
    "Match",
    "NullMeasurement",
    "PredictedEvent",
    "Operation",
    "OperationStream",
    "analyse",
    "barcode_from",
    "coarse_bar_of",
    "dionysus_barcode",
    "event_log",
    "event_metrics",
    "event_report",
    "ephemeral_counts",
    "events_from_barcode",
    "match_events",
    "index_set",
    "measure_null",
    "presence_runs",
    "wilson_interval",
    "zero_hits_upper_bound",
    "operation_stream",
    "replay",
    "snapshot_complex",
    "union_complex",
    "zigzag_complexes",
]
