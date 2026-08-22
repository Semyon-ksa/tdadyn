"""The Dionysus 2 adapter, and its agreement with the oracle.

The danger of this layer is not that the backend fails to run.  It is that it runs, returns
a plausible barcode with shifted indices, and the shift is indistinguishable in a report
from a systematic localisation error of the algorithm (``docs/zigzag.md`` §2.4).  So the
index convention is validated **first on the toys**, where the answer is written out by
hand, and only then on the datasets.

Marked ``slow_zigzag`` and guarded by ``importorskip``: ``make fast`` must pass in an
environment without the ``[zigzag]`` extra, and ``make zigzag`` there must skip rather than
fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn.reference import barcode, module_from_complexes
from tdadyn.zigzag import index_set, presence_runs, zigzag_complexes
from tdadyn.zigzag.backend import dionysus_barcode

pytestmark = pytest.mark.slow_zigzag

pytest.importorskip(
    "dionysus",
    reason="the zigzag backend is an optional dependency: pip install -e '.[zigzag]'",
)

# The toys, built the same way as in the oracle's own tests; see tests/reference/test_oracle
# for what each one is for.  Duplicated deliberately rather than imported: a test of the
# backend that borrowed the oracle's fixtures would share a failure mode with it.
TWO_VERTICES = [(0,), (1,)]
ONE_EDGE = [(0,), (1,), (0, 1)]
THREE_VERTICES = [(0,), (1,), (2,)]
TRIANGLE_BOUNDARY = THREE_VERTICES + [(0, 1), (1, 2), (0, 2)]
_V = [(v,) for v in range(5)]
ONE_CYCLE = _V + [(0, 2), (1, 2), (0, 3), (1, 3)]
THETA = ONE_CYCLE + [(0, 4), (1, 4)]
THETA_CUT = _V + [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4)]


def toy(*snapshots):
    from tdadyn.reference import SimplicialComplex

    built = [SimplicialComplex.from_simplices(s) for s in snapshots]
    diagram = [built[0]]
    for left, right in zip(built, built[1:]):
        union = {s for src in (left, right) for dim in src.faces for s in src.faces[dim]}
        diagram.append(SimplicialComplex.from_simplices(union))
        diagram.append(right)
    return diagram


def build_args(config):
    return {
        "max_homology_dim": config.zigzag.max_homology_dim,
        "max_nodes": config.topology_check.reference_max_nodes,
    }


# --- the convention, on answers known by hand ---------------------------------------


def test_the_backend_reproduces_the_barcode_of_t1():
    """``docs/zigzag.md`` §5.1 writes it out: ``{[0,2], [0,0], [2,2]}`` in snapshot terms."""
    bars = dionysus_barcode(toy(TWO_VERTICES, ONE_EDGE, TWO_VERTICES), max_homology_dim=1)
    assert bars[0] == ((0, 0), (0, 4), (4, 4))
    assert bars[1] == ()


def test_the_backend_finds_the_loop_of_t2_with_half_integer_ends():
    bars = dionysus_barcode(toy(THREE_VERTICES, TRIANGLE_BOUNDARY, THREE_VERTICES),
                            max_homology_dim=1)
    assert bars[1] == ((1, 3),)


def test_the_backend_decides_the_pairing_of_t3_the_way_the_oracle_does():
    """T3 is the fork of §3.2, and the two pairings have identical coverings.

    This is what stands in for T4 here.  T4 is a module assembled from matrices with no
    complexes behind it, so the backend cannot be fed it at all; T3 is the same
    discriminating test realised as data, and passing it means the backend is not merely
    counting coverings.
    """
    complexes = toy(ONE_CYCLE, THETA, THETA_CUT)
    assert dionysus_barcode(complexes, max_homology_dim=1)[1] == ((0, 3), (1, 4))
    assert barcode(module_from_complexes(complexes, 1)) == ((0, 3), (1, 4))


def test_a_shifted_convention_would_disagree_with_the_oracle():
    """Negative control: the agreement below is not a comparison of something with itself.

    Dionysus reports half-open intervals, so the closed right end is ``death - 1``.  Dropping
    that subtraction still produces a plausible barcode — every bar one longer — and nothing
    but a comparison would notice.
    """
    complexes = toy(ONE_CYCLE, THETA, THETA_CUT)
    honest = dionysus_barcode(complexes, max_homology_dim=1)[1]
    shifted = tuple(sorted((birth, death + 1) for birth, death in honest))
    assert shifted != barcode(module_from_complexes(complexes, 1))


# --- presence runs ------------------------------------------------------------------


def test_a_simplex_that_leaves_and_returns_gets_two_intervals():
    """The deliberate repeat of state (``method.md`` §2.1) reaches the backend as two runs.

    Handing Dionysus a single interval spanning the gap would silently glue the split and
    the merge of the component together, and the :math:`H_0` bar between them would vanish.
    """
    complexes = toy(ONE_EDGE, TWO_VERTICES, ONE_EDGE)
    runs = presence_runs(complexes)

    # Positions are 0, ½, 1, 1½, 2.  The edge is in snapshots 0 and 2, and therefore also
    # in both unions next to them; it is absent only at snapshot 1, position 2.  So the gap
    # is one position wide and the runs are [0,2) and [3,5) — the unions are part of why a
    # simplex that leaves can come back without the diagram ever being wrong.
    assert runs[(0, 1)] == ((0, 2), (3, 5))
    assert runs[(0,)] == ((0, 5),)

    # And the gap really is a gap: two runs, not one spanning interval.
    assert len(runs[(0, 1)]) == 2
