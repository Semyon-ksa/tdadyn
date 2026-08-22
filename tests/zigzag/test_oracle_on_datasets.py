"""The oracle on family 2: the criterion of Z2.

Marked ``slow_zigzag`` — the first tests to use that marker.  The generalized rank is
quadratic in the number of indices, and at 119 indices a full barcode costs seconds rather
than milliseconds.

Nothing here transcribes an expectation.  Bar **supports** are derived from the Betti
curve, which the reference computes independently and which has been trusted since S2;
the **pairing** — the part the Betti curve is blind to — is forced by an algebraic argument
computed in the test body.  Copying the barcode out of ``docs/zigzag.md`` would check that
the code matches the document, not that the document is right, and §5.1a was in fact wrong
by a half-step until this session.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn.reference import betti_curve, rank_mod2
from tdadyn.zigzag import event_log

pytestmark = pytest.mark.slow_zigzag


def ends_from_the_betti_curve(curve):
    """Birth and death positions read off a Betti curve.

    A birth is where the curve rises (or the very start, if it starts non-zero); a death is
    the last position before it falls (or the very end).  This fixes the multiset of
    endpoints and says nothing about which birth goes with which death — that is exactly
    the blindness of ``docs/zigzag.md`` §3.4, and it is why this helper cannot be the whole
    expectation.
    """
    births, deaths = [], []
    previous = 0
    for position, value in enumerate(curve):
        births.extend([position] * max(0, value - previous))
        deaths.extend([position - 1] * max(0, previous - value))
        previous = value
    deaths.extend([len(curve) - 1] * previous)
    return sorted(births), sorted(deaths)


def compose_forward(module, start, stop):
    """The composite :math:`V_{start} \\to V_{stop}`, refusing if an arrow points back."""
    matrix = np.eye(module.dimensions[start], dtype=bool)
    for position in range(start, stop):
        forward, arrow = module.arrows[position]
        assert forward, f"arrow {position} points backwards; the composite is undefined"
        matrix = ((arrow.astype(np.int64) @ matrix.astype(np.int64)) % 2).astype(bool)
    return matrix


# --- supports -----------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1])
def test_the_bars_end_where_the_betti_curve_says_they_do(
    family_two_dataset, config, dim, zigzag_of
):
    complexes, _ = zigzag_of.complexes(family_two_dataset)
    bars = zigzag_of.bars(family_two_dataset, dim)
    births, deaths = ends_from_the_betti_curve(betti_curve(complexes, dim))
    assert sorted(birth for birth, _ in bars) == births
    assert sorted(death for _, death in bars) == deaths


@pytest.mark.parametrize("dim", [0, 1])
def test_the_bars_covering_an_index_reproduce_its_betti_number(
    family_two_dataset, config, dim, zigzag_of
):
    """L3 against L1 on real data.  Necessary, and not sufficient — see the pairing test."""
    complexes, _ = zigzag_of.complexes(family_two_dataset)
    bars = zigzag_of.bars(family_two_dataset, dim)
    for position, expected in enumerate(betti_curve(complexes, dim)):
        covering = sum(1 for birth, death in bars if birth <= position <= death)
        assert covering == expected, f"position {position}"


def test_the_supports_are_the_ones_the_construction_predicts(
    cycle_lifecycle, config, zigzag_of
):
    """Half-integer ends are not a defect; they follow from which way a transition goes.

    Closing the path at snapshot 12 *adds* an edge, so ``K_11½ = K_12`` already carries the
    class and the bar starts at ``11½``.  Cutting ``(6,7)`` at 41 *removes* one, so
    ``K_40½ = K_40`` still carries it and the bar ends at ``40½``.  In :math:`H_0` the signs
    are opposite — a birth comes from a removal — so ``[48, 54]`` has integer ends.
    """
    _, indices = zigzag_of.complexes(cycle_lifecycle)
    as_text = {
        dim: tuple(
            f"[{indices[birth]}, {indices[death]}]"
            for birth, death in zigzag_of.bars(cycle_lifecycle, dim)
        )
        for dim in (0, 1)
    }
    assert as_text[0] == ("[0, 59]", "[48, 54]")
    assert as_text[1] == ("[11½, 40½]", "[26½, 59]")


def test_the_metric_dataset_supports(geometric_loop, config, zigzag_of):
    _, indices = zigzag_of.complexes(geometric_loop)
    as_text = {
        dim: tuple(
            f"[{indices[birth]}, {indices[death]}]"
            for birth, death in zigzag_of.bars(geometric_loop, dim)
        )
        for dim in (0, 1)
    }
    assert as_text[0] == ("[0, 59]",)
    # The second loop is a different class, not a continuation of the first: between them
    # there is no class at all, and intervals are connected (docs/zigzag.md 5.1a).
    assert as_text[1] == ("[0, 19½]", "[44½, 59]")


# --- the pairing: the criterion of the session --------------------------------------


def test_the_pairing_is_forced_by_the_algebra_and_the_oracle_finds_it(
    cycle_lifecycle, config, zigzag_of
):
    """The 12<->41 pair, computed rather than taken from ``docs/method.md`` §2.2a.

    The Betti curve leaves two pairings open, and both are built here from its endpoints.
    The algebra chooses: at the peak the class carried in from the left and the class that
    survives to the right span **different lines**, so the space splits as their direct sum
    and the bar arriving from the left cannot continue past the peak.

    This is the canonical part of the claim.  A test on the identity of the class would not
    be: the basis of :math:`H_1` of a theta graph is not canonical.
    """
    complexes, _ = zigzag_of.complexes(cycle_lifecycle)
    module = zigzag_of.module(cycle_lifecycle, 1)
    births, deaths = ends_from_the_betti_curve(betti_curve(complexes, 1))
    assert len(births) == len(deaths) == 2

    peak = deaths[0]
    left = compose_forward(module, births[0], peak)
    forward, arrow = module.arrows[peak]
    assert not forward, "the transition out of the peak must be a removal"
    right = arrow

    # Different lines in a two-dimensional space: the sum of the ranks is the dimension.
    assert module.dimensions[peak] == 2
    assert rank_mod2(left) == rank_mod2(right) == 1
    assert rank_mod2(np.column_stack([left, right])) == 2

    forced = ((births[0], deaths[0]), (births[1], deaths[1]))
    other = ((births[0], deaths[1]), (births[1], deaths[0]))
    assert zigzag_of.bars(cycle_lifecycle, 1) == forced
    assert zigzag_of.bars(cycle_lifecycle, 1) != other


def test_the_pairing_the_oracle_rejects_is_not_ruled_out_below_l3(
    cycle_lifecycle, config, zigzag_of
):
    """Both pairings agree with L1 and with L2, so nothing cheaper could have decided.

    Written out because a reader who sees the pairing test pass could conclude that the
    Betti curve was enough.  ``docs/zigzag.md`` §3.4 says it is not, and here is that
    statement on the project's own data rather than on a constructed example.
    """
    complexes, _ = zigzag_of.complexes(cycle_lifecycle)
    module = zigzag_of.module(cycle_lifecycle, 1)
    curve = betti_curve(complexes, 1)
    births, deaths = ends_from_the_betti_curve(curve)
    rejected = ((births[0], deaths[1]), (births[1], deaths[0]))

    for position, expected in enumerate(curve):
        covering = sum(1 for birth, death in rejected if birth <= position <= death)
        assert covering == expected, "the rejected pairing already fails L1"
    for position, rank in enumerate(module.arrows):
        crossing = sum(
            1 for birth, death in rejected if birth <= position and death >= position + 1
        )
        assert crossing == rank_mod2(rank[1]), "the rejected pairing already fails L2"


# --- reconciliation with the truth log ----------------------------------------------


def test_the_barcode_accounts_for_every_event_in_the_truth_log(
    family_two_dataset, config, zigzag_of
):
    """The oracle's barcode and the truth log describe the same events.

    Written in Z2 with the translation rule spelled out inline, because the layer did not
    exist yet.  It goes through :func:`event_log` now: the rule of ``docs/zigzag.md`` §4.2
    is introduced in exactly one place, and a test carrying its own copy would be a second
    implementation — the shift by one is precisely what this project keeps to a single site.

    What it still checks is what it checked then, and it is not what Z4's metrics check:
    that the **oracle's** barcode reconciles with the log.  Z4 measures the backend's.
    """
    _, indices = zigzag_of.complexes(family_two_dataset)
    bars = {
        dim: zigzag_of.bars(family_two_dataset, dim)
        for dim in range(config.zigzag.max_homology_dim + 1)
    }
    predicted = {
        (event.snapshot, event.dim, event.type)
        for event in event_log(bars, indices=indices).uncensored()
    }
    logged = {
        (event.snapshot, event.dim, event.type)
        for event in family_two_dataset.ground_truth.events
    }
    assert predicted == logged


# --- negative controls ---------------------------------------------------------------


def test_the_breathing_radius_produces_no_bar_end(geometric_loop, config, zigzag_of):
    """``geometric_loop`` snapshots 4, 43 and 56: the complex changes, the barcode does not.

    Named snapshot by snapshot (``docs/zigzag.md`` §5.4).  A detector that fires on any
    change of the complex is caught precisely here.
    """
    _, indices = zigzag_of.complexes(geometric_loop)
    breathing = {indices.index(index) for index in indices
                 if index.is_snapshot and index.snapshot in (4, 43, 56)}
    for dim in (0, 1):
        for birth, death in zigzag_of.bars(geometric_loop, dim):
            assert birth not in breathing, f"a bar is born at a breathing snapshot, H_{dim}"
            assert death not in breathing, f"a bar dies at a breathing snapshot, H_{dim}"
