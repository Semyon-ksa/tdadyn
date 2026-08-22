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
from tdadyn.zigzag import (
    coarse_bar_of,
    event_log,
    operation_stream,
    presence_runs,
    zigzag_complexes,
)
from tdadyn.zigzag.backend import barcode_from, dionysus_barcode

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


# --- agreement on the datasets: the criterion of the session ------------------------


def oracle_and_backend(dataset, config, zigzag_of, dim):
    complexes, _ = zigzag_of.complexes(dataset)
    backend = dionysus_barcode(
        complexes, max_homology_dim=config.zigzag.max_homology_dim
    )[dim]
    return zigzag_of.bars(dataset, dim), backend


@pytest.mark.parametrize("dim", [0, 1])
def test_the_oracle_and_the_backend_agree_bar_for_bar(
    family_two_dataset, config, zigzag_of, dim
):
    """Two-sided agreement, whole barcodes compared including multiplicity.

    Not the number of bars, not the Betti curves, not the arrow coverings: those do not
    distinguish ``{[1,4],[2,3]}`` from ``{[1,3],[2,4]}`` (``docs/zigzag.md`` §3.4), and a
    comparison that used them would pass on a backend that got the pairing wrong.
    """
    oracle, backend = oracle_and_backend(family_two_dataset, config, zigzag_of, dim)
    assert backend == oracle


def test_the_backend_reproduces_the_pairing_of_twelve_and_fortyone(
    cycle_lifecycle, config, zigzag_of
):
    """The clause of Result 2 that Z2 could not close: the pair found by the **backend**.

    As in Z2 the two candidate pairings are built from the endpoints rather than typed out,
    so the test states which of them the backend picked instead of restating the document.
    """
    _, backend = oracle_and_backend(cycle_lifecycle, config, zigzag_of, 1)
    assert len(backend) == 2
    births = sorted(birth for birth, _ in backend)
    deaths = sorted(death for _, death in backend)
    forced = ((births[0], deaths[0]), (births[1], deaths[1]))
    crossed = ((births[0], deaths[1]), (births[1], deaths[0]))
    assert backend == forced
    assert backend != crossed


def test_the_backend_reports_nothing_above_the_requested_dimension(
    geometric_loop, config, zigzag_of
):
    """The 69 bars of dimension 2 are an artefact of truncation, not topology.

    ``geometric_loop`` has triangles, so Dionysus computes a two-dimensional diagram; the
    complexes stop at dimension 2, so :math:`\\partial_3` is missing and every closed
    2-chain is counted as a cavity (``method.md`` §2.3).  The adapter must not pass them on.
    """
    complexes, _ = zigzag_of.complexes(geometric_loop)
    bars = dionysus_barcode(complexes, max_homology_dim=config.zigzag.max_homology_dim)
    assert set(bars) == {0, 1}

    # And the artefact is real: asked for one dimension more, they appear.
    with_artefacts = dionysus_barcode(complexes, max_homology_dim=2)
    assert len(with_artefacts[2]) > 50

    # They contradict a number that was computed properly: the generator built its complexes
    # to dimension 3, with the boundary operator these lack, and recorded beta_2 = 0
    # everywhere. That is what makes them artefacts rather than a finding.
    assert set(geometric_loop.ground_truth.betti_curve[2]) == {0}


def test_relabelling_the_nodes_leaves_the_backend_barcode_alone(
    cycle_lifecycle, config, zigzag_of
):
    """``docs/zigzag.md`` §5.5.  Labels are not randomised in the data, so this is a test."""
    rng = np.random.default_rng(20260201)
    order = rng.permutation(cycle_lifecycle.snapshots.shape[1])
    permuted = type(cycle_lifecycle)(
        snapshots=cycle_lifecycle.snapshots[:, order][:, :, order],
        ground_truth=cycle_lifecycle.ground_truth,
    )
    _, original = oracle_and_backend(cycle_lifecycle, config, zigzag_of, 1)
    _, shuffled = oracle_and_backend(permuted, config, zigzag_of, 1)
    assert shuffled == original


def test_gluing_the_presence_runs_together_destroys_a_bar(
    cycle_lifecycle, config, zigzag_of
):
    """Why :func:`presence_runs` returns runs and not one spanning interval.

    The edge cut at 48 and restored at 55 is absent in between.  Handed over as a single
    interval it is never absent, the component never splits, and the :math:`H_0` bar between
    the two events disappears — leaving a barcode that is still well formed.
    """
    import dionysus

    complexes, _ = zigzag_of.complexes(cycle_lifecycle)
    runs = presence_runs(complexes)
    glued = {simplex: ((run[0][0], run[-1][1]),) for simplex, run in runs.items()}
    assert glued != runs, "the dataset must contain a simplex that leaves and returns"

    order = sorted(glued, key=lambda simplex: (len(simplex), simplex))
    filtration = dionysus.Filtration([list(simplex) for simplex in order])
    times = [[bound for run in glued[simplex] for bound in run] for simplex in order]
    _, diagrams, _ = dionysus.zigzag_homology_persistence(filtration, times)
    spoilt = len(list(diagrams[0]))

    honest, _ = oracle_and_backend(cycle_lifecycle, config, zigzag_of, 0)
    assert len(honest) == 2
    assert spoilt == 1, "gluing the runs lost the H_0 bar of the split component"


# --- the fine path: where ephemeral bars live ---------------------------------------


def fine_states(stream):
    """The complex after each operation, from the empty one onwards."""
    from tdadyn.reference import SimplicialComplex

    present, states = set(), [SimplicialComplex(faces={})]
    for operation in stream.operations:
        if operation.kind == "insert":
            present.add(operation.simplex)
        else:
            present.discard(operation.simplex)
        states.append(SimplicialComplex.from_simplices(present))
    return states


def test_the_fine_path_translates_back_to_the_coarse_one(family_two_dataset, config, zigzag_of):
    """Feeding the backend one time per **operation** must give the same coarse answer.

    This is the strongest available test of the index translation: the fine path sees 448
    states where the coarse one sees 119, and the two agree only if
    :func:`coarse_bar_of` is right at both ends.
    """
    complexes, _ = zigzag_of.complexes(family_two_dataset)
    stream = operation_stream(
        zigzag_complexes(family_two_dataset, **build_args(config))
    )
    states = fine_states(stream)

    for dim in range(config.zigzag.max_homology_dim + 1):
        fine = dionysus_barcode(states, max_homology_dim=config.zigzag.max_homology_dim)[dim]
        translated = sorted(
            (span[0].doubled, span[1].doubled)
            for span in (coarse_bar_of(stream, birth, death) for birth, death in fine)
            if span is not None
        )
        assert tuple(translated) == zigzag_of.bars(family_two_dataset, dim), f"H_{dim}"


EPHEMERAL = {
    ("cycle_lifecycle", 0): (23, 0),
    ("cycle_lifecycle", 1): (0, 0),
    ("geometric_loop", 0): (23, 0),
    ("geometric_loop", 1): (48, 94),
}
"""Measured ``(in the prologue, outside it)`` counts of ephemeral bars.

Kept split because the two halves mean different things.  The prologue assembles the first
complex from nothing, so its ephemeral bars count the order in which that assembly happens —
lexicographic, fixed in Z1, and arbitrary (``docs/zigzag.md`` §2.3).  Publishing the total as
a property of the dataset would report that order as if it were data.
"""


def test_the_ephemeral_bars_are_counted_and_split(family_two_dataset, config, zigzag_of):
    """Ephemeral bars are invisible to the coarse path by construction, and are counted here.

    They are born and die inside one transition, so no index of the diagram sees them and
    the oracle cannot produce them at all.  Only the fine path can, which is why it exists.
    """
    name = family_two_dataset.ground_truth.generator
    stream = operation_stream(
        zigzag_complexes(family_two_dataset, **build_args(config))
    )
    states = fine_states(stream)
    prologue = len(stream.prologue)

    for dim in range(config.zigzag.max_homology_dim + 1):
        fine = dionysus_barcode(states, max_homology_dim=config.zigzag.max_homology_dim)[dim]
        ephemeral = [
            (birth, death)
            for birth, death in fine
            if coarse_bar_of(stream, birth, death) is None
        ]
        inside = sum(1 for _, death in ephemeral if death <= prologue)
        outside = len(ephemeral) - inside
        assert (inside, outside) == EPHEMERAL[(name, dim)], f"H_{dim}"


def test_no_bar_of_either_dataset_is_invisible(family_two_dataset, config, zigzag_of):
    """An invisible bar spans no snapshot index at all (``docs/zigzag.md`` §4.3).

    Measured: none on either dataset.  Recorded as a measurement rather than assumed — the
    count is a characteristic of the union rule and has to be known before any of this
    reaches a connectome.
    """
    _, indices = zigzag_of.complexes(family_two_dataset)
    bars = {
        dim: zigzag_of.bars(family_two_dataset, dim)
        for dim in range(config.zigzag.max_homology_dim + 1)
    }
    # Counted by the event layer rather than by a predicate written again here: what makes a
    # bar invisible is part of the translation rule and lives in one place.
    assert event_log(bars, indices=indices).invisible_bars == 0


# --- the dispatcher -----------------------------------------------------------------


def test_the_dispatcher_selects_the_implementation_the_config_names(
    cycle_lifecycle, config, zigzag_of
):
    complexes, _ = zigzag_of.complexes(cycle_lifecycle)
    ceiling = config.zigzag.max_homology_dim
    through_backend = barcode_from(complexes, backend="dionysus", max_homology_dim=ceiling)
    through_oracle = barcode_from(complexes, backend="reference", max_homology_dim=ceiling)
    assert through_backend == through_oracle
    assert through_oracle[1] == zigzag_of.bars(cycle_lifecycle, 1)


def test_the_configured_backend_is_one_the_dispatcher_knows(config):
    assert config.zigzag.backend in ("reference", "dionysus", "fzz")


def test_fzz_says_why_it_is_absent_rather_than_failing_obscurely():
    """It is accepted by the schema and has no adapter, and the message has to say why.

    The reason is not that it is hard to install — it is that a second backend was never the
    check.  The oracle computes the answer another way; a second implementation of the same
    class of algorithm only adds a vote (``docs/zigzag.md`` §5.3).
    """
    with pytest.raises(NotImplementedError, match="oracle"):
        barcode_from(toy(TWO_VERTICES, ONE_EDGE, TWO_VERTICES), backend="fzz",
                     max_homology_dim=1)


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown backend"):
        barcode_from(toy(TWO_VERTICES, ONE_EDGE, TWO_VERTICES), backend="gudhi",
                     max_homology_dim=1)
