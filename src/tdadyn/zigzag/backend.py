"""Adapter to Dionysus 2 — the backend, which is the thing being checked.

Its role is exactly one: to be **verified against the oracle**, not to verify anything.
The oracle in :mod:`tdadyn.reference.oracle` computes the barcode a different way — the
generalized rank and Mobius inversion — so when the two disagree, the adapter is at fault
until shown otherwise (``docs/plan_task2.md`` §Z3).  A second backend would only add a vote:
two implementations of the same class of algorithm cannot say which of them is right.

One module rather than a ``backends/`` package.  fast-zigzag was demoted to optional in Z0
because the check is the oracle and not a second backend, so a directory would promise a
plurality that is not there.

**The import of ``dionysus`` happens inside the function.** ``pip install -e .`` must work
without the optional extra and the fast test target must pass in such an environment; an
import at module scope would break both.
"""

from __future__ import annotations


Barcode = tuple[tuple[int, int], ...]
"""Bars as closed intervals of positions, sorted, with repetition for multiplicity."""


def presence_runs(complexes) -> dict[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """For each simplex, the maximal runs of positions where it is present.

    Parameters
    ----------
    complexes : sequence of SimplicialComplex
        The diagram, in index order.

    Returns
    -------
    dict
        Simplex to a tuple of ``(start, stop)`` half-open runs of positions.

    Notes
    -----
    A simplex may leave and come back, and on ``cycle_lifecycle`` exactly one does: the edge
    that is cut to split a component and restored to merge it again.  The repeat of state is
    deliberate (``method.md`` §2.1).  Reporting one interval spanning the gap instead of two
    would glue the split and the merge together and erase the :math:`H_0` bar between them —
    quietly, since the result would still be a well-formed barcode.
    """
    positions: dict[tuple[int, ...], list[int]] = {}
    for position, complex_ in enumerate(complexes):
        for dim in complex_.faces:
            for simplex in complex_.faces[dim]:
                positions.setdefault(simplex, []).append(position)

    runs: dict[tuple[int, ...], tuple[tuple[int, int], ...]] = {}
    for simplex, present in positions.items():
        collected, start, previous = [], present[0], present[0]
        for position in present[1:]:
            if position == previous + 1:
                previous = position
            else:
                collected.append((start, previous + 1))
                start = previous = position
        collected.append((start, previous + 1))
        runs[simplex] = tuple(collected)
    return runs


def dionysus_barcode(complexes, *, max_homology_dim: int) -> dict[int, Barcode]:
    """Zigzag barcode of a diagram of complexes, computed by Dionysus 2.

    Parameters
    ----------
    complexes : sequence of SimplicialComplex
        The diagram in index order; positions are ``0 .. N-1`` and the answer comes back in
        the same positions, so the caller keeps the meaning of the indices.
    max_homology_dim : int
        Highest homology dimension to report.

    Returns
    -------
    dict of int to Barcode
        One barcode per dimension from 0 to ``max_homology_dim``, bars as **closed**
        intervals of positions, sorted, with repetition.

    Raises
    ------
    RuntimeError
        If Dionysus is not installed.

    Notes
    -----
    Two conventions are translated here, both validated on the toys before any dataset:

    * a simplex present on positions ``[a, b]`` is handed over as the time interval
      ``[a, b+1)``, one entry per run (:func:`presence_runs`);
    * a bar comes back half-open and becomes the closed ``[birth, death - 1]``; an infinite
      death becomes the last position.

    **Dimensions above** ``max_homology_dim`` **are dropped rather than passed on.**  The
    complexes are truncated at ``max_homology_dim + 1``, so :math:`\\partial_{k+2}` is absent
    and every closed chain of the top dimension is miscounted as a cavity
    (``method.md`` §2.3).  Measured on ``geometric_loop``: Dionysus returns 69 such bars in
    dimension 2 while the truth carries none.  They are dropped here rather than by the
    caller because ``cycle_lifecycle`` has no triangles at all and shows none of them, so a
    check on one dataset would not have caught them.
    """
    try:
        import dionysus
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "the Dionysus backend is an optional dependency and is not installed; "
            "run: pip install -e '.[zigzag]'"
        ) from error

    if max_homology_dim < 0:
        raise ValueError(f"max_homology_dim must be >= 0, got {max_homology_dim}")
    complexes = list(complexes)
    if len(complexes) < 2:
        raise ValueError(f"a zigzag needs at least two complexes, got {len(complexes)}")

    runs = presence_runs(complexes)
    # Faces before cofaces: sorting by dimension guarantees it, and lexicographic order
    # inside a dimension keeps the input reproducible.
    order = sorted(runs, key=lambda simplex: (len(simplex), simplex))
    filtration = dionysus.Filtration([list(simplex) for simplex in order])
    times = [[bound for run in runs[simplex] for bound in run] for simplex in order]

    _, diagrams, _ = dionysus.zigzag_homology_persistence(filtration, times)

    last = len(complexes) - 1
    barcodes: dict[int, Barcode] = {}
    for dim in range(max_homology_dim + 1):
        if dim >= len(diagrams):
            barcodes[dim] = ()
            continue
        bars = [
            (int(point.birth), last if point.death == float("inf") else int(point.death) - 1)
            for point in diagrams[dim]
        ]
        barcodes[dim] = tuple(sorted(bars))
    return barcodes




def barcode_from(complexes, *, backend: str, max_homology_dim: int) -> dict[int, Barcode]:
    """Compute the barcode with the implementation named by ``config.zigzag.backend``.

    Parameters
    ----------
    complexes : sequence of SimplicialComplex
        The diagram in index order.
    backend : {'reference', 'dionysus'}
        Which implementation to use.  ``'fzz'`` is accepted by the config schema but has no
        adapter: it is not on PyPI and is built from source only if Dionysus fails to fit
        the budget, which it does not (``docs/plan_task2.md`` §Z3).
    max_homology_dim : int
        Highest homology dimension to report.

    Returns
    -------
    dict of int to Barcode
        One barcode per dimension.

    Notes
    -----
    ``'reference'`` selects the **oracle**, and the name is worth being careful about: the
    oracle is path B, the independent check, not an implementation of the pipeline.  Choosing
    it here means "take the answer from the checker", which is a legitimate thing to want and
    is not the same as having checked anything.

    For that reason the agreement tests do **not** go through this function.  They call both
    implementations directly: a comparison routed through a switch would be testing the
    switch.
    """
    if backend == "dionysus":
        return dionysus_barcode(complexes, max_homology_dim=max_homology_dim)
    if backend == "reference":
        from tdadyn.reference import barcode, module_from_complexes

        return {
            dim: barcode(module_from_complexes(complexes, dim))
            for dim in range(max_homology_dim + 1)
        }
    if backend == "fzz":
        raise NotImplementedError(
            "fast-zigzag has no adapter: it is absent from PyPI and is built from source "
            "only if Dionysus does not fit the time budget, which it does — measured at "
            "16 ms per dataset against seconds for the oracle. The check here is the "
            "oracle, not a second backend (docs/zigzag.md 5.3)"
        )
    raise ValueError(f"unknown backend {backend!r}")
