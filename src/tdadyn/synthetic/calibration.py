"""Calibration of the significance criterion — ``docs/validation.md`` §2.

The old rule "persistent enough relative to :math:`\\mathrm{diam}(X)`" was found unfit by
measurement (``docs/method.md`` §1.2a): it gives false negatives on anisotropic objects
and false positives on compact ones, and the two ranges overlap in units of the diameter,
so no constant repairs it.  The replacement is chosen here rather than guessed.

This module is **not a test**.  It runs the table of cases from ``docs/validation.md`` §2
over every seed in ``topology_check.calibration_seeds``, at :math:`\\sigma = 0` and at the
configured noise level, and reports for each candidate rule whether it is right on every
row.  A rule is accepted only if it is — tuning one object is what the whole exercise is
against.

The candidate rules live here rather than in :mod:`tdadyn.synthetic.persistence` because
they are research artefacts; the accepted one moves there once calibration has run.

Run it with::

    python -m tdadyn.synthetic.calibration --config config.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .io import read_json, write_json
from .persistence import pairwise_distances, persistence_diagrams
from .rng import rng_for
from .static_clouds import (
    klein_bottle_betti,
    make_circle,
    make_figure_eight,
    make_hopf_link,
    make_klein_bottle,
    make_sphere,
    make_torus,
    make_two_circles,
)


@dataclass(frozen=True)
class CalibrationCase:
    """One row of the table in ``docs/validation.md`` §2.

    Parameters
    ----------
    name : str
        Label for the report.
    expected : dict of int to int
        Expected Betti numbers, from ``docs/method.md`` §1.2.
    coeff_field : int
        Field the expectation is stated for.
    build : callable
        ``(config, seed, noise_sigma) -> numpy.ndarray`` returning the point cloud.
    """

    name: str
    expected: dict[int, int]
    coeff_field: int
    build: Callable[[Config, int, float], np.ndarray]


@dataclass(frozen=True)
class Spectrum:
    """Persistences of one dimension, sorted descending, split by finiteness."""

    finite: np.ndarray
    n_infinite: int
    diameter: float


def build_cases(config: Config) -> list[CalibrationCase]:
    """Assemble the calibration table.

    Parameters
    ----------
    config : Config
        Loaded configuration; all geometry comes from it.

    Returns
    -------
    list of CalibrationCase
        The rows of ``docs/validation.md`` §2, in table order.

    Notes
    -----
    The null-model row is required by the table but family 4 is only built in S8, so a
    minimal stand-in lives here: uniform points in a cube, which has no topology to find.
    Its expectation is ``(1, 0, 0)`` — the one infinite bar of :math:`H_0` is always
    significant, so "zero in every dimension" means no *features* beyond the single
    component.  The full null models of family 4 stay in S8.
    """
    clouds = config.static_clouds
    field = config.topology_check.coeff_field

    def points_of(dataset):
        return dataset.points

    def count(generator: str) -> int:
        return clouds.n_points_by_generator.get(generator, clouds.n_points)

    return [
        CalibrationCase(
            "circle",
            {0: 1, 1: 1, 2: 0},
            field,
            lambda config, seed, sigma: points_of(
                make_circle(
                    count("circle"), clouds.circle.radius, sigma, field, rng_for(seed, "circle")
                )
            ),
        ),
        CalibrationCase(
            "two_circles",
            {0: 2, 1: 2, 2: 0},
            field,
            lambda config, seed, sigma: points_of(
                make_two_circles(
                    count("two_circles"),
                    clouds.two_circles.radius,
                    clouds.two_circles.separation,
                    sigma,
                    field,
                    rng_for(seed, "two_circles"),
                )
            ),
        ),
        CalibrationCase(
            "figure_eight",
            {0: 1, 1: 2, 2: 0},
            field,
            lambda config, seed, sigma: points_of(
                make_figure_eight(
                    count("figure_eight"),
                    clouds.figure_eight.scale,
                    sigma,
                    field,
                    rng_for(seed, "figure_eight"),
                )
            ),
        ),
        CalibrationCase(
            "torus",
            {0: 1, 1: 2, 2: 1},
            field,
            lambda config, seed, sigma: points_of(
                make_torus(
                    count("torus"),
                    clouds.torus.R,
                    clouds.torus.r,
                    sigma,
                    clouds.torus.sampling,
                    field,
                    rng_for(seed, "torus"),
                )
            ),
        ),
        CalibrationCase(
            "sphere",
            {0: 1, 1: 0, 2: 1},
            field,
            lambda config, seed, sigma: points_of(
                make_sphere(
                    count("sphere"),
                    clouds.sphere.radius,
                    sigma,
                    clouds.sphere.sampling,
                    field,
                    rng_for(seed, "sphere"),
                )
            ),
        ),
        *[
            CalibrationCase(
                f"klein_bottle p={test_field}",
                klein_bottle_betti(test_field),
                test_field,
                lambda config, seed, sigma: points_of(
                    make_klein_bottle(
                        count("klein_bottle"),
                        clouds.klein_bottle.a,
                        clouds.klein_bottle.b,
                        sigma,
                        clouds.klein_bottle.sampling,
                        config.topology_check.coeff_field,
                        rng_for(seed, "klein_bottle"),
                    )
                ),
            )
            for test_field in config.topology_check.coeff_field_test_values
        ],
        CalibrationCase(
            "hopf_link",
            {0: 2, 1: 2, 2: 0},
            field,
            lambda config, seed, sigma: points_of(
                make_hopf_link(
                    count("hopf_link"),
                    clouds.hopf_link.radius,
                    clouds.hopf_link.offset,
                    sigma,
                    field,
                    rng_for(seed, "hopf_link"),
                )
            ),
        ),
        CalibrationCase(
            "null_uniform_cube",
            {0: 1, 1: 0, 2: 0},
            field,
            lambda config, seed, sigma: _uniform_cube(
                clouds.n_points, sigma, rng_for(seed, "null_uniform_cube")
            ),
        ),
    ]


def _uniform_cube(n_points: int, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """The null stand-in: uniform points in a cube, with the same noise applied."""
    points = rng.uniform(0.0, 1.0, (n_points, 3))
    return points + rng.normal(0.0, noise_sigma, points.shape)


def spectra_of(points: np.ndarray, *, max_homology_dim: int, coeff_field: int) -> dict[int, Spectrum]:
    """Persistence spectra of a point cloud, one per dimension.

    Parameters
    ----------
    points : numpy.ndarray
        The cloud.
    max_homology_dim : int
        Highest homology dimension.
    coeff_field : int
        Prime field of coefficients.

    Returns
    -------
    dict of int to Spectrum
        Sorted persistences per dimension, with the infinite bars counted separately.
    """
    distances = pairwise_distances(points)
    diagrams = persistence_diagrams(
        distances, max_homology_dim=max_homology_dim, coeff_field=coeff_field
    )
    diameter = float(distances.max())
    spectra = {}
    for dim, diagram in diagrams.items():
        if diagram.size == 0:
            spectra[dim] = Spectrum(np.empty(0), 0, diameter)
            continue
        lifetimes = diagram[:, 1] - diagram[:, 0]
        infinite = ~np.isfinite(lifetimes)
        finite = np.sort(lifetimes[~infinite])[::-1]
        spectra[dim] = Spectrum(finite, int(infinite.sum()), diameter)
    return spectra


# --- candidate rules -------------------------------------------------------------
#
# Each takes a Spectrum and returns the number of significant features in that
# dimension, infinite bars included.  The parameters are shared by the whole family;
# a rule that needs a different constant per object is not a candidate.


def rule_diam_ratio(spectrum: Spectrum, *, significance_ratio: float, **_: float) -> int:
    """The old rule: persistence at least ``significance_ratio`` of the diameter."""
    threshold = significance_ratio * spectrum.diameter
    return spectrum.n_infinite + int((spectrum.finite >= threshold).sum())


def rule_gap(spectrum: Spectrum, *, gap_min_ratio: float, **_: float) -> int:
    """Cut at the largest multiplicative gap over the whole spectrum.

    Kept as a candidate to record *why* it fails: with hundreds of nearly equal dust bars
    the largest **multiplicative** ratio lands deep in the tail, between two microscopic
    values, and the rule reports hundreds of features.  A gap search needs a floor under
    it before it means anything.
    """
    return spectrum.n_infinite + _cut_at_largest_gap(spectrum.finite, gap_min_ratio)


def rule_gap_with_floor(
    spectrum: Spectrum, *, gap_min_ratio: float, gap_noise_floor_ratio: float, **_: float
) -> int:
    """Discard dust below a noise floor, then cut at the largest gap.

    The floor is a fraction of the diameter, which is not a return to the old rule: at
    0.04 of the diameter it only removes bars that are numerically certain to be dust —
    the torus cavity (0.53 against a floor of 0.22) and the sphere's noise loops (0.23 to
    0.35 against a floor of 0.085) both survive it.  The discriminating is left to the
    scale-free part of the rule.
    """
    return spectrum.n_infinite + _cut_at_largest_gap(
        _above_floor(spectrum, gap_noise_floor_ratio), gap_min_ratio
    )


def rule_tail_gap(
    spectrum: Spectrum,
    *,
    tail_min_ratio: float,
    gap_min_ratio: float,
    gap_noise_floor_ratio: float,
    **_: float,
) -> int:
    """Compare the top bar with the bulk of the spectrum, then cut at the largest gap.

    The single largest adjacent gap cannot tell a spectrum with one real feature from a
    smooth staircase of noise: measured on 600 points, the sphere in :math:`H_1`
    (expected 0) and the Klein bottle over Z/3 in :math:`H_1` (expected 1) give 1.20x and
    1.32x.  Referring the top bar to the median of the tail separates them — 1.20x
    against 1.83x on the same runs — because it measures how far the leader stands out
    from the noise the object itself provides, not from its nearest neighbour.
    """
    finite = _above_floor(spectrum, gap_noise_floor_ratio)
    if finite.size == 0:
        return spectrum.n_infinite
    tail = finite[1:]
    if tail.size == 0:
        return spectrum.n_infinite + 1
    tail_scale = float(np.median(tail))
    if tail_scale <= 0 or finite[0] / tail_scale < tail_min_ratio:
        return spectrum.n_infinite
    return spectrum.n_infinite + max(1, _cut_at_largest_gap(finite, gap_min_ratio))


def rule_gap_sentinel(
    spectrum: Spectrum, *, gap_min_ratio: float, gap_noise_floor_ratio: float, **_: float
) -> int:
    """Cut at the largest gap, counting the step down into the dust as a gap too.

    Fixes what the calibration showed to be fatal for :func:`rule_gap_with_floor`: two
    disjoint circles leave exactly two nearly equal bars above the floor, there is no gap
    *between* them, and the rule reported zero loops where there are two.  A spectrum of
    :math:`k` features followed by dust has its real gap at the very end, between the
    last feature and the first discarded bar, so that step has to be part of the search.

    When nothing at all falls below the floor the sentinel is zero and the whole
    above-floor spectrum counts, which is the right reading: there is no dust to separate
    the features from.
    """
    floor = gap_noise_floor_ratio * spectrum.diameter
    finite = spectrum.finite
    above = finite[finite >= floor]
    if above.size == 0:
        return spectrum.n_infinite
    below = finite[finite < floor]
    sentinel = float(below[0]) if below.size else 0.0
    extended = np.concatenate([above, [sentinel]])
    ratios = extended[:-1] / np.where(extended[1:] > 0, extended[1:], np.inf)
    tail_is_empty = sentinel == 0.0
    if tail_is_empty:
        ratios = np.concatenate([ratios[:-1], [np.inf]])
    best = int(np.argmax(ratios))
    if ratios[best] < gap_min_ratio:
        return spectrum.n_infinite
    return spectrum.n_infinite + best + 1


def _above_floor(spectrum: Spectrum, gap_noise_floor_ratio: float) -> np.ndarray:
    """Bars above the dust floor, still sorted descending."""
    floor = gap_noise_floor_ratio * spectrum.diameter
    return spectrum.finite[spectrum.finite >= floor]


CANDIDATE_RULES: dict[str, Callable[..., int]] = {
    "diam_ratio": rule_diam_ratio,
    "gap": rule_gap,
    "gap_with_floor": rule_gap_with_floor,
    "tail_gap": rule_tail_gap,
    "gap_sentinel": rule_gap_sentinel,
}


def _cut_at_largest_gap(persistences: np.ndarray, gap_min_ratio: float) -> int:
    """Position of the largest multiplicative gap, or 0 if no gap is convincing."""
    if persistences.size == 0:
        return 0
    positive = persistences[persistences > 0]
    if positive.size == 0:
        return 0
    if positive.size == 1:
        return 1
    ratios = positive[:-1] / positive[1:]
    best = int(np.argmax(ratios))
    if ratios[best] < gap_min_ratio:
        return 0
    return best + 1


# --- the run ---------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrumRecord:
    """One measured combination of case, seed and noise level."""

    label: str
    case_name: str
    expected: dict[int, int]
    spectra: dict[int, Spectrum]


def collect_spectra(config: Config, cases: list[CalibrationCase]) -> list[SpectrumRecord]:
    """Measure every combination of case, seed and noise level.

    This is the expensive half — persistent homology of every cloud in the table, about
    eight minutes for the full set.  It is deliberately separate from rule evaluation so
    that a parameter sweep costs nothing once the spectra are on disk.
    """
    topology = config.topology_check
    records = []
    for case in cases:
        for seed in topology.calibration_seeds:
            for sigma in (0.0, config.static_clouds.noise_sigma):
                points = case.build(config, seed, sigma)
                spectra = spectra_of(
                    points,
                    max_homology_dim=topology.max_homology_dim,
                    coeff_field=case.coeff_field,
                )
                label = f"{case.name} seed={seed} sigma={sigma}"
                print(f"измерено: {label}", flush=True)
                records.append(SpectrumRecord(label, case.name, dict(case.expected), spectra))
    return records


def save_spectra(records: list[SpectrumRecord], path: Path) -> None:
    """Write measured spectra so that sweeps do not recompute them."""
    payload = [
        {
            "label": record.label,
            "case_name": record.case_name,
            "expected": {str(dim): value for dim, value in record.expected.items()},
            "spectra": {
                str(dim): {
                    "finite": spectrum.finite.tolist(),
                    "n_infinite": spectrum.n_infinite,
                    "diameter": spectrum.diameter,
                }
                for dim, spectrum in record.spectra.items()
            },
        }
        for record in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def load_spectra(path: Path) -> list[SpectrumRecord]:
    """Read spectra written by :func:`save_spectra`."""
    return [
        SpectrumRecord(
            item["label"],
            item["case_name"],
            {int(dim): value for dim, value in item["expected"].items()},
            {
                int(dim): Spectrum(
                    np.asarray(entry["finite"], dtype=np.float64),
                    int(entry["n_infinite"]),
                    float(entry["diameter"]),
                )
                for dim, entry in item["spectra"].items()
            },
        )
        for item in read_json(path)
    ]


def evaluate(
    records: list[SpectrumRecord],
    rules: dict[str, Callable[..., int]],
    parameters: dict[str, float],
    *,
    verbose: bool = True,
) -> dict[str, list[str]]:
    """Score every rule against every record.

    Returns
    -------
    dict of str to list of str
        Rule name to the combinations it got wrong; an empty list means the rule is
        right on the whole table, which is the only acceptable outcome.
    """
    failures: dict[str, list[str]] = {name: [] for name in rules}
    for record in records:
        if verbose:
            print(f"\n{record.label}  ожидание={_as_tuple(record.expected)}")
            for dim in sorted(record.spectra):
                spectrum = record.spectra[dim]
                head = np.round(spectrum.finite[:5], 3)
                print(f"   H{dim} inf={spectrum.n_infinite} спектр={head}")
        for name, rule in rules.items():
            got = {dim: rule(record.spectra[dim], **parameters) for dim in sorted(record.spectra)}
            ok = got == record.expected
            if not ok:
                failures[name].append(f"{record.label}: {_as_tuple(got)}")
            if verbose:
                print(f"   {name:16} -> {_as_tuple(got)}  {'ok' if ok else 'МИМО'}")
    return failures


def report(failures: dict[str, list[str]], total: int) -> None:
    """Print the summary of a calibration run."""
    print("\n" + "=" * 72)
    print(f"Итог по {total} комбинациям (случай, сид, sigma):")
    for name, missed in failures.items():
        print(f"  {name:16} верно {total - len(missed):3d}/{total}")
        for failure in missed[:6]:
            print(f"        {failure}")


def sweep(
    records: list[SpectrumRecord],
    rules: dict[str, Callable[..., int]],
    grid: dict[str, list[float]],
    base: dict[str, float],
) -> list[tuple[str, dict[str, float], int]]:
    """Score a grid of parameter settings for each rule.

    Returns
    -------
    list of tuple
        ``(rule name, parameters, number of failures)``, sorted by failures ascending.
    """
    from itertools import product

    names = sorted(grid)
    results = []
    for values in product(*(grid[name] for name in names)):
        parameters = {**base, **dict(zip(names, values))}
        failures = evaluate(records, rules, parameters, verbose=False)
        for rule_name, missed in failures.items():
            results.append((rule_name, dict(zip(names, values)), len(missed)))
    results.sort(key=lambda item: item[2])
    return results


def _as_tuple(betti: dict[int, int]) -> tuple[int, ...]:
    return tuple(betti[dim] for dim in sorted(betti))


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Calibration of the significance criterion")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--cache", type=Path, default=None, help="where spectra are stored")
    parser.add_argument(
        "--collect", action="store_true", help="recompute the spectra instead of reading the cache"
    )
    parser.add_argument("--sweep", action="store_true", help="score a grid of parameters")
    parser.add_argument(
        "--tail-min-ratio", type=float, default=1.6, help="parameter of the tail_gap candidate"
    )
    parser.add_argument("--case", action="append", help="restrict to named cases")
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    cache = arguments.cache or config.output_root / "calibration_spectra.json"

    if arguments.collect or not cache.exists():
        cases = build_cases(config)
        if arguments.case:
            wanted = set(arguments.case)
            cases = [case for case in cases if case.name in wanted]
        records = collect_spectra(config, cases)
        save_spectra(records, cache)
        print(f"спектры записаны в {cache}")
    else:
        records = load_spectra(cache)
        if arguments.case:
            wanted = set(arguments.case)
            records = [record for record in records if record.case_name in wanted]
        print(f"спектры прочитаны из {cache}: {len(records)} комбинаций")

    topology = config.topology_check
    base = {
        "significance_ratio": topology.significance_ratio,
        "gap_min_ratio": topology.gap_min_ratio,
        "gap_noise_floor_ratio": topology.gap_noise_floor_ratio,
        "tail_min_ratio": arguments.tail_min_ratio,
    }

    if arguments.sweep:
        grid = {
            "gap_min_ratio": [1.4, 1.6, 1.8, 2.0, 2.5, 3.0],
            "gap_noise_floor_ratio": [0.02, 0.04, 0.06, 0.08, 0.12],
            "tail_min_ratio": [1.4, 1.6, 2.0, 2.5, 3.0],
        }
        results = sweep(records, CANDIDATE_RULES, grid, base)
        print(f"\nЛучшие настройки по {len(records)} комбинациям:")
        for rule_name, parameters, missed in results[:15]:
            print(f"  {rule_name:16} промахов {missed:3d}  {parameters}")
        return

    failures = evaluate(records, CANDIDATE_RULES, base)
    report(failures, len(records))


if __name__ == "__main__":
    main()
