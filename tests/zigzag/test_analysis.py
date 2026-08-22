"""Null models and noise: measuring, not confirming a zero.

The lesson this module exists to honour is S8's: a rate measured on a handful of trials
shows nothing, and "no false positives" without the number of trials is not a result.  So
every rate here arrives with its denominator attached, and the confidence bounds are
computed rather than quoted.
"""

from __future__ import annotations

import pathlib

import pytest

from tdadyn.zigzag.analysis import (
    FalsePositiveRate,
    analyse,
    ephemeral_counts,
    measure_null,
    wilson_interval,
    zero_hits_upper_bound,
)


# --- statistics ---------------------------------------------------------------------


def test_wilson_reproduces_the_bounds_method_md_already_publishes():
    """``method.md`` §1.2a states 4.3% and 10.2%, computed off-line in S8.

    Reproducing them from code is the strongest check available for this helper: the
    numbers were derived independently, are already in the document, and nothing in the
    package could compute them until now.
    """
    assert wilson_interval(3, 200)[1] == pytest.approx(0.043, abs=0.0005)
    assert wilson_interval(12, 200)[1] == pytest.approx(0.102, abs=0.0005)


def test_the_interval_covers_the_observed_rate_and_narrows_with_trials():
    lower, upper = wilson_interval(50, 100)
    assert lower < 0.5 < upper
    narrow = wilson_interval(500, 1000)
    assert (narrow[1] - narrow[0]) < (upper - lower)


def test_zero_hits_still_get_an_upper_bound():
    lower, upper = wilson_interval(0, 60)
    assert lower == 0.0
    assert upper > 0.0


def test_three_trials_without_a_miss_mean_sixty_three_percent_not_zero():
    """The number ``docs/zigzag.md`` §6 and ``method.md`` quote, now computed.

    Three seeds giving zero bound the true rate at 63%, not at zero.  The familiar rule of
    three — ``3/n`` — is the large-sample form of the same bound and is useless at ``n = 3``,
    where it returns 1; the exact expression is used instead and agrees with the rule where
    the rule applies.
    """
    assert zero_hits_upper_bound(3) == pytest.approx(0.632, abs=0.001)
    assert zero_hits_upper_bound(60) == pytest.approx(0.049, abs=0.001)
    assert zero_hits_upper_bound(60) == pytest.approx(3 / 60, abs=0.002)


def test_a_rate_cannot_be_built_without_its_denominator():
    """A frequency printed without the number of trials is the failure S8 catalogued."""
    with pytest.raises(TypeError):
        FalsePositiveRate(hits=5)


def test_a_rate_reports_its_own_bounds():
    rate = FalsePositiveRate(hits=3540, trials=3600)
    assert rate.rate == pytest.approx(0.9833, abs=0.0001)
    lower, upper = rate.interval
    assert lower < rate.rate < upper
    assert "3600" in str(rate), "the denominator must survive into the printed form"


def test_a_rate_with_no_hits_reports_the_bound_rather_than_zero():
    rate = FalsePositiveRate(hits=0, trials=60)
    assert rate.rate == 0.0
    assert rate.interval[1] > 0.0
    assert rate.upper_bound_without_hits() == pytest.approx(0.049, abs=0.001)


def test_impossible_counts_are_refused():
    with pytest.raises(ValueError, match="trials"):
        FalsePositiveRate(hits=1, trials=0)
    with pytest.raises(ValueError, match="hits"):
        FalsePositiveRate(hits=61, trials=60)


# --- the pipeline on one dataset ----------------------------------------------------

DATASETS = pytest.mark.slow_zigzag


@DATASETS
def test_the_pipeline_on_a_clean_dataset_reproduces_result_two(
    family_two_dataset, config, zigzag_of
):
    report = analyse(
        family_two_dataset, config=config, truth=family_two_dataset.ground_truth.events
    )
    assert report.metrics.recall == pytest.approx(1.0)
    assert report.metrics.precision == pytest.approx(1.0)
    assert report.invisible_bars == 0
    assert report.ephemeral_bars is None, "the fine path was not asked for"


@DATASETS
def test_the_fine_path_fills_in_the_ephemeral_count(cycle_lifecycle, config):
    """Off by default, and ``None`` rather than zero when it was not run.

    Zero for "we did not look" is the class of number this project has spent sessions
    removing; the report says so instead.
    """
    blind = analyse(cycle_lifecycle, config=config,
                    truth=cycle_lifecycle.ground_truth.events)
    full = analyse(cycle_lifecycle, config=config,
                   truth=cycle_lifecycle.ground_truth.events, with_fine_path=True)
    assert blind.ephemeral_bars is None
    assert full.ephemeral_bars == 23
    assert full.ephemeral_outside_prologue() == 0


@DATASETS
def test_the_diagram_is_built_once(cycle_lifecycle, config, monkeypatch):
    """The mistake made while measuring for this session, pinned so it cannot come back.

    Indexing the diagram inside a comprehension over the indices rebuilds the whole thing
    per index — 119 times — and turns tenths of a second into seconds.
    """
    import tdadyn.zigzag.analysis as module

    calls = []
    original = module.zigzag_complexes

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "zigzag_complexes", counted)
    analyse(cycle_lifecycle, config=config, truth=cycle_lifecycle.ground_truth.events)
    assert len(calls) == 1


# --- the null model -----------------------------------------------------------------


@pytest.fixture(scope="session")
def null_measurement(config):
    """The null model measured once, over seeds and realisations.

    Three calibration seeds times ``null_models.n_realizations``.  The seeds come from
    ``topology_check.calibration_seeds`` — another section's key, but the same seeds the
    whole project is calibrated on; a ``zigzag.seeds`` of its own would let them drift.
    """
    from tdadyn.synthetic import build_dynamic_graph, build_null_graph

    datasets = [
        build_null_graph(config, build_dynamic_graph("cycle_lifecycle", config, seed=seed),
                         repeat=repeat)
        for seed in config.topology_check.calibration_seeds
        for repeat in range(config.null_models.n_realizations)
    ]
    return measure_null(datasets, config=config), len(datasets)


@DATASETS
def test_the_null_model_fires_almost_everywhere(null_measurement, config):
    """Not "no false positives" — a measured rate, with its denominator.

    ``method.md`` §4.2 says a random graph at this density honestly carries cycles: it has
    :math:`\\beta_0 = 4..5` and :math:`\\beta_1 = 2..5` on every snapshot, and the edges are
    redrawn independently, so classes are born and die continuously.  Every snapshot of the
    null model is a topological event, and the measurement says so.

    The assertion is deliberately about the **shape** of the answer rather than its exact
    value: a rate far above zero, over thousands of trials.  If someone adds a significance
    criterion the rate will drop and this test will fail — correctly, because that would be
    a deliberate change and the document would have to change with it.
    """
    measurement, realisations = null_measurement
    assert measurement.per_snapshot.trials >= 3000
    assert measurement.per_snapshot.rate > 0.5
    assert measurement.events_per_snapshot > 1.0

    lower, upper = measurement.per_snapshot.interval
    assert lower < measurement.per_snapshot.rate < upper
    assert realisations == (
        len(config.topology_check.calibration_seeds) * config.null_models.n_realizations
    )


@DATASETS
def test_both_denominators_are_reported_and_the_wider_one_is_wider(null_measurement):
    """Snapshots inside a realisation are not independent; realisations are.

    Per snapshot there are thousands of trials and a narrow interval that assumes an
    independence the data does not have.  Per realisation the trials are genuinely
    independent and there are sixty, so the interval is wide.  Publishing only the first
    would overstate the precision; only the second would throw away most of the data.
    """
    measurement, _ = null_measurement
    per_snapshot = measurement.per_snapshot.interval
    per_realisation = measurement.per_realisation.interval
    assert (per_realisation[1] - per_realisation[0]) > (per_snapshot[1] - per_snapshot[0])


@DATASETS
def test_the_false_positives_appear_in_both_dimensions(null_measurement):
    """Neither dimension is silent, so neither is a special case of the other."""
    measurement, _ = null_measurement
    assert set(measurement.events_by_dimension) == {0, 1}
    assert all(count > 0 for count in measurement.events_by_dimension.values())


def test_the_family_one_threshold_is_not_the_quantity_measured_here(config):
    """``null_models.max_false_positive_rate`` names this and means something else.

    It is calibrated for the significance criterion on uniform point clouds — the share of
    realisations where that criterion invented a feature, measured at 1.5% with a bound of
    4.3% (``method.md`` §1.2a).  The zigzag measures the share of snapshots on which an
    event is reported, and it is near one.  A test that compared them would report a failure
    of a threshold that was never about this.
    """
    assert config.null_models.max_false_positive_rate == pytest.approx(0.05)
    import tdadyn.zigzag.analysis as module
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    assert "max_false_positive_rate" in source, "the reason must be written down"
    assert "not consulted" in source or "is **not** consulted" in source
