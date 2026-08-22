"""Null models and noise: measuring, not confirming a zero.

The lesson this module exists to honour is S8's: a rate measured on a handful of trials
shows nothing, and "no false positives" without the number of trials is not a result.  So
every rate here arrives with its denominator attached, and the confidence bounds are
computed rather than quoted.
"""

from __future__ import annotations

import pytest

from tdadyn.zigzag.analysis import (
    FalsePositiveRate,
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
