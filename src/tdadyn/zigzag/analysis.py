"""Null models and noise: the pipeline on a dataset, and rates with their denominators.

Two things live here, and they share one principle.

**A frequency is never reported without the number of trials.** S8 spent a session on this:
an estimate taken on sixty trials showed nothing, "zero misses on three seeds" turned out to
bound the true rate at 63% rather than at zero, and both mistakes were invisible until the
denominator was written down.  :class:`FalsePositiveRate` therefore cannot be constructed
without one, and its printed form carries it.

**The bounds are computed, not quoted.**  Until now the project's confidence bounds — 4.3%,
10.2%, 63% — were derived off-line and copied into ``method.md``.  Nothing in the package
could reproduce them, which is exactly the state to avoid: a number in prose
that is a memory of a calculation rather than its output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .backend import dionysus_barcode
from .construction import zigzag_complexes
from .events import EventLog, EventReport, event_log, event_report
from .indices import index_set
from .operations import coarse_bar_of, operation_stream

DEFAULT_CONFIDENCE = 0.95


def wilson_interval(
    hits: int, trials: int, *, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    hits, trials : int
        Successes and attempts.
    confidence : float, default 0.95
        Two-sided confidence level.

    Returns
    -------
    tuple of float
        Lower and upper bound, clipped to ``[0, 1]``.

    Notes
    -----
    Wilson rather than the normal approximation because the rates here sit near the ends of
    the range, where the normal interval leaves the unit interval and reports negative
    frequencies.  It reproduces the bounds ``method.md`` §1.2a publishes for 3/200 and
    12/200, which is what its test checks.

    The interval assumes the trials are independent.  Where they are not — snapshots inside
    one realisation share edges with their neighbours — it is narrower than the truth, and
    the caller has to say so rather than let the number stand unqualified.
    """
    _check_counts(hits, trials)
    z = _z_for(confidence)
    proportion = hits / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def zero_hits_upper_bound(trials: int, *, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Upper bound on a rate when no hit was observed at all.

    Parameters
    ----------
    trials : int
        Number of attempts, all of them misses.
    confidence : float, default 0.95

    Returns
    -------
    float
        The largest rate consistent with observing nothing: :math:`1 - \\alpha^{1/n}`.

    Notes
    -----
    The familiar rule of three, :math:`3/n`, is the large-sample form of this and agrees
    with it from a few dozen trials onwards — at ``n = 60`` both give about 0.05.  At small
    ``n`` the rule breaks down: it returns 1 for three trials, while the exact bound is
    **0.63**, which is the number ``method.md`` uses to reject "zero misses on three seeds"
    as a claim.  The exact form is used so that the same function serves both ends.
    """
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    alpha = 1 - confidence
    return 1 - alpha ** (1 / trials)


@dataclass(frozen=True)
class FalsePositiveRate:
    """A measured rate that cannot be separated from its denominator.

    Both fields are required, and the denominator survives into ``__str__``: a frequency
    printed without the number of trials is the failure S8 catalogued, and making it
    impossible to construct is cheaper than remembering not to.
    """

    hits: int
    trials: int

    def __post_init__(self) -> None:
        _check_counts(self.hits, self.trials)

    @property
    def rate(self) -> float:
        return self.hits / self.trials

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.hits, self.trials)

    def upper_bound_without_hits(self) -> float | None:
        """The bound that applies when nothing was observed; ``None`` when something was.

        Kept separate from :attr:`interval` because the two answer different questions, and
        conflating them is how "we saw nothing" becomes "the rate is zero".
        """
        if self.hits:
            return None
        return zero_hits_upper_bound(self.trials)

    def __str__(self) -> str:
        lower, upper = self.interval
        return (
            f"{self.rate:.4f} ({self.hits}/{self.trials}, 95% [{lower:.4f}, {upper:.4f}])"
        )


def _check_counts(hits: int, trials: int) -> None:
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    if hits < 0 or hits > trials:
        raise ValueError(f"hits must lie between 0 and trials, got {hits} of {trials}")


def _z_for(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence}")
    # Inverse normal CDF by bisection: no scipy here, and the reference layer's habit of
    # obvious code over fast code applies to anything whose number ends up in a report.
    target = 1 - (1 - confidence) / 2
    low, high = 0.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2
        if 0.5 * (1 + math.erf(middle / math.sqrt(2))) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2
