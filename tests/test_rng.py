"""Derived randomness streams: reproducible, independent, free of global state."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from tdadyn.synthetic import rng_for

SEED = 20260201

_PROBE = (
    "import json;"
    "from tdadyn.synthetic import rng_for;"
    "print(json.dumps(rng_for(20260201, 'torus', 2).normal(size=4).tolist()))"
)


def draw(base_seed: int = SEED, stream: str = "torus", repeat: int = 0) -> np.ndarray:
    return rng_for(base_seed, stream, repeat).normal(size=16)


def test_same_arguments_give_bitwise_identical_draws():
    assert draw().tobytes() == draw().tobytes()


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (("torus", 0), ("sphere", 0)),
        (("torus", 0), ("torus", 1)),
        (("torus", 1), ("torus", 2)),
    ],
)
def test_different_streams_are_independent(left, right):
    first = draw(stream=left[0], repeat=left[1])
    second = draw(stream=right[0], repeat=right[1])
    assert first.tobytes() != second.tobytes()
    # Independent, not merely different: an offset copy of the same stream would
    # still compare unequal.
    assert abs(float(np.corrcoef(first, second)[0, 1])) < 0.9


def test_different_base_seeds_give_different_streams():
    assert draw(base_seed=1).tobytes() != draw(base_seed=2).tobytes()


def test_global_numpy_seed_does_not_affect_the_stream():
    expected = draw()
    np.random.seed(12345)
    np.random.normal(size=1000)
    assert draw().tobytes() == expected.tobytes()


def test_stream_is_independent_of_pythonhashseed():
    """The stream name must not be hashed with the built-in ``hash``.

    ``hash`` is randomised per process, so a stream keyed on it would silently
    produce different data on every run.
    """
    outputs = set()
    for hash_seed in ("0", "1"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1


@pytest.mark.parametrize(
    ("base_seed", "stream", "repeat", "error"),
    [
        (-1, "torus", 0, ValueError),
        (SEED, "", 0, ValueError),
        (SEED, "torus", -1, ValueError),
        (SEED, "torus", True, TypeError),
        (True, "torus", 0, TypeError),
    ],
)
def test_invalid_arguments_are_rejected(base_seed, stream, repeat, error):
    with pytest.raises(error):
        rng_for(base_seed, stream, repeat)
