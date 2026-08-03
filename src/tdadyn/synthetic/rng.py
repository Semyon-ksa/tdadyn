"""Derived randomness streams: one independent generator per generator and repeat.

A stream is addressed by *name*, never by the ordinal of a ``SeedSequence.spawn``
call.  With ordinals, adding a new generator shifts the streams of every generator
after it and silently changes datasets that were already published.
"""

from __future__ import annotations

import hashlib

import numpy as np

_DIGEST_BYTES = 16
_WORD_BYTES = 4


def rng_for(base_seed: int, stream: str, repeat: int = 0) -> np.random.Generator:
    """Derive an independent generator for a named stream.

    Parameters
    ----------
    base_seed : int
        Base seed from ``config.yaml``.
    stream : str
        Stable name of the stream, normally the generator name (``"torus"``).  Do not
        use the dataset id: it depends on the parameters, so changing any parameter
        would renumber the noise streams and break comparability across noise levels.
    repeat : int, default 0
        Repeat index for repeated realisations (``noise.n_repeats``,
        ``null_models.n_realizations``).

    Returns
    -------
    numpy.random.Generator
        A generator that depends only on the three arguments, reproducibly across
        processes and machines.

    Notes
    -----
    The name is hashed with ``blake2b`` rather than the built-in ``hash``: the latter
    is randomised per process via ``PYTHONHASHSEED`` and would make the streams
    irreproducible between runs.
    """
    if not isinstance(base_seed, (int, np.integer)) or isinstance(base_seed, bool):
        raise TypeError(f"base_seed must be an int, got {type(base_seed).__name__}")
    if base_seed < 0:
        raise ValueError(f"base_seed must be >= 0, got {base_seed}")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a non-empty string")
    if not isinstance(repeat, (int, np.integer)) or isinstance(repeat, bool):
        raise TypeError(f"repeat must be an int, got {type(repeat).__name__}")
    if repeat < 0:
        raise ValueError(f"repeat must be >= 0, got {repeat}")

    entropy = [int(base_seed), *_stream_words(stream), int(repeat)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def base_seed_of(rng: np.random.Generator) -> int:
    """Recover the base seed a generator's stream was derived from.

    Parameters
    ----------
    rng : numpy.random.Generator
        A generator produced by :func:`rng_for`, or seeded with a plain integer.

    Returns
    -------
    int
        The base seed, which is the first element of the entropy :func:`rng_for` builds.

    Notes
    -----
    Generators need the seed to record it in the ground truth, and taking it from the
    stream itself is what keeps the two from disagreeing.  Passing ``seed`` and ``rng``
    as separate arguments would let a caller record one seed while generating from
    another, and the dataset would then be irreproducible from its own metadata.
    """
    seed_seq = getattr(rng.bit_generator, "seed_seq", None)
    if seed_seq is None:
        raise ValueError("the generator carries no seed sequence; build it with rng_for")
    entropy = seed_seq.entropy
    if isinstance(entropy, (int, np.integer)):
        return int(entropy)
    if isinstance(entropy, (list, tuple)) and entropy:
        return int(entropy[0])
    raise ValueError(f"cannot recover a base seed from entropy {entropy!r}")


def _stream_words(stream: str) -> list[int]:
    """Hash a stream name into 32-bit words for ``SeedSequence`` entropy."""
    digest = hashlib.blake2b(stream.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
    return [
        int.from_bytes(digest[offset : offset + _WORD_BYTES], "big")
        for offset in range(0, _DIGEST_BYTES, _WORD_BYTES)
    ]
