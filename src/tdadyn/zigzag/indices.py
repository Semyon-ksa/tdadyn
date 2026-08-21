"""Coarse indices of the zigzag diagram: snapshots and the unions between them.

The index set is :math:`I = \\{0, \\tfrac12, 1, \\tfrac32, \\dots, T-1\\}`
(``docs/zigzag.md`` §2.1): integer indices are the snapshots :math:`K_i`, half-integer
ones the intermediate unions :math:`K_{i+1/2} = K_i \\cup K_{i+1}`.

**Half-integers are stored doubled.**  :class:`CoarseIndex` wraps a single ``int``: ``2i``
for snapshot ``i``, ``2i + 1`` for the union that follows it.  The arithmetic is then
exact, and "snapshot or intermediate" is a parity test rather than a comparison of floats.

**It is a type and not a bare int on purpose.**  An off-by-one in this place shows up in
the report as a systematic localisation error of the algorithm and is indistinguishable
from one (``docs/zigzag.md`` §2.4).  A bare ``int`` lets a snapshot index be passed where a
coarse one is expected without a word; a separate type does not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class CoarseIndex:
    """One index of the zigzag diagram, stored doubled.

    Parameters
    ----------
    doubled : int
        ``2i`` for snapshot ``i``, ``2i + 1`` for the union of snapshots ``i`` and
        ``i + 1``.  Build through :meth:`at_snapshot` or :meth:`between` rather than
        directly, so that the doubling is applied in one place.
    """

    doubled: int

    @classmethod
    def at_snapshot(cls, snapshot: int) -> "CoarseIndex":
        """The index of snapshot ``snapshot``."""
        _check_position(snapshot)
        return cls(doubled=2 * snapshot)

    @classmethod
    def between(cls, snapshot: int) -> "CoarseIndex":
        """The index of the union of ``snapshot`` and ``snapshot + 1``."""
        _check_position(snapshot)
        return cls(doubled=2 * snapshot + 1)

    @property
    def is_snapshot(self) -> bool:
        """True at an integer index, that is at a snapshot of the dataset."""
        return self.doubled % 2 == 0

    @property
    def snapshot(self) -> int:
        """The snapshot this index refers to.

        Raises
        ------
        ValueError
            At an intermediate index.  Half-integer ends are never rounded
            (``docs/zigzag.md`` §4.2): a class born at :math:`i + 1/2` exists in neither
            snapshot but exists in their union, and rounding would report a plausible
            event at a wrong time instead of an unusual one at the right time.
        """
        if not self.is_snapshot:
            raise ValueError(
                f"{self} is an intermediate index and belongs to no snapshot; it lies "
                "between snapshots "
                f"{self.doubled // 2} and {self.doubled // 2 + 1}. Half-integer ends are "
                "not rounded (docs/zigzag.md 4.2)"
            )
        return self.doubled // 2

    @property
    def left_snapshot(self) -> int:
        """The snapshot at or before this index; defined at every index."""
        return self.doubled // 2

    def __str__(self) -> str:
        return str(self.left_snapshot) if self.is_snapshot else f"{self.left_snapshot}½"


def index_set(n_snapshots: int) -> tuple[CoarseIndex, ...]:
    """Every coarse index of a zigzag over ``n_snapshots`` snapshots, in order.

    Parameters
    ----------
    n_snapshots : int
        Length of the snapshot sequence; at least two.

    Returns
    -------
    tuple of CoarseIndex
        The :math:`2T - 1` indices, alternating snapshot and intermediate, starting and
        ending at a snapshot.
    """
    _check_position(n_snapshots)
    if n_snapshots < 2:
        raise ValueError(
            f"a zigzag needs at least two snapshots, got {n_snapshots}: with one there "
            "is no arrow and no intermediate union"
        )
    return tuple(CoarseIndex(doubled=doubled) for doubled in range(2 * n_snapshots - 1))


def _check_position(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an int, got {type(value).__name__}: {value!r}")
    if value < 0:
        raise ValueError(f"expected a non-negative index, got {value}")
