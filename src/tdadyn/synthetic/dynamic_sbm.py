"""Family 3: a stochastic block model whose communities merge and split on schedule.

**The truth here is the partition, not a barcode**, and that is a decision rather than an
omission (``docs/method.md`` §3.1).  Three things follow, and each is the reason the
obvious alternative was rejected:

* At ``p_out = 0.02`` on blocks of fifteen the blocks are stitched together by dozens of
  edges, so the graph is connected throughout and **merging two communities changes no
  homology at all**.  :math:`\\beta_0` does move on one of the three calibration seeds —
  but because a single vertex happened to draw no edges, which tracks sampling noise and
  not the schedule.
* Recording a *measured* Betti curve would be truth as pipeline output.  Family 2 gets
  away with a computed curve because the curve is **also predicted by the construction**
  and the test compares two sources; here there is no prediction to compare against.
* Lowering ``p_out`` until the blocks separate would make the answer known only *with
  high probability*, and a bad seed would quietly ship a dataset with a wrong truth.

So this generator writes ``truth_kind="partition"``: the community labels per snapshot
and the schedule that produced them, both known in advance and exactly.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from tdadyn.core.contracts import FORMAT_VERSION, DynamicGraphDataset, GroundTruth
from tdadyn.core.naming import register_name_order
from tdadyn.core.rng import base_seed_of

register_name_order(
    "dynamic_sbm",
    [
        "n_nodes",
        "n_snapshots",
        "p_in",
        "p_out",
        "block_sizes",
        ("sched", ["merge_at", "split_at"]),
    ],
)

DYNAMIC_SBM_NOTES = (
    "Истина — разбиение на сообщества по снимкам и расписание слияния/деления, и ничего "
    "сверх того (docs/method.md, 3.1). Числа Бетти здесь НЕ истина: при p_out=0.02 блоки "
    "прошиты десятками рёбер, граф связен на всех снимках, и слияние сообществ не "
    "является гомологическим событием. Соседние разбиения связаны измельчением."
)


def make_dynamic_sbm(
    n_nodes: int,
    n_snapshots: int,
    block_sizes: tuple[int, ...],
    p_in: float,
    p_out: float,
    merge_at: int,
    split_at: int,
    rng: np.random.Generator,
) -> DynamicGraphDataset:
    """Sample a sequence of SBM snapshots whose communities merge and then split again.

    Parameters
    ----------
    n_nodes : int
        Number of nodes; ``block_sizes`` must sum to it.
    n_snapshots : int
        Length of the sequence.
    block_sizes : tuple of int
        Sizes of the base communities, at least two of them so that a merge is possible.
    p_in, p_out : float
        Edge probabilities within and between communities; ``p_out`` must be the smaller,
        or the blocks are not communities.
    merge_at : int
        Snapshot from which the first two communities are one.
    split_at : int
        Snapshot from which they are separate again.
    rng : numpy.random.Generator
        The only source of randomness, passed straight to
        :func:`networkx.stochastic_block_model`.

    Returns
    -------
    DynamicGraphDataset
        Snapshots of shape ``(n_snapshots, n_nodes, n_nodes)``, unweighted and symmetric,
        with ``truth_kind="partition"`` and community labels per snapshot.

    Notes
    -----
    The partitions of consecutive snapshots are related by refinement, so the transitions
    really are a merge and a split rather than a re-cut (``docs/method.md`` §3.1a).  The
    schedule is deterministic; only the edges are random.
    """
    _check_arguments(n_nodes, n_snapshots, block_sizes, p_in, p_out, merge_at, split_at)

    partitions = [_partition_at(index, block_sizes, merge_at, split_at) for index in range(n_snapshots)]
    snapshots = np.zeros((n_snapshots, n_nodes, n_nodes), dtype=np.float64)
    labels = np.zeros((n_snapshots, n_nodes), dtype=np.int64)

    for index, partition in enumerate(partitions):
        probabilities = [
            [p_in if row == column else p_out for column in range(len(partition))]
            for row in range(len(partition))
        ]
        graph = nx.stochastic_block_model(list(partition), probabilities, seed=rng)
        snapshots[index] = nx.to_numpy_array(graph, dtype=np.float64)
        labels[index] = _labels_of(partition)

    params = {
        "n_nodes": int(n_nodes),
        "n_snapshots": int(n_snapshots),
        "p_in": float(p_in),
        "p_out": float(p_out),
        "block_sizes": [int(size) for size in block_sizes],
        "merge_at": int(merge_at),
        "split_at": int(split_at),
    }
    ground_truth = GroundTruth(
        format_version=FORMAT_VERSION,
        family="dynamic_graph",
        truth_kind="partition",
        generator="dynamic_sbm",
        params=params,
        seed=base_seed_of(rng),
        # Recorded although `partition` does not require it: the snapshots are still a
        # graph someone will threshold, and the direction should not have to be guessed
        # (docs/method.md, 2.3).
        edge_rule="weight_above",
        edge_threshold=0.5,
        community_labels=labels,
        notes=DYNAMIC_SBM_NOTES,
    )
    return DynamicGraphDataset(snapshots=snapshots, ground_truth=ground_truth)


def _partition_at(
    index: int, block_sizes: tuple[int, ...], merge_at: int, split_at: int
) -> tuple[int, ...]:
    """Block sizes in force at one snapshot.

    Between ``merge_at`` and ``split_at`` the first two communities are one block; the
    result is therefore a coarsening of the base partition, and the base partition is a
    refinement of it — which is what makes the transitions a merge and a split.
    """
    if merge_at <= index < split_at:
        return (block_sizes[0] + block_sizes[1], *block_sizes[2:])
    return tuple(block_sizes)


def _labels_of(partition: tuple[int, ...]) -> np.ndarray:
    """Community label of every node, blocks laid out contiguously."""
    return np.repeat(np.arange(len(partition), dtype=np.int64), partition)


def _check_arguments(
    n_nodes: int,
    n_snapshots: int,
    block_sizes: tuple[int, ...],
    p_in: float,
    p_out: float,
    merge_at: int,
    split_at: int,
) -> None:
    """Validate what the schedule cannot repair on its own."""
    if len(block_sizes) < 2:
        raise ValueError(
            f"at least two communities are needed for a merge, got {list(block_sizes)}"
        )
    if sum(block_sizes) != n_nodes:
        raise ValueError(
            f"block_sizes sum to {sum(block_sizes)} but n_nodes is {n_nodes}"
        )
    if not 0.0 <= p_out < p_in <= 1.0:
        raise ValueError(
            f"need 0 <= p_out ({p_out}) < p_in ({p_in}) <= 1, otherwise the blocks are "
            "not communities"
        )
    if not 1 <= merge_at < split_at < n_snapshots:
        raise ValueError(
            f"need 1 <= merge_at ({merge_at}) < split_at ({split_at}) < n_snapshots "
            f"({n_snapshots}); a change on snapshot 0 has no preceding snapshot"
        )
