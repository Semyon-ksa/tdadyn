"""Cloud to weighted network and back — the bridge to task 3.

The fast half: no persistent homology, only the properties that decide the answer
quietly.  ``docs/method.md`` §1.3 names three of them — kernel width, monotonicity of the
inverse transform, and the k window of the geodesic construction — and each gets its
negative control here, because a guard that never fires proves nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from tdadyn.synthetic import pairwise_distances
from tdadyn.synthetic.network import (
    COLLAPSE_FLOOR,
    collapsed_weight_fraction,
    distances_from_similarity,
    gaussian_kernel,
    geodesic_diameter_ratio,
    is_monotone_decreasing,
    kernel_width,
    knn_geodesic_distances,
    knn_graph,
)

from conftest import STATIC_GENERATORS, build_static_cloud

DISCONNECTED_BY_CONSTRUCTION = ("two_circles", "hopf_link")


@pytest.fixture(params=STATIC_GENERATORS)
def cloud_distances(request, config):
    return request.param, pairwise_distances(build_static_cloud(request.param, config).points)


# --- kernel width ----------------------------------------------------------------


def test_configured_width_keeps_the_kernel_alive(cloud_distances, config):
    """The guard from docs/method.md §1.3, on every object of family 1."""
    _, distances = cloud_distances
    network = config.network_from_cloud
    sigma = kernel_width(distances, median_fraction=network.kernel_sigma_median_fraction)
    weights = gaussian_kernel(distances, sigma=sigma)
    assert collapsed_weight_fraction(weights) <= network.max_collapsed_weight_fraction


def test_an_absolute_width_collapses_the_torus(config):
    """The negative control, and the actual lesson of the step.

    The old config carried an absolute width of 0.5.  On the torus, whose diameter is
    5.71, that puts 60% of the weights below 1e-6 — the geometry is destroyed before any
    inverse transform is applied.  The failure belongs to the kernel, not to ``1 - w``.
    """
    distances = pairwise_distances(build_static_cloud("torus", config).points)
    collapsed = collapsed_weight_fraction(gaussian_kernel(distances, sigma=0.5))
    assert collapsed > 0.5
    assert collapsed > config.network_from_cloud.max_collapsed_weight_fraction


def test_width_scales_with_the_object(config):
    """A fraction of the median adapts; an absolute number cannot."""
    fraction = config.network_from_cloud.kernel_sigma_median_fraction
    widths = {
        name: kernel_width(
            pairwise_distances(build_static_cloud(name, config).points),
            median_fraction=fraction,
        )
        for name in ("figure_eight", "torus")
    }
    assert widths["torus"] > 2 * widths["figure_eight"]


# --- inverse transforms ----------------------------------------------------------


@pytest.mark.parametrize("transform", ["sqrt_neg_log", "one_minus"])
def test_both_transforms_are_monotonically_decreasing(transform, config):
    """Monotonicity keeps the order simplices enter the filtration."""
    assert is_monotone_decreasing(transform, sigma=1.0)


def test_one_minus_is_not_forbidden(config):
    """It is what task 3 will use as d = 1 - |r| (docs/method.md §1.3).

    Dropping it from the harness would mean not testing the function that will run on
    real data.
    """
    distances = pairwise_distances(build_static_cloud("circle", config).points)
    sigma = kernel_width(
        distances, median_fraction=config.network_from_cloud.kernel_sigma_median_fraction
    )
    weights = gaussian_kernel(distances, sigma=sigma)
    recovered = distances_from_similarity(weights, transform="one_minus")
    assert np.array_equal(recovered, recovered.T)
    assert np.all(np.diagonal(recovered) == 0.0)


def test_sqrt_neg_log_inverts_the_kernel_exactly(config):
    """With sigma supplied the round trip returns the original distances."""
    distances = pairwise_distances(build_static_cloud("circle", config).points)
    sigma = kernel_width(distances, median_fraction=0.65)
    weights = gaussian_kernel(distances, sigma=sigma)
    recovered = distances_from_similarity(weights, transform="sqrt_neg_log", sigma=sigma)
    assert np.allclose(recovered, distances, atol=1e-8)


def test_sqrt_neg_log_without_sigma_is_refused(config):
    weights = np.array([[1.0, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match="needs a positive sigma"):
        distances_from_similarity(weights, transform="sqrt_neg_log")


def test_unknown_transform_is_refused():
    weights = np.array([[1.0, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match="unknown transform"):
        distances_from_similarity(weights, transform="exp")


# --- the geodesic construction ---------------------------------------------------


def test_geodesic_distances_are_exactly_symmetric(config):
    """scipy's shortest_path is not bitwise symmetric; the module fixes that.

    Without the explicit symmetrisation the distance validator rejects the matrix — this
    was measured, not assumed.
    """
    distances = pairwise_distances(build_static_cloud("torus", config).points)
    geodesic = knn_geodesic_distances(
        distances, k=config.network_from_cloud.knn_k, symmetrize="union"
    )
    assert np.array_equal(geodesic, geodesic.T)
    assert np.all(np.diagonal(geodesic) == 0.0)


def test_geodesic_exceeds_the_ambient_metric(config):
    """On a curved manifold the intrinsic distance is the longer one."""
    distances = pairwise_distances(build_static_cloud("torus", config).points)
    geodesic = knn_geodesic_distances(
        distances, k=config.network_from_cloud.knn_k, symmetrize="union"
    )
    assert np.all(geodesic >= distances - 1e-9)
    ratio = geodesic_diameter_ratio(geodesic, distances)
    assert ratio >= config.network_from_cloud.min_geodesic_diameter_ratio


def test_short_circuiting_is_caught_by_the_diameter_guard(config):
    """The negative control for the upper end of the k window.

    As k grows, chords appear across the holes and the intrinsic metric degenerates into
    the ambient one.  The guard costs no persistent homology.
    """
    distances = pairwise_distances(build_static_cloud("torus", config).points)
    huge_k = distances.shape[0] // 2
    geodesic = knn_geodesic_distances(distances, k=huge_k, symmetrize="union")
    assert geodesic_diameter_ratio(geodesic, distances) < (
        config.network_from_cloud.min_geodesic_diameter_ratio
    )


def test_a_disconnected_graph_is_an_explicit_error(config):
    """Substituting a large number for infinity would feed rubbish to the pipeline."""
    distances = pairwise_distances(build_static_cloud("two_circles", config).points)
    with pytest.raises(ValueError, match="falls into 2 components"):
        knn_geodesic_distances(distances, k=config.network_from_cloud.knn_k, symmetrize="union")


@pytest.mark.parametrize("name", DISCONNECTED_BY_CONSTRUCTION)
def test_several_components_leave_the_graph_in_pieces_at_the_default_k(name, config):
    """A set with beta_0 > 1 is disconnected at the k that suits the surfaces.

    Its components are separated by empty space, so ``knn_k`` leaves the graph in pieces
    and the construction must either take a larger k or step back entirely
    (``docs/method.md`` §1.3).
    """
    distances = pairwise_distances(build_static_cloud(name, config).points)
    for k in (20, config.network_from_cloud.knn_k):
        graph = knn_graph(distances, k=k, symmetrize="union")
        assert connected_components(csr_matrix(graph), directed=False)[0] == 2


def test_a_larger_k_connects_two_circles(config):
    """For well-separated components a larger k is enough; the override records it."""
    distances = pairwise_distances(build_static_cloud("two_circles", config).points)
    override = config.network_from_cloud.knn_k_by_generator["two_circles"]
    graph = knn_graph(distances, k=override, symmetrize="union")
    assert connected_components(csr_matrix(graph), directed=False)[0] == 1


def test_the_override_for_two_circles_keeps_the_metric_alive(config):
    """A larger k both connects the graph and clears the short-circuit guard."""
    network = config.network_from_cloud
    distances = pairwise_distances(build_static_cloud("two_circles", config).points)
    geodesic = knn_geodesic_distances(
        distances,
        k=network.knn_k_by_generator["two_circles"],
        symmetrize=network.knn_symmetrize,
    )
    assert geodesic_diameter_ratio(geodesic, distances) >= network.min_geodesic_diameter_ratio


def test_the_hopf_link_has_no_working_k_and_carries_no_override(config):
    """Bridges between components create loops that outlive the noise floor here.

    ``docs/method.md`` §1.3: m bridges add (m - 1) spurious cycles, and how long they live
    depends on how close the components come.  The Hopf link is the extreme — its circles
    pass right against each other, so at every k that connects the graph the spurious
    class competes with the real ones.  A per-generator override would look like a
    calibrated value where no working value exists, so the config deliberately has none.
    """
    from tdadyn.synthetic.calibration import GEODESIC_OUT_OF_DOMAIN

    assert "hopf_link" in GEODESIC_OUT_OF_DOMAIN
    assert "hopf_link" not in config.network_from_cloud.knn_k_by_generator


def test_bridges_between_components_create_spurious_loops():
    """The mechanism, on a flag complex the reference implementation can measure.

    Two disjoint 6-cycles joined by m bridges between non-adjacent pairs:
    beta_1 = 2 + (m - 1) for m >= 1, which is why connecting components is not free.
    """
    from tdadyn.synthetic.reference import betti_numbers_of_graph

    def two_cycles_with_bridges(n_bridges):
        adjacency = np.zeros((12, 12))
        for base in (0, 6):
            for index in range(6):
                left, right = base + index, base + (index + 1) % 6
                adjacency[left, right] = adjacency[right, left] = 1.0
        for left, right in [(0, 6), (2, 8), (4, 10), (1, 7)][:n_bridges]:
            adjacency[left, right] = adjacency[right, left] = 1.0
        return adjacency

    expected_beta_1 = {0: 2, 1: 2, 2: 3, 3: 4, 4: 5}
    for n_bridges, expected in expected_beta_1.items():
        betti = betti_numbers_of_graph(
            two_cycles_with_bridges(n_bridges),
            threshold=0.5,
            max_homology_dim=2,
            max_nodes=30,
        )
        assert betti[1] == expected, f"{n_bridges} bridges"
        assert betti[0] == (2 if n_bridges == 0 else 1)


def test_union_connects_where_mutual_does_not(config):
    """Why union is the adopted rule: it widens the window from below."""
    distances = pairwise_distances(build_static_cloud("sphere", config).points)
    small_k = 3
    union = knn_graph(distances, k=small_k, symmetrize="union")
    mutual = knn_graph(distances, k=small_k, symmetrize="mutual")
    union_components = connected_components(csr_matrix(union), directed=False)[0]
    mutual_components = connected_components(csr_matrix(mutual), directed=False)[0]
    assert union_components <= mutual_components
    assert (union > 0).sum() >= (mutual > 0).sum()


def test_knn_graph_rejects_an_impossible_k(config):
    distances = pairwise_distances(build_static_cloud("circle", config).points)
    with pytest.raises(ValueError, match="k must lie in"):
        knn_graph(distances, k=distances.shape[0], symmetrize="union")


def test_collapse_floor_is_the_documented_one():
    assert COLLAPSE_FLOOR == 1e-6


# --- topology of the network, against the cloud ----------------------------------


def diagram_structure(diagrams):
    """Reduce a barcode to what a monotone reparametrisation must preserve.

    A monotone ``phi`` maps every bar ``(b, d)`` to ``(phi(b), phi(d))``: the classes are
    the same, and the order of all births and deaths is the same.  Persistences are *not*
    preserved, so comparing them would be comparing the wrong thing — that is the whole
    point of ``docs/method.md`` §1.3.
    """
    structure = {}
    for dim, diagram in diagrams.items():
        if diagram.size == 0:
            structure[dim] = (0, ())
            continue
        finite = diagram[np.isfinite(diagram[:, 1])]
        endpoints = np.concatenate([finite[:, 0], finite[:, 1]])
        structure[dim] = (len(diagram), tuple(np.argsort(np.argsort(endpoints)).tolist()))
    return structure


@pytest.mark.slow
@pytest.mark.parametrize("transform", ["sqrt_neg_log", "one_minus"])
def test_the_network_keeps_the_structure_of_the_diagram(transform, config):
    """Both transforms preserve which classes exist and in what order.

    Compared by structure, not by persistence: roadmap S5 asks for exactly this
    distinction, because a monotone map keeps the first and distorts the second.
    """
    from tdadyn.synthetic import persistence_diagrams

    topology = config.topology_check
    network = config.network_from_cloud
    distances = pairwise_distances(build_static_cloud("circle", config).points)
    sigma = kernel_width(distances, median_fraction=network.kernel_sigma_median_fraction)
    weights = gaussian_kernel(distances, sigma=sigma)
    recovered = distances_from_similarity(weights, transform=transform, sigma=sigma)

    original = persistence_diagrams(
        distances, max_homology_dim=1, coeff_field=topology.coeff_field
    )
    through_network = persistence_diagrams(
        recovered, max_homology_dim=1, coeff_field=topology.coeff_field
    )
    assert diagram_structure(original) == diagram_structure(through_network)


@pytest.mark.slow
@pytest.mark.parametrize("transform", ["sqrt_neg_log", "one_minus"])
def test_the_network_recovers_the_expected_betti_numbers(transform, config):
    """With a healthy kernel both transforms reproduce the table."""
    from tdadyn.synthetic import significant_betti

    topology = config.topology_check
    network = config.network_from_cloud
    expected = {"circle": {0: 1, 1: 1, 2: 0}, "torus": {0: 1, 1: 2, 2: 1}}
    for name, table in expected.items():
        distances = pairwise_distances(build_static_cloud(name, config).points)
        sigma = kernel_width(distances, median_fraction=network.kernel_sigma_median_fraction)
        weights = gaussian_kernel(distances, sigma=sigma)
        recovered = distances_from_similarity(weights, transform=transform, sigma=sigma)
        assert significant_betti(
            recovered,
            max_homology_dim=topology.max_homology_dim,
            coeff_field=topology.coeff_field,
            criterion=topology.criterion,
            significance_ratio=topology.significance_ratio,
            gap_noise_floor_ratio=topology.gap_noise_floor_ratio,
            gap_min_ratio=topology.gap_min_ratio,
        ) == table, f"{name} through {transform}"


@pytest.mark.slow
def test_the_geodesic_metric_recovers_the_expected_betti_numbers(config):
    """The Euclidean criterion parameters were *verified* on the geodesic metric.

    ``docs/method.md`` §1.3: they are not inherited, they were re-calibrated, and the
    zero-miss region turned out to contain them.  The Hopf link is outside the domain of
    the construction and is not part of this check.
    """
    from tdadyn.synthetic import significant_betti

    topology = config.topology_check
    network = config.network_from_cloud
    expected = {"torus": {0: 1, 1: 2, 2: 1}, "klein_bottle": {0: 1, 1: 2, 2: 1}}
    for name, table in expected.items():
        distances = pairwise_distances(build_static_cloud(name, config).points)
        geodesic = knn_geodesic_distances(
            distances,
            k=network.knn_k_by_generator.get(name, network.knn_k),
            symmetrize=network.knn_symmetrize,
        )
        assert significant_betti(
            geodesic,
            max_homology_dim=topology.max_homology_dim,
            coeff_field=topology.coeff_field,
            criterion=topology.criterion,
            significance_ratio=topology.significance_ratio,
            gap_noise_floor_ratio=topology.gap_noise_floor_ratio,
            gap_min_ratio=topology.gap_min_ratio,
        ) == table, name
