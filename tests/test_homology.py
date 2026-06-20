"""Validate the from-scratch persistent homology against GUDHI and by hand."""
import numpy as np
import pytest

from topotransport.homology import (
    bottleneck_distance,
    compute_persistence,
)

gudhi = pytest.importorskip("gudhi")


def _gudhi_h1(P, max_edge):
    rc = gudhi.RipsComplex(points=P, max_edge_length=max_edge)
    st = rc.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    h1 = [[b, d] for dim, (b, d) in st.persistence() if dim == 1 and np.isfinite(d)]
    h0 = [[b, d] for dim, (b, d) in st.persistence() if dim == 0 and np.isfinite(d)]
    return np.array(h0).reshape(-1, 2), np.array(h1).reshape(-1, 2)


def test_unit_square_h1_is_exact():
    """A unit square has one H1 loop: born at side (1) and filled at diagonal."""
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    pd = compute_persistence(sq, max_dim=1)
    assert pd.h1.shape == (1, 2)
    assert pd.h1[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert pd.h1[0, 1] == pytest.approx(np.sqrt(2), abs=1e-9)
    # exactly one essential (infinite) H0 component
    assert np.sum(~np.isfinite(pd.h0[:, 1])) == 1


def test_matches_gudhi_on_random_clouds():
    rng = np.random.default_rng(0)
    for _ in range(8):
        P = rng.normal(size=(16, 2))
        me = float(np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)).max())
        mine = compute_persistence(P, max_dim=1, max_edge=me)
        g0, g1 = _gudhi_h1(P, me)
        assert bottleneck_distance(mine.finite(1), g1) < 1e-9
        md0 = np.sort(mine.finite(0)[:, 1])
        gd0 = np.sort(g0[:, 1])
        assert len(md0) == len(gd0)
        assert np.max(np.abs(md0 - gd0)) < 1e-9


def test_circle_has_one_loop():
    theta = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    circ = np.c_[np.cos(theta), np.sin(theta)]
    pd = compute_persistence(circ, max_dim=1)
    assert len(pd.finite(1)) == 1
    assert pd.max_persistence(1) > 0.8


def test_bottleneck_is_a_metric_like_zero_self():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(12, 2))
    d = compute_persistence(P).finite(1)
    assert bottleneck_distance(d, d) == pytest.approx(0.0, abs=1e-12)


def test_two_far_clusters_have_two_components():
    a = np.array([[0, 0], [0.1, 0], [0, 0.1]])
    b = a + np.array([20.0, 0.0])
    pd = compute_persistence(np.vstack([a, b]), max_dim=1)
    # at a scale above within-cluster spacing but below the 20-unit gap: 2 comps
    assert pd.n_components_at(1.0) == 2
