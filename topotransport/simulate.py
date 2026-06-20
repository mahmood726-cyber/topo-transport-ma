"""
simulate.py -- Known-truth generators for validating the topology + transport.

Each generator returns (modifiers, y, v, truth) where truth records the planted
structure so tests can assert the method recovers it.
"""
from __future__ import annotations

import numpy as np


def two_clusters(n_per=12, sep=6.0, sd=0.4, seed=0):
    """Two well-separated subpopulations -> H0 should see 2 components.

    Scalar I^2 cannot distinguish this from a single diffuse population.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0], sd, size=(n_per, 2))
    b = rng.normal([sep, 0], sd, size=(n_per, 2))
    M = np.vstack([a, b])
    # CATE linear in x[0]; two clusters give bimodal effects.
    true_cate = 0.1 + 0.05 * M[:, 0]
    v = rng.uniform(0.01, 0.05, size=len(M))
    y = true_cate + rng.normal(0, np.sqrt(v))
    return M, y, v, {"n_components": 2}


def donut(n=40, r=4.0, width=0.5, seed=0):
    """Points on an annulus -> one persistent H1 hole at the centre.

    A target at the centre is INSIDE the convex hull (so a hull test passes) but
    inside the hole, so it must be flagged GAP. This is the hero demonstration.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    rad = r + rng.normal(0, width, n)
    M = np.c_[rad * np.cos(theta), rad * np.sin(theta)]
    true_cate = 0.2 + 0.03 * M[:, 0] - 0.02 * M[:, 1]
    v = rng.uniform(0.01, 0.04, size=n)
    y = true_cate + rng.normal(0, np.sqrt(v))
    center = np.array([0.0, 0.0])
    edge = np.array([r, 0.0])
    return M, y, v, {"hole_center": center, "supported_point": edge}


def uniform_blob(n=40, seed=0):
    """A single convex blob -> no significant holes, one component."""
    rng = np.random.default_rng(seed)
    M = rng.uniform(-3, 3, size=(n, 2))
    true_cate = 0.15 + 0.04 * M[:, 0]
    v = rng.uniform(0.01, 0.04, size=n)
    y = true_cate + rng.normal(0, np.sqrt(v))
    return M, y, v, {"n_components": 1, "holes": 0}


def linear_cate(n=30, beta=(0.1, 0.05, -0.03), sd_x=2.0, seed=0):
    """Known linear CATE so meta-regression must recover `beta`."""
    rng = np.random.default_rng(seed)
    M = rng.normal(0, sd_x, size=(n, 2))
    b0, b1, b2 = beta
    true_cate = b0 + b1 * M[:, 0] + b2 * M[:, 1]
    v = rng.uniform(0.005, 0.02, size=n)
    y = true_cate + rng.normal(0, np.sqrt(v))
    return M, y, v, {"beta": np.array(beta)}
