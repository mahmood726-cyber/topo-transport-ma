"""Validate the topological support certificate, incl. the hull-vs-topology gap."""
import numpy as np
import pytest

from topotransport import simulate
from topotransport.support import (
    certify_support,
    count_subpopulations,
    in_convex_hull,
    significant_holes,
)


def test_donut_centre_is_a_topological_gap_that_convex_hull_misses():
    """The hero result: the centre is INSIDE the convex hull (so a hull-only
    positivity check passes) but lies in a confirmed hole -> graded GAP."""
    M, y, v, truth = simulate.donut(n=44, r=4.0, width=0.5, seed=1)
    centre = truth["hole_center"]
    assert in_convex_hull(M, centre)  # convex hull says "supported"
    cert = certify_support(M, centre, n_boot=80, seed=1)
    assert cert.grade == "GAP"
    assert cert.topological_gap
    assert any(h.significant for h in cert.holes)


def test_donut_edge_is_supported():
    M, y, v, truth = simulate.donut(n=44, r=4.0, width=0.5, seed=1)
    cert = certify_support(M, truth["supported_point"], n_boot=80, seed=1)
    assert cert.grade in ("GOLD", "SILVER")
    assert not cert.topological_gap


def test_uniform_blob_has_no_significant_holes():
    M, y, v, _ = simulate.uniform_blob(n=40, seed=2)
    holes, _c = significant_holes(M, n_boot=80, seed=2)
    assert not any(h.significant for h in holes)
    # an interior point of a convex blob is genuinely supported
    cert = certify_support(M, [0.0, 0.0], n_boot=80, seed=2)
    assert cert.grade in ("GOLD", "SILVER")
    assert not cert.topological_gap


def test_two_clusters_detected_as_two_subpopulations():
    M, y, v, truth = simulate.two_clusters(seed=2)
    assert count_subpopulations(M) == truth["n_components"]


def test_blob_and_donut_are_single_components():
    assert count_subpopulations(simulate.uniform_blob(seed=4)[0]) == 1
    assert count_subpopulations(simulate.donut(n=44, seed=4)[0]) == 1


def test_far_target_is_extrapolation():
    M, y, v, _ = simulate.uniform_blob(n=40, seed=2)
    cert = certify_support(M, [50.0, 50.0], n_boot=40, seed=2)
    assert cert.grade == "NONE"
    assert not cert.in_hull


def test_certificate_supported_property():
    M, y, v, _ = simulate.uniform_blob(n=40, seed=2)
    assert certify_support(M, [0.0, 0.0], n_boot=40, seed=2).supported
    assert not certify_support(M, [99.0, 99.0], n_boot=40, seed=2).supported
