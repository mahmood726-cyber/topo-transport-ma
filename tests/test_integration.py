"""End-to-end: the supported_transport convenience path and its contract."""
import numpy as np

from topotransport import simulate, supported_transport


def test_supported_transport_end_to_end_on_donut():
    """A target in the donut hole gets an estimate AND a GAP warning grade,
    while a supported target on the ring is graded GOLD/SILVER."""
    M, y, v, truth = simulate.donut(n=44, seed=1)
    tr_gap, cert_gap, fit = supported_transport(
        y, v, M, truth["hole_center"], scale="RD", n_boot=80, seed=1
    )
    tr_ok, cert_ok, _ = supported_transport(
        y, v, M, truth["supported_point"], scale="RD", n_boot=80, seed=1
    )
    assert cert_gap.grade == "GAP"
    assert cert_ok.grade in ("GOLD", "SILVER")
    # both still return a numeric transported estimate
    assert np.isfinite(tr_gap.estimate)
    assert np.isfinite(tr_ok.estimate)


def test_estimate_and_certificate_are_independent_layers():
    """The transport estimate exists regardless of support grade -- the topology
    annotates confidence, it does not gate the point estimate."""
    M, y, v, truth = simulate.donut(n=44, seed=3)
    tr, cert, fit = supported_transport(y, v, M, [99.0, 99.0], scale="RD", n_boot=40)
    assert np.isfinite(tr.estimate)
    assert cert.grade == "NONE"
