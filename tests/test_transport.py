"""Validate meta-regression + transport against metafor anchors and known truth."""
import numpy as np
import pytest

from topotransport import simulate
from topotransport.datasets import load_bcg
from topotransport.transport import fit_meta_regression, transport


def test_bcg_intercept_only_matches_metafor():
    """RE (REML) overall logRR on dat.bcg -- documented metafor output."""
    d = load_bcg()
    f = fit_meta_regression(d["logRR"], d["var"], None, scale="logRR")
    assert f.beta[0] == pytest.approx(-0.7145, abs=5e-3)
    assert f.tau2 == pytest.approx(0.3132, abs=1e-2)


def test_bcg_metaregression_on_latitude_matches_metafor():
    """Mixed-effects meta-regression on absolute latitude -- metafor anchors:
    intercept 0.2515, ablat slope -0.0292, tau^2 0.0764 (Viechtbauer 2010, JSS)."""
    d = load_bcg()
    f = fit_meta_regression(d["logRR"], d["var"], d["ablat"], scale="logRR")
    assert f.converged
    assert f.beta[0] == pytest.approx(0.2515, abs=5e-3)
    assert f.beta[1] == pytest.approx(-0.0292, abs=2e-3)
    assert f.tau2 == pytest.approx(0.0764, abs=1e-2)


def test_bcg_transport_monotone_in_latitude():
    """Vaccine efficacy increases with latitude: transported logRR decreases."""
    d = load_bcg()
    f = fit_meta_regression(d["logRR"], d["var"], d["ablat"], scale="logRR")
    e13 = transport(f, [13]).estimate
    e44 = transport(f, [44]).estimate
    e55 = transport(f, [55]).estimate
    assert e13 > e44 > e55  # higher latitude -> stronger protection (more negative)


def test_recovers_known_linear_cate_with_coverage():
    """Recovered coefficients should cover the planted truth within ~2.5 SE."""
    M, y, v, truth = simulate.linear_cate(n=80, beta=(0.1, 0.05, -0.03), seed=7)
    f = fit_meta_regression(y, v, M, scale="RD")
    se = np.sqrt(np.diag(f.cov_beta))
    z = np.abs(f.beta - truth["beta"]) / se
    assert np.all(z < 2.5), f"z-scores {z}"


def test_reml_small_when_no_heterogeneity():
    """With effects generated from a single line + only sampling error, tau^2~0."""
    rng = np.random.default_rng(1)
    n = 60
    M = rng.normal(0, 2, size=(n, 1))
    v = np.full(n, 0.02)
    y = 0.1 + 0.05 * M[:, 0] + rng.normal(0, np.sqrt(v))
    f = fit_meta_regression(y, v, M, scale="RD")
    assert f.tau2 < 0.02


def test_gls_beta_matches_statsmodels():
    """Independent GLS implementation must agree on beta and its covariance."""
    sm = pytest.importorskip("statsmodels.api")
    d = load_bcg()
    f = fit_meta_regression(d["logRR"], d["var"], d["ablat"], scale="logRR")
    sigma = np.diag(d["var"] + f.tau2)
    X = f.X
    gls = sm.GLS(d["logRR"], X, sigma=sigma).fit()
    assert np.allclose(gls.params, f.beta, atol=1e-6)
    # GLS cov uses a scale factor; compare the unscaled (X' Sigma^-1 X)^-1
    assert np.allclose(gls.normalized_cov_params, f.cov_beta, atol=1e-6)


def test_real_2d_bcg_transport_and_support():
    """Real 2D effect-modifier space (latitude x log baseline risk): the method
    runs end to end, raises no false hole, and grades targets sensibly."""
    from topotransport.datasets import bcg_modifier_cloud
    from topotransport.support import certify_support, significant_holes

    d = load_bcg()
    M = bcg_modifier_cloud(standardize=True)
    fit = fit_meta_regression(
        d["logRR"], d["var"], np.c_[d["ablat"], np.log(d["baseline_risk"])],
        scale="logRR",
    )
    assert fit.converged
    # both modifiers contribute; transport returns a finite estimate
    tr = transport(fit, [40.0, np.log(0.02)])
    assert np.isfinite(tr.estimate)
    # the real evidence base is one connected blob with no significant hole
    holes, _ = significant_holes(M, n_boot=80, seed=0)
    assert not any(h.significant for h in holes)
    # interior target supported; far target is extrapolation
    assert certify_support(M, [0.0, 0.0], n_boot=60, seed=0).grade in ("GOLD", "SILVER")
    assert certify_support(M, [6.0, 6.0], n_boot=60, seed=0).grade == "NONE"


def test_collapsibility_guard_refuses_odds_ratio():
    M, y, v, _ = simulate.linear_cate(n=20, seed=2)
    f = fit_meta_regression(y, v, M, scale="logOR")
    with pytest.raises(ValueError, match="non-collapsible"):
        transport(f, [0.0, 0.0])


def test_baseline_risk_sign_flip_warning_on_ratio_scale():
    rng = np.random.default_rng(5)
    n = 20
    baseline = rng.uniform(0.05, 0.6, n)  # SD well above 0.1
    other = rng.normal(0, 1, n)
    M = np.c_[baseline, other]
    v = np.full(n, 0.02)
    y = rng.normal(0, 0.1, n)
    f = fit_meta_regression(y, v, M, scale="RR")
    res = transport(f, [0.3, 0.0], baseline_risk_idx=0)
    assert any("flip sign" in w for w in res.warnings)
