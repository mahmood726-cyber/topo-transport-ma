"""
transport.py -- Effect-modifier meta-regression and transport to a target.

The transported effect is the classical generalizability estimand
    tau_target = E_target[ CATE(x) ] = beta0 + beta' E_target[x]
where CATE(x) is approximated by a random-effects meta-regression of study
effects on study-level effect modifiers x (REML tau^2 via Fisher scoring, the
metafor default). The estimator is deliberately separate from the topology so it
stands on its own as ordinary causal-transport meta-regression; topology
(support.py) only certifies *whether the target is in the evidence support*.

Collapsibility guard (from the project's statistics rules)
----------------------------------------------------------
The odds ratio is non-collapsible, so a transported OR does not correspond to
any population causal contrast. We therefore refuse to transport on the OR/logOR
scale and direct the user to a risk difference or risk ratio. When the modifiers
include baseline risk and the across-study SD of baseline risk exceeds 0.1, a
ratio-scale transport can flip sign relative to the average causal effect, so we
raise an RD sensitivity flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COLLAPSIBLE = {"RD", "MD", "SMD"}          # safe to transport directly
RATIO_SCALES = {"RR", "logRR", "HR", "logHR"}
NON_COLLAPSIBLE = {"OR", "logOR"}


@dataclass
class MetaRegression:
    beta: np.ndarray          # (p,) coefficients, beta[0] = intercept
    cov_beta: np.ndarray      # (p, p)
    tau2: float
    X: np.ndarray             # (n, p) design (intercept + modifiers)
    modifiers: np.ndarray     # (n, p-1) raw modifier matrix (no intercept)
    y: np.ndarray
    v: np.ndarray
    scale: str
    converged: bool
    n_iter: int

    def cate(self, x: np.ndarray) -> float:
        """Predicted conditional effect at modifier vector x (length p-1)."""
        xf = np.concatenate([[1.0], np.atleast_1d(x)])
        return float(xf @ self.beta)


@dataclass
class TransportResult:
    estimate: float
    se: float
    ci: tuple[float, float]
    target_x: np.ndarray
    scale: str
    warnings: list[str]


def _reml_tau2(y, X, v, max_iter=100, tol=1e-7):
    """REML estimate of tau^2 by Fisher scoring (metafor's default method)."""
    n, p = X.shape
    tau2 = max(np.var(y) - np.mean(v), 0.0)  # crude start
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        W = np.diag(1.0 / (v + tau2))
        XtW = X.T @ W
        A = XtW @ X
        Ainv = np.linalg.pinv(A)
        P = W - W @ X @ Ainv @ XtW
        Py = P @ y
        # score U and Fisher information for tau^2 (dSigma/dtau2 = I)
        U = -0.5 * np.trace(P) + 0.5 * (Py @ Py)
        info = 0.5 * np.trace(P @ P)
        if info <= 0:
            break
        step = U / info
        new = tau2 + step
        if new < 0:
            new = 0.0
        if abs(new - tau2) < tol:
            tau2 = new
            converged = True
            break
        tau2 = new
    return tau2, converged, it


def fit_meta_regression(y, v, modifiers=None, scale="RD", method="REML"):
    """Random-effects meta-regression of study effects on effect modifiers.

    Parameters
    ----------
    y : (n,) effect estimates on `scale`.
    v : (n,) sampling variances of y.
    modifiers : (n, q) study-level effect modifiers, or None for intercept-only.
    scale : effect scale label (see COLLAPSIBLE / RATIO_SCALES / NON_COLLAPSIBLE).
    method : currently only "REML".
    """
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    n = len(y)
    if modifiers is None:
        M = np.zeros((n, 0))
    else:
        M = np.atleast_2d(np.asarray(modifiers, float))
        if M.shape[0] != n:
            M = M.T
    X = np.column_stack([np.ones(n), M]) if M.size else np.ones((n, 1))

    if method != "REML":
        raise ValueError("only REML is implemented")
    tau2, converged, it = _reml_tau2(y, X, v)

    W = np.diag(1.0 / (v + tau2))
    A = X.T @ W @ X
    Ainv = np.linalg.pinv(A)
    beta = Ainv @ X.T @ W @ y
    cov_beta = Ainv
    return MetaRegression(
        beta=beta, cov_beta=cov_beta, tau2=tau2, X=X, modifiers=M,
        y=y, v=v, scale=scale, converged=converged, n_iter=it,
    )


def _check_collapsibility(scale, modifiers, baseline_risk_idx, warnings):
    if scale in NON_COLLAPSIBLE:
        raise ValueError(
            f"Refusing to transport on non-collapsible scale '{scale}'. "
            "A transported odds ratio has no population causal interpretation; "
            "convert effects to a risk difference (RD) or risk ratio (RR) first."
        )
    if scale in RATIO_SCALES and baseline_risk_idx is not None and modifiers.size:
        br = modifiers[:, baseline_risk_idx]
        if np.std(br) > 0.1:
            warnings.append(
                f"Baseline-risk SD across studies = {np.std(br):.3f} > 0.1 on a "
                f"ratio scale ('{scale}'): the transported ratio can flip sign "
                "versus the average causal effect. Re-run on the RD scale as a "
                "sensitivity analysis."
            )


def transport(
    fit: MetaRegression,
    target_x,
    target_x_cov=None,
    level=0.95,
    baseline_risk_idx=None,
):
    """Transport a fitted meta-regression to a target effect-modifier profile.

    target_x : (q,) target mean effect-modifier vector.
    target_x_cov : optional (q, q) sampling covariance of the target means;
        propagated into the SE via the delta method when supplied.
    baseline_risk_idx : index into modifiers that is baseline risk (for the
        ratio-scale sign-flip guard).
    """
    from scipy.stats import norm

    warnings: list[str] = []
    target_x = np.atleast_1d(np.asarray(target_x, float))
    _check_collapsibility(fit.scale, fit.modifiers, baseline_risk_idx, warnings)

    xf = np.concatenate([[1.0], target_x])
    est = float(xf @ fit.beta)
    var = float(xf @ fit.cov_beta @ xf)
    if target_x_cov is not None:
        # delta method: Var += beta_mods' Cov(x*) beta_mods
        bm = fit.beta[1:]
        var += float(bm @ np.asarray(target_x_cov, float) @ bm)
    se = float(np.sqrt(max(var, 0.0)))
    z = norm.ppf(0.5 + level / 2.0)
    ci = (est - z * se, est + z * se)
    return TransportResult(est, se, ci, target_x, fit.scale, warnings)
