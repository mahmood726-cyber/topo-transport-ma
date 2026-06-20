"""
topotransport -- Topological Transportability for Meta-Analysis.

A geometry-aware positivity certificate for transported meta-analytic effects.
The transport estimator (effect-modifier meta-regression) and the topological
support certificate (persistent homology of the effect-modifier cloud) are kept
separate so each is independently valid; together they yield a
*Topologically-Supported Transported Effect*: a transported estimate paired with
a Gold/Silver/Bronze/GAP/None support grade.
"""
from .homology import PersistenceDiagram, bottleneck_distance, compute_persistence
from .support import Hole, SupportCertificate, certify_support, significant_holes
from .transport import (
    MetaRegression,
    TransportResult,
    fit_meta_regression,
    transport,
)

__version__ = "0.1.0"

__all__ = [
    "compute_persistence", "bottleneck_distance", "PersistenceDiagram",
    "fit_meta_regression", "transport", "MetaRegression", "TransportResult",
    "certify_support", "significant_holes", "SupportCertificate", "Hole",
    "__version__",
]


def supported_transport(y, v, modifiers, target_x, scale="RD",
                        baseline_risk_idx=None, n_boot=100, seed=0):
    """Convenience: fit, transport, and certify support in one call.

    Returns (TransportResult, SupportCertificate, MetaRegression).
    """
    fit = fit_meta_regression(y, v, modifiers=modifiers, scale=scale)
    tr = transport(fit, target_x, baseline_risk_idx=baseline_risk_idx)
    cert = certify_support(fit.modifiers, target_x, n_boot=n_boot, seed=seed)
    return tr, cert, fit
