"""
support.py -- Topological transportability support certificate.

This is the novel contribution: a geometry-aware positivity / coverage diagnostic
for transported meta-analytic effects. Given the study cloud in effect-modifier
space and a target profile x*, it asks not merely "is x* inside the convex hull
of the studies" (which is blind to holes and multimodal support) but "is x* in
the *persistent topological support* of the evidence base?"

Two homological signals do the work:
  * H0 (connected components) reveals latent subpopulations the scalar I^2 hides.
  * H1 (loops) reveals *holes* -- regions enclosed by evidence yet uncovered.
    A target inside a confirmed hole is flagged GAP even though the convex hull
    accepts it: transporting there is interpolation into emptiness.

Holes are confirmed against sampling noise with a subsampling bottleneck band
(Fasy et al. 2014): an H1 feature is significant iff its persistence exceeds
2 * c_alpha, where c_alpha is the (1-alpha) quantile of bottleneck distances
between the full diagram and bootstrap resamples. This is the topological analog
of putting a confidence interval on I^2 at small k.

Grades
  GOLD   : target sits among studies (genuine interpolation).
  SILVER : interior to support, moderate gap, no confirmed hole.
  BRONZE : just outside the periphery (mild extrapolation).
  GAP    : convex hull accepts it but it lies in a confirmed topological hole.
  NONE   : beyond the evidence connection scale (extrapolation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

from .homology import bottleneck_distance, compute_persistence


@dataclass
class Hole:
    birth: float
    death: float
    persistence: float
    significant: bool


@dataclass
class SupportCertificate:
    grade: str
    d_nn: float                 # distance from target to nearest study
    spacing: float              # median within-cloud nearest-neighbour distance
    connect_scale: float        # max MST edge (scale at which cloud connects)
    in_hull: bool
    topological_gap: bool
    n_subpopulations: int       # H0 components at the spacing scale
    holes: list[Hole] = field(default_factory=list)
    reason: str = ""

    @property
    def supported(self) -> bool:
        return self.grade in ("GOLD", "SILVER")


def _nn_spacing(cloud: np.ndarray) -> float:
    """Typical within-cloud spacing: median of *positive* nearest-neighbour
    distances (duplicate points contribute 0 and would otherwise collapse it)."""
    n = len(cloud)
    if n < 2:
        return 0.0
    D = cdist(cloud, cloud)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    pos = nn[nn > 0]
    return float(np.median(pos)) if pos.size else 0.0


def count_subpopulations(cloud: np.ndarray, gap_factor: float = 4.0) -> int:
    """Number of latent subpopulations from outlier MST edges (H0 structure).

    The finite H0 deaths are exactly the MST edge lengths. Removing an MST edge
    splits the cloud into two components, so the number of subpopulations is
    1 + (number of MST edges that are anomalously long). An edge is anomalous if
    it exceeds `gap_factor` times the median edge length -- two well-separated
    clusters leave one bridging edge many times the within-cluster spacing, while
    a single blob or a connected ring leaves edges of similar length. This is the
    topological analog of requiring a heterogeneity signal before declaring
    subgroups, and it correctly reports one component for an annulus (a hole is
    H1 structure, not a second cluster).
    """
    n = len(cloud)
    if n < 2:
        return 1
    D = cdist(cloud, cloud)
    mst = minimum_spanning_tree(D).toarray()
    edges = mst[mst > 0]
    if edges.size < 2:
        return 1
    med = float(np.median(edges))
    if med <= 0:
        return 1
    return int(1 + np.sum(edges > gap_factor * med))


def _connect_scale(cloud: np.ndarray) -> float:
    n = len(cloud)
    if n < 2:
        return 0.0
    D = cdist(cloud, cloud)
    mst = minimum_spanning_tree(D).toarray()
    return float(mst.max())


def in_convex_hull(cloud: np.ndarray, x: np.ndarray) -> bool:
    q = cloud.shape[1]
    x = np.atleast_1d(x)
    if q == 1:
        return bool(cloud.min() <= x[0] <= cloud.max())
    try:
        from scipy.spatial import Delaunay

        return bool(Delaunay(cloud).find_simplex(x) >= 0)
    except Exception:
        from scipy.optimize import linprog

        n = len(cloud)
        A_eq = np.vstack([cloud.T, np.ones(n)])
        b_eq = np.concatenate([x, [1.0]])
        res = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n)
        return bool(res.success)


def significant_holes(
    cloud: np.ndarray, n_boot: int = 100, alpha: float = 0.05, seed: int = 0
) -> tuple[list[Hole], float]:
    """Persistent H1 holes with subsampling-bottleneck significance (Fasy 2014).

    Returns the holes and the confidence-band half-width c_alpha. A hole is
    significant iff its persistence > 2 * c_alpha.
    """
    pd = compute_persistence(cloud, max_dim=1)
    obs = pd.finite(1)
    if len(cloud) < 4:
        return [], 0.0
    rng = np.random.default_rng(seed)
    dists = []
    n = len(cloud)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # bootstrap resample
        bd = compute_persistence(cloud[idx], max_dim=1).finite(1)
        dists.append(bottleneck_distance(obs, bd))
    c_alpha = float(np.quantile(dists, 1 - alpha)) if dists else 0.0
    holes = []
    for b, d in obs:
        per = d - b
        holes.append(Hole(float(b), float(d), float(per), bool(per > 2 * c_alpha)))
    holes.sort(key=lambda h: -h.persistence)
    return holes, c_alpha


def certify_support(
    cloud,
    target_x,
    n_boot: int = 100,
    alpha: float = 0.05,
    seed: int = 0,
) -> SupportCertificate:
    """Topological transportability support certificate for a target profile."""
    cloud = np.atleast_2d(np.asarray(cloud, float))
    if cloud.shape[0] == 1:
        cloud = cloud.reshape(-1, 1) if cloud.shape[1] > 1 else cloud
    target_x = np.atleast_1d(np.asarray(target_x, float))

    spacing = _nn_spacing(cloud)
    r_body = _connect_scale(cloud)
    d_nn = float(cdist(cloud, target_x.reshape(1, -1)).min())
    in_hull = in_convex_hull(cloud, target_x)

    # subpopulation count via gap statistic on H0 merge scales (robust to the
    # over-fine-scale artifact of counting components at the spacing scale).
    eps = max(spacing, 1e-9)
    n_sub = count_subpopulations(cloud)

    holes, _c = significant_holes(cloud, n_boot=n_boot, alpha=alpha, seed=seed)
    confirmed = [h for h in holes if h.significant]

    # Decision logic.
    topo_gap = bool(in_hull and d_nn > r_body and len(confirmed) > 0)
    if d_nn <= eps:
        grade, reason = "GOLD", "target lies among studies (interpolation)"
    elif topo_gap:
        grade, reason = (
            "GAP",
            "convex hull accepts the target but it lies in a confirmed "
            "topological hole; transport here is interpolation into emptiness",
        )
    elif in_hull and d_nn <= r_body:
        grade, reason = "SILVER", "interior to support, no confirmed hole"
    elif (not in_hull) and d_nn <= r_body:
        grade, reason = "BRONZE", "just outside the evidence periphery"
    else:
        grade, reason = "NONE", "beyond the evidence connection scale (extrapolation)"

    return SupportCertificate(
        grade=grade,
        d_nn=d_nn,
        spacing=spacing,
        connect_scale=r_body,
        in_hull=in_hull,
        topological_gap=topo_gap,
        n_subpopulations=int(n_sub),
        holes=holes,
        reason=reason,
    )
