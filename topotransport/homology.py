"""
homology.py -- Self-contained persistent homology for small point clouds.

Vietoris-Rips filtration + standard boundary-matrix reduction over GF(2),
computing H0 (connected components) and H1 (loops/holes). Meta-analytic point
clouds are tiny (k = tens of studies), so building the complex up to dimension 2
is cheap and an exact, dependency-light implementation is preferable to a
compiled library. The test-suite validates this engine against GUDHI as an
independent oracle (see tests/test_homology.py) -- we trust the math because we
checked it, not because a library said so.

Filtration convention (Vietoris-Rips):
  filt(vertex)        = 0
  filt(edge {i,j})    = d(i, j)
  filt(triangle ...)  = max pairwise distance among the three vertices

Persistence pairs (over GF(2)):
  H0 pair = (vertex born at 0) killed by (edge)      -> merge of components
  H1 pair = (edge that creates a cycle) killed by (triangle) -> a loop filled in
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PersistenceDiagram:
    """Birth/death pairs by homological dimension.

    h0, h1: arrays of shape (m, 2) with columns [birth, death]; death may be
    np.inf for an essential (never-dying) class.
    """

    h0: np.ndarray
    h1: np.ndarray
    meta: dict = field(default_factory=dict)

    def finite(self, dim: int) -> np.ndarray:
        d = self.h0 if dim == 0 else self.h1
        if d.size == 0:
            return d.reshape(0, 2)
        return d[np.isfinite(d[:, 1])]

    def persistences(self, dim: int) -> np.ndarray:
        f = self.finite(dim)
        return (f[:, 1] - f[:, 0]) if f.size else np.array([])

    def max_persistence(self, dim: int) -> float:
        p = self.persistences(dim)
        return float(p.max()) if p.size else 0.0

    def n_components_at(self, scale: float) -> int:
        """Number of H0 classes alive at a given filtration scale."""
        born = self.h0[:, 0] <= scale
        alive = ~(self.h0[:, 1] <= scale)
        return int(np.sum(born & alive))


def _pairwise(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.maximum((diff * diff).sum(-1), 0.0))


def compute_persistence(
    points: np.ndarray,
    max_dim: int = 1,
    max_edge: float | None = None,
) -> PersistenceDiagram:
    """Persistent homology of a point cloud via Vietoris-Rips.

    Parameters
    ----------
    points : (n, d) array.
    max_dim : highest homology dimension to compute (1 -> H0 and H1).
        Triangles (2-simplices) are always built when max_dim >= 1 because they
        are what *kill* H1 cycles.
    max_edge : optional cap on edge length; edges longer than this (and any
        simplex containing them) are excluded. Caps complex size for large n.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return PersistenceDiagram(np.zeros((0, 2)), np.zeros((0, 2)))
    D = _pairwise(points)
    if max_edge is None:
        max_edge = float(D.max()) if n > 1 else 0.0

    # Build simplices as (filtration, dim, vertex-tuple).
    simplices: list[tuple[float, int, tuple[int, ...]]] = []
    for i in range(n):
        simplices.append((0.0, 0, (i,)))
    edge_filt: dict[tuple[int, int], float] = {}
    for i, j in itertools.combinations(range(n), 2):
        w = D[i, j]
        if w <= max_edge:
            edge_filt[(i, j)] = w
            simplices.append((w, 1, (i, j)))
    if max_dim >= 1:
        for i, j, k in itertools.combinations(range(n), 3):
            e = (D[i, j], D[i, k], D[j, k])
            w = max(e)
            if w <= max_edge:
                simplices.append((w, 2, (i, j, k)))

    # Order: by filtration, then dimension (faces before cofaces on ties),
    # then vertex tuple for determinism.
    simplices.sort(key=lambda s: (s[0], s[1], s[2]))
    index_of: dict[tuple[int, int, tuple[int, ...]], int] = {}
    for idx, s in enumerate(simplices):
        index_of[(s[1], s[2])] = idx
    filt = np.array([s[0] for s in simplices])
    dims = np.array([s[1] for s in simplices])

    # Boundary columns as sorted lists of row indices (GF(2)).
    columns: list[list[int]] = []
    for _, dim, verts in simplices:
        if dim == 0:
            columns.append([])
        elif dim == 1:
            a, b = verts
            columns.append(sorted((index_of[(0, (a,))], index_of[(0, (b,))])))
        else:  # triangle -> its three edges
            a, b, c = verts
            faces = [
                index_of[(1, (a, b))],
                index_of[(1, (a, c))],
                index_of[(1, (b, c))],
            ]
            columns.append(sorted(faces))

    # Standard reduction. pivot[row] = column index that currently has that low.
    pivot: dict[int, int] = {}
    low_of: list[int | None] = [None] * len(simplices)

    def low(col: list[int]) -> int | None:
        return col[-1] if col else None

    def symdiff(a: list[int], b: list[int]) -> list[int]:
        # symmetric difference of two sorted lists -> sorted list (GF(2) add).
        out: list[int] = []
        ia = ib = 0
        while ia < len(a) and ib < len(b):
            if a[ia] == b[ib]:
                ia += 1
                ib += 1
            elif a[ia] < b[ib]:
                out.append(a[ia])
                ia += 1
            else:
                out.append(b[ib])
                ib += 1
        out.extend(a[ia:])
        out.extend(b[ib:])
        return out

    for j in range(len(simplices)):
        col = columns[j]
        l = low(col)
        while l is not None and l in pivot:
            col = symdiff(col, columns[pivot[l]])
            l = low(col)
        columns[j] = col
        if l is not None:
            pivot[l] = j
            low_of[j] = l

    h0: list[tuple[float, float]] = []
    h1: list[tuple[float, float]] = []
    paired_birth = set(pivot.keys())  # simplices that are a 'low' => births that die

    for j in range(len(simplices)):
        l = low_of[j]
        if l is not None:
            b, d = filt[l], filt[j]
            if d > b:  # drop zero-persistence pairs
                if dims[l] == 0:
                    h0.append((b, d))
                elif dims[l] == 1:
                    h1.append((b, d))
    # Essential classes: zero columns that are never used as a pivot 'low'.
    for j in range(len(simplices)):
        if not columns[j] and j not in paired_birth:
            if dims[j] == 0:
                h0.append((filt[j], np.inf))
            elif dims[j] == 1:
                h1.append((filt[j], np.inf))

    h0a = np.array(sorted(h0), dtype=float).reshape(-1, 2) if h0 else np.zeros((0, 2))
    h1a = np.array(sorted(h1), dtype=float).reshape(-1, 2) if h1 else np.zeros((0, 2))
    return PersistenceDiagram(h0a, h1a, meta={"n": n, "max_edge": max_edge})


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Bottleneck distance between two finite persistence diagrams.

    Each off-diagonal point may be matched to a point in the other diagram or to
    its own projection on the diagonal. Implemented as binary search over
    candidate thresholds with a bipartite-feasibility check. Sizes here are tiny
    (a handful of H1 features), so this is inexpensive.
    """
    A = np.asarray(dgm1, dtype=float).reshape(-1, 2)
    B = np.asarray(dgm2, dtype=float).reshape(-1, 2)
    A = A[np.isfinite(A).all(1)]
    B = B[np.isfinite(B).all(1)]
    if len(A) == 0 and len(B) == 0:
        return 0.0

    def linf(p, q):
        return max(abs(p[0] - q[0]), abs(p[1] - q[1]))

    def diag(p):  # L-inf distance to diagonal
        return abs(p[1] - p[0]) / 2.0

    na, nb = len(A), len(B)
    # Cost matrix over (A points + diagonal slots for B) x (B points + diagonal for A).
    # Build candidate thresholds.
    cands = {0.0}
    for p in A:
        cands.add(diag(p))
    for q in B:
        cands.add(diag(q))
    for p in A:
        for q in B:
            cands.add(linf(p, q))
    cand_list = sorted(cands)

    from scipy.sparse.csgraph import maximum_bipartite_matching
    from scipy.sparse import csr_matrix

    # Left nodes: A points (0..na-1) and "diagonal-for-B" (na..na+nb-1).
    # Right nodes: B points (0..nb-1) and "diagonal-for-A" (nb..nb+na-1).
    size = na + nb

    def feasible(t: float) -> bool:
        if size == 0:
            return True
        # Left nodes:  A points 0..na-1, then diagonal-of-B nodes na..na+nb-1.
        # Right nodes: B points 0..nb-1, then diagonal-of-A nodes nb..nb+na-1.
        rows, cols = [], []
        for i in range(na):
            for j in range(nb):
                if linf(A[i], B[j]) <= t:
                    rows.append(i)
                    cols.append(j)
            if diag(A[i]) <= t:  # A[i] matched to the diagonal
                rows.append(i)
                cols.append(nb + i)
        for j in range(nb):
            if diag(B[j]) <= t:  # B[j] matched to the diagonal
                rows.append(na + j)
                cols.append(j)
        # diagonal-left to diagonal-right always allowed (cost 0)
        for j in range(nb):
            for i in range(na):
                rows.append(na + j)
                cols.append(nb + i)
        data = np.ones(len(rows))
        g = csr_matrix((data, (rows, cols)), shape=(size, size))
        match = maximum_bipartite_matching(g, perm_type="column")
        return bool(np.all(match != -1))

    lo = 0
    hi = len(cand_list) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(cand_list[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(cand_list[lo])
