"""
datasets.py -- Real meta-analytic data with a known effect modifier.

dat.bcg: the BCG vaccine trials for tuberculosis (Colditz et al. 1994), the
canonical dataset distributed with the R package `metafor`. Absolute latitude
(`ablat`) is the textbook effect modifier: vaccine efficacy increases away from
the equator. We store the raw 2x2 counts and DERIVE the log risk ratio and its
variance, so nothing is hand-entered as an "effect size" -- the numbers are
reproducible from the counts and are cross-checked against documented metafor
output in tests/test_transport.py.

Columns: author, year, tpos, tneg, cpos, cneg, ablat, alloc
  tpos/tneg = TB+/TB- among vaccinated; cpos/cneg = among controls.
"""
from __future__ import annotations

import numpy as np

_BCG = [
    ("Aronson", 1948, 4, 119, 11, 128, 44, "random"),
    ("Ferguson & Simes", 1949, 6, 300, 29, 274, 55, "random"),
    ("Rosenthal et al", 1960, 3, 228, 11, 209, 42, "random"),
    ("Hart & Sutherland", 1977, 62, 13536, 248, 12619, 52, "random"),
    ("Frimodt-Moller et al", 1973, 33, 5036, 47, 5761, 13, "alternate"),
    ("Stein & Aronson", 1953, 180, 1361, 372, 1079, 44, "alternate"),
    ("Vandiviere et al", 1973, 8, 2537, 10, 619, 19, "random"),
    ("TPT Madras", 1980, 505, 87886, 499, 87892, 13, "random"),
    ("Coetzee & Berjak", 1968, 29, 7470, 45, 7232, 27, "random"),
    ("Rosenthal et al", 1961, 17, 1699, 65, 1600, 42, "systematic"),
    ("Comstock et al", 1974, 186, 50448, 141, 27197, 18, "systematic"),
    ("Comstock & Webster", 1969, 5, 2493, 3, 2338, 33, "systematic"),
    ("Comstock et al", 1976, 27, 16886, 29, 17825, 33, "systematic"),
]


def load_bcg():
    """Return BCG trials as a dict of numpy arrays with derived logRR and var.

    logRR = log( [tpos/(tpos+tneg)] / [cpos/(cpos+cneg)] )
    var(logRR) = 1/tpos - 1/(tpos+tneg) + 1/cpos - 1/(cpos+cneg)
    """
    author = np.array([r[0] for r in _BCG], dtype=object)
    year = np.array([r[1] for r in _BCG])
    tpos = np.array([r[2] for r in _BCG], float)
    tneg = np.array([r[3] for r in _BCG], float)
    cpos = np.array([r[4] for r in _BCG], float)
    cneg = np.array([r[5] for r in _BCG], float)
    ablat = np.array([r[6] for r in _BCG], float)
    alloc = np.array([r[7] for r in _BCG], dtype=object)

    n1 = tpos + tneg
    n0 = cpos + cneg
    logRR = np.log((tpos / n1) / (cpos / n0))
    var = 1.0 / tpos - 1.0 / n1 + 1.0 / cpos - 1.0 / n0
    # Baseline TB risk among controls -- a second, genuine effect modifier
    # (derived from the counts, not hand-entered). Together with latitude it
    # gives a *real* two-dimensional effect-modifier space.
    baseline_risk = cpos / n0
    return {
        "author": author, "year": year,
        "tpos": tpos, "tneg": tneg, "cpos": cpos, "cneg": cneg,
        "ablat": ablat, "alloc": alloc, "baseline_risk": baseline_risk,
        "logRR": logRR, "var": var,
    }


def bcg_modifier_cloud(standardize=True):
    """Real 2D effect-modifier cloud for BCG: (absolute latitude, log baseline
    risk). Standardized to unit SD per axis by default, since the two modifiers
    have incommensurate units (degrees vs log-probability) and any distance-based
    topology must not be dominated by the larger-scale axis.
    """
    d = load_bcg()
    ablat = d["ablat"]
    logbr = np.log(d["baseline_risk"])
    if standardize:
        a = (ablat - ablat.mean()) / ablat.std()
        b = (logbr - logbr.mean()) / logbr.std()
    else:
        a, b = ablat, logbr
    return np.c_[a, b]
