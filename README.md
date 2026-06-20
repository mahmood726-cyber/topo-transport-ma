# topo-transport-ma

**Topological Transportability for Meta-Analysis** — a geometry-aware positivity
certificate for transported meta-analytic effects.

When a pooled effect is generalized to a new target population, *positivity*
(overlap of the studies' effect-modifier distribution with the target) is almost
always assumed and rarely tested. This package computes the **persistent
homology of the effect-modifier space** and uses it as the positivity diagnostic,
catching targets that fall in a **hole** of the evidence base — a case a convex-hull
overlap check waves through as "supported."

## What's novel

Two well-developed ideas are bridged here for the first time:

- **Topological data analysis** (persistent homology) — used elsewhere for the
  *shape* of data, never as a transportability positivity certificate.
- **Causal transportability** (Pearl/Bareinboim; Dahabreh; Westreich) — uses
  selection diagrams and weighting, but checks positivity only informally.

The bridge: represent studies as a point cloud in effect-modifier space and read
two homological signals.

| Signal | Meaning for transport |
|---|---|
| **H0** (connected components) | latent subpopulations the scalar I² cannot see |
| **H1** (loops / holes) | regions *enclosed* by evidence yet uncovered — where transport is interpolation into emptiness |

A target is graded **GOLD / SILVER / BRONZE / GAP / NONE**. The decisive case is
**GAP**: the convex hull accepts the target, but it lies in a bootstrap-confirmed
hole, so transport there is unjustified.

## Two independent layers

1. **Transport estimator** — `τ_target = β₀ + β'·E_target[x]` from a
   random-effects meta-regression (REML τ² via Fisher scoring, metafor's method).
   Stands alone as ordinary causal-transport meta-regression.
2. **Topological support certificate** — persistent homology of the modifier
   cloud, holes confirmed against sampling noise by a subsampling-bottleneck band
   (Fasy et al. 2014). Annotates the estimate; never gates the point estimate.

## Built-in statistical guards

- **Odds ratios are refused** for transport: a transported OR has no population
  causal interpretation (non-collapsibility). Use RD or RR.
- **Ratio-scale sign-flip flag**: if baseline-risk SD across studies > 0.1, a
  ratio-scale transport can flip sign vs. the average causal effect → RD
  sensitivity flag.

## Validation

Everything is checked against independent oracles and known truth:

- Persistent homology matches **GUDHI** to machine precision (bottleneck = 0)
  across random clouds; the unit square gives the hand-computable H1 `[1, √2]`.
- Meta-regression reproduces **metafor** on the real BCG dataset to 4 decimals
  (latitude slope −0.0292, intercept 0.2515, τ² 0.0764) and **statsmodels** GLS
  on β and its covariance.
- The donut simulation demonstrates the GAP case the convex hull misses.
- Real 2D space (BCG latitude × log baseline risk, `bcg_modifier_cloud()`) runs
  end-to-end and correctly reports **no false hole** (connected evidence).

```bash
pip install numpy scipy            # runtime
pip install gudhi statsmodels pytest   # validation oracles
python -m pytest -q                # 22 tests
```

## Quick start

```python
import numpy as np
from topotransport import supported_transport, fit_meta_regression, transport
from topotransport.datasets import load_bcg

d = load_bcg()
fit = fit_meta_regression(d["logRR"], d["var"], d["ablat"], scale="logRR")
tr  = transport(fit, target_x=[35.0])          # transport to latitude 35°
print(np.exp(tr.estimate), tr.ci)              # transported risk ratio + CI

# one-shot: estimate + topological support grade
tr, cert, fit = supported_transport(
    d["logRR"], d["var"], d["ablat"].reshape(-1, 1), target_x=[35.0], scale="logRR")
print(cert.grade, cert.reason)
```

## Interactive capsule

`docs/index.html` is a self-contained, offline, self-auditing dashboard
(click to place a target on the donut; slide latitude on BCG). Its self-audit
panel recomputes every claim in-browser and shows a Bronze/Silver/Gold badge.

## Layout

```
topotransport/
  homology.py    persistent homology (VR + GF(2) reduction) + bottleneck distance
  transport.py   REML meta-regression, transport estimand, collapsibility guards
  support.py     topological support certificate (H0 subpops, H1 holes, grades)
  simulate.py    known-truth generators (two clusters, donut, blob, linear CATE)
  datasets.py    BCG vaccine trials (Colditz 1994), logRR derived from counts
tests/           22 tests vs GUDHI, metafor, statsmodels, and known truth
docs/index.html  self-auditing offline capsule
```

## License

MIT.
