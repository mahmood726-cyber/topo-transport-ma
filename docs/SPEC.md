# SPEC — topo-transport-ma

## Portfolio recon (required by workflow rules)

`find-related-repos.py "topology persistent homology transportability
meta-analysis"` returned only **token-only** matches (score 3:
`fragility-atlas`, `ma-workbench`). A direct disk scan of `C:\Projects`, however,
revealed eight transportability/topology repos the manifest had not indexed
(registry drift). Each was inspected:

| Repo | Topology? | Transport? | Space it operates on |
|---|---|---|---|
| `evidence-topology` | ✅ persistent homology (ripser) | ❌ | evidence-*quality* scores (audit/consistency/robustness/stability/power) |
| `transport-ma` | ❌ | ✅ causal transportability | — |
| `transportabilitycalc` | ❌ | ✅ composite penalty index | demographic/temporal mismatch heuristic |
| `nmatransport`, `hta-transportability`, `global-transportability-{atlas,multiverse}` | ❌ | ✅ NMA / HTA / country-level | covariate scoring, not geometry |

**Reused (concepts, named):** the persistent-homology / Vietoris–Rips approach
demonstrated in `evidence-topology`; the transport estimand framing from
`transport-ma`; the generalizability-penalty framing contrasted against
`transportabilitycalc`.

**Net-new (this repo):**
1. Topology computed on the **effect-modifier space**, not on quality scores.
2. A **transportability positivity certificate** derived from persistent **H1
   holes** plus the **convex-hull-vs-Rips disagreement** (the GAP grade).
3. **Bootstrap-gated hole confirmation** (subsampling-bottleneck band, Fasy 2014).
4. **Collapsible-scale transport guard** (OR refused; ratio-scale sign-flip flag).
5. Integration into a single estimand, the **Topologically-Supported Transported
   Effect**, with a self-contained from-scratch homology engine (no compiled
   dependency) validated against GUDHI.

No existing repo computes topology of the effect-modifier space or uses topology
as the transport positivity certificate. This is the bridge between
`evidence-topology` (technique) and `transport-ma` (problem), on a new space and
for a new purpose.

## Method

### Transport estimand
`τ_target = E_target[CATE(x)] = β₀ + β'·E_target[x]`, with CATE approximated by a
random-effects meta-regression of study effects `y_i` (variance `v_i`) on
study-level modifiers `x_i`. τ² by REML via Fisher scoring (metafor default):
with `Σ = diag(v)+τ²I`, `W=Σ⁻¹`, `P = W − WX(X'WX)⁻¹X'W`, iterate
`τ² ← τ² + U/I`, `U = −½tr(P) + ½y'PPy`, `I = ½tr(P²)`, clamped at 0.
SE of the transported effect by delta method on `Cov(β̂)` (+ optional target-mean
covariance).

### Topological support certificate
- Vietoris–Rips filtration on the modifier cloud; persistence by GF(2)
  boundary-matrix reduction (H0, H1).
- **Subpopulations**: `1 + #{MST edges > 4×median edge}` (H0 gap statistic).
- **Holes**: finite H1 features, significant iff persistence > 2·c_α where c_α is
  the (1−α) quantile of bottleneck distances between the full diagram and
  bootstrap resamples (Fasy et al. 2014).
- **Grade** for target x*: with spacing `s` (median NN distance), connection
  scale `r_body` (max MST edge), `d = d_NN(x*)`, `inHull`:
  - `d ≤ s` → **GOLD**
  - `inHull ∧ d > r_body ∧ confirmed hole` → **GAP**
  - `inHull ∧ d ≤ r_body` → **SILVER**
  - `¬inHull ∧ d ≤ r_body` → **BRONZE**
  - else → **NONE**

## Validation contract (numerical baselines)
- GUDHI bottleneck = 0 across random clouds; unit-square H1 = `[1, √2]`.
- metafor BCG anchors: intercept 0.2515, ablat slope −0.0292, τ² 0.0764.
- statsmodels GLS: β and `(X'Σ⁻¹X)⁻¹` to 1e-6.
- Donut: centre `inHull=True`, grade `GAP`; ring point `GOLD/SILVER`.

## Boundaries / honest limits
- H1 holes need ≥2 effect modifiers. Two real-data demos: BCG **1D** (latitude →
  H0 + interval coverage) and BCG **2D** (latitude × log baseline risk →
  `bcg_modifier_cloud()`), which is a single connected blob with **no significant
  hole** — the method correctly raises no false alarm and grades targets by
  geometric support. The 2D **donut** (simulation) is the controlled GAP demo; a
  clean hole in a real evidence base is rare, which is itself a finding.
- Persistent H1 at small k is noisy — hence the bootstrap gate; with k<4 no holes
  are reported.
- Aggregate-data meta-regression carries ecological-bias risk; the estimator is
  no better than the modifiers available at study level.
