# E156-PROTOCOL — Topological Transportability for Meta-Analysis

- **Project:** topo-transport-ma
- **Built:** 2026-06-20
- **Primary estimand:** the Topologically-Supported Transported Effect — a
  transported meta-regression estimate paired with a persistent-homology support
  grade (GOLD / SILVER / BRONZE / GAP / NONE).
- **Dashboard:** `docs/index.html` (offline, self-auditing)

## Body (7 sentences, single paragraph)

When a meta-analysis is generalized to a target population, does that population
actually lie inside the evidence base, and can topology reveal coverage gaps that
a convex-hull overlap check waves through? We study the thirteen-trial BCG
tuberculosis-vaccine meta-analysis (Colditz 1994), in which absolute latitude is
the canonical effect modifier, alongside known-truth simulations. We estimate the
transported effect by random-effects meta-regression on study-level effect
modifiers and, independently, compute the Vietoris–Rips persistent homology of
the modifier point cloud, grading a target by whether it sits inside a confirmed
homological hole. On BCG our REML meta-regression reproduces metafor to four
decimals (latitude slope −0.0292, τ²=0.076), while in simulation a target inside
an annulus's hole is correctly graded unsupported even though it lies within the
convex hull. The homology engine matches GUDHI to machine precision, holes are
confirmed against sampling noise by a subsampling-bottleneck band, and odds-ratio
transport is refused as non-collapsible. Persistent homology thus turns
transportability positivity from an untested assumption into a computable,
geometry-aware certificate that separates interpolation, extrapolation, and
interpolation-into-a-gap. The H1 hole signal requires at least two effect
modifiers and an adequate study count; with a single modifier only H0 structure
and interval coverage apply.

## Validation summary

- Persistent homology vs GUDHI: bottleneck distance 0 (machine precision).
- Meta-regression vs metafor (BCG): intercept 0.2515, slope −0.0291, τ² 0.0763.
- Meta-regression vs statsmodels GLS: β and covariance match to 1e-6.
- 22/22 tests pass; capsule self-audit GOLD (8/8 in-browser checks).
