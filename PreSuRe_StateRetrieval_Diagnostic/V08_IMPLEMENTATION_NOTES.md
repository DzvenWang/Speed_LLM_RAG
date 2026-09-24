# v0.8 Implementation Notes — Intervention-Aware Predictive Utility

v0.8 keeps the validated PatchSTG-aligned Top-K retrieval from v0.6 and replaces the v0.7 metadata-only utility representation with features that describe the **actual proposed intervention**.

## New feature families

1. **Candidate state** — candidate patch mean/std, recent statistics and dynamics.
2. **Transport** — affine scale/shift, overlap support and clipping indicators.
3. **Proposal magnitude** — number of filled cells, input change magnitude, anchor jumps and patch-state shift.
4. **Local coherence** — compatibility with observed neighboring PatchSTG patches and candidate neighbor context.
5. **Frozen PatchSTG response** — prediction shift and prehead latent shift, including patch-local versus outside-patch spillover.

All v0.8 model features are inference-available. Hidden current history and future targets remain historical labels or evaluation outcomes only.

## Risk-aware selectors

- `logistic_intervention`
- `cost_sensitive_gate`
- `histgb_mean_intervention`
- `histgb_q20_intervention`
- `risk_expected_intervention`

Regression targets use utility normalized by the historical baseline MAE. Calibration still optimizes true historical additive forecast utility and patch-count abstention using only the immediately preceding time block.

## Default candidate budget

The intervention-aware forward pass defaults to Top-20 candidates per patch. This preserves a large v0.6 Forecast-Utility Oracle while substantially reducing the extra PatchSTG forward cost relative to Top-50. Use `--deep-top-k 50 --max-k 50` for a full candidate-pool run.
