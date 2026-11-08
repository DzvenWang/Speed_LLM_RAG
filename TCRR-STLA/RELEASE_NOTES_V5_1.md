# TCRR-STLA v5.2.0: CFRM → SCRT → CRSI

C1 / CFRM: Counterfactual Failure Response Modeling.

C2 / SCRT: Support-conditioned Residual Transfer. The final candidate weights
are shared by both Normal and CF corrections. Their bounded contributions are
allocated symmetrically, preserving the full bounded proposal exactly.

C3 / CRSI: Counterfactual Response-guided Selective Intervention. Two separate
correction gates, directional response consistency, Normal/CF sign conflict,
functional-neighborhood + Horizon-block adjustment, and utility/harm supervision
for each correction source and its interaction.

All three C1/C2/C3 switches remain orthogonal. Mxyz with z=0 uses the full
bounded C2 proposal with gate=1; z=1 learns two gates and an optional validation-
calibrated rejection threshold. The original five-scale controller is retained
as a capacity comparison through c3_variant=scale. Variants require comparable
training budgets for publication-grade attribution.

New run output uses v5 paths and distinct checkpoint version 5.2.0. No former
TCRR model outputs, adapters or replay banks are loaded. Automatically generated
validation analyses and four-severity tests remain in the formal PBS.

Tests: run `python -m pytest -q TCRR-STLA/tests`.
Run true-data GPU smoke before Gadi formal training. This version has not been
shown to improve forecasting accuracy until real Speed/Flow trials complete.
