# FARE v3 architecture and experimental semantics

## Shapes

- Traffic input/history `[B,H,N]`, mask `[B,H,N]`, base prediction `[B,P,N]`, pooled STLA hidden `[B,N,d_base]`.
- Support `[B,N,P,64]`, directional coverage `[B,N,P]`, observed boundary and component extent `[B,N]`.
- Region candidates `[B,G,64]` and node shortlist `[B,N,16]`; episode IDs refer to common past/future/residual memory rows.
- Two historical proposal types `[B,N,16,2,P]` (Residual and Evolution).
- Set-condition benefit estimate `[B,N,16,2,P]`; final selected episodes `[B,N,8]`.
- Evidence weight `[B,N,8,2,P]` and Query/Node/Horizon Null `[B,N,P]`.
- Final control alpha `[B,N,P]` rescales `evidence prediction − null prediction`.

## Explicit training targets

- Train-only LOO keep utility `abs(y−pred_removed) − abs(y−pred_full)`.
- Train-only replacement gain `abs(y−pred_full) − abs(y−pred_replaced)`.
- Train-only risk gain `abs(y−pred_null) − abs(y−pred_full)`.
- Train-only action labels derived from minimum MAE among several alpha values **within four-step future horizon blocks**.

Utility Teacher outputs do not appear in normal inference inputs. The final prediction is always computed from current masked history, Backbone and eligible historical episodes only.

## Reporting scope

16 inference interventions all use exactly one Full FARE best checkpoint. LOO transfers dropped weight to Null; replacement preserves the original slot weight. A fixed-checkpoint positive delta is evidence that the trained model relies on the module; it **is not** an effect estimate for separately retraining without it. Uncertainty intervals are paired block-bootstrap summaries by origin-day group. No automatically generated p-value or improvement guarantee is claimed from a single seed.

## Physical scope

Graph-backed support encodes directional observability only when benchmark directional-neighbor metadata exists. The learned lag/kernel is a soft proxy for propagation compatibility, not a calibrated kinematic-wave solver. Speed and Flow share architecture but train independently; Flow-only traffic regimes should not be relabeled as fully observed physical congestion-wave classifications.
