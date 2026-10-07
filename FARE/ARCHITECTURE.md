# FARE v0.2 — Full-Strength Restoration Experts

FARE is **Failure-Adaptive Restoration Experts for Traffic Forecasting under Incomplete Sensing**.

v0.2 deliberately contains **no MoE/router**. The purpose is to establish whether strong, information-distinct restoration experts actually provide complementary downstream forecasting utility before learning a routing policy.

## Input/output contract

Input per query:
- incomplete normalized history `x_obs`: `[12, N]`
- `history_observed`: cells visible at inference
- `artificial_missing`: benchmark-hidden cells that FARE is allowed to restore
- TOD/DOW history features
- graph structure from Base_SupportBench

Every expert returns a complete estimate for **every artificial-missing cell**. It is a hard invariant that:
- observed cells are never changed;
- originally invalid/non-artificial cells are never changed;
- evaluation truth `x_full` and future `y` never enter expert inference decisions.

The restored history is passed to the frozen STLAformer checkpoint. The primary criterion is downstream forecast MAE/RMSE, not restoration MAE alone.

## Experts

### E_C — Context Expert
A learned masked spatiotemporal restorer trained on formal-train Core-5 failures. It alternates temporal multi-head attention and graph message passing with node/TOD/DOW embeddings. It receives only the current incomplete query and endogenous graph/time context.

Training objective:
`L = L_restore + lambda_f * L_forecast`, where `L_forecast` backpropagates through the frozen STLAformer into the restorer.

### E_S — Structural Expert
A causal multi-day tensor-completion expert. It constructs `[node, support_day, query_slot]` tensors from the chronological formal-training archive only and performs TMCP-style tensor completion with observed-value projection after every iteration. Default: 28 same-slot historical days, official-style firm singular-value shrinkage and adaptive theta.

### E_R — Retrieval Expert
External historical support expert.
1. The trained Context encoder embeds chronological train windows and the current incomplete query.
2. A learned embedding index retrieves causal Top-K candidates.
3. Candidate/node reranking uses observed target mismatch, graph-neighbour mismatch, support coverage and learned similarity.
4. A learned per-cell gate fuses external support with the Context estimate.
5. Training includes downstream STLAformer forecast loss.

This avoids the v0.1 failure mode of copying the single most similar historical trace.

### E_F — Backbone-Aware Expert
A forecast-conditioned residual restorer. It receives the frozen STLAformer forecast produced from the incomplete query as an inference-visible feature, together with the strong Context restoration. A graph-temporal network learns a missing-cell residual optimized heavily for downstream forecast MAE.

This is the explicitly forecast-oriented expert: it asks how missing history should be reconstructed given what the final predictor currently believes about the future.

## Oracles

Two diagnostic upper bounds are reported:
- `Expert Oracle`: best of `{Context, Structural, Retrieval, Backbone-Aware}` per query.
- `Safe Oracle`: best of `{Backbone, Context, Structural, Retrieval, Backbone-Aware}` per query.

The Safe Oracle answers whether adaptive intervention has real headroom when abstention is allowed. Backbone is a safety reference, not a restoration expert.

## Dependency policy

Runtime dependencies are only:
- `Base_SupportBench`
- `Backbone`

`Baseline` is comparison-only. FARE v0.2 does not import or call Baseline and does not depend on any prior FARE, STLA_DASR, or Support_Oracle artifact.
