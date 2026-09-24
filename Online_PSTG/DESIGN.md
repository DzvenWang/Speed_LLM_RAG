# Online_PSTG v0.1.0 — Strictly Causal Online PatchSTG

## Goal
Provide a strong and fair online-learning baseline for later Online-RAG experiments. The PatchSTG architecture and offline checkpoint are unchanged; only model parameters are incrementally updated from labels that have already matured in the stream.

## Causal protocol
For query i with raw `window_start=s_i`, history H, and horizon P:

- forecast origin = `s_i + H`
- full future becomes available at `s_i + H + P`
- case i may enter the online replay buffer only when `future_end_i <= current_origin`

The online update is executed **before** the current query prediction, but only from previously matured cases. The current query is enqueued only after its prediction and cannot train the model until its complete future has matured.

## Variants
- `frozen`: no adaptation; sanity reference.
- `head`: update only `regression_conv` (recommended first baseline).
- `last_block_head`: update last spatial encoder block and regression head.
- `full`: update the complete PatchSTG; included as an upper-capacity but higher-forgetting baseline.

## Replay
A rolling buffer contains only matured stream cases. Default capacity 256 queries. Each causal update takes 2 optimizer steps from uniformly sampled recent cases.

## Evaluation
Every stream query is predicted twice:
1. frozen offline PatchSTG;
2. causally adapted online PatchSTG.

The same query order and masks are used. Outputs include overall/masked/support MAE/RMSE, per-horizon masked MAE, per-query metrics, update traces, and four chronological stream segments.

## What this baseline is for
Later Online-RAG should be compared against this baseline, not only against frozen PatchSTG. If Online-RAG improves while avoiding repeated backbone retraining, its contribution is much easier to isolate.
