# TCRR-STLA v2.0 — Failure-conditioned Counterfactual Residual Transfer

Complete standalone research project: data adapters, Train-only retrieval, STLAformer, counterfactual replay, train, validate, evaluate, ablate, diagnose, PBS. This is **fresh v2.0 code** and does not load TCRR v1 / Mask_RAFT Full checkpoints.

## Implemented mechanisms

1. **Train-only statistical neighbor graph** from differenced historical training inputs (`traffic_neighbors.py`). It identifies predictive support; it does not assert physical road topology. No sensor-index roll is treated as road adjacency.
2. **Dual-stage traffic retrieval**: multi-period observations and coarse mask-robust traffic signatures; forecast-conditioned candidate shortlist; matched-mask source comparisons before selecting candidates (`traffic_index.py`, `forecast_pool.py`). Train futures are permissible *historical evidence*; test targets are not used.
3. **Failure-conditioned candidate representation**: candidate histories are encoded under the query observation mask, augmented by Train-only graph support and missing-age (`traffic_v2.py`).
4. **Node-horizon residual transfer**: separate Normal and query-mask Failure residual scaling and learned candidate-level gain/harm estimates `[B,K,P,N]` (`model.py`).
5. **Selective mixture correction**: MAE-oriented mixture gain/harm, bounded corrections, candidate None route and risk control.
6. **Formal protocol**: same Backbone STLAformer Incomplete checkpoint family and Base_SupportBench evaluation; MAE primary, RMSE and MAPE secondary. Inference cannot access query future labels.

## Gadi deployment

Unzip from the `Speed_LLM_RAG` repository root; retain your existing `Base_SupportBench`, `Backbone` and `Backbone_exps` (no copies bundled). Use the connected **TFP** environment.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
unset V23_CHECKPOINT RESUME_CHECKPOINT
python -u TCRR-STLA/scripts/preflight.py --target speed --dataset TraffiDent_Net_06 --backbone-checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt
qsub -N TCRR2_speed -v TARGET=speed,SEED=10,BACKBONE_MODE=incomplete,ABLATION=full,BACKBONE_CHECKPOINT=Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt TCRR-STLA/pbs/formal.pbs
qsub -N TCRR2_flow -v TARGET=flow,SEED=10,BACKBONE_MODE=incomplete,ABLATION=full,BACKBONE_CHECKPOINT=Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt TCRR-STLA/pbs/formal.pbs
```

Each PBS job uses 1 V100, 12 CPU, 32 GB, 48 hours, timestamped live logs, best.pt/last.pt, automated evaluation and same-checkpoint retrieval audit. Outputs are isolated in `TCRR-STLA_exps`, `TCRR-STLA_cache`, `TCRR-STLA_logs`, not in Mask_RAFT/RAST project folders.

## Evaluate and diagnose without retraining

```bash
python -u TCRR-STLA/scripts/evaluate.py --checkpoint /path/to/new/v2/run/best.pt --batch-size 8
python -u TCRR-STLA/scripts/diagnose.py --checkpoint /path/to/new/v2/run/best.pt --batch-size 8
```

The diagnosis uses *the same trained checkpoint* to run `full`, `no_evidence`, `no_mask_replay`, `no_traffic_retrieval`, `no_mask_matched_retrieval`, `no_candidate_utility`, and `no_mixture_utility`. These are inference ablations, not substitutes for independently trained structural ablations.

## Tests

```bash
cd TCRR-STLA
PYTHONPATH=.:tests python -m pytest -q
```

All v2 files are **source code only**. This release was checked with synthetic CPU tests, source compilation and PBS Bash syntax; a real Gadi V100 end-to-end training run and final MAE improvement have **not yet been verified**. Do not report performance claims without formal experiments. MAPE uses the Base_SupportBench evaluator's existing validity convention.
