# Mask_RAST-STLA v2.3.0

Independent, **breaking** full project for incomplete-sensing retrieval of Train-only spatial-regional future evolution. Old checkpoints, indexes and caches are not supported. Depends on shared `Backbone` / `Base_SupportBench`, **not** Baseline, FARE or Support_Oracle.

## New method

1. `graph.py` builds a dynamic Train-only positive lagged statistical support graph and regions (not falsely claimed to be an observed road propagation graph). `encoders.py` derives global temporal/spatial keys and regional keys with separately measured regional internal-coverage and graph-neighbour support. AMP-sensitive aggregation is performed in FP32.
2. `memory.py` builds first-stage **Global Temporal / Global Spatial / per-region** Train-only indexes and legal temporal-overlap filtering. `model.py` applies support-weighted regional candidate scores and learns Temporal / Spatial / Regional / None routing.
3. For a target with observed historical data, retrieved future evolution is anchored to the same last-valid timestep. For a **fully unobserved** target, `seasonal.py` computes a **Train-only hourly-of-week future reference**, transfers the candidate's future innovation to the STLA forecast, and permits a nonzero H1 correction. No historical imputation head is used.
4. `trainer.py` trains forecast and Base-preservation losses, signed candidate utility supervision, composite-correction risk loss, **Train-matched incomplete Query ↔ complete Memory contrastive alignment** and an EMA memory encoder. EMA updates the *whole* memory encoder, not just a projector.
5. Full mode correction is always evidence-anchored. A separately trained no-retrieval control has a distinct forecast adapter and is not a gate-toggle proxy.

The research objective is **≥6% Avg(s20–s90) MAE reduction compared with RAST-STLA**, for both speed and flow; this release has **no new real Formal performance claim**.

## Protocol and limits

- `TraffiDent_Net_01/02/03/04/06`, dynamic node count and region count, `speed/flow`, Train s40, Test `s20/s40/s60/s90`, history/horizon 12/12, Seed 10 default.
- Memory uses **only complete historical Train windows and their true Train futures**. Query inference uses incomplete historical readings and masks, STLA forecast, and training memory. Test future never enters retrieval or prediction.
- Train-only hourly-week statistics are derived from future records that exist in Train Memory; the temporal reference uses candidate's **historical** time context. This is a research variant that needs an H1/Regional-Outage ablation.
- Graph is statistical unless explicit road topology is supplied via a separate implementation; no directed physical propagation is claimed.
- Checkpoint selector is single s40 Validation MAE, not a Test-informed or fictitious multi-severity Validation criterion.

## Gadi

Unpack next to `Backbone` and `Base_SupportBench`:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export STLA_S40_SPEED="$(pwd)/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export STLA_S40_FLOW="$(pwd)/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
python -u Mask_RAST-STLA/scripts/preflight.py --target speed --backbone-checkpoint "$STLA_S40_SPEED"
qsub -N MRAST23_speed -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_SPEED" Mask_RAST-STLA/pbs/formal.pbs
qsub -N MRAST23_flow  -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_FLOW" Mask_RAST-STLA/pbs/formal.pbs
# Independent NoRetrieval controls
qsub -N MRAST23_noret_s -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,ABLATION=no_retrieval,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_SPEED" Mask_RAST-STLA/pbs/formal.pbs
qsub -N MRAST23_noret_f -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,ABLATION=no_retrieval,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_FLOW" Mask_RAST-STLA/pbs/formal.pbs
```

Gadi PBS resources: `gpuvolta`, 1 V100 **with 12 CPUs**, 32GB memory, 48h walltime, timestamped live log and GPU CSV. `last.pt` supports exact-v2.3 same-config resume via PBS `RESUME_CHECKPOINT=/absolute/path/last.pt`. Automatic full Train → Evaluate → Audit is enabled.

## Independent Retrieval Audit

```bash
python -u Mask_RAST-STLA/scripts/evaluate_retrieval.py \
   --checkpoint /absolute/full_run/best.pt \
   --no-retrieval-checkpoint /absolute/noret_run/best.pt \
   --severities 20 40 60 90 --reuse
```

Outputs under `retrieval_audit/`: `paired_summary.csv`, `paired_horizon_family.csv`, `audit_manifest.json`.
This uses Full vs Internal Base, Full vs `no_future`, Full vs `global_only`, `cross_history` deterministic Train-case mismatch, and **independently trained** No-retrieval. Existing `shuffled` is a separate same-query rank-shuffle diagnostic. Reports matched MAE/RMSE, relative effects, negative transfer, directly missing versus observed nodes, missing family and all 12 horizons.

## Key code

- `mask_rast_stla/graph.py`, `encoders.py`: Train-only positive lag-aware statistical graph, region assignment and observability support.
- `mask_rast_stla/memory.py`: legally time-separated global/regional indexes.
- `mask_rast_stla/seasonal.py`: Train-only innovation reference and no-observation H1 evidence.
- `mask_rast_stla/model.py`: global + regional evidence, time alignment and risk gate.
- `mask_rast_stla/trainer.py`: momentum memory alignment, forecast, ranking, risk and Base preservation.

## Tests / honesty

`python -m pytest Mask_RAST-STLA/tests -q` verifies normal and AMP/autocast paths, region aggregation, nonzero fully masked H1 historical evidence and paired alignment gradients. We have **not** run actual Gadi V100 full formal training in the packaging environment; no MAE improvement is guaranteed. The new region retrieval is based on Train-only statistical association, not a verified physical network topology. Further cross-seed/cross-dataset tests and no-retrieval controls are required for publication-level attribution.


## Parallel STLA initialization modes (2026-10-11)

Submit with `BACKBONE_MODE=incomplete` (STLA path contains `/formal/`) or
`BACKBONE_MODE=complete` (STLA path contains `/formal_complete/`).
Both modes use the **same incomplete-sensing Base_SupportBench inputs**; only
the initial STLA checkpoint changes. Experiment roots, live logs and
region graph/index storage
are separated by the mode component. Run the provided repository-root
`submit_mask_retrieval_dual_backbone.sh` to submit eight independent PBS jobs.
Do not interpret differences between complete/incomplete initializations as
Retrieval-only gains: compare independently trained NoRetrieval controls.
