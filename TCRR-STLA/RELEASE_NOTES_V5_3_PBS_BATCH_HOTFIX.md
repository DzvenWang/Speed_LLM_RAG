# TCRR-STLA v5.3.0 — PBS Parameter Forwarding + Full-Batch Smoke Hotfix

**Patch scope:** PBS environment forwarding and optional isolated full-batch smoke profiling. The canonical version remains **5.3.0**, and normal formal C1/C2/C3 computation and all existing cache formats remain unchanged.

## Defects fixed

1. `pbs/submit_pipeline_v5.sh` previously omitted `BATCH_SIZE` in its `qsub -v` environment. Passing `BATCH_SIZE=64` in the caller did not affect the training PBS child.
2. `TCRR_TRAIN_WORKERS`, `TCRR_EVAL_WORKERS`, and the three diagnostic batch settings were also silently dropped.
3. `SMOKE=1` could not be safely sent through the three-stage pipeline launcher: smoke generates no formal checkpoints for the dependent DIAGNOSTICS and TEST jobs.
4. Standard `--smoke` uses batch 3, so it cannot validate the memory of the formal 64-sample C12 backward pass.

## Implemented

- The launcher now validates and forwards `BATCH_SIZE` (1..512), `TCRR_TRAIN_WORKERS` (0..8), `TCRR_EVAL_WORKERS` (0..8), `DIAGNOSTIC_BATCH_SIZE` (1..512), `DIAGNOSTIC_MAX_BATCHES` (0+), and `DIAGNOSTIC_PRIMARY_MAX_BATCHES` (0+).
- Invalid values are rejected **before** submitting any PBS job.
- Normal mode still submits TRAIN -> DIAGNOSTICS -> TEST with `afterok` dependencies. The printed batch size reflects the selected setting.
- `SMOKE=1` submits **one** isolated short-duration job under `/v5/smoke/`, with no formal dependents.
- `SMOKE=1 SMOKE_PROFILE_FULL_BATCH=1 BATCH_SIZE=64` performs S0 and C12/controller training steps using **a true 64-item microbatch**. Because smoke retrieval and replay cover only 24 distinct queries, this mode *duplicates smoke query indices* to build the 64-item batch. This tests network memory footprint and gradient execution but **must not** be interpreted as representative throughput or accuracy. Standard smoke remains at batch 3.
- The optional profiling mode uses **smoke-only** S0 adapters and caches. Formal S0 identity, training sample counts, masks, replay protocol, epochs and canonical version remain unchanged.
- Training logs show the effective size with `V5_PLAN ... batch_size=64 smoke_full_batch=True` plus per-epoch peak GPU allocation and CPU RSS.

## Gadi examples

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export TFP_ENV_ROOT=/g/data/mh20/zw1165/envs/TFP
export SPEED_CKPT="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"

# Isolated memory stress test; not a formal experiment.
TARGET=speed BACKBONE_MODE=incomplete BACKBONE_CHECKPOINT="$SPEED_CKPT" \
  BATCH_SIZE=64 SMOKE=1 SMOKE_PROFILE_FULL_BATCH=1 \
  bash TCRR-STLA/pbs/submit_pipeline_v5.sh

# Formal pipeline with environment forwarding after the profiling run succeeds.
TARGET=speed BACKBONE_MODE=incomplete BACKBONE_CHECKPOINT="$SPEED_CKPT" \
  BATCH_SIZE=64 TCRR_TRAIN_WORKERS=0 \
  bash TCRR-STLA/pbs/submit_pipeline_v5.sh
```

**Do not change batch size in a running formal experiment.** Doing so changes the optimizer update count and S0 Adapter identity. Keep old runs at their original batch size; start a new run for the new batch size.

## Verification

- Mock-qsub integration tests validate all three PBS jobs receive the same forwarded variables, the new full-batch smoke submits exactly one job, and invalid overrides fail before submission.
- Unit tests validate the effective smoke batch size (3 vs 64) and repetition of isolated smoke query indices.
- Existing project suite and Python/shell syntax checks pass on a CPU-only machine. Real Gadi V100 memory and throughput must still be measured separately.
