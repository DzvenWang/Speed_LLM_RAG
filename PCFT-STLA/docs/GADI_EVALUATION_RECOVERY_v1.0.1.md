# PCFT-STLA v1.0.1 — Gadi OOM / SIGBUS recovery

This is the **complete PCFT-STLA project** (not a patch). It fixes runtime
resource allocation; the PCFT architecture, parameter keys, retrieval scoring,
Train-only graph, checkpoint format, MAE/RMSE/MAPE protocol, and original
STLA-only initialization policy are unchanged.

## What the October 18 logs showed

- `PCFT_speed.o181395434`: training completed and `best.pt` was saved, best
  epoch 39 / validation MAE 1.523005. The automatic evaluator reused
  training `batch_size=64`, enabled GPU repository preload, and failed with
  CUDA OOM at `memory_encoder` / `graph_k`.
- `PCFT_flow.o181395435`: best model saved at epoch 38 / validation MAE
  16.938402, but training failed in epoch 54 with worker SIGBUS. The
  `DataLoader` previously used 4 persistent workers and prefetch 2.

## Changes

- **Evaluation now defaults to batch 8, zero DataLoader workers, CPU-backed
  repository**. It retains the Train-only retrieval index on GPU, if CUDA is
  available, and auto-retries a whole severity with batches 4, 2, 1 on CUDA
  OOM. Incomplete attempts never write partial metrics.
- New formal runs use `num_workers=0` and no persistent workers by default,
  avoiding worker shared-memory pressure. Training model and loss unchanged.
- Existing `best.pt` is kept as v1.0.0. `VERSION=1.0.1` identifies the
  *source runtime hotfix*, not a new checkpoint format.
- `evaluate_retrieval.py` automatically uses the safer evaluator for every
  evidence intervention. The P0 Candidate Probe also uses safe settings.
- Test files inherited from the original project have their internal module
  imports repaired.
- `scripts/check_saved_checkpoint.py` verifies original backbone file SHA256,
  checkpoint target, and Train-only memory index **without retraining**.

## Deployment

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
unzip -o PCFT-STLA_v1.0.1_Gadi_Resource_Fix_Complete.zip
conda activate TFP
```

## Validate existing checkpoints

```bash
python -u PCFT-STLA/scripts/check_saved_checkpoint.py --target speed \
  --checkpoint PCFT-STLA_exps/speed/incomplete/TraffiDent_Net_06/formal/20261017_181821_seed10_full/best.pt

python -u PCFT-STLA/scripts/check_saved_checkpoint.py --target flow \
  --checkpoint PCFT-STLA_exps/flow/incomplete/TraffiDent_Net_06/formal/20261017_183001_seed10_full/best.pt
```

Both `best.pt` and `best_memory_index.npz` must exist. If a checkpoint's
original STLA checkpoint moved, use `--backbone-checkpoint` pointing to the
*identical* original STLAformer `best.pt` (SHA256 checked).

## Submit two independent recovery Evaluate jobs

```bash
qsub -V -N PCFT_ES \
  -v TARGET=speed,CHECKPOINT=PCFT-STLA_exps/speed/incomplete/TraffiDent_Net_06/formal/20261017_181821_seed10_full/best.pt,EVAL_BATCH_SIZE=8 \
  PCFT-STLA/pbs/evaluate.pbs

qsub -V -N PCFT_EF \
  -v TARGET=flow,CHECKPOINT=PCFT-STLA_exps/flow/incomplete/TraffiDent_Net_06/formal/20261017_183001_seed10_full/best.pt,EVAL_BATCH_SIZE=8 \
  PCFT-STLA/pbs/evaluate.pbs
```

These jobs reuse saved weights, never start a training job. In a tight GPU
allocation set `EVAL_BATCH_SIZE=4`. Batch size should not change prediction
values beyond ordinary floating-point kernel tolerances; it is not a model
hyperparameter and is not chosen based on Test MAE.

## Outputs

Each model writes only within its own `PCFT-STLA_exps/...`:

```
evaluation/s20/full/summary.json
evaluation/s40/full/summary.json
evaluation/s60/full/summary.json
evaluation/s90/full/summary.json
evaluation/severity_summary.full.csv
evaluation/severity_summary.csv
```

The outputs include MAE (primary), RMSE, MAPE, failure-family and directly
masked metrics. Training is not required for these outputs.

## Additional experiments

Formal training stays backbone-only and isolated from `Mask_RAFT-STLA` and
`Mask_RAST-STLA`.

`pbs/formal.pbs` requests one V100, twelve CPUs, 32 GB RAM. Automatic
Evaluate will use the safe runtime independently of its training batch size.

For Retrieval Audit, `scripts/evaluate_retrieval.py --checkpoint ... --reuse`
reuses Full outputs and evaluates missing interventions with the same safe
settings. The optional P0 candidate probe defaults to batch 8 and CPU-backed
Train-only repository. These are **diagnostic** ablations, not new training.

## Validation completed locally

Source compiles; all PBS shell scripts pass `bash -n`; 29 CPU tests pass.
Actual Gadi evaluation still must be run. No claim of improved MAE is made by
this resource-only hotfix.
