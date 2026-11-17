# CoRSTA v0.2.2 — AMP PairedTeacher Hotfix1

## Observed failure on Gadi

- Speed job `183460398`: `RuntimeError: CF/Residual target decomposition broken` at `corsta_v2/counterfactual/teacher.py:32` during first `probe_size` (before training Epoch 1).
- Flow job `183460403`: same traceback.
- Both loaded 388-node data and completed initial Normal-stage selection. Both exited with status 1. This is **not an OOM**.

## Root cause

`trainer.py` calls the PairedTeacher from inside `torch.autocast(device_type='cuda', dtype=torch.float16)`.
The old teacher computed `cf = pf - pm` when FP16 predictions were still half-precision. It then computed `res = y.float() - pf`, which promotes to FP32. Adding FP16 CF to FP32 residual can disagree with `y.float() - pm` by ~1e-3 or more because the FP16 subtraction has already rounded. `torch.allclose(... atol=2e-5, rtol=2e-5)` therefore fails for valid forecasts.

## Exact code changes

Only `corsta_v2/counterfactual/teacher.py` is modified; adds `tests/v0_2/test_teacher_amp_hotfix.py`. All Teacher predictions/hidden outputs are converted to FP32 **before** CF/residual/latent target subtraction. Non-finite predictions and labels remain fail-fast with informative messages; the decomposition check uses a float32-ULP scale-aware tolerance. No change to Student forward, STLA weights, CF/Residual target semantics, optimizer stages, epoch settings, or dataset.

## On Gadi

Copy the Hotfix1 patch ZIP to the project root `/scratch/mh20/zw1165/Speed_LLM_RAG`, and run:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate /g/data/mh20/zw1165/envs/TFP

# Recommended: first save the original teacher implementation.
cp CoRSTA/corsta_v2/counterfactual/teacher.py \
   CoRSTA/corsta_v2/counterfactual/teacher.py.pre_amp_hotfix1.bak

unzip -o CoRSTA_v0.2.2_AMP_Teacher_Hotfix1_Patch.zip

# Verify with the same TFP environment:
PYTHONPATH="$PWD/CoRSTA:$PWD/Backbone:$PWD" \
 python -m pytest -q CoRSTA/tests/v0_2/test_teacher_amp_hotfix.py
```

If you had already started a separate/new run of v0.2.2 (different from the two failed jobs), do not edit code during its active training. The `code_sha256` run identity will change, invalidating Checkpoint resume identity for a different code version.

Both failed jobs ended during the initial batch probe (before any training epoch), so `CORSTA_V2_STAGE=all` can be submitted again after the patch. `normal/best.pt` may already exist as a preliminary Epoch 0 checkpoint; it should not be treated as a trained checkpoint. The initial selection logic will overwrite it when beginning the patched run (assuming `last.pt` does not exist).

### Re-submit Speed / Flow (`all`: train → lock → test → report)

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate /g/data/mh20/zw1165/envs/TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export BACKBONE_MODE=incomplete
export SPEED_STLA="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export FLOW_STLA="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
export SPEED_S0="$PWD/CoRSTA_exps/speed/incomplete/TraffiDent_Net_06/formal/seed10/bootstrap_v7/s0.pt"
export FLOW_S0="$PWD/CoRSTA_exps/flow/incomplete/TraffiDent_Net_06/formal/seed10/bootstrap_v7/s0.pt"

qsub -N C022H1_speed \
  -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$SPEED_STLA",MASK_PROJ_CHECKPOINT="$SPEED_S0",BACKBONE_MODE=incomplete,PREDICTION_MODE=normal_anchored,CORSTA_V2_STAGE=all,BATCH_INITIAL=32 \
  CoRSTA/pbs/v0_2/train.pbs

qsub -N C022H1_flow \
  -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$FLOW_STLA",MASK_PROJ_CHECKPOINT="$FLOW_S0",BACKBONE_MODE=incomplete,PREDICTION_MODE=normal_anchored,CORSTA_V2_STAGE=all,BATCH_INITIAL=32 \
  CoRSTA/pbs/v0_2/train.pbs
```

## Validation conducted locally

- 63 Python tests passed (`tests/v0_2` and old CoRSTA/v7/D1 regression suites).
- Speed and Flow *synthetic data* E2E each passed `all`: train, lock, TEST, report.
- Direct numerical reproduction demonstrated original FP16 subtraction fails at `2e-5`, and Hotfix1 FP32-based targets pass.
- **Not yet run against real TraffiDent STLA/TFP V100**. After re-submission, inspect logs to confirm Batch Probe and Epoch 1 succeed.
