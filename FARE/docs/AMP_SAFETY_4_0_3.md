# FARE v4.0.3 — V100 FP16 Safety and Fail-Closed Submission

## Why the Gadi job 181327616 failed

The job on 2026-10-16 completed support teacher generation (`queries=128`) but stopped at the **first training batch**:

```
FARE/fare/retrieval.py:23
RuntimeError: value cannot be converted to type at::Half without overflow
```

`torch.autocast(device_type='cuda', dtype=torch.float16)` makes retrieval score tensors FP16. The old code wrote `-1e7` into the FP16 tensor via `masked_fill`. Finite FP16 values are limited to approximately ±65504; this is a code bug. The existing FP32 unit tests could never detect it.

## Changes in v4.0.3

- `fare/numerics.py`: shared `mask_invalid_logits`, returning FP32 scores before negative sentinel masking.
- `fare/retrieval.py`: all region/node Top-K score masks and duplicates use the safe primitive. Zero-vector cosine normalization uses FP32 with a safe epsilon.
- `fare/transfer.py`: path masked logits and temporal softmax are FP32 under AMP.
- `fare/utility.py`: candidate, pairwise Keep/Swap and final evidence+Null masked logits and softmax inputs use FP32 under AMP.
- `tests/test_fp16_amp_safety.py`: real FP16 chunked Top-K, no-lawful-history, full model AMP forward/backward, Teacher and interventions.
- `tests/test_production_shape_amp.py`: Net-06-scale N=388, B=2, P=12, region/node/final K=64/16/8, FP16 AMP forward/backward.
- `scripts/pre_submit_check.py`: CPU safety gate using the active environment, four evaluation caches, and actual frozen STLAformer vendor Checkpoint strict loading + one real data CPU forward.
- `scripts/submit_gadi.py`: **default** pre-submit gate; failure prevents `qsub`. `--check-only` verifies without submitting.
- `fare/cuda_preflight.py`: on an allocated CUDA node, runs the actual vendor+FARE FP16 forward/backward with a small transient chronology-safe Train sample before building the expensive 16k-Episode Memory or running any training epoch. It restores RNG state; its test-only residuals are never saved.
- Training checkpoint metadata version updated to `4.0.3`; v4.0.0–4.0.3 remain acceptable for resume where usable.

## First action on Gadi (no GPU queue)

Use the patch over an idle v4.0.2 source tree:

```bash
unzip -o FARE_v4.0.3_AMP_Safety_PreSubmit_Overlay.zip
conda activate TFP
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q FARE/tests

python -u FARE/scripts/submit_gadi.py --check-only \
  --config FARE/configs/pilot.yaml --dataset TraffiDent_Net_06 \
  --target speed --seed 10 \
  --checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt
```

Expected end marker: `FARE_PRE_SUBMIT_GATE_PASS` and `FARE_CHECK_ONLY_NO_QSUB`.

Submit the same command **without** `--check-only` only after success; the gate will run again by default.

## Scope of confidence

The submit gate runs on CPU with CPU autocast FP16, including production-shaped tensors. It reproduces and prevents the logged FP16 negative-sentinel overflow, strictly loads the real STLAformer vendor and checkpoint, and runs the real backbone once in CPU float32. When the GPU job starts, an additional small real CUDA AMP forward/backward check runs before expensive Memory building (enabled by `training.cuda_preflight: true`).

Local tests do **not** certify that the real Gadi V100 CUDA preflight passes until it has run there. Even a successful CUDA preflight does not prove full-epoch stability, 48-hour completion, forecasting accuracy or numerical consistency with the old vendor. Do not claim such validation until compute-node tests succeed.

The failed job stopped before a training checkpoint was reported; any fully saved compatible Memory directory may be reused. The existing in-sample residual warning is a Pilot research limitation, not the cause of this crash.
