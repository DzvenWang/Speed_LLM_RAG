# TCRR-STLA v5.3.0 — Resource optimisation, without a new forecasting method

This release preserves CFRM, SCRT and CRSI forward mathematics and the
three independent model switches. The implementation target remains frozen
STLAformer + independent v5 Adapter/Retrieval/Normal Teacher/CF Replay caches.

## Implemented

1. **Shared multi-switch inference** (`TCRRSTLAV5.forward_modes`): identical
   Normal/CF inputs and Support features are computed once for a batch. For
   all eight Mxyz modes, two C1 passes and four C2 passes replace eight of
   each. C3 continues to run independently for the ON/OFF variants. Training
   still uses its original differentiable single-mode forward implementation.
   Shared and independent outputs were compared numerically in regression tests.
2. **Validation Replay reuse**: `evaluate_v5.py` loads the identical,
   source-mask-verified `val` CF Replay Bank built by training, using the same
   retrieval shortlist and teacher/source fingerprint. Correct and shuffled CF
   evaluate on the cached values. Wrong-mask still runs a NEW, scientifically
   required frozen teacher intervention and cannot be replaced with cached
   correct-mask outputs.
3. **Frozen Teacher memory**: training uses the existing frozen STLAformer as
   teacher, rather than deep-copying it to a second GPU module. Frozen weights,
   eval mode and Teacher SHA stay the same.
4. **Evaluation CPU memory**: ground truth, base predictions and valid masks
   are accumulated once per query batch, not eight times for eight modes.
5. **Optional test CF Replay**: `evaluate_v5.py --cache-test-replay` writes
   `test_s20`, `test_s40`, `test_s60`, `test_s90` caches with separate
   fingerprints. This is *off by default* because large test banks use disk.
6. **Profiling**: CSV/JSON and live logs now include epoch and evaluation
   throughput, GPU allocated/reserved peak, CPU RSS high-water mark, elapsed
   time, cache usage and whether shared forward was used.
7. **DataLoader tuning**: `TCRR_TRAIN_WORKERS` and `TCRR_EVAL_WORKERS` can be
   set to 0–8 (train) / a supported nonnegative count (eval). Default stays 0
   so memory-mapped arrays do not silently multiply CPU RAM usage.

## Preserved

- Source-valid original-mask intersection CF protocol and wrong-mask checks.
- Train-only memory, original frozen backbone and strict checkpoint identity.
- Eight model switches, negative controls, stage selection, calibration
  isolation and three-stage `afterok` PBS workflow.
- FP32 Replay by default: no rounding/accuracy trade-off introduced.
- K=12, all training queries, original epochs, Speed/Flow and four severities.

## Not claimed

These changes do not guarantee a particular speed-up or GPU memory reduction
on Gadi: use the recorded real V100 telemetry for quantitative comparisons.
GPU OOM batch-size fallback, test Replay FP16 quantisation and C3-variant
process coalescing are **not automatically enabled**. They require separate
robustness/numerical protocols; `BATCH_SIZE` remains the explicit stable knob.

## Usage

See `pbs/submit_pipeline_v5.sh` for the unchanged automatic TRAIN →
DIAGNOSTICS → TEST submission workflow.

```
# Optional, AFTER locating your v5.3 run's checkpoint:
python TCRR-STLA/scripts/evaluate_v5.py --checkpoint /path/best_controller.pt \
  --split validation --modes M000 M001 M010 M011 M100 M101 M110 M111 \
  --controller-mode auto --diagnostics

# Pure numerical regression comparator; slower but equivalent:
python TCRR-STLA/scripts/evaluate_v5.py --checkpoint /path/best_controller.pt \
  --split validation --modes M000 M001 M010 M011 M100 M101 M110 M111 \
  --controller-mode auto --disable-shared-modes --output-tag independent_reference
```
