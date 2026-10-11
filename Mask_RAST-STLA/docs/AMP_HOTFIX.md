# Mask_RAST-STLA v2.2.0 AMP hotfix

## Failure
Both MRAST22 Speed and Flow PBS jobs failed at `encoders.py:76` on the first training batch with `index_add_(): self (Half) and source (Float) must have the same scalar type`.

## Change
Region pooling now casts numerator/denominator inputs to float32 and accumulates in float32 before `region_proj`. Autocast still applies to the rest of the model. No training hyperparameters, retrieval functions, or memory schema were changed.

## Validation
`python -m pytest tests/test_rast_v2.py -q`: 3 passed on CPU, including CPU bfloat16 autocast forward/backward through region pooling and a full mock model. CUDA/V100 FP16 training remains to be validated on Gadi.

## Installation
From project repository root:
`unzip -o Mask_RAST-STLA_v2.2.0_AMP_HOTFIX.zip`
Or apply only the source patch:
`patch -p0 < MRAST22_amp_hotfix.patch`
These are alternative methods; do not use both.

This package intentionally preserves VERSION 2.2.0; the output filename identifies the AMP hotfix build.
