# Migration: Backbone v0.3.1 -> v0.4.0

Purpose: add recent numerical candidates PatchSTG (KDD 2025) and DST-SGNN (IJCAI 2025) before Formal selection.

## What remains frozen
- Shared Protocol Development stage and its stage hash.
- Existing screening cache and four completed Stage-1A results.
- STAEformer/HimNet/BigST/STDN vendor source and adapters.
- Selection rule: within 2% Overall Val MAE, then support-dependent Val MAE, 0.5% support tie -> masked MAE, then efficiency/parameters.
- Test remains locked.

## Compatibility
v0.4.0 explicitly permits the existing v0.3.1 **screening** cache/results. This exception is screening-only. Formal and Evaluation caches must be created by v0.4.0.

## New execution path
1. Install `timm==1.0.12` in TFP if absent.
2. Run `scripts/setup_official_models.sh` to extend the source lock with the two new repositories.
3. Run `scripts/preflight.py --model PatchSTG` and `--model DST-SGNN` on a GPU allocation/job for the latter.
4. Submit only the two new screening runs with `scripts/submit_recent_screening.sh`.
5. Run `scripts/summarize_screening.py` after all six candidates have valid results.
6. Only then build Formal Protocol/cache and submit the new Top-2.
