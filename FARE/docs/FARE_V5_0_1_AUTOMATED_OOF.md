# FARE 5.0.1 — Chronological STLAformer OOF Producer Pipeline

The new pipeline uses the *actual* pinned Backbone STLAformer adapter, official
Backbone training recipe and schema-v3 Base_SupportBench Train cache. Neither
`Backbone` nor `Base_SupportBench` is modified. Their source files and
`Backbone/vendor/STLAformer/model/STLformer.py` must exist in the workspace.

## Scientific contract

- Historical episodes are identified by source-cache **row ID plus all available
  manifest identity fields**, not by `window_start` alone. Every scored view is
  preserved, including views sharing the same timestamp.
- Four expanding-window time folds by default, warmup 20%; strict no-label-overlap
  temporal split between producer fit, producer internal validation, and scored
  episodes. Configurable purge (default 12 steps).
- Every Fold trains a **new STLAformer**, once, on genuine prior Train windows.
  It **never** trains on Base_SupportBench val or test. Internal fold validation
  also comes from the chronologically earlier train split.
- Each Fold estimates normalization from its own valid *fit prefix*. The full
  Base_SupportBench normalization is algebraically inverted and remapped, then
  fold predictions are converted back into the FARE/Backbone reference scale.
- The SAME incomplete-trained producer predicts (a) failed-observation input
  and (b) complete-available historical input. This yields paired residuals and
  failure-induced forecast deltas; the second is not a separately trained
  complete-input model.
- A producer's actual training procedure writes its time provenance to `best.pt`.
  The exporter checks the checkpoint plan hash, fold ID, fit/validation label
  endpoint, source-cache checksum and time gap. Sidecar metadata pins all
  producer checkpoint hashes and cache identity.
- Outputs omit the first warmup portion of Train: no fake "OOF" forecasts are
  generated for that period. FARE Train queries can still cover all Train rows.
- `schema_version=6` Memory stores per-Episode `row_id`, `episode_key`,
  `family`, `severity`, `fold_id` and all original forecast arrays. Old memory
  and old `window_start` keyed `.npz` predictions are not accepted.

## Recommended Gadi commands

From `/scratch/mh20/zw1165/Speed_LLM_RAG` after `conda activate TFP`:

```bash
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/FARE:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
python -m pytest -q FARE/tests

# This does not submit or require a GPU.
python -u FARE/scripts/submit_oof.py --target both --plan-only

# Single PBS submission; processes Speed then Flow sequentially on one V100.
python -u FARE/scripts/submit_oof.py --target both
```

Gadi: gpuvolta, 1 V100, 12 CPU, 32GB RAM, 48h. Output appears at:

- `FARE_cache/oof_v501/{speed,flow}/TraffiDent_Net_06/fold_plan.json`
- `FARE_cache/oof_v501/{speed,flow}/TraffiDent_Net_06/fold_NN/best.pt`
- `FARE_cache/oof_speed_paired.npz` and `.json`
- `FARE_cache/oof_flow_paired.npz` and `.json`
- `FARE_cache/{speed,flow}/TraffiDent_Net_06/memory_v501_10_50000`
- `FARE_logs/oof/*.live.log`

The pipeline automatically writes `last.pt` per fold **after each epoch**;
rerunning the same submission with identical configuration skips completed folds
and resumes an interrupted fold from its saved optimizer/scheduler and model
state. It does not treat a mere `best.pt` as proof that the fold completed.
If source-cache, plan, or reference Checkpoint changes, clean the affected
`FARE_cache/oof_v501` folder before starting a new experiment.

Once the OOF pipeline completes, run the FARE v5 Formal preflight and training:

```bash
python -u FARE/scripts/preflight.py --config FARE/configs/formal.yaml --target speed \
  --oos-forecast-file FARE_cache/oof_speed_paired.npz
python -u FARE/scripts/preflight.py --config FARE/configs/formal.yaml --target flow \
  --oos-forecast-file FARE_cache/oof_flow_paired.npz

python -u FARE/scripts/submit_gadi.py --target speed \
  --oos-forecast-file FARE_cache/oof_speed_paired.npz
python -u FARE/scripts/submit_gadi.py --target flow \
  --oos-forecast-file FARE_cache/oof_flow_paired.npz
```

## HPC limitations

* Training 8 real Fold producers (4 per target) is compute-intensive. **One
  48-hour PBS allocation may not be enough.** All completed folds and epoch
  progress remain on disk; submit again with identical options to resume.
* Do not train Fold producers on the login node. `--plan-only` is CPU-only.
* The log/state proves which data the *provided training code* consumed. It
  does not certify an arbitrary externally supplied `best.pt` as OOF.
* Historical Data's real `x_full_norm` means all *genuinely available* train
  history with synthetic masks removed, not access to unobserved truth.
* Test caches are never used in Fold selection, training or Memory formation.
* Before interpreting improvements as causal, keep learned-ranker comparisons
  to Matched Random/Shuffled on locked validation and later independent test.
