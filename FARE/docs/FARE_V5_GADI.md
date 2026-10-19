# FARE v5 Gadi operations

**Location**: `/scratch/mh20/zw1165/Speed_LLM_RAG` (adapt if different).
**Python**: selected active TFP executable; do not invoke system `python`.

## Before replacing v4

Finish/cancel running v4 jobs and archive the old code/results:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
mv FARE FARE_v404_backup
unzip FARE_v5.0.0_Full_UtilityAligned.zip
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/FARE:${PYTHONPATH:-}"
python -m pytest -q FARE/tests
```

This ZIP contains the *complete* FARE package, scripts, configs, tests and PBS,
not an Overlay. The workspace's Base_SupportBench and Backbone are external
research dependencies and are not duplicated.

## Paired OOF source required

`best.pt` from the existing formal full-train STLAformer is suitable as the
*initial frozen base model*, but **not** sufficient to generate OOF historical
residual labels retrospectively. Chronological fold STLAformer producer models
must be trained separately using appropriately earlier Train data. See README
for exact manifest and `build_oos_predictions.py` use.

## Check before qsub

```bash
python -u FARE/scripts/preflight.py --config FARE/configs/formal.yaml \
  --target speed --oos-forecast-file FARE_cache/oof_speed_paired.npz

python -u FARE/scripts/preflight.py --config FARE/configs/formal.yaml \
  --target flow --oos-forecast-file FARE_cache/oof_flow_paired.npz
```

Also check `--help` of `scripts/submit_gadi.py`. It runs its own CPU precheck;
submit only when all resource/data checks pass.

## PBS full training

```bash
python -u FARE/scripts/submit_gadi.py \
  --config FARE/configs/formal.yaml --target speed \
  --oos-forecast-file FARE_cache/oof_speed_paired.npz

python -u FARE/scripts/submit_gadi.py \
  --config FARE/configs/formal.yaml --target flow \
  --oos-forecast-file FARE_cache/oof_flow_paired.npz
```

The helper passes exact `FARE_PYTHON`, `FARE_ENV_PREFIX` and workspace paths into
PBS. Jobs use `gpuvolta: 1 GPU/12 CPUs/32 GB/48h`, one target per job.
Live logs are under `FARE_logs/<target>/<dataset>/`; experiment outputs under
`FARE_exps/<target>/<dataset>/formal/<run>/`.

**No duration guarantee**: new multi-candidate Teacher, full Memory and time-
aware evidence encoding can be slower than v4.0.4. Run the smoke/profile pass
and check actual V100 GPU memory before committing full formal training.

## Common blockers

- `Missing paired OOF source`: make real chronological OOF forecasts first.
- `Memory provenance mismatch`: old/stale Memory cannot be reused; remove only
  newly built invalid v5 Memory, not your archived v4 experiments.
- `Checkpoint and dataset node count mismatch`: select the proper Net checkpoint.
- `Walltime exceeds 48h`: checkpoints are saved each epoch; resume the v5 `last.pt`
  through `--resume`, with unchanged Backbone/Memory hashes.
- `CUDA OOM`: benchmark V100 peak and adjust batch size with a documented
  matched-budget retraining plan. Do not silently alter dataset/seed.

## Keep v4 diagnostics separate

The historical `FARE404_diagnostic_results_181508551` is prior exploratory
Test-based model analysis, **not** a v5 validation metric. Do not compare raw
pilot diagnostic MAE to full formal Test averages without identical sampling.
