# FARE v2.0.4 — Pilot Validation + Test

Replace the old `FARE/` folder with this package only **after old PBS jobs have stopped** (or deploy under a separate project checkout). Preserve `FARE_exps` and `FARE_logs` separately.

FARE v2.0.4 keeps all v2.0.3 Forecast Utility Retrieval training components. The key change is **Pilot trains on Train, selects a checkpoint on Validation, and then automatically reports Validation AND Test s20/s40/s60/s90 results**. Retrieval evidence ablations are evaluated on both splits by default.

## Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
python -u FARE/scripts/preflight.py --target speed --config FARE/configs/pilot.yaml
python -u FARE/scripts/preflight.py --target flow  --config FARE/configs/pilot.yaml
qsub -N FARE_P204_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32 FARE/pbs/train_pilot.pbs
qsub -N FARE_P204_flow  -v TARGET=flow,SEED=10,BATCH_SIZE=32 FARE/pbs/train_pilot.pbs
```

The default uses 8,000 sampled Train windows, train-only Memory 16,000, coarse Top-64/final Top-8, BS32, Retrieval 8 / Recovery 5 / Injection 12, stops before Partial/Full, and stays in the same 24-hour PBS allocation for the later Evaluate and Attribution phases. Hardware monitoring remains active.

## Pilot validation vs Test

- Validation: the training-protocol validation data (typically s40), used for checkpoint selection.
- Test: evaluation caches for s20, s40, s60, s90; Test is not used by optimizers or checkpoint selection.
- The Pilot Test analyses are exploratory. Repeatedly changing the model based on Pilot Test numbers invalidates the interpretation of that Test as an untouched final holdout.

## Files to inspect

- `pilot_val_test_metrics.md`: Backbone, Candidate and FARE MAE/RMSE, by Val and Test severity and Test macro average.
- `retrieval_attribution/retrieval_gain_val_test.md`: full vs similarity-only, utility-off, random and no-memory, per Test severity.
- `retrieval_attribution/{val,test}/ablation_summary.csv`: machine-readable fixed-checkpoint intervention scores.
- `FARE_logs/<target>/<dataset>/*.train.live.log`: live progress, resource samples and training diagnostics.

## Runtime controls

```bash
# Auto metrics on Val+Test; skip 25 potentially expensive retrieval interventions:
qsub -N FARE_P204_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32,FARE_AUTO_ABLATIONS=0 FARE/pbs/train_pilot.pbs

# Val+Test metrics but attribution on Val only:
qsub -N FARE_P204_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32,FARE_ABLATION_SPLITS=val FARE/pbs/train_pilot.pbs
```

Other controls: `EVAL_SPLITS=val:test`, `SEVERITIES=20:40:60:90`, `FARE_RESOURCE_INTERVAL=30`, `FARE_PROGRESS_EVERY=100`. Do not use `EVAL_SPLITS=val:test` if Test caches have not been built; the job will fail visibly, rather than silently skip Test.
