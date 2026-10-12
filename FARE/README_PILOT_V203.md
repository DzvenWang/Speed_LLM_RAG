# FARE v2.0.3 — Retrieval-first Pilot

## Setup

Unpack the ZIP into the `Speed_LLM_RAG` project root so the single top-level folder is `FARE/`. Back up the old `FARE/` folder first, and do not overwrite code in use by running PBS jobs. The external `Base_SupportBench` and `Backbone` projects remain dependencies, just as in v2.0.2.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
python -u FARE/scripts/preflight.py --target speed --config FARE/configs/pilot.yaml
python -u FARE/scripts/preflight.py --target flow --config FARE/configs/pilot.yaml
```

## Recommended Pilot, BS32

```bash
qsub -N FARE_RU_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32,FARE_PROGRESS_EVERY=100 FARE/pbs/train_pilot.pbs
qsub -N FARE_RU_flow -v TARGET=flow,SEED=10,BATCH_SIZE=32,FARE_PROGRESS_EVERY=100 FARE/pbs/train_pilot.pbs
```

- Dataset Net-06, 8,000 time-spread training windows, 16,000 train-only memory windows, validation from the same SupportBench protocol.
- Retrieval epochs 8; Recovery epochs 5; Injection epochs 12. **Partial/Full intentionally not run by default**, based on their earlier validation degradation.
- Full Top-64 / Top-8 retrieval and STLAformer Injection, no module removal.
- Hardware sampling (every 30s), online diagnostics, evaluation and five Validation retrieval interventions are enabled.
- The previous BS64 job failed OOM; use 32 here.

### Optional conservative fine-tune study (after successful main Pilot)

```bash
qsub -N FARE_FT_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32,FARE_CONFIG=FARE/configs/pilot_finetune.yaml FARE/pbs/train_pilot.pbs
```

This is an **independent training run**, not an automatic resume of the first Pilot; more conservative partial/full learning rates and retrieval freeze are configured. Do not compare models from different validation protocols.

## Outputs

- Live log: `FARE_logs/<target>/<dataset>/*.train.live.log`.
- Epoch metrics and snapshots: `FARE_exps/<target>/<dataset>/pilot/<stamp>_seed10/` (see printed `run_dir`).
- Automatic fixed-checkpoint attribution:
  - `retrieval_attribution/ablation_summary.csv`
  - `retrieval_attribution/retrieval_gain.md`
  - Per-mode `val_<mode>_summary.json`
- Training objective diagnostics in `FARE_EPOCH`, including:
  - `coarse_utility_true_positive_fraction`
  - `coarse_utility_sign_accuracy`
  - `coarse_utility_expected_gain`
  - `coarse_utility_oracle_gain`
  - `coarse_utility_oracle_positive_groups`
  - `ranking_selected_error` vs `ranking_uniform_error`.
- `FARE_RESOURCE_EPOCH` and `FARE_DIAG` are preserved.

## How to interpret Retrieval gains

1. First verify real memory outperforms random, similarity-only, utility-off and no-memory on **the same validation instances**. A positive MAE advantage is the desired evidence, not guaranteed.
2. Then check whether the forecast-utility head predicts which candidate sets actually contain positive historical-analog improvement. Accuracy around class prevalence without positive ranking gain is not enough.
3. Inspect Negative Transfer and direct masked nodes. A successful ranking that does not lower final MAE is insufficient.
4. Only after consistent Validation improvement across seeds should full-data Formal and Test be considered. Fixed-checkpoint interventions are complemented by matched **retrained** ablations for paper claims.

## Manual ablations

```bash
python -u FARE/scripts/ablate.py --checkpoint /path/to/candidate_best.pt --split val \
  --modes full similarity_memory utility_off random_memory no_memory
```

**This is a new training objective and a different retrieval inference function**: v2.0.2 model checkpoints should not be resumed for v2.0.3 experiments. Keeping them as baselines is encouraged.
