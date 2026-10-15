# EvalOnly — sequential single-GPU re-evaluation (2026-10-13)

Applies to the uploaded Backbone v1.3.1 and Baseline v1.4.7, with the MAPE-enabled Base_SupportBench. **It never trains, fine-tunes, or overwrites best.pt.**

## Install

Extract this zip into `/scratch/mh20/zw1165/Speed_LLM_RAG/` so it replaces the existing `EvalOnly/` folder. No other project code needs changing.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
bash EvalOnly/submit_all.sh --dry-run
```

## Whole project, one GPU per project

```bash
# Backbone: ONE PBS job, ONE V100 GPU, evaluations run sequentially
bash EvalOnly/submit_all.sh --submit-backbone

# Baseline: ONE PBS job, ONE V100 GPU, evaluations run sequentially
bash EvalOnly/submit_all.sh --submit-baseline

# Both: TWO jobs total, EACH has one V100 (the jobs may run concurrently)
bash EvalOnly/submit_all.sh --submit-all
```

The original per-model batch submission behavior is retained separately as `bash EvalOnly/submit_per_model_legacy.sh --submit-backbone` etc. **Do not** use this legacy command if you want one GPU total per project.

The scanner only evaluates existing completed runs. The full current Gadi dry-run identified **14 Backbone** evaluations (12 incomplete + STLAformer speed/flow complete), and **22 Baseline** evaluations (11 speed + 11 flow); skipped combinations are not trained. Both groups include all found s20/s40/s60/s90, 5 failure types, MAE/RMSE/MAPE and masked metrics.

The PBS group job calls the existing `backbone_eval_only.pbs` / `baseline_eval_only.pbs` scripts one at a time, with no nested qsub and no additional GPU allocations. Resources: `gpuvolta`, `ngpus=1`, `ncpus=12`, `mem=48GB`, `walltime=48:00:00` per group job.

### Filter a batch

```bash
# One GPU for two Backbone models, speed only
TARGETS='speed' BACKBONE_MODELS='STLAformer PatchSTG' \
  bash EvalOnly/submit_all.sh --submit-backbone

# One GPU for selected Baseline methods, flow only
TARGETS='flow' BASELINE_METHODS='RAFT-STLAformer RAST-STLAformer' \
  bash EvalOnly/submit_all.sh --submit-baseline
```

Available scanner environment filters: `TARGETS`, `BACKBONE_MODELS`, `BACKBONE_MODES`, `BASELINE_METHODS`, `REFERENCE_MODES`, and `SEED` (default 10).

## Single evaluation with an explicitly supplied model weight

By default this command **only previews** the qsub arguments. Add `--submit` to really queue it.

```bash
# Single Backbone checkpoint (incomplete)
bash EvalOnly/submit_one.sh \
  --project backbone --model STLAformer --target speed --seed 10 \
  --checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt \
  --mode incomplete --submit

# Single Backbone checkpoint (complete)
bash EvalOnly/submit_one.sh \
  --project backbone --model STLAformer --target flow --seed 10 \
  --checkpoint Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal_complete/20261009_170304_seed10/best.pt \
  --mode complete --submit

# Single Baseline checkpoint; the --checkpoint is the BASELINE'S OWN best.pt,
# not its reference STLAformer best.pt
bash EvalOnly/submit_one.sh \
  --project baseline --method RAFT-STLAformer --backbone STLAformer \
  --target speed --seed 10 --mode incomplete \
  --checkpoint Baseline_exps/speed/RAFT-STLAformer/STLAformer/TraffiDent_Net_06/formal/20261006_155442_seed10/best.pt \
  --submit

# Alternative: explicit Baseline formal run directory
bash EvalOnly/submit_one.sh \
  --project baseline --method GinAR --target flow --seed 10 \
  --run-dir Baseline_exps/flow/GinAR/TraffiDent_Net_06/formal/20261002_113149_seed10 \
  --submit
```

Baselines that use a reference Backbone need both `--backbone STLAformer|PatchSTG` and the correct `--mode`. For complete-referenced Baseline runs, use `--mode complete` and specify the Baseline's own `best.pt` under its `.../complete/.../formal/...` path. No complete-referenced Baseline Formal runs were detected in the supplied dry-run.

## Monitoring / resuming

```bash
qstat -u $USER
find EvalOnly_logs -type f -name '*.live.log' -printf '%T@ %p\n' | sort -nr | head -8
find Backbone_logs Baseline_logs -name '*.reeval.*.live.log' -o -name 'BaselineReEval*.live.log'
```

On submission, a frozen task manifest is saved in `EvalOnly/runs/<date>_<id>/backbone.jsonl` or `baseline.jsonl`. Per-item PASS markers are written under `<manifest-parent>/state/<manifest-stem>/`. The group continues after any individual failure and returns nonzero overall if a model failed. Its `latest_summary.json` reports successes and errors. Old PASS markers are skipped when the exact same manifest is restarted.

**To resume after walltime or failure, use the *same* manifest** (replace placeholder with the actual path printed at submission):

```bash
qsub -N BB_Eval1GPU -v KIND=backbone,MANIFEST=/scratch/mh20/zw1165/Speed_LLM_RAG/EvalOnly/runs/<run-id>/backbone.jsonl,SPEED_LLM_RAG_ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG EvalOnly/pbs/group_eval_only.pbs

qsub -N BL_Eval1GPU -v KIND=baseline,MANIFEST=/scratch/mh20/zw1165/Speed_LLM_RAG/EvalOnly/runs/<run-id>/baseline.jsonl,SPEED_LLM_RAG_ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG EvalOnly/pbs/group_eval_only.pbs
```

Group walltime is capped at 48 h. If 14/22 evaluations exceed 48 h in sequence, resume the same manifest to finish remaining entries. Do not start two simultaneous copies of the same manifest.

## Result paths and limitations

- Backbone evaluations write new timestamped folders in `Backbone_exps/<target>/<model>/TraffiDent_Net_06/checkpoint_evaluation/...`.
- Baseline re-evaluation still writes to existing evaluation result paths in `Baseline_exps/.../evaluation/...` and may overwrite CSVs and predictions. **Back up the relevant evaluation folders** before submitting if those results need preserving.
- Per-model live logs remain in `Backbone_logs/<target>/<model>/` and `Baseline_logs/<target>/`; aggregate live logs are in `EvalOnly_logs/`.
- Older Baseline checkpoints may fail identity or SHA256 checks against current code. The batch logs the error and continues. It never retrains the checkpoint.
- This covers 6 Backbone model names and 11 Baseline evaluation methods supported by the current generic evaluators. Additional experimental methods without generic evaluator support cannot be silently included.
- On zero/near-zero target positions (`abs(y) <= 1`), MAPE excludes those positions per the shared MAPE protocol. MAE/RMSE definitions are unchanged.
- Gadi CUDA/PBS training runtime is not available in this authoring environment; use dry-run and optionally test one model to validate the actual shared cache/environment before starting both full batches.
