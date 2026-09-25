# Explicit PatchSTG checkpoint evaluation

This path evaluates any explicitly supplied PatchSTG `best.pt` on the current
20/40/60/90 evaluation caches. It does **not** edit or depend on
`Backbone_exps/FROZEN_BACKBONE.json`, so historical evaluation cannot contaminate
the later official frozen-model evaluation.

Current historical checkpoint to evaluate:

```text
Backbone_exps/PatchSTG/TraffiDent_Net_06/formal/20260913_150736_seed10/best.pt
```

## Local / Windows: all severities

Run from the `Speed_LLM_RAG` root:

```powershell
python Backbone\scripts\run_checkpoint_evaluations.py --checkpoint Backbone_exps\PatchSTG\TraffiDent_Net_06\formal\20260913_150736_seed10\best.pt --expected-seed 10
```

Run one severity only:

```powershell
python Backbone\scripts\evaluate_checkpoint.py --checkpoint Backbone_exps\PatchSTG\TraffiDent_Net_06\formal\20260913_150736_seed10\best.pt --severity 40 --expected-seed 10
```

The checkpoint path may be absolute, relative to the current directory, or relative
to the `Speed_LLM_RAG` root.

## GADI: all severities

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
bash submit_checkpoint_evaluations.sh \
  Backbone_exps/PatchSTG/TraffiDent_Net_06/formal/20260913_150736_seed10/best.pt \
  10
```

One severity only:

```bash
qsub -v \
SEVERITY=40,SEED=10,CHECKPOINT=/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_exps/PatchSTG/TraffiDent_Net_06/formal/20260913_150736_seed10/best.pt \
pbs/evaluate_checkpoint_severity.pbs
```

Live logs are written directly under:

```text
/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_logs/
```

with names such as:

```text
PatchSTG.checkpoint_eval.s40.seed10.<jobid>.<timestamp>.live.log
```

Every live-log line is timestamped by `timestamp_stream.py`.

Results are intentionally separated from the official frozen evaluation:

```text
Backbone_exps/PatchSTG/TraffiDent_Net_06/checkpoint_evaluation/20260913_150736_seed10/s20/...
Backbone_exps/PatchSTG/TraffiDent_Net_06/checkpoint_evaluation/20260913_150736_seed10/s40/...
Backbone_exps/PatchSTG/TraffiDent_Net_06/checkpoint_evaluation/20260913_150736_seed10/s60/...
Backbone_exps/PatchSTG/TraffiDent_Net_06/checkpoint_evaluation/20260913_150736_seed10/s90/...
```

## Later new model

For an ad-hoc evaluation of a new `best.pt`, use the same explicit-checkpoint
commands with the new checkpoint path. Results are automatically separated by the
checkpoint's parent formal-run directory name.

For the final official experiment, keep using the normal formal/freeze path and
`FROZEN_BACKBONE.json`, then run the existing frozen severity evaluation workflow.
This keeps historical results and final model results distinct.


## Evaluation access policy

Backbone v1.0.3 removes the Protocol grant-file gate. Evaluation is allowed whenever the evaluation cache is frozen and passes cache/hash/contract validation.

## Logging

`run_checkpoint_evaluations.py` automatically creates `Backbone_logs/PatchSTG.checkpoint_eval.all.seed<seed>.<stamp>.live.log` locally. GADI PBS jobs create per-severity timestamped live logs in the same directory.
