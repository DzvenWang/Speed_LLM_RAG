# Reference Backbone migration — Baseline v1.4.0

## Old lifecycle
```text
Backbone Formal -> freeze_checkpoint.py -> FROZEN_BACKBONE.json -> Baseline reads selected_model
```

## New lifecycle
```text
Backbone Formal -> best.pt (+ model-local latest pointer)
Baseline Formal --backbone STLAformer -> resolve compatible STLAformer best.pt
                                      -> record exact path + SHA256
Baseline evaluation                  -> reload exact recorded checkpoint
```

## Selection precedence
1. `--backbone-checkpoint <path>` when explicitly supplied.
2. `BASELINE_BACKBONE_CHECKPOINT` environment variable.
3. `<formal>/LATEST_CHECKPOINT.txt`.
4. `<formal>/LATEST_RUN.txt/best.pt`.
5. Latest compatible `<timestamp>_seed10/best.pt` under the explicitly selected model.

Every candidate is validated against model name, target, seed, node count/model contract, and the shared Formal cache identity before use.

## Examples
```bash
qsub -v METHOD=ImputeFormer,TARGET=speed,SEED=10,BACKBONE=STLAformer -N Impute_speed Baseline/pbs/methods/formal_primary.pbs
qsub -v METHOD=LRTC-TMCP,TARGET=flow,SEED=10,BACKBONE=STLAformer -N LRTC_flow Baseline/pbs/methods/formal_primary.pbs
```

Exact-checkpoint local example:
```bash
python -u Baseline/baseline/runners/lrtc_tmcp_runner.py \
  --stage formal --seed 10 --backbone STLAformer \
  --backbone-checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/<run>/best.pt
```


## v1.4.1 matched retrieval controls

RAST-STLAformer and RAFT-STLAformer now use the same explicit reference-Backbone provenance contract. They do not inspect a global frozen model. Each Formal run records the exact STLAformer checkpoint path, SHA256, model name and reference seed; evaluation reloads those values from the completed Formal run.


## Portable paths (v1.4.2)
Reference checkpoint paths are stored relative to `SPEED_LLM_RAG_ROOT` (for example `Backbone_exps/speed/STLAformer/.../best.pt`). They are not tied to a Windows drive or a Gadi `/scratch/...` prefix. Old absolute paths remain readable and are rebased to the current project root when the original host path does not exist.

## Matched retrieval control safety (v1.4.3)

`RAFT-STLAformer` starts exactly from the selected STLAformer forecast and learns a single scalar `alpha` in `final = base + alpha * (raft_fused - base)`. The scalar is initialized to zero, saved in the Formal checkpoint, and restored for evaluation. v1.4.4 extends the same compatibility rule to RAST and to PatchSTG matched controls; native RAFT/RAST remain unchanged.


### v1.4.4 retrieval-control rule

`RAFT-PatchSTG`, `RAFT-STLAformer`, `RAST-PatchSTG`, and `RAST-STLAformer` are matched-backbone controls. They use a zero-initialized scalar residual calibration around the selected reference Backbone. Native `RAFT` and native `RAST` remain unchanged.
