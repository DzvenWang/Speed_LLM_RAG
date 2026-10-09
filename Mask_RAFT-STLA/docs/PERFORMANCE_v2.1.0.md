# Mask_RAFT / Mask_RAST-STLA v2.1.0 — performance and reliability release

## Important: v2.1 is new code, not a resumption of a v2.0 optimization state

* Keeps STLAformer and the historical Future/Residual Retrieval research objective.
* Continues to use the user's original s40-trained STLAformer checkpoints; setting `BACKBONE_CHECKPOINT` explicitly is mandatory for comparable Formal experiments.
* Does NOT implement an imputer, shorten the history/horizon, subsample training queries, or reduce retrieval candidate counts.
* **RAFT fixed-teacher mode does change the model definition:** old v2.0 replay used the currently adapted STLA, while v2.1 (default) uses the *original frozen* s40 STLA as a consistent historical replay teacher. This enables exact reuse of replay values across epochs. Disable `runtime.frozen_replay_teacher` to retain dynamic behavior; dynamic mode will be far slower.
* `runtime.amp=true` + `training.batch_size=32` change numerical precision / optimizer step count, so do not describe them as mathematically identical training to batch-16 FP32 v2.0.
* All improvements to accuracy are unverified. Formal GPU execution is required.

## Main code changes

### RAFT
* `mask_raft_stla/replay_cache.py` (NEW): immutable teacher replay bank for all Train/Validation queries, using every original Top-3 x 3 resolutions candidate. Float16 `.npy` mmap storage with optional GPU residency and cache source fingerprint. Rebuild on checkpoint/index/mask/Train Memory change.
* `trainer.py` / `evidence.py`: keep full candidate sequence / mask-conditioned residual, use bank instead of nine STLA forwards per training sample in every epoch.
* `training.batch_size: 32`, `retrieval.replay_batch_size: 64`, `runtime.preload_retrieval_memory_gpu: true`, AMP FP16.
* The **original 50,000 queries** and **full 80 epochs** remain.

### RAST
* `data.py`: GPU residency for Train historical Future and validity masks, in addition to history, time and day features. Avoid per-batch CPU→GPU copies on this fast path.
* `training.batch_size: 32`, AMP FP16.
* Memory-index rebuild policy and ALL Temporal / Regional / None candidate evidence remain unchanged. Validation index is rebuilt at the correct end-of-epoch model state; no stale index used for model selection.

### Both
* `checkpoint.py` (NEW): atomic per-epoch `best.pt` + `last.pt`, optimizer/AMP scaler/RNG/history restoration via `--resume`.
* `trainer.py`: records `train_seconds`, `val_seconds`, `train_samples_per_second`, `peak_gpu_gib` in EPOCH logs and history CSV.
* `pbs/formal.pbs`: `RESUME_CHECKPOINT` and `BATCH_SIZE` override, timestamped logs, 12 CPUs, one V100, 32GB memory, 48h.
* `scripts/evaluate.py`: uses checkpoint's AMP config; RAFT evaluation explicitly uses the same frozen replay teacher as training; RAST eval preloads entire Train Repository when GPU memory permits.

## Expected resources (not yet measured on actual v2.1 Gadi jobs)

Net-06, Seed=10, 50,000 Train/5,000 Val, 80 epochs, one Tesla V100-SXM2 32GB per run:

| Run | Old v2.0 epoch | New v2.1 epoch estimate | New total 80 epochs incl. initialization/eval |
|---|---|---|---|
| RAFT Speed | 45–50 min estimated | 8–18 min | ~12–27 GPU hours |
| RAFT Flow | 45–50 min estimated | 8–18 min | ~12–27 GPU hours |
| RAST Speed | 12.2 min measured | 7–12 min | ~10–18 GPU hours |
| RAST Flow | 12.3 min measured | 7–12 min | ~10–18 GPU hours |

Ranges are planning estimates, **not benchmark results**. RAFT includes one-time frozen replay construction which can require ~20–90 minutes on top of batch training. The overall estimate includes approximately 0.5–2 hours for retrieval index construction and final four-severity evaluation when needed. Fine-tuning phases 11–35 and 36–80 may have different costs and RAM demands. Early stopping may finish faster. The GPU resource is mostly compute-saturated already; more memory use does not guarantee proportional speedup.

RAFT cache disk requirements Net-06: Train [50000,9,12,388] float16 ≈ 3.90 GiB; Val [5000,9,12,388] ≈ 0.39 GiB per target (Speed and Flow each need their own cache). GPU cache can add ~4.3 GiB memory; fallbacks use CPU memmap.

## Stop previous four PBS jobs (run manually on Gadi)

```
qdel 180882769 180882771 180882795 180882797
qstat -u "$USER"
```

## Deploy

Back up the two old code directories if they exist, then unpack `Mask_RAFT_RAST_STLA_v2.1.0_COMPLETE.zip` into `/scratch/mh20/zw1165/Speed_LLM_RAG`, preserving Base_SupportBench and Backbone as they are. The archive contains `Mask_RAFT-STLA/` and `Mask_RAST-STLA/`. Checkpoint outputs in `*_exps` are not touched.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export STLA_S40_SPEED="$(pwd)/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export STLA_S40_FLOW="$(pwd)/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
test -f "$STLA_S40_SPEED" && test -f "$STLA_S40_FLOW" || exit 1

qsub -N MRAFT21_speed -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,BACKBONE_CHECKPOINT="$STLA_S40_SPEED" Mask_RAFT-STLA/pbs/formal.pbs
qsub -N MRAFT21_flow  -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,BACKBONE_CHECKPOINT="$STLA_S40_FLOW"  Mask_RAFT-STLA/pbs/formal.pbs
qsub -N MRAST21_speed -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,BACKBONE_CHECKPOINT="$STLA_S40_SPEED" Mask_RAST-STLA/pbs/formal.pbs
qsub -N MRAST21_flow  -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,BACKBONE_CHECKPOINT="$STLA_S40_FLOW"  Mask_RAST-STLA/pbs/formal.pbs
```

## Resume training after 48h/preemption

`best.pt` and `last.pt` are written after every completed epoch. Resume with the exact `last.pt` path from that run:

```bash
qsub -N MRAFT21_speed_resume -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,BACKBONE_CHECKPOINT="$STLA_S40_SPEED",RESUME_CHECKPOINT="/absolute/path/to/Mask_RAFT-STLA_exps/speed/TraffiDent_Net_06/formal/your_run/last.pt" Mask_RAFT-STLA/pbs/formal.pbs
```

`--resume` will reject v2.0.0 checkpoints and incompatible dataset/seed/retrieval/backbone configurations. For RAST, per-best Memory Index is also saved to enable evaluation if stopped mid-run.

## Testing performed without actual Backbone/sensing cache

- Static Python compile: PASS for both projects.
- Existing RAFT/RAST torch unit tests: PASS.
- Replay Bank fixed-teacher vs dynamic replay equivalence (FP16 tolerance), cache hit, source fingerprint invalidation, atomic best/last and restore: PASS on synthetic tensors.
- Two-epoch simulated full trainers with mock STLA and mock SupportBench cache: PASS for both projects, including checkpoint and validation saving.
- **Not performed**: real Gadi GPU runs, real STLAformer formal dataset fit, CUDA AMP numeric convergence, observed v2.1 acceleration. Use the first epoch's logged timing before trusting the walltime estimate.
