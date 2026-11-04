# CoRSTA-D1: one diagnostic job per target

**Independent read-only add-on to CoRSTA v0.1.0 / Hotfix2.** Run **one GPU PBS job for Speed and one for Flow**; each job completes all D1 analyses. Do not submit six individual experiments. No model retraining, no TEST access, no cache rebuilding.

## Requirements

- Same Gadi project root as original formal experiment: `/scratch/mh20/zw1165/Speed_LLM_RAG`.
- `TFP` conda environment, original Backbone and Base_SupportBench installed.
- Completed `CoRSTA_exps/{speed,flow}/incomplete/TraffiDent_Net_06/formal/seed10/` with `best_warmup.pt`, `best_joint.pt`, `last_joint.pt`, `model_lock.json`, and a valid bootstrap (`best_c1.pt`, TRAIN/VAL caches, TRAIN/VAL paired Replay, TRAIN graph).
- The original STLA checkpoint and original `CoRSTA/configs/*.yaml`; do not change their contents.

## First: optional CPU check-only (NO forward, no GPU)

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export BACKBONE_MODE=incomplete

python -u CoRSTA/diagnostics/run_d1.py --check-only \
  --target speed --dataset TraffiDent_Net_06 --seed 10 \
  --backbone-checkpoint "$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"

python -u CoRSTA/diagnostics/run_d1.py --check-only \
  --target flow --dataset TraffiDent_Net_06 --seed 10 \
  --backbone-checkpoint "$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
```

## Actual one-shot GPU PBS submissions

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"

qsub -N CoRSTA_D1_speed \
  -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_MODE=incomplete,BATCH_INITIAL=64,D1_GRAD_QUERIES=128,D1_GRAD_BATCH=8,BACKBONE_CHECKPOINT="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt" \
  CoRSTA/pbs/diagnose_d1.pbs

qsub -N CoRSTA_D1_flow \
  -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_MODE=incomplete,BATCH_INITIAL=64,D1_GRAD_QUERIES=128,D1_GRAD_BATCH=8,BACKBONE_CHECKPOINT="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt" \
  CoRSTA/pbs/diagnose_d1.pbs
```

The PBS walltime is **8 hours requested**, not an estimate of actual usage. Each task requests a single V100 32GB, 12 CPUs, 64GB system RAM. The inference batch probes 64, optionally 128 if there is headroom, otherwise 32/16. Gradient batch is fixed 8 to control the cost of 4 objective-specific gradient calculations.

If the job encounters an inference OOM, it restarts the full Selection-only evaluation with the next smaller allowed batch, never modifies original checkpoints. The gradient audit is read-only. Every line in the generated live log receives a UTC timestamp.

## Outputs

A single target job creates:

```
CoRSTA_diagnostics/D1_Net06_seed10/speed/
    source_manifest.json
    batch_probe.json
    selection_query_metrics.csv
    prediction_metrics.json
    prediction_headroom.json
    candidate_utility.json
    fusion_statistics.json
    c2_interventions.json
    horizon_mae.json
    student_drift.json
    gradient_conflict.json
    architecture_decisions.json
    FINAL_DIAGNOSIS.md
    D1_COMPLETE.json
```

Flow uses the analogous `flow/` folder. Logs:

```
CoRSTA_logs/diagnostics/D1/speed/incomplete/
CoRSTA_logs/diagnostics/D1/flow/incomplete/
```

**Full Selection** is used by default. `--max-selection`/`D1_MAX_SELECTION` is for debugging only and marks the report as partial, *not* a valid full diagnostic.

After both tasks finish:

```bash
bash CoRSTA/diagnostics/pack_d1.sh
```

Upload the resulting `CoRSTA_diagnostics/CoRSTA_D1_speed_flow_*.tar.gz`. It contains compact JSON/CSV, code and logs only, not `.pt`, `.npy` or `.npz`.

## Diagnostic definitions and limitations

- `teacher_backbone` is the original S0 prediction from the fixed teacher cache. `bootstrap_c1` uses the **original best_c1**, not Warm-up's changed C1. `warmup_refined` and `last_joint_refined` use their respective saved checkpoints. All are evaluated on the **same validation Selection windows**, not Locked Check or TEST.
- Prediction Oracle is computed at query and element levels over `{teacher_backbone, bootstrap_c1, warmup_refined}`. It uses real Selection future labels solely to measure upper bounds; it **must not** be deployed, trained to match directly, or reported as a prediction result.
- Candidate Oracle applies existing bounded transported corrections and a minimum allowed candidate valid-element coverage (`0.50` by default); unavailable candidate positions use the baseline prediction. Compare `transport_normal_only` and `transport_normal_plus_cf` for the marginal value of CF.
- Inference interventions disable C1/C2 Bridge, graph branches or all feedback **without retraining** and report model sensitivity, not fully retrained ablation gains.
- Fusion audit measures actual relative norm of injected C1/C2 features; graph mix reflects current trained gates and is not a causal explanation.
- The gradient audit reads **TRAIN only**, evaluates `best_warmup` and `last_joint` on the same ~128 deterministic TRAIN queries, computes 4 separate objective gradients, and **never steps an optimizer**. Gradient cosine values are snapshots, not the historical training trajectory.
- Student/module swap tests are diagnostic hybrids (`joint_student_warm_rest`, `warm_student_joint_rest`), potentially out-of-distribution; their errors only suggest which updated component drives regression.
- `architecture_decisions.json` contains **heuristic** recommendations for v0.2, not hypothesis-test proofs. The overall research acceptance target is macro-MAE gain >=5% against the strongest matching baselines, separately for Speed and Flow.

### Artifact immutability

D1 adds files only under `CoRSTA/diagnostics/` and `CoRSTA/pbs/diagnose_d1.pbs`, with documentation/tests. It intentionally **does not modify** original `CoRSTA/scripts/*.py`, `CoRSTA/corsta/*.py`, or `CoRSTA/configs/*.yaml`. These are used to construct the old training source digest and are essential to checkpoint identity. The original `model_lock.json`, TEST results, weights and Replay are never changed by D1.

## CPU PBS full diagnostic (GPU-free alternative)

The original GPU D1 job remains available. For the **full** D1 on a CPU compute node,
use `CoRSTA/pbs/diagnose_d1_cpu.pbs`. This performs the SAME Selection-only evaluation,
C1 candidate audit, inference interventions and TRAIN gradient audit; it is **not**
`--check-only`. Expect materially longer runtime than on a V100. No retraining occurs.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"

qsub -N CoRSTA_D1_CPU_speed \
  -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_MODE=incomplete,BATCH_INITIAL=64,D1_GRAD_QUERIES=128,D1_GRAD_BATCH=8,BACKBONE_CHECKPOINT="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt" \
  CoRSTA/pbs/diagnose_d1_cpu.pbs

qsub -N CoRSTA_D1_CPU_flow \
  -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_MODE=incomplete,BATCH_INITIAL=64,D1_GRAD_QUERIES=128,D1_GRAD_BATCH=8,BACKBONE_CHECKPOINT="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt" \
  CoRSTA/pbs/diagnose_d1_cpu.pbs
```

CPU PBS requests 24 CPUs, 96 GB memory, 24 h walltime (subject to Gadi queue
limits). PyTorch uses 16 intra-op threads by default, leaving cores for I/O;
`D1_CPU_THREADS` and `D1_WORKERS` can be overridden. The full CPU run uses a
conservative inference batch of 16 even if `BATCH_INITIAL=64`; `D1_GRAD_BATCH=8`
is independent. This differs from the GPU memory-probe/autobatch policy.

The CPU and GPU D1 outputs are saved to the same target-specific directory,
so do **not** submit both CPU and GPU diagnostics for the same target simultaneously.
If a target D1 already completed, use a new `D1_OUTPUT_DIR`; never overwrite it.
No measurements on real Gadi CPU resources are yet available; the 24h request is a
limit, not a throughput forecast. CPU is often cheaper in GPU allocation but may
consume considerable CPU service units due to full-selection multi-pass inference.
