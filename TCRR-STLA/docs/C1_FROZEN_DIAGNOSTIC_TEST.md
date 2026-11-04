# TCRR-STLA v7.1 C1 — ONE frozen CPU diagnostic + held-out Test

## Purpose and non-negotiable protocol

This release adds a **separate, source-hash-checked evaluation program** without editing training source (`tcrr_stla/v7/*.py`, `scripts/run_v7.py`, existing PBS or YAML). It loads the **frozen best_c1.pt** from the completed official run. It **never trains** or selects parameters from held-out Test. Existing `model_lock.json`, joint C2 checkpoints and `test/s*.json` are neither required nor modified.

Selection diagnostic includes Learned vs Uniform/Permute/Top-1/None (holding learned None mass fixed), normal-component-only and CF-component-only (post-mixture Shapley attribution), shuffled CF, zeroed support inputs, target/predicted weight entropy, utility ranking/Spearman, pairwise accuracy, harm AUC, failure-family/coverage/horizon slices, harmful-query rate, oracle regret for diagnosis only, and paired time-block bootstrap. These *frozen interventions are not equivalent to separately trained ablations*. No diagnostic output is used to modify the checkpoint.

Test evaluates only four **predeclared** comparators per severity: original frozen STLAformer (no Mask Adapter), S0 frozen adapted Backbone (called `backbone` in original v7 trainer), Normal, and C1. Reports Overall MAE, RMSE, Masked MAE, H12 MAE, per-horizon MAE, failure family and coverage groups, query-level csv and paired confidence intervals. Four severities are equally weighted in the final macro average.

All Test queries use the original TRAIN-only candidate memory. Missing Test Teacher/Replay cache is generated using the original source-hashed v7.1 pipeline, stored only in `teacher_cache/test_sXX/` and `paired_replay/test_sXX/`, and resumed using the existing per-chunk `DONE.json` + file SHA256. No labeled Test target appears in inference, selection or replay retrieval. Labels are read only after frozen forward to compute metrics.

## What is in the archive

This archive contains the **full original v7.1.0 project** plus these add-ons:

- `scripts/c1_frozen_analysis.py` — `verify|diagnostic|test|all|aggregate`, CPU or CUDA, no model changes
- `pbs/c1_frozen_cpu.pbs` — Gadi `normal` CPU queue (12 CPUs, 64GB, 48h)
- `pbs/c1_frozen_gpu_test.pbs` — optional V100 **Test-only** cache preparation and frozen inference
- `pbs/submit_c1_frozen.sh` — local prequeue verification and PBS submission, source SHA handshake
- `tests/test_c1_frozen_addon.py` — numeric/masking/bootstrap contracts
- `docs/C1_FROZEN_DIAGNOSTIC_TEST.md` — this guide

**User's model/data/checkpoints not included.** Paths assumed below are the previously-used Gadi paths; adjust checkpoint paths if yours differ.

## Installation without disturbing an already trained run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/TCRR-STLA:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
export TFP_ENV_ROOT=/g/data/mh20/zw1165/envs/TFP

# Upload archive to the repository root first, then install WITHOUT overwriting
# any existing original v7.1 source or checkpoints.
unzip -n TCRR_STLA_v7.1.0_C1_Frozen_Diagnostic_Test_Full.zip
python -m pytest -q TCRR-STLA/tests
```

Do NOT modify source files while a PBS job is queued or running. `verify` checks the *recorded run* code digest and existing checkpoint SHA256; if another implementation was used in your Gadi run it deliberately fails instead of guessing. Never delete `run_identity.json` or manipulate its hash to pass a check.

## Before submitting (CPU preflight)

```bash
export STLA_SPEED="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export STLA_FLOW="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"

export RUN_SPEED="$PWD/TCRR-STLA_exps/speed/incomplete/TraffiDent_Net_06/v7_1_0_c1/formal/seed10"
export RUN_FLOW="$PWD/TCRR-STLA_exps/flow/incomplete/TraffiDent_Net_06/v7_1_0_c1/formal/seed10"

python -u TCRR-STLA/scripts/c1_frozen_analysis.py \
  --mode verify --target speed --backbone-checkpoint "$STLA_SPEED" --run-dir "$RUN_SPEED"
python -u TCRR-STLA/scripts/c1_frozen_analysis.py \
  --mode verify --target flow --backbone-checkpoint "$STLA_FLOW" --run-dir "$RUN_FLOW"
```

`verify` checks training version/code SHA, S0 and C1 identities, TRAIN functional/road graph identities, Selection Teacher and Replay file hashes, and all four Test **source data shapes**. It does **not** need a GPU or create Test cache data.

## Recommended: ONE planned CPU-only run per target

Both following submissions execute one full Selection diagnostic then the four Severity Test evaluations, using the existing frozen checkpoint and original replay data preparation. The two jobs are independent; either can resume on the same run directory after preemption. They don't need to run simultaneously.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export MODE=all
export QUEUE=cpu
bash TCRR-STLA/pbs/submit_c1_frozen.sh speed "$STLA_SPEED" 10
bash TCRR-STLA/pbs/submit_c1_frozen.sh flow  "$STLA_FLOW"  10
```

**Important:** Gadi may apply project-level queue and walltime limits; if `qsub` rejects the 48h CPU walltime, edit only the **NEW** `pbs/c1_frozen_cpu.pbs` resource lines and resubmit. Source digest of the frozen training checkpoint is not affected by this new PBS file. **First-time CPU Test replay generation may take substantially longer than the Selection diagnostic**. Every completed chunk is SHA-checked/reused on the next run. If runtime is excessive, switch to the faster GPU Test-only option below *after the CPU diagnostic finishes*.

## Separate CPU diagnostic and CPU TEST (optional)

```bash
MODE=diagnostic QUEUE=cpu bash TCRR-STLA/pbs/submit_c1_frozen.sh speed "$STLA_SPEED" 10
MODE=diagnostic QUEUE=cpu bash TCRR-STLA/pbs/submit_c1_frozen.sh flow  "$STLA_FLOW"  10

# After diagnostic, use one per-target Test job (no model retraining):
MODE=test QUEUE=cpu bash TCRR-STLA/pbs/submit_c1_frozen.sh speed "$STLA_SPEED" 10
MODE=test QUEUE=cpu bash TCRR-STLA/pbs/submit_c1_frozen.sh flow  "$STLA_FLOW"  10
```

## Faster Test cache path (optional GPU PBS, only if CPU is too slow)

```bash
MODE=test QUEUE=gpu bash TCRR-STLA/pbs/submit_c1_frozen.sh speed "$STLA_SPEED" 10
MODE=test QUEUE=gpu bash TCRR-STLA/pbs/submit_c1_frozen.sh flow  "$STLA_FLOW"  10
```

GPU jobs are **inference only**. They build the same immutable source-bound Test cache, resume incomplete chunks and use the same frozen checkpoint. Use only one active job per target/run at a time; otherwise two jobs can contend for the same Test cache directory.

## Direct interactive CPU execution (no PBS, useful for small cached checks)

```bash
python -u TCRR-STLA/scripts/c1_frozen_analysis.py \
  --mode diagnostic --target speed --backbone-checkpoint "$STLA_SPEED" \
  --run-dir "$RUN_SPEED" --device cpu --threads 8 --eval-batch-size 8

# Use existing prepared Test caches only; fails fast if a cache is missing.
python -u TCRR-STLA/scripts/c1_frozen_analysis.py \
  --mode test --cache-only --target speed --backbone-checkpoint "$STLA_SPEED" \
  --run-dir "$RUN_SPEED" --device cpu --threads 8
```

`--severities 20:40:60:90` accepts colons to be safely carried by `qsub -v`; direct CLI also accepts commas. `--force-output` recomputes report files for the **same frozen checkpoint** after explicit identity validation; do not set it in first-time formal evaluation. For smoke-only synthetic runs see `tests/integration_synthetic_v7.py`; those numbers are not research metrics.

## Output locations

```text
TCRR-STLA_exps/{speed|flow}/incomplete/TraffiDent_Net_06/v7_1_0_c1/formal/seed10/
  best_c1.pt                    # unchanged
  run_identity.json             # unchanged
  diagnostic_c1/
    summary.json                # candidate interventions, weight/utility/harm, CI
    query_methods.csv           # per-query paired errors and grouping keys
  test_c1_only/
    s20.json
    s20_query_methods.csv
    s40.json
    s40_query_methods.csv
    s60.json
    s60_query_methods.csv
    s90.json
    s90_query_methods.csv
    summary.json                # equal-severity averages
    RESULTS.md                  # readable comparison table
```

Compute logs:

```bash
find TCRR-STLA_logs -path '*v7_1_0_c1/frozen/*' -type f -name '*.log'
qstat -u "$USER"
```

## Interpretation and limitations

1. C1 Test results are **not yet known** until the user's actual Gadi experiment completes. Current synthetic e2e numbers are only API/shape integration checks.
2. Original STLAformer is constructed with the official frozen Backbone and an exactly zeroed Mask Adapter; S0 is the cached output from the user's trained Mask Adapter. Same Test input, output scaling and mask is used for each baseline.
3. `shuffled_cf`, `support_occlusion`, `normal_component_only`, and `cf_component_only` are **frozen interventions**, not re-trained model ablations. They are excluded from TEST method comparison.
4. Selection diagnostics use the original fixed chronological Selection partition (first 60% distinct validation windows); they do not inspect the Locked Check or use TEST outcomes for selection.
5. A bootstrap interval groups queries into blocks of 288 5-min steps (1 day), with alternate custom width via `--block-steps`; it is descriptive and does not cure all temporal dependence.
6. A 48h CPU job may require requeueing because TEST cache preparation is expensive; each successful cache chunk and severity report is reused without repeated training.
