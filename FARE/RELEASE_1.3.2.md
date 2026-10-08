# FARE v1.3.2 — Same-GPU Training→Evaluate + Optional Pilot Test

## Changes

**Default Gadi submission:** One PBS job **per target**, not one Train job plus a held Evaluate job. After successful Train, the same `train_formal.pbs` GPU allocation starts `scripts/evaluate.py` as a separate process. The training model and memory CUDA allocations are freed first. Evaluation failure makes the Train PBS job fail rather than printing a false PASS.

**Checkpoint selection remains Validation-based.** Pilot/Test uses only the selected `candidate_best.pt`; the held-out Test split is never used in the training loss, memory, retrieval labels, early stopping, alpha tuning, or checkpoint selection. Running a Pilot Test for debugging/analysis is technically supported, but if its results influence architecture or hyperparameter selection, later evaluation on the **same** Test cannot be presented as an unbiased final result.

**PBS Conda initialization:** `pbs/_activate_tfp.sh` explicitly sources `conda.sh` from the parent of `Speed_LLM_RAG` (e.g. `/scratch/mh20/zw1165/miniconda/etc/profile.d/conda.sh`) or from `FARE_CONDA_SH`, then runs `conda activate TFP`. A bare `conda` executable is not considered sufficient.

No STLAformer/FARE architecture, model weights, losses, or training hyperparameters changed. Existing v1.3.1 checkpoints remain compatible.

## Gadi commands

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"

# 2 submitted GPU jobs total, both Auto Evaluate only Validation (default Pilot).
python -u FARE/scripts/submit_gadi.py train \
  --dataset TraffiDent_Net_06 --targets speed flow --seed 10 \
  --config FARE/configs/pilot.yaml

# 2 submitted GPU jobs total, each Train -> Validation -> Test (s20/s40/s60/s90).
python -u FARE/scripts/submit_gadi.py train \
  --dataset TraffiDent_Net_06 --targets speed flow --seed 10 \
  --config FARE/configs/pilot.yaml --eval-splits val test

# Formal Train -> Test s20/s40/s60/s90 in the same GPU job.
python -u FARE/scripts/submit_gadi.py train \
  --dataset TraffiDent_Net_06 --targets speed flow --seed 10

# To preview without submitting:
python -u FARE/scripts/submit_gadi.py train \
  --dataset TraffiDent_Net_06 --targets speed flow --seed 10 \
  --config FARE/configs/pilot.yaml --eval-splits val test --dry-run
```

Test severity subset: `--severities 20 40`. To skip Evaluate entirely: `--no-auto-evaluate`. `--no-auto-evaluate` must not be combined with `--eval-splits`.

### Output

Each target writes:

```
FARE_exps/<target>/<dataset>/pilot/<run_id>/
  candidate_best.pt
  memory.npz
  posttrain_evaluation.json  # pass/failed, completed splits, pilot_test_exploratory
  evaluation_candidate/
    val_full_summary.json
    val_full_predictions.npz
    s20/s20_full_summary.json
    s40/s40_full_summary.json
    s60/s60_full_summary.json
    s90/s90_full_summary.json
```

The TEST folders exist only when Test is requested. All evaluation outputs are checkpoint-exact, split-separated, and use the existing Base_SupportBench cache/metrics contract.

### Operational caution

**Training plus all selected evaluations must fit the same 48-hour GPU allocation.** If training reaches the PBS walltime, the Evaluate step cannot run and should be submitted manually later. Pilot with four severities adds compute; use fewer severities for a purely operational test if needed. A separate `submit_gadi.py evaluate` command remains supported. All pre-existing Gadi job submissions are unchanged by updating files on disk: PBS has already stored their submitted scripts; cancel/resubmit if you specifically need the new workflow.

The optional `--finalize` OOF→Calibration→Final Evaluate workflow remains a **separate, explicitly requested** dependent chain, because its out-of-fold training can exceed a single GPU walltime. Do not use `--finalize` if the requirement is strictly one PBS job per target.

## Local CLI / manual Evaluate

Direct local `train.py --config FARE/configs/pilot.yaml --auto-evaluate --eval-splits val test --eval-severities 20 40 60 90` uses the same post-training code path. Manual Gadi `submit_gadi.py evaluate --checkpoint ... --split test --severities 20 40 60 90` remains unchanged.

## Test coverage

- Fake qsub: one PBS job per target, no `-W` afterok when using normal `train`.
- Fake PBS: `--eval-splits` and `--eval-severities` propagated to `train.py`.
- Real `train.py` with fake model/data/evaluator: Train → Val → Test, checkpoint/memory reuse, unchanged `PBS_JOBID`, manifest PASS, and nonzero failure propagation.
- Synthetic FARE training/ablation/gradient suites remain available.
