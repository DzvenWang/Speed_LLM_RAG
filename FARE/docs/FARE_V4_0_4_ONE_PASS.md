# FARE v4.0.4 — full source + one-pass diagnostic

This is the **complete FARE project** from the supplied `FARE(1).zip` with integrated native source diagnostics. The original v4.0.3 train, memory, model, evaluate and Gadi formal training functionality remains intact; no independent no-evidence training is run.

## New features

- `scripts/diagnose_once.py`: one GPU job, sequential speed & flow, four severities, five actual failure families, frozen same-checkpoint counterfactual predictions.
- `fare/diagnostics_once/native_export.py`: **real** FARE `prepare`, `SupportDataset`, `FARE.intervene_cached`, and `predict_with_selection` integration. No user-written plugin needed.
- `scripts/diagnose_preflight.py`: CPU check **before qsub** of both FARE checkpoints, backbone/memory paths, resolved evaluation caches and all five manifest families.
- `pbs/diagnose_once.pbs`: one V100, 12 CPUs, 32 GB RAM, 48 hours maximum; two targets run sequentially in the same job.
- Exported full, no-evidence, hidden-only, forecast-only, no gate, null, similarity, random, shuffled, no support demand, no direction, no propagation alignment, no set interaction, and no counterfactual utility.
- Selected-slot oracle computed by isolating each selected evidence member with identical frozen FARE parameters. **This is not an exhaustive search-pool oracle.**
- Deterministic query selection per target × severity × actual manifest family; provenance SHA256 and leakage timestamp checks; incremental resumable per-stratum NPZ exports and structured reports.

## Mandatory decision on data split

This version defaults to `split=test` **only** because the existing Base_SupportBench severity caches are the test evaluation caches (`cache_evaluation_sXX.yaml`). The config explicitly sets `allow_exploratory_test=true`. These diagnostic numbers are **exploratory** and must **not** be treated as unbiased test-set performance after decisions based on them. Do not claim independent held-out architectural validation using this run. To use true validation for architecture selection, create severity-specific validation caches and set `split=val`, `allow_exploratory_test=false` in `configs/diagnose_once.json`; no silent test fallback is implemented.

## Install safely

From the `Speed_LLM_RAG` root, **after earlier PBS jobs using FARE finish**:

```bash
# Keep original experiment/checkpoints and caches; replace source only.
mv FARE FARE_backup_before_v404
unzip FARE_v4.0.4_Full_OnePass_Integrated.zip
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/FARE:${PYTHONPATH:-}"
conda activate TFP
python -m pytest -q FARE/tests
```

## Checkpoint selection — actual trained FARE, NOT Backbone STLA

```bash
find FARE_exps/speed -path '*/best.pt' -print
find FARE_exps/flow -path '*/best.pt' -print
export FARE_SPEED_CHECKPOINT="$(pwd)/FARE_exps/speed/TraffiDent_Net_06/formal/<YOUR_RUN>/best.pt"
export FARE_FLOW_CHECKPOINT="$(pwd)/FARE_exps/flow/TraffiDent_Net_06/formal/<YOUR_RUN>/best.pt"
```

Paths above are placeholders until the user's Gadi run IDs are known. They must be real FARE trained `best.pt` checkpoints; the code reads their original config and frozen Backbone and Memory paths. If the original checkpoint's dependencies were moved, restore them before submission.

## Zero-GPU-queue preflight (run on Gadi login node)

```bash
python -u FARE/scripts/diagnose_preflight.py \
  --config FARE/configs/diagnose_once.json \
  --speed-checkpoint "$FARE_SPEED_CHECKPOINT" \
  --flow-checkpoint "$FARE_FLOW_CHECKPOINT"
```

Fails before qsub if any severity cache lacks the chosen split or manifest family, or if a checkpoint/dependency is missing. The true STLA forward can only be tested on a GPU; the CPU check cannot guarantee actual cluster inference success.

## One PBS job, not two

```bash
export FARE_PYTHON="$(python -c 'import sys; print(sys.executable)')"
qsub -N FARE404_DIAG \
  -v FARE_SPEED_CHECKPOINT="$FARE_SPEED_CHECKPOINT",FARE_FLOW_CHECKPOINT="$FARE_FLOW_CHECKPOINT",FARE_PYTHON="$FARE_PYTHON",SEED=10 \
  FARE/pbs/diagnose_once.pbs
```

Runs exactly once per target within one PBS allocation. Defaults to 64 sampled queries in each of 40 target×severity×family strata (2560 total across two targets). The query sample is deterministic and stratified **by the actual manifest family**. Headline full-test metrics from the pre-existing original FARE full evaluation remain separate from these sampled counterfactuals.

Optional environment variables: `DIAG_MAX_QUERIES`, `DIAG_BOOTSTRAP`, `DIAG_CONFIG`, `DIAG_INPUT_DIR`, `DIAG_OUTPUT_DIR`, `OMP_NUM_THREADS`.

## Outputs

`FARE_diagnostics/diagnose_once_<JOBID>/` with:

- `preflight.json`, `manifest.json`, `overall_comparison.csv`, `module_contribution.csv`, `diagnostic_report.json`, `diagnostic_report.md`, `decisions.json`.
- Source NPZs under `FARE_diagnostics/one_pass_inputs/` and each target's `*_source_audit.json`.
- Live PBS output + `FARE_logs/diagnostic/FARE404.*.live.log`.

All interventions use one unchanged set of trainable model weights; do **not** present inference-time mode switches as independently retrained ablation experiments. Selected-slot counterfactuals are a **limited** bound over selected evidence, not an upper bound over all legal retrieval candidates. Checkpoint-in-sample residual memory remains exploratory and is not automatically converted to OOF by this release.

## Test status

All included tests run on the synthetic fixtures and local CPU; no claim is made that the real Gadi datasets/FARE checkpoint were executed in this environment.
