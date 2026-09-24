# PatchSTG_Oracle v1.1.0

Controlled experiments for identifying and repairing PatchSTG's incomplete-sensing failure mode on `TraffiDent_Net_06`.

## Experiments

### 1. Mask-aware PatchSTG

`run_observation_aware.py --variant mask_aware` adds a small observation-semantic branch after the official PatchSTG embedding. Its four inputs are the observed indicator, causal missing-run length, observation age at the forecast origin, and per-node history coverage.

The original seed-10 PatchSTG is frozen. The new branch is zero-initialized, and the run aborts unless its epoch-zero prediction reproduces the reference checkpoint within `1e-6`. This isolates the value of telling PatchSTG which normalized zeros are missing values.

### 2. True cell coverage restoration curve

`run_coverage_curve.py` evaluates coverage `[0, .25, .50, .75, 1]`. For each query it chooses deterministic nested subsets of `artificial_missing AND finite(x_full_norm)`, copies the actual cached values into those cells, and marks exactly those cells observed.

This is a cell-coverage intervention. It does not scale the amplitude of every hidden value. The old `run_oracle.py` interpolation curve remains available only as a legacy sensitivity result; `run_coverage_curve.py` is the experiment to use for coverage claims.

The curve can evaluate the frozen reference, mask-aware model, and state-auxiliary model using identical restored cell sets.

### 3. Missing-state reconstruction auxiliary task

`run_observation_aware.py --variant state_aux` uses the same mask branch and adds a head that reconstructs all 12 historical states. Reconstruction loss is calculated only on genuinely benchmark-hidden cells with finite ground truth. The reconstructed signal is never fed into the forecast path, so any forecasting change comes from representation shaping rather than direct oracle replacement.

The report includes reconstruction MAE in normalized units and improvement over the cache fill-value baseline.

## Controlled protocol

- Input history and supervision come from `ExperimentCache`; naturally invalid cells are excluded.
- Both learned variants use the same frozen PatchSTG checkpoint and original all-valid future MAE objective.
- Model selection and early stopping use only the deterministic Development subset's masked MAE.
- Final is evaluated after the best epoch is fixed.
- Forecast reports include overall, directly masked, support-node, and per-horizon metrics.
- Checkpoints record the base checkpoint SHA256 and cache identity. Coverage evaluation rejects mismatches.

## Run all three experiments locally

From PowerShell:

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG\PatchSTG_Oracle
.\run_local.ps1 -Stage development
```

To check the local environment without starting training:

```powershell
.\run_local.ps1 -Stage development -PreflightOnly
```

or from `cmd.exe`:

```bat
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG\PatchSTG_Oracle
run_local.bat development
```

The local runner automatically uses `%USERPROFILE%\AppData\Local\miniconda3\envs\TFP\python.exe` when present. It runs `mask_aware` and `state_aux` sequentially so both do not compete for the same local GPU, then evaluates the shared coverage curve. Paths are resolved from the current repository; the `/scratch/...` paths in the YAML do not need to be edited.

Outputs are written under:

```text
../PatchSTG_Oracle_exps/TraffiDent_Net_06/development/<RUN_TAG>/
  mask_aware/{best.pt,summary.json}
  state_aux/{best.pt,summary.json}
  coverage_curve/summary.json
```

Individual commands are:

```bash
python scripts/run_observation_aware.py --stage development --run-tag RUN --variant mask_aware
python scripts/run_observation_aware.py --stage development --run-tag RUN --variant state_aux
python scripts/run_coverage_curve.py --stage development --run-tag RUN \
  --model mask_aware=/path/to/mask_aware/best.pt \
  --model state_aux=/path/to/state_aux/best.pt
```

The original complete-history ceiling can still be reproduced locally with:

```powershell
python scripts/run_oracle.py --stage development --run-tag complete_history_local
```

The PBS files remain available for reproducing the same protocol on Gadi, but they are not used by `run_local.ps1`.
