# FARE v0.2.3

**Failure-Adaptive Restoration Experts for Traffic Forecasting under Incomplete Sensing**

v0.2.3 addresses the failure exposed by the v0.2.2 smoke: prediction-control gates stayed near their conservative initialization and Retrieval's listwise rank loss stayed near the random/uniform baseline. The four expert architectures are unchanged. The learning mechanism is changed to **two-stage restoration/control training with explicit counterfactual forecast-utility supervision**.

## Runtime dependencies

Only `Base_SupportBench + Backbone`. `Baseline` is comparison-only. No previous FARE code or artifacts are runtime dependencies.

## What changed in v0.2.3

- **Stage A — Restoration capability**: Context, Retrieval and Backbone-Aware train their raw dense restoration while the prediction-control head is frozen.
- **Stage B — Prediction control**: the restoration backbone is frozen; each neural expert learns its cell-wise control gate from offline counterfactual alpha labels.
- For every formal train/val label query, frozen STLAformer evaluates `alpha in {0,.25,.5,.75,1}` and labels the alpha with maximum downstream forecast utility. Evaluation truth is never used.
- Gate bias is initialized from the mean training alpha label instead of always starting at `sigmoid(-3)=0.047`.
- Retrieval candidate learning uses **pairwise utility ranking**. Candidate utility magnitude no longer gets flattened by a softmax temperature. Pairwise random logistic loss is `ln(2)=0.6931`, random accuracy is `0.5`.
- Structural TMCP remains a full causal 28-day expert with validation-only intervention calibration.

All experts still produce a dense raw estimate for every `artificial_missing` cell. Observed and original-invalid cells remain immutable.

## Smoke

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:SEED = "10"
$env:TRAFFIDENT_TARGET = "speed"

New-Item -ItemType Directory -Force -Path "FARE_logs\speed" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

python -u FARE\scripts\run_pipeline.py `
    --target speed `
    --smoke 2>&1 |
    Tee-Object -FilePath "FARE_logs\speed\FARE.v0.2.3.speed.smoke.$ts.live.log"
```

Expected new diagnostics include:

```text
CONTEXT_RESTORER_EPOCH ... raw_gain=...
CONTEXT_ALPHA_UTILITY ...
CONTEXT_CONTROL_INIT mean_alpha=... bias=...
CONTEXT_CONTROL_EPOCH ... gate=... alpha_mae=...
RETRIEVAL_RESTORER_EPOCH ... rank=... rank_acc=... val_rank_acc=...
RETRIEVAL_ALPHA_UTILITY ...
RETRIEVAL_CONTROL_INIT mean_alpha=... bias=...
RETRIEVAL_CONTROL_EPOCH ... control_gate=... alpha_mae=... rank_acc=...
BACKBONE_AWARE_RESTORER_EPOCH ...
BACKBONE_AWARE_ALPHA_UTILITY ...
BACKBONE_AWARE_CONTROL_EPOCH ...
STRUCTURAL_CALIBRATION_PASS alpha=...
CONTROLLED_EXPERT_SUMMARY ...
FARE_V023_PIPELINE_PASS
```

Smoke uses the same model architecture and objectives but fewer rows/epochs. It is an engineering/learning-mechanism diagnostic, not a paper result.

## Formal training on Gadi/V100

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
python -u FARE\scripts\run_pipeline.py `
    --target speed `
    --stage train 2>&1 |
    Tee-Object -FilePath "FARE_logs\speed\FARE.v0.2.3.speed.train.$ts.live.log"
```

Then evaluate s40+s90:

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
python -u FARE\scripts\run_pipeline.py `
    --target speed `
    --stage eval 2>&1 |
    Tee-Object -FilePath "FARE_logs\speed\FARE.v0.2.3.speed.eval.$ts.live.log"
```

Artifacts are isolated under `FARE_artifacts/<target>/TraffiDent_Net_06/v0.2.3/{smoke|formal}/`.

## Leakage contract

- Structural and Retrieval external support is chronological formal-train only.
- Retrieval support windows finish before the current query.
- `x_full` / future `y` are permitted only for formal train/val supervision and final evaluation diagnostics.
- Counterfactual alpha labels and Retrieval utility labels are built only from formal train/val.
- Evaluation truth never enters expert inference, gate prediction, Retrieval ranking, or Structural calibration.
