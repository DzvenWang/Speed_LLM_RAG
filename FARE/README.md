# FARE v0.2.4

**FARE: Failure-Adaptive Restoration Experts for Traffic Forecasting under Incomplete Sensing**

v0.2.4 replaces scalar-alpha-to-dense-gate supervision with **Forecast Intervention Landscapes + Hierarchical Prediction Control**.

## Core idea

Every expert still produces a dense raw estimate for every `artificial_missing` cell. Prediction control is now split into two identifiable decisions:

1. **Query-level Forecast Intervention Landscape** predicts downstream utility for every allowed intervention strength `alpha`.
2. **Cell-level Reliability** predicts where the raw restoration is locally trustworthy.

The query controller selects `alpha` from the predicted utility landscape. Reliability redistributes that intervention across missing cells while preserving approximately the selected query-level mean strength.

This avoids the v0.2.3 failure mode where a scalar `alpha*` label supervised a `[T,N]` gate and the network converged to a nearly constant mean gate.

## Experts

- Context: full spatiotemporal learned restoration.
- Structural: causal 28-day TMCP-style completion with validation utility landscape calibration.
- Retrieval: pairwise downstream-utility candidate ranking + learned support fusion + intervention landscape.
- Backbone-Aware: forecast-conditioned residual restoration + intervention landscape.

No MoE/router is trained in v0.2.4. Router work remains blocked until expert/safe-oracle headroom is materially useful.

## Training contract

Stage A trains raw restoration (and Retrieval support ranking/fusion). Stage B freezes the restorer and trains only the hierarchical controller. Counterfactual utility curves use formal train/val future truth only; evaluation truth is never used for control decisions.

Cell reliability is supervised by restoration correctness on formal train/val artificial-missing cells. This target answers *where* restoration is trustworthy; query-level forecast utility answers *whether/how much* it should affect STLAformer.

## Smoke

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:SEED = "10"
$env:TRAFFIDENT_TARGET = "speed"
New-Item -ItemType Directory -Force -Path "FARE_logs\speed" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
python -u FARE\scripts\run_pipeline.py --target speed --smoke 2>&1 |
  Tee-Object -FilePath "FARE_logs\speed\FARE.v0.2.4.speed.smoke.$ts.live.log"
```

Smoke is an engineering/learning-signal check, not a performance estimate.
