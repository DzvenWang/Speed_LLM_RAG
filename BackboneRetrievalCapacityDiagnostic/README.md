# Backbone–Retrieval Capacity Diagnostic v0.1.2

Independent diagnostic for the `Speed_LLM_RAG` project. It does **not** modify or retrain `Backbone` or `Base_SupportBench`.

## Question

Does the marginal benefit of the **same conventional historical retrieval signal** shrink as standalone numerical forecasting performance becomes stronger?

The controlled pipeline is:

`Train-only historical memory -> one backbone-independent retriever -> frozen Backbone predictions -> scalar validation-calibrated fusion`.

The retriever is deliberately not PreSuRe. It never reads a backbone latent, a backbone forecast, `direct_mask`, or any future target when producing deployable Top-K candidates.

## Root/path contract

Both Windows and Gadi use `SPEED_LLM_RAG_ROOT`.

- Windows root: `C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG`
- Gadi root: `/scratch/mh20/zw1165/Speed_LLM_RAG`
- code: `<root>/BackboneRetrievalCapacityDiagnostic`
- results: `<root>/BackboneRetrievalCapacityDiagnostic_exps`
- live logs: `<root>/BackboneRetrievalCapacityDiagnostic_logs`

## Reused Backbone artifacts

The diagnostic auto-discovers the latest completed Formal seed-10 run for each of:

`STAEformer, HimNet, BigST, STDN, PatchSTG, STLAformer`.

It requires `best.pt`, `val_predictions.npz`, and the matching s20/s40/s60/s90 `test_predictions.npz`. No backbone training occurs here.

## Retrieval protocol

- memory: Formal **Train s40 only**;
- validation: Formal s40 Val;
- test: s20 / s40 / s60 / s90 evaluation caches;
- query information: `x_obs_norm`, explicit historical observation mask, TOD/DOW only;
- learned-free key: 3 temporal bins, each storing per-node observed mean and observation fraction;
- temporal candidate pre-filter: same weekday/weekend type, TOD ±1 slot, with safe fallback if the pool is too small;
- value: historical H12 future from Train memory;
- K: 1/5/10/20/50;
- temperature: 0.25/0.5/1.0;
- K and temperature are chosen **once from retrieval-only s40 validation MAE**, independent of all backbones;
- fusion: `(1-alpha) * backbone + alpha * retrieval`;
- one per-backbone `alpha` is selected on s40 validation and then frozen across all test severities;
- a single shared-alpha control is also reported.

## Main outputs

Each target produces:

- `calibration/retrieval_grid.csv`
- `calibration/retrieval_best.json`
- `calibration/backbone_alpha.csv`
- `calibration/shared_alpha.json`
- `sXX/topk.npz`
- `sXX/retrieval_predictions.npz`
- `sXX/oracle_choices.npz`
- `sXX/capacity_results.csv`
- `capacity_results.csv`
- `capacity_correlations.csv`
- `REPORT.json`

`capacity_results.csv` reports standalone MAE/Masked-MAE, calibrated fusion gain, shared-alpha gain, paired-bootstrap 95% CI, retrieval oracle, and per-query oracle-fusion upper bound.

## Local Windows

PowerShell:

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:SEED = "10"

python -u BackboneRetrievalCapacityDiagnostic\scripts\preflight.py --target speed --severities 40

# First bounded real-data smoke: no backbone retraining.
python -u BackboneRetrievalCapacityDiagnostic\scripts\run_diagnostic.py `
  --target speed --severities 40 --max-memory 5000 --max-queries 256 --bootstrap-samples 200

# Full speed diagnostic.
python -u BackboneRetrievalCapacityDiagnostic\scripts\run_diagnostic.py --target speed

# Full flow diagnostic.
python -u BackboneRetrievalCapacityDiagnostic\scripts\run_diagnostic.py --target flow
```

## Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT=$PWD
export TFP_ENV_ROOT=$CONDA_PREFIX
export SEED=10

python -u BackboneRetrievalCapacityDiagnostic/scripts/preflight.py --target speed
python -u BackboneRetrievalCapacityDiagnostic/scripts/preflight.py --target flow

BackboneRetrievalCapacityDiagnostic/pbs/submit.sh speed
BackboneRetrievalCapacityDiagnostic/pbs/submit.sh flow
```

This diagnostic is CPU-only because it reuses saved Backbone predictions. The PBS template therefore uses `normal`, 12 CPU cores and 64 GB RAM rather than consuming a V100.

## Interpretation

The primary relation is, separately for each target and severity:

`standalone Backbone MAE` vs `retrieval gain (%)`.

A positive Spearman coefficient means higher standalone MAE (weaker measured backbone) tends to receive larger retrieval gain. Do not use architecture labels themselves as a proxy for strength; use measured standalone error.

Two outcomes matter:

1. **Actual gain and oracle-fusion gain both shrink for strong backbones**: ordinary pattern retrieval is genuinely redundant with strong parametric forecasting.
2. **Actual gain is small but oracle-fusion gain remains large**: useful candidates exist and selection/fusion, rather than information availability, is the bottleneck.


## v0.1.2 fix

- Correctly broadcasts Base_SupportBench node-level `direct_mask` `[Q,N]` across forecast horizons `[Q,P,N]` for Masked MAE/RMSE and masked oracle analysis.
- Synthetic smoke now uses the real `[Q,N]` direct-mask contract and tests masked oracle execution.
