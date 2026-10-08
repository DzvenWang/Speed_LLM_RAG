# FARE v1.3.0 — Connected Observability, Comparable Ranking, Pilot Diagnostics

This is a complete replacement of the standalone FARE project, developed from v1.2.0. Runtime dependencies are durable `Backbone` and `Base_SupportBench`, never `Support_Oracle`, `STLA_DASR`, `Mask_RAFT-STLA`, or `Mask_RAST-STLA`.

## Fixed research issues

1. `ranking_selected_error`, `ranking_uniform_error`, and `ranking_oracle_error` are now computed on the *same target-label-eligible legal candidates*. The label-valid portion of learned weight mass is logged as `ranking_target_valid_weight_mass`. Ranking soft-label cross-entropy likewise renormalizes on that support.
2. `ObservabilityEncoder` output is actually used by retrieval-query embedding, state-recovery gate and reliability estimation; reliability then enters STLA support adapters and utility/risk controller. `test_v130.py` audits gradients, allowing for intentionally zero-initialized support paths on step one.
3. New **fixed-pool** offline diagnostics keep Top-K memory IDs unchanged and compare learned, uniform, state-oracle, and optionally per-rank one-hot weighting. `state_oracle` **reads Validation masked target labels for analysis only** and must never be used to train or calibrate an inference policy.
4. Time-spread Pilot uses 8k train rows, 16k train-only memory entries and full val; evaluation of Pilot defaults to **Validation only**. Run tags/artefacts are separate from Formal. For multi-target submission use `--config FARE/configs/pilot.yaml`.
5. The entire Speed/Flow joint Smoke, five-stage training, strict State/Innovation interventions, matched No-Memory retraining and Gadi afterok automatic/manual Evaluate workflow are retained.

## Local tests

```powershell
python -u FARE/tests/test_v130.py
python -u FARE/scripts/smoke_all.py --dataset TraffiDent_Net_06 --seed 10
python -u FARE/scripts/diagnose_gradients.py --dataset TraffiDent_Net_06 --targets speed flow
```

## Pilot (not Formal, Test is NOT used for selection)

```powershell
python -u FARE/scripts/train.py --config FARE/configs/pilot.yaml --dataset TraffiDent_Net_06 --target speed --seed 10
python -u FARE/scripts/train.py --config FARE/configs/pilot.yaml --dataset TraffiDent_Net_06 --target flow --seed 10
```

Use `--variant no_memory` to retrain an architecture-matched No-Memory control. The separate runs are saved to the `pilot`/`pilot_no_memory` directories when submitted through Gadi; when running locally, `--run-dir` can isolate such controls explicitly.

## Gadi PBS automatic Evaluate

```bash
python -u FARE/scripts/submit_gadi.py train --dataset TraffiDent_Net_06 --targets speed flow --seed 10 --config FARE/configs/pilot.yaml
# Formal (auto Evaluate s20/s40/s60/s90 on separate afterok jobs):
python -u FARE/scripts/submit_gadi.py train --dataset TraffiDent_Net_06 --targets speed flow --seed 10
# Formal matched baseline:
python -u FARE/scripts/submit_gadi.py train --dataset TraffiDent_Net_06 --targets speed flow --seed 10 --variant no_memory
# Explicit manual Evaluate:
python -u FARE/scripts/submit_gadi.py evaluate --checkpoint /absolute/path/to/candidate_best.pt
```

Pilot automatic PBS Evaluate uses `split=val` to avoid Test leakage. Formal auto Evaluate uses Test s20/s40/s60/s90.

## Offline fixed-pool diagnostics

```powershell
python -u FARE/scripts/diagnose_fixed_pool.py --checkpoint "FARE_exps/speed/TraffiDent_Net_06/pilot/RUN_ID/candidate_best.pt" --max-batches 2 --forecast-probe
```

Outputs: `fixed_pool_summary.json`, `fixed_pool_ranking.csv` and (with probe) `fixed_pool_forecast.csv`.

### Scientific/engineering caveats

- v1.2.x checkpoints are **not compatible** with v1.3.0: observability adapter dimensions/weights changed. Re-train with this version; never silently load older weights with `strict=False`.
- The fixed-pool Oracle is a privileged analysis upper reference **within existing Top-K candidates**, not an achievable Test result or deployable policy.
- A future pilot/full training run, not 4-row Smoke, must establish whether state-recovery ranking improves forecast MAE/RMSE.
- The reported gradient audit tests connectivity, not statistical validity or the efficacy of the retrieval candidates.
- Gadi 48h walltime is not a guarantee of end-to-end Formal completion. Pilot must be used to estimate per-epoch runtime; resume if needed.
