# FARE v1.1.0 Full — Forecast-Calibrated Retrieval Integration

This is a **complete replacement project package**, not a differential patch.
It is based on the working v1.0.4 five-stage FARE research implementation.
It retains all modules, scripts, five-net and speed/flow support, and Gadi workflows.

## Motivation from the Net-06 Speed/Flow smoke

In v1.0.4, the retrieved candidate correction was nonzero and Train retrieval coverage was >0.9, but alpha=1 candidate forecasts were worse than base. A held-out **smoke** alpha sweep suggested small positive corrections helped; the most helpful alpha differed between speed and flow. This is a **diagnostic** result, not a statistically sound performance claim.

## Actual changes

1. `fare.control.UtilityRiskController`: bounded correction `0 <= alpha <= max_alpha`, initially 0.12, maximum default 0.60. Learns a per-position/horizon dynamic correction, uses inference-safe reliability/candidate/risk/utility features, and remains compatible with purged OOF controller training. Same architecture for speed and flow; no test-selected target-specific constants.
2. `fare.model.FARE`: the *reported* `final_gate` is now **exactly** the effective gate used in `prediction`. Validation calibration multiplier is clipped to the alpha maximum, preventing overshoot and inconsistent diagnostics. `correction_delta` is logged as normalized proposal.
3. `fare.losses`: add small SmoothL1 `candidate` auxiliary supervision (default weight 0.08) and SmoothL1 `correction_energy` stabilizer (default 0.003) **only in forecast phases**, preserving retrieval/recovery phase objectives. This constrains overlarge proposals without requiring alpha=1 candidate forecasts to win.
4. `fare.evaluation`: added `final_MAE_gain`, helpful/harmful fractions for final vs base, effective-gate distribution and `effective_gate` in optional validation/test predictions arrays.
5. `scripts/alpha_sweep.py`: calculate validation-only scalar alpha vs MAE/RMSE from `val_full_predictions.npz`. Writes `alpha_sweep.csv`/`.json`, rejects test filename by default. Do **not** use smoke alpha optima as published results.
6. `scripts/calibrate.py`: align clipped multiplier sweep with *the exact inference formula*, extend multiplier candidates so targets that need stronger corrections can be calibrated on the full validation split. Save `calibration_grid.csv`/`.json` and calibrated `best.pt`.
7. `fare.oof`: OOF optimal-correction targets clipped to controller alpha bound, rather than training an impossible alpha=1 target.
8. All evaluation/OOF/calibration paths now recover exact `config` from the checkpoint, including custom alpha bounds and loss settings, instead of silently applying the current default YAML; resume also restores saved config.
9. Gadi `submit_gadi.py train --config <path>` passes a shared custom YAML to the train PBS; post-train Evaluate uses checkpoint config. Defaults are unchanged: independent afterok s20/s40/s60/s90 candidate Evaluate; manual Evaluate; optional `--finalize` OOF+Calibration+Final Evaluate.
10. New regression tests for gradients into dynamic policy, exact calibrated-gate/prediction identity, safety bound, no candidate fallback, validation sweep, exact config recovery, and synthetic staged training.

## Compatibility and limitations

- **v1.0.4 checkpoints are NOT structurally compatible** (new policy parameters and buffer). Start a fresh v1.1.0 run. Keep old runs as controls.
- Existing `Base_SupportBench` caches, STLAformer Backbone checkpoints, and graph metadata are reused. Old FARE operational code is not an external runtime dependency.
- A newly trained `candidate_best.pt` is not an OOF-trained final model; run `build_oof.py`, `calibrate.py` to obtain the final `best.pt`.
- Candidate/Final gains must be compared with the same underlying STLA/FARE Base and matching validation/test protocol.
- Smoke is for correctness only. No real Net-06/01-04 formal accuracy gains have been verified in the package test environment.

## Windows commands

```powershell
# Both speed + flow five-stage smoke
python -u FARE\scripts\smoke_all.py --dataset TraffiDent_Net_06 --seed 10

# Independently inspect any trained checkpoint on FULL validation
python -u FARE\scripts\evaluate.py --checkpoint "<run_dir>\candidate_best.pt" --split val --output "<run_dir>\val_diagnostic"
python -u FARE\scripts\alpha_sweep.py --predictions "<run_dir>\val_diagnostic\val_full_predictions.npz"

# Build true temporally purged OOF, calibrate VALIDATION, evaluate TEST
python -u FARE\scripts\build_oof.py --checkpoint "<run_dir>\candidate_best.pt"
python -u FARE\scripts\calibrate.py --checkpoint "<run_dir>\controller_best.pt"
python -u FARE\scripts\evaluate.py --checkpoint "<run_dir>\best.pt" --split test --severities 20 40 60 90
```

## Gadi

```bash
python -u FARE/scripts/submit_gadi.py train --dataset TraffiDent_Net_06 --targets speed flow --seed 10
# Queue independent train -> candidate Evaluate by default.

# Optional full OOF/Validation calibration/final Test chain
python -u FARE/scripts/submit_gadi.py train --dataset TraffiDent_Net_06 --targets speed flow --seed 10 --finalize

# Manual Evaluate of a checkpoint
python -u FARE/scripts/submit_gadi.py evaluate --checkpoint /absolute/path/to/candidate_best.pt --checkpoint-kind candidate

# Pre-submit check
python -u FARE/scripts/submit_gadi.py train --dataset TraffiDent_Net_06 --targets speed flow --seed 10 --dry-run
```

## Research test priority

1. Speed/Flow joint smoke with nonzero Train retrieval; `final_gate` finite & bounded.
2. Formal validation Base/Candidate/Final and full-validation alpha sweep, before any Test tuning.
3. Candidate + automatic s20/s40/s60/s90 Test (same dataset protocol).
4. Purged OOF Utility/Risk, Validation calibration, independent final Test.
5. Random Memory / retrieval-off / state-only / innovation-only inference ablations, followed by matched-budget retrained ablations.
6. Repeat in Net-01,02,03,04 and additional seeds.
