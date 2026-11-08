# CoRSTA v0.2.1 — Multi-Mask CF Error-Contrast + Forecast Residual

**Research hypothesis:** Under *the same actual traffic history and future*, changing only the historical sensing mask causes identifiable changes in the forecasting model's **error**, not merely its forecast. These error-degradation patterns can train a deployment-time CF estimator and a benefit-aware acceptance gate. Residual prediction remains the second principal mechanism, conditioned on the already applied CF correction.

**Status:** Full source patch and synthetic CPU tests. No real Gadi / real STLA checkpoint end-to-end validation and **no measured 5% gain**. Do not treat the mathematical teacher error contrast as deployable accuracy or a physical causal traffic effect.

## 1. What is new compared to v0.2.0?

For an original **TRAIN** history `X_full` and true future `Y`, create two distinct, structured, source-valid-subset masks `M_a` and `M_b` per training query. The SAME frozen STLA Teacher produces:

```
P_F = Teacher(X_full, original_valid)
P_a = Teacher(X_full * M_a, M_a)
P_b = Teacher(X_full * M_b, M_b)
D_a = P_F - P_a                                  # forecasting response
U_a = abs(P_a - Y) - abs(P_F - Y)                  # error degradation
U_b = abs(P_b - Y) - abs(P_F - Y)
U_a - U_b = abs(P_a - Y) - abs(P_b - Y)            # mask-to-mask error ranking
R_F = Y - P_F                                    # separate Residual target
```

The **signed** error difference `(P_a-Y)-(P_F-Y)` equals `P_a-P_F` exactly, so it is not new supervision beyond forecast difference. `U` is what adds actual loss-impact information; it may be negative and is **not monotone in mask severity**.

### CF network now has two extra heads

- `error_gain_head`: predicts `U_a` using **only masked-view** STLA/CF/graph tokens; trained across both views and on their pairwise error rankings. It is **not fed `Y` or the full teacher forecast at inference**.
- `acceptance_head`: predicts whether the **actual Student CF candidate** `P_CF_raw` will improve the protected base `P0`. It receives current masked-view tokens plus the predicted reference gain and candidate forecast change. Its TRAIN target is `abs(P0-Y)-abs(P_CF_raw-Y)`, **not** hypothetical teacher gain.

CF representation is produced inside the pre-existing STLA latent adapter. Its **raw** response is trained against `P_F-P_a`, even when the action gate suppresses the candidate; therefore a low-benefit scenario cannot trivially erase the CF representation-learning objective.

At deployment:

```
P0 = frozen_STLA(X_observed, M)
P_CF_raw = frozen_STLA_tail(H0 + CF_latent_shift)
u_hat = CF_error_gain_head(masked_features)
g_CF = sigmoid(CF_acceptance_head(masked_features, u_hat, P_CF_raw-P0))
P_CF = P0 + g_CF * (P_CF_raw-P0)
P_CF_plus_R = P_CF + [STLA_tail(CF_latent_state + Residual_shift) - STLA_tail(CF_latent_state)]
P_final = P_CF + g_R * (P_CF_plus_R-P_CF)
```

The Residual branch receives the **actually applied** `P_CF`. Its latent delta is decoded as an incremental effect relative to the same raw CF latent reference; therefore the Residual branch cannot accidentally re-insert the CF correction that the acceptance gate suppressed. The Residual branch may use raw CF latent features as covariates, but its **output correction is strictly incremental**.

## 2. Train-only data boundary

`CoRSTASplit(..., split='train', contrast_views=True)` emits `contrast_x` and `contrast_mask` plus the original masked view, `x_full` and `Y`. All masks are subsets of `original_valid`. Half of pairs compare **different spatial patterns at identical severity** and half compare **different patterns and severities**. Mask sampling is deterministic in `(seed, epoch, sample index, severity, pattern)`; no future is used to **generate** the mask.

`CoRSTAv2.forward()` still accepts **only** `{'x','mask','tod','dow'}`. `validation`/`test` continue to get only their masked views and a future label available only to metric computation. Full TRAIN history and future labels appear only in loss/Teacher code; **they are never inputs to `error_gain_head` or `acceptance_head` at deployment**.

## 3. Training details

Existing stages retained:

- `cf`: train latent CF response, teacher-error degradation prediction, candidate acceptance and the two-view comparison.
- `residual`: freeze CF; forecast Residual after the **applied** CF; auxiliary target `Y-P_F`.
- `joint`: jointly adapt CF + Residual (+ optional graph) and repeat multi-mask CF comparison.
- `control`: train the final **Residual** benefit controller; CF acceptance was already trained during CF/Joint.

Loss includes forecast-first MAE/RMSE in physical units, Huber full-view forecast-response, optional CF latent alignment, auxiliary complete-view Residual, **two-view teacher error-gain**, **soft pairwise error ordering**, **secondary CF response** and **Student candidate acceptance**. The added loss weights/temperature are in `configs/v0_2/corsta_v2.yaml` and should be calibrated on TRAIN/Selection, not TEST. `cf_error_temperature` is in normalized target units.

Stage batch probing is updated to include the *secondary* CF forward/backward, accounting for extra memory; V100 performance and feasible batch size are **not yet established**. The extra branch is not called in validation/test/inference.

Per-epoch log `CORSTA_V021_EPOCH` includes `loss_breakdown` with contrast magnitudes. Validation per severity now additionally reports `cf_diagnostics` containing actual applied/candidate harm rates, average acceptance gate, predicted reference error gain, and actual masked-view gain. `cf_diagnostics` uses validation future labels **only for measurement**, never for gate inputs or gradient updates.

## 4. Important controls for a convincing experiment

- `model.use_cf_error_gate: false`: turn OFF new CF acceptance action without deleting CF response prediction (for controlled ablation).
- `training.multimask_error_contrast: false`: remove the second masked view; can separately zero `cf_error_gain_weight` and `cf_acceptance_weight` to remove their single-view supervision.
- `cf_error_gain_weight: 0` and `cf_error_rank_weight: 0`: remove Teacher-error contrast supervision; compare matched-parameter alternatives.
- Keep `training stage_epochs`, optimizer steps, batch budget, and random seeds matched when comparing interventions. Two-view training has more compute; compare also against a two-augmentation ordinary STLA/Residual baseline.
- TRAIN `R_F=Y-P_F` from a Teacher fitted on overlapping TRAIN windows is an in-sample residual and may be optimistic. Use temporally embargoed cross-fitting or held-out TRAIN calibration to test residual-generalization claims.
- Graph **Road/CF edge union and typed-edge semantics** from TCRR v7/v8 comparison are NOT implemented in this CF-specific patch; they require a separate graph revision. The physical road graph is now built regardless of `use_road_graph` for **mask-generation consistency**, while model graph message passing can be disabled.

### Core success criterion

The hypothesis is supported only if multi-mask error-contrast training beats (i) CF response-only, (ii) matching-capacity single-mask error-gain learning, (iii) same-budget two-mask standard augmentation, and (iv) CF+Residual without this learning, **on held-out fixed mask scenarios**. Better correlation with Teacher error labels alone is insufficient; measure real forecast MAE/H12/masked MAE, CF harm frequency, calibration, and s20/s40/s60/s90 separately.

## 5. Install and execution

Unzip `CoRSTA_v0.2.1_MultiMask_CF_ErrorContrast_Patch.zip` over the **same v0.2.0 project** or unzip the full project into `Speed_LLM_RAG/CoRSTA/`. Legacy v0.1 is preserved. v0.2.1 outputs are isolated at `CoRSTA_v021_exps` and may not resume v0.2.0 checkpoints because the CF module state changed.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export BACKBONE_MODE=incomplete
export SPEED_STLA="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export FLOW_STLA="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
export SPEED_S0="$PWD/CoRSTA_exps/speed/incomplete/TraffiDent_Net_06/formal/seed10/bootstrap_v7/s0.pt"
export FLOW_S0="$PWD/CoRSTA_exps/flow/incomplete/TraffiDent_Net_06/formal/seed10/bootstrap_v7/s0.pt"

# Confirm S0 paths are present on your system before submitting.
qsub -N CoRSTA021_preS -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$SPEED_STLA",MASK_PROJ_CHECKPOINT="$SPEED_S0",BACKBONE_MODE=incomplete CoRSTA/pbs/v0_2/preflight_cpu.pbs
qsub -N CoRSTA021_preF -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$FLOW_STLA",MASK_PROJ_CHECKPOINT="$FLOW_S0",BACKBONE_MODE=incomplete CoRSTA/pbs/v0_2/preflight_cpu.pbs

# TRAIN only; do not run locked TEST repeatedly while designing the model.
qsub -N CoRSTA021_speed -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$SPEED_STLA",MASK_PROJ_CHECKPOINT="$SPEED_S0",BACKBONE_MODE=incomplete,CORSTA_V2_STAGE=train,BATCH_INITIAL=64 CoRSTA/pbs/v0_2/train.pbs
qsub -N CoRSTA021_flow -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$FLOW_STLA",MASK_PROJ_CHECKPOINT="$FLOW_S0",BACKBONE_MODE=incomplete,CORSTA_V2_STAGE=train,BATCH_INITIAL=64 CoRSTA/pbs/v0_2/train.pbs
```

On GPU, allow stage Batch Probe to reduce Batch 64 if needed; the multi-mask CF stage has additional forward cost. Once the exact model and config are fixed, separately lock and test using `CORSTA_V2_STAGE=lock` and `CORSTA_V2_STAGE=test`.

## 6. Local verification

```
PYTHONPATH=. pytest -q tests
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tests/v0_2/integration_synthetic_v2.py
```

Local synthetic checks use a fake STLA/cache and **do not** establish performance on real Gadi STLA/V100 or TraffiDent.
