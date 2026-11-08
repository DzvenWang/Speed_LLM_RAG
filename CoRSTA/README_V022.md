# CoRSTA v0.2.2 — Full-to-Masked Normal-Anchored Multi-Failure CF

## Scope and research claim

**Two principal prediction mechanisms only:** counterfactual observation-failure response (CF), and normal full-view forecast residual (R). **No historical retrieval, no candidate-bank k, no future labels in inference.** This version extends v0.2.1 with (1) online K-view multi-failure masks including `entry_random`, (2) frozen Normal Residual Anchor learned from full-view TRAIN supervision, (3) actual incremental CF utility and multi-mask gain ranking relative to that fixed anchor.

**Not established:** any gain on real TraffiDent/Gadi or the 5% Speed/Flow target. Local synthetic tests verify code behavior, not research performance. v7/v8-inspired typed multi-relation graph and train-fold out-of-fold Teacher residuals are **not** implemented by this CF-focused update; the existing v0.2.1 road/optional CF graph is preserved.

## Exact model/target semantics

TRAIN has an original valid full history `X_full`, synthetic failure subsets `M_j`, and the same future `Y` for all views. The same frozen STLA Teacher produces:

```text
P_full = Teacher(X_full, original_valid)
P0_j   = Teacher(X_full masked by M_j, M_j)
CF*_j  = P_full - P0_j
R_N*   = Y - P_full                  # shared across failure views
```

The model, using ONLY currently masked history:

```text
H0_j  = frozen_STLA_prefix(X_masked_j, M_j)
P0_j  = frozen_STLA_tail_head(H0_j)
Rhat_N_j = frozen_STLA_tail_head(H0_j + normal_latent_j) - P0_j
P_N_j = P0_j + Rhat_N_j              # Normal Residual Anchor
Dhat_CF_j = frozen_STLA_tail_head(H0_j + cf_latent_j) - P0_j
P_candidate_j = P_N_j + Dhat_CF_j
u_hat_j = anchor_gain_head(masked_features_j, P_N_j, Dhat_CF_j)
g_CF_j = sigmoid(anchor_acceptance_head(masked_features_j, u_hat_j))
P_final_j = P_N_j + g_CF_j * Dhat_CF_j
```

`P_N_j` is frozen **after** Normal Stage (the residual, shared failure encoder, residual spatial graph, and STLA weights). It depends on current Mask and is therefore not a mask-independent forecast; "fixed" means fixed *parameters and computation* for a given input while CF trains. Actual CF incremental utility:

```text
U_anchor_j = abs(P_N_j - Y) - abs(P_candidate_j - Y)
```

Use TRAIN future ONLY to calculate supervision. The mask-to-mask ranking compares the SAME TRAIN history and future across K views, and labels are not assumed monotone in mask severity.

### New stages

1. `normal`: train failure encoder + normal forecast residual + optional residual spatial encoder. **Train-only** main objective uses `P_N + (P_full-P0)` against Y, plus semantic residual target `R_N*=Y-P_full`. For formal training, Normal checkpoint selection uses a **temporally embargoed TRAIN-only calibration subset** and its Teacher-forced residual-proxy MAE; this subset is excluded from Normal-stage optimizer data. Small smoke experiments fall back to deployable Validation `P_N` MAE. Validation itself always measures deployable `P_N` only, with no full-view Teacher. Normal-only MAE need not improve until CF is added.
2. `anchor_cf`: load best Normal checkpoint; freeze Normal and STLA; train CF latent response and gain heads using final raw `P_N + Dhat_CF`, Teacher CF response, multi-mask Teacher-error and actual Normal-anchor utility ranking. Four views by default (primary plus three extras); no bank or lookup.
3. `anchor_control`: freeze CF candidate, Normal anchor and STLA; train only CF `anchor_gain_head` and `anchor_acceptance_head` to select useful candidate corrections. Control initialized close to candidate identity, not mathematically exact 1.0. Report both raw candidate and applied result.

Legacy `cf_first` mode is retained and should be invoked with `--config CoRSTA/configs/v0_2/corsta_v2.yaml`. Its four stages remain `cf`, `residual`, `joint`, `control`. The new config defaults to `normal_anchored`. V0.2.1 run identities are not compatible with changed source; use original v0.2.1 package to resume old experiments. Do not overwrite existing checkpoints.

## Training input / output schemas

`CoRSTASplit(train, num_views=4, contrast_views=True)` yields:

```text
x [B,L,N], mask [B,L,N], tod/dow [B,L]
view_x [B,3,L,N], view_mask [B,3,L,N]
view_severity [B,3], view_kind [B,3]
x_full [B,L,N] and original_valid [B,L,N]  # TRAIN only
y_norm/y_raw [B,P,N] and future_valid [B,P,N]   # loss only
```

Failure types `entry_random`, `sensor`, `temporal`, `regional`, `directional`, combined with severity s20/s40/s60/s90; examples deterministic by `(seed, epoch, index, severity, kind, view_id)`. The v0.2.1 *validation* failure types are deliberately left unchanged (sensor/regional/directional/temporal) to preserve comparable fixed Selection cases; `entry_random` enters TRAIN now. A distinct Entry Random validation benchmark can be added later with separate scenario identity.

`CoRSTAv2.forward(observed)` accepts **exactly** `x`, `mask`, `tod`, `dow`—never `x_full` or `Y`. It returns normalized `backbone`, `normal_prediction`, `normal_cf_candidate`, `prediction`, plus `delta_cf_candidate`, `anchor_gain_pred`, `cf_accept_logit`, `cf_accept_gate`. Evaluation additionally computes absolute MAE/RMSE using target-specific TRAIN statistics.

## Gadi setup

Place `CoRSTA/` under `/scratch/mh20/zw1165/Speed_LLM_RAG/`, beside `Backbone/` and `Base_SupportBench/`. The zip does **not** contain pretrained checkpoints, S0 mask projector, Backbone vendor, complete raw data, or any real trained model. Verify S0 location; it is required and checked by backbone digest.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export BACKBONE_MODE=incomplete
export SPEED_STLA="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export FLOW_STLA="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
export SPEED_S0="$PWD/CoRSTA_exps/speed/incomplete/TraffiDent_Net_06/formal/seed10/bootstrap_v7/s0.pt"
export FLOW_S0="$PWD/CoRSTA_exps/flow/incomplete/TraffiDent_Net_06/formal/seed10/bootstrap_v7/s0.pt"

# Verify before qsub
ls -lh "$SPEED_STLA" "$FLOW_STLA" "$SPEED_S0" "$FLOW_S0"

qsub -N C022_preS -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$SPEED_STLA",MASK_PROJ_CHECKPOINT="$SPEED_S0",BACKBONE_MODE=incomplete CoRSTA/pbs/v0_2/preflight_cpu.pbs
qsub -N C022_preF -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$FLOW_STLA",MASK_PROJ_CHECKPOINT="$FLOW_S0",BACKBONE_MODE=incomplete CoRSTA/pbs/v0_2/preflight_cpu.pbs

# Submit only after both preflights succeed. No locked TEST in this stage.
qsub -N C022_speed -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$SPEED_STLA",MASK_PROJ_CHECKPOINT="$SPEED_S0",BACKBONE_MODE=incomplete,CORSTA_V2_STAGE=train,BATCH_INITIAL=32 CoRSTA/pbs/v0_2/train.pbs
qsub -N C022_flow -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$FLOW_STLA",MASK_PROJ_CHECKPOINT="$FLOW_S0",BACKBONE_MODE=incomplete,CORSTA_V2_STAGE=train,BATCH_INITIAL=32 CoRSTA/pbs/v0_2/train.pbs
```

Training uses `CoRSTA_v022_exps/{target}/incomplete/{dataset}/formal/seed{seed}/normal|anchor_cf|anchor_control`; logs use `CoRSTA_v022_logs`. GPU batch probing runs full K-view forward+backward at anchor_cf, initially 32 recommended for V100; no batch size/epoch duration claims before real profiling. The four-severity fixed Selection controls checkpoint selection. **Only after experimental design and checkpoint are locked**, use `CORSTA_V2_STAGE=lock`, then `test`, then `report`, rather than `all` for real experiments. `all` is intended for controlled synthetic smoke integration.

### Local tests

```bash
cd CoRSTA
PYTHONPATH=. pytest -q tests/v0_2 tests/test_corsta.py tests/test_v7_core.py tests/test_v7_regressions.py tests/test_preflight_pbs.py tests/test_d1_diagnostics.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python tests/v0_2/integration_synthetic_v022.py
```

The synthesis-based integration additionally checks training stage reuse, model lock, test and report; it does not measure generalization. For v0.2.1 regression:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python tests/v0_2/integration_synthetic_v2.py
```

## Scientific limitations / follow-up experiments

- Compare `cf_first` vs `normal_anchored` with fixed mask scenarios. Distinguish changed ordering from additional multi-view/error-contrast supervision using ablations.
- Paired-view teacher differences are *algorithmic observation response*, not physical causality; paired full-target supervision does not guarantee recoverability.
- Complete-view Teacher errors on TRAIN may be optimistic if Teacher trained on the same overlapping windows. Formal evidence needs temporally embargoed out-of-fold teacher residual targets or independent TRAIN calibration, with no Validation/Test fitting.
- Normal-only output is not guaranteed to improve over STLA; forecast selection should prioritize **final frozen-anchor + CF MAE**, with Normal stage monitored separately.
- Stored Road/CF graph code is unchanged from v0.2.1 (including the known road/CF duplicate edge merging limitation). Do not attribute any v0.2.2 gain to a graph revision.
- No full-data use at deployment. `val_x_full_norm.npy` may only build fixed VAL masked scenarios, not serve as validation Teacher privilege.
