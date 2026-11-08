# CoRSTA v0.2.0 — Counterfactual + Forecast Residual (No Retrieval)

**Status:** Full research implementation with synthetic CPU integration validated. The real official STLAformer + Gadi V100 + TraffiDent data have **not** been exercised in this environment. There are no v0.2 accuracy claims. Target: **Speed and Flow each ≥5% Macro-Overall MAE gain versus strong comparable baselines**, subject to independent validation.

This is a **standalone v2 modeling/training path** inside the existing `CoRSTA/` tree. It does not import/use `corsta.v7`, Historical Memory, Top-K, Replay or legacy C1/Transport. It reuses only three verified v0.1 utility files for loading STLA and reading the official cache: `corsta/stla.py`, `corsta/student.py`, `corsta/data.py` (plus existing UTC timestamp helper).

## 1. Model & data boundaries

Two distinct learnable mechanisms:

1. **CF branch** predicts the observation-loss response `P_T(full) - P_T(masked)`, by updating the STLA prefix representation. It produces `P_CF` through the **frozen, differentiable STLA Tail + Head**.
2. **Residual branch** conditions on the **actually applied** `P_CF` and visible historical innovations, predicting a second latent correction for `P_CF+R`. Auxiliary residual target is `Y - P_T(full)`; the final loss against `Y` is primary so the residual branch can also repair CF estimation error.

The fixed Teacher and protected base STLA use the *same pretrained parameters*, including the separately trained S0 mask projector. Paired targets satisfy `Y - P_T(masked) = [P_T(full)-P_T(masked)] + [Y-P_T(full)]`.

The model `forward()` accepts **only** `{x,mask,tod,dow}`; it rejects full history or future targets. The Teacher is a separate TRAIN-only module. All augmentation removes observations from `original_valid`; artificial masking never creates observations. Hidden original values may not influence a fixed masked prediction.

The controller accepts/discards **Residual** correction after CF has actually been applied; the CF branch has its own latent confidence gate. The previous mistake of training residual corrections assuming a CF update that was never applied is avoided.

Spatial road graph: real road direction + distance, sparse fixed source-to-target candidates, per-query availability bias, and explicit `no graph` option. Graph outputs are zero-initialized so enabling graph during the joint stage initially exactly preserves the learned non-graph model. Optional TRAIN-only CF-sensitivity + realized forecast-utility graph is a separate offline build, not retrieval.

## 2. External dependencies (not included in ZIP)

Place the project alongside the existing:

```
Speed_LLM_RAG/
  CoRSTA/
  Backbone/                  # original Backbone + vendor STLAformer
  Base_SupportBench/         # existing protocol & shared cache
  Backbone_exps/             # pretrained speed/flow best.pt
  CoRSTA_exps/               # v0.1 bootstrap S0 checkpoint
```

Must have the original official Backbone STLA `best.pt` and the matching `bootstrap_v7/s0.pt` from the **same checkpoint identity** (sha256 verified). This project **never re-trains legacy C1**.

The official formal shared cache requires (for each train and val split) `*_x_obs_norm.npy`, `*_history_observed.npy`, `*_artificial_missing.npy` when available, `*_y_norm.npy`, `*_y_raw.npy`, `*_future_valid.npy`, `*_direct_mask.npy`, `*_tod_hist.npy`, `*_dow_hist.npy`, `*_manifest.csv`. TRAIN additionally requires `train_x_full_norm.npy` (genuine raw history before artificial masks).

**Four-severity validation** uses `val_x_full_norm.npy` **only to construct fixed s20/s40/s60/s90 masked inputs**. `val_x_full_norm` is not forwarded to Teacher, Student, or loss targets. If missing, CPU Preflight deliberately fails: an s20 view cannot be reconstructed by removing more data from s40. Data availability and parity need checking on Gadi; no TEST data are ever substituted for Validation. This is distinct from deployment, where no complete history exists.

## 3. Setup / CPU preflight

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export BACKBONE_MODE=incomplete

export SPEED_STLA="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export SPEED_S0="$PWD/CoRSTA_exps/speed/incomplete/TraffiDent_Net_06/formal/seed10/bootstrap_v7/s0.pt"

export FLOW_STLA="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
export FLOW_S0="$PWD/CoRSTA_exps/flow/incomplete/TraffiDent_Net_06/formal/seed10/bootstrap_v7/s0.pt"

qsub -N CoRSTA_v2_preS -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$SPEED_STLA",MASK_PROJ_CHECKPOINT="$SPEED_S0",BACKBONE_MODE=incomplete CoRSTA/pbs/v0_2/preflight_cpu.pbs
qsub -N CoRSTA_v2_preF -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$FLOW_STLA",MASK_PROJ_CHECKPOINT="$FLOW_S0",BACKBONE_MODE=incomplete CoRSTA/pbs/v0_2/preflight_cpu.pbs
```

**Verify both S0 paths exist:** old CoRSTA v0.1 layout is used as the default. If old S0 lives elsewhere, change `SPEED_S0`/`FLOW_S0` accordingly. CPU Preflight verifies STLA forward equivalence, real cache shapes, S0 identity and four-severity validation view availability.

## 4. Formal Speed and Flow — one GPU PBS per Target

```bash
qsub -N CoRSTA_v2_speed -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$SPEED_STLA",MASK_PROJ_CHECKPOINT="$SPEED_S0",BACKBONE_MODE=incomplete,BATCH_INITIAL=64,CORSTA_V2_STAGE=all CoRSTA/pbs/v0_2/train.pbs
qsub -N CoRSTA_v2_flow  -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$FLOW_STLA",MASK_PROJ_CHECKPOINT="$FLOW_S0",BACKBONE_MODE=incomplete,BATCH_INITIAL=64,CORSTA_V2_STAGE=all CoRSTA/pbs/v0_2/train.pbs
```

Each job executes `CF -> Residual -> Joint (Road Graph) -> Control -> Lock -> TEST s20/s40/s60/s90 -> Report`, automatically **resuming** completed stages/checkpoints after interruption. Use the **same source/config/environment and same run dir** for resume. The job must have a CUDA GPU; there is no silent CPU fallback. User may also set `CORSTA_V2_STAGE=cf|residual|joint|control|lock|test|report` to control stages. The default `all` performs TEST **only after** a locked checkpoint is written. For full scientific rigor, it is preferable to run `train`, inspect Selection/Locked Validation, then issue a single locked `test` after pre-registering the final model. Never tune using previously seen TEST results.

Default run dirs:

```
CoRSTA_v2_exps/speed/incomplete/TraffiDent_Net_06/formal/seed10/
CoRSTA_v2_exps/flow/incomplete/TraffiDent_Net_06/formal/seed10/
```

Log dirs:

```
CoRSTA_v2_logs/speed/incomplete/
CoRSTA_v2_logs/flow/incomplete/
```

Each line carries UTC ISO 8601 time from the existing timestamp filter. GPU run requests 1× V100/12 CPUs/64GB RAM/48h. Automatic CUDA batch probe tests the *real* Teacher + Student + Loss forward/backward at 64, tries 128 if safe, otherwise backs off 32/16. Stage and batch checkpoints include RNG/optimizer/scaler states; an OOM rolls back to the last completed epoch before retrying with a smaller batch. CPU synthetic smoke is available with `--cpu --smoke`.

## 5. Graph options

Default: Road graph enabled in Joint+Control (`use_road_graph: true`, `use_cf_graph: false`). A non-graph CF-only and Residual stage precedes graph adaptation. Do not call the road graph a causal graph. Real edges come from `data/processed/traffident/base_supportbench/<target>/<dataset>/core/nodes.csv` and `graph/{adjacency,distance}.npy` (hashes locked in run identity).

To use the **optional** CF observation-response graph, first build it on a sampled **TRAIN-only** subset, with an independent GPU PBS (`CoRSTA/pbs/v0_2/prepare_cf_graph.pbs`) or run `scripts/v0_2/prepare_cf_graph.py` directly. The builder calculates how withholding historical node j changes predictions on i; edge selection favors *forecast gain*, not large response alone. Source j -> target i orientation is checked. This measures predictor dependence, not physical/causal traffic propagation.

Enable `model.use_cf_graph: true` in a **new configuration/run dir** and pass `CF_GRAPH_PATH` with the `.npz`. No cached legacy Retrieval or `CoRSTA/corsta/v7` import is used. `FORMAL_CACHE` is explicitly required by graph PBS to avoid guessing private paths; use the validated shared-cache directory from the CPU preflight/Backbone outputs. CF graph compilation may be expensive; it is optional and **not necessary** for first v0.2 experiments.

## 6. Model, targets, and loss

Model tensor shapes for Net_06 (N=388, not hard-coded):

```
Observed x, mask         [B,L,N]    (L=12)
STLA prefix H0          [B,L,N,64]
Failure history         [B,L,N,16]
CF horizon tokens       [B,P,N,32] (P=12)
CF latent shift         [B,L,N,64]
CF forecast P_CF        [B,P,N]
Residual horizon tokens [B,P,N,32]
Residual latent shift   [B,L,N,64]
Final forecast          [B,P,N]
```

CF target `P_T^F - P_T^M` and residual semantic target `Y - P_T^F` are **derived only on TRAIN**. They are not assumed fully predictable from masked history. Forecast loss dominates, while CF response, feature recovery and residual semantics are auxiliary. Residual Training/Joint are optimized against true future `Y`, allowing correction of imperfect CF estimates. Controller is fitted only after a stable candidate; its labels reflect **actual** candidate improvement, not hypothetical teacher improvement.

One cross-attention per node for temporal horizon representation and reverse latent shift; dynamic graph messages are sparse `[B,P,N,K,32]`, never an unbounded dense all-node Transformer. The frozen STLA Tail remains differentiable wrt modified latent states, even though its parameters are not updated.

The model does **not** modify original pretrained STLA Head weights. Instead it adds a trainable failure/input latent adapter plus sequential CF/Residual latent adapters. This is an intentional internal STLA representation adaptation, preserving the strong original output projection. Further LoRA/unfreezing would require a separate matched ablation and is not silently activated.

## 7. Selection, TEST protocol, and target

Train masks cover s20/s40/s60/s90 using deterministic sample/epoch kinds. Validation severity views are **fixed** independently from training and retain chronological Selection and Locked Check groups with a ≥24-step embargo. Checkpoints are selected by **four-severity Macro-Overall MAE**, not by s40 alone. The evaluation stage loads official four severity TEST caches and uses true test labels only after model lock.

Current previously observed baseline (same protocol, Net_06, Seed10): Speed MAE 1.7455, Flow MAE 20.0677. Research targets for ≥5% against that baseline: Speed ≤1.6582, Flow ≤19.0643. **These are targets, NOT v0.2 results.** Comparing against the strongest existing matched baseline and multiple seeds/nets is mandatory before claiming publication-quality evidence.

## 8. Local test suite

```
python -m pytest -q CoRSTA/tests/v0_2/test_v2_core.py
python -u CoRSTA/tests/v0_2/integration_synthetic_v2.py
```

The integration uses a fake STLA and synthetic cache to verify real v2 stages, locks, test gating, resume/reuse and report. **It is not a substitute for running `preflight_cpu.pbs` against your official STLA checkpoint and actual Base_SupportBench files.**

## 9. Export lightweight diagnostics

```
bash CoRSTA/scripts/v0_2/pack_v2.sh
```

This packs JSON/CSV/logs/configs/v2 code and excludes checkpoint weights and large NumPy arrays.
