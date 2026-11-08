# v9.3.0 — Full-Observation Multi-Failure Counterfactual Learning

This **full source distribution** extends v9.2.0; it does not replace historical retrieval files or historical results. No real Gadi Speed/Flow forecasting performance is claimed without completing actual jobs.

## Scientific contract

TRAIN-only, same sample and frozen Backbone/Adapter:

* Reference `B_ref=F(X_ref,M_ref)`; `R_ref=Y-B_ref`
* Official/synthetic observation view `B_m=F(X_ref*M_m,M_m)`; `D_cf=B_ref-B_m`
* `R_masked=R_ref+D_cf`, exactly in float32
* `Damage=abs(Y-B_m)-abs(Y-B_ref)` (can be negative).
* Pair identity: `D_a-D_b=B_b-B_a`.

TRAIN views: original official view + deterministic rotated synthetic family × severity cells. The 20 cells are Entry Random/Sensor/Temporal/Regional/Directional crossed with 20/40/60/90. Views share the reference. Official `Base_SupportBench` masks mean **True=extra-missing** and are inverted before STLA. Original invalid positions never become valid. The training split is chronological with purge. Response scoring uses held-out TRAIN Query IDs; VAL uses its *official* mask only and never its complete-history Teacher.

At inference only `x_obs, observed, tod, dow, base, context` are accepted; there is **no Query-specific Retrieval**. Graph is CF-only, with original v7 physical road construction and train-only functional correlations when selected. The CF student has an independent learnable 12-step temporal-mask pattern encoder, so it can distinguish outage temporal structures even when mean coverage is similar.

`incremental`: `R_strong` trained against masked residual; then frozen. CF is independently supervised against Full-vs-Masked Response and incrementally gated. No double-adding full CF unconditionally.

`decomposed`: `R_ref` trained against reference residual, CF against full-minus-masked; validates the strict decomposition research hypothesis.

`single`: residual-only control can be run by setting all CF stage epochs to zero (and no CF train); use separate output, keep capacity matching protocol in independent research experiment.

## Three-stage integrity

1. Teacher preparation: independent frozen STLA mask adapter, single reference forward per TRAIN query; repeated masked interventions. Chunked crash-resumable NPY cache with identity and source hashes.
2. Residual/CF: stage-specific optimizer groups; `normal_query` and `residual_head` frozen after stage zero; exact normal prediction fingerprint/probe each epoch.
3. Selection/check: validation chronological selection, calibration and locked check are separated. Calibration chooses one scalar factor over CF increment only. Training future labels may supervise losses; never consumed by prediction forward.

## Gadi

The project shares `Backbone`, `Base_SupportBench` and the v5.3 cache/adapter protocol. Place `TCRR-STLA` beneath `$SPEED_LLM_RAG_ROOT`. The original STLA `best.pt` is required, and full/cache layout generation must already have completed.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export STLA_SPEED="$(pwd)/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
# PREPARE (adapt checkpoint and train-only CF teacher + official val)
qsub -N V93_SPD_PREP -v TARGET=speed,SEED=10,V93_STAGE=prepare,BACKBONE_CHECKPOINT="$STLA_SPEED" TCRR-STLA/pbs/formal_v93.pbs
# TRAIN after prepare succeeds, with the same adapter identity and teacher root.
qsub -N V93_SPD_TRAIN -v TARGET=speed,SEED=10,V93_STAGE=train,BACKBONE_CHECKPOINT="$STLA_SPEED" TCRR-STLA/pbs/formal_v93.pbs
```

Replace the Speed checkpoint with the actual Flow `best.pt` for Flow. The example Speed path came from previous jobs; verify existence and checkpoint identity rather than trusting a fixed stamp. Use `V93_TEACHER_ROOT`, `V93_RUN_DIR`, and `V93_RESUME` to explicitly reuse caches and resume a run. `V93_STAGE=all` combines preparation and training in one PBS job but may require more than one walltime; separate jobs are recommended. Code refuses to treat a mismatched cache as valid.

### CPU smoke (requires actual Backbone/vendor/cache)

```bash
python -u TCRR-STLA/scripts/train_v93.py --target speed --dataset TraffiDent_Net_06 \
  --backbone-checkpoint "$STLA_SPEED" --stage all --smoke
```

### Source tests (do not require Gadi)

```bash
cd TCRR-STLA
python -m pytest -q tests/test_v93_*.py
```

## Operational caveats

* Source package contains code **only**, not user research data, trained checkpoints or official STLA vendor source. No v9.3 job has been run on Gadi here.
* Numeric correction is bounded by `c*tanh(x/c)` and is monitored in formal evaluation. CF labels are unbounded difference signals. For Flow, evaluate saturation and consider a controlled bound ablation.
* No-Retrieval creates no new inference-time sensor evidence. Any improvement is attributable to training supervision / inductive bias and must beat an independently trained capacity-controlled single Head.
* Two views per query do not cover all 20 combinations individually; rotating views guarantees coverage across train Queries. Complete grid is a separate TRAIN diagnostic experiment if needed.
* Real dataset must contain benchmark GPS, distance and official mask layouts for selected graph modes and synthetic interventions.
