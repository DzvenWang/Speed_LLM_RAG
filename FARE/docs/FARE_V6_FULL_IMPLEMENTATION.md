# FARE v6.0.0 — Full implementation and research validation guide

## Research contract

- Target: **Transportation Research Part C**.
- Two non-substitutable goals: improve strong **STLAformer** Masked MAE/RMSE (internal MAE target >=2%, stretch >=4%) **and** substantiate three methodological contributions.
- C1: Failure-aware predictive support (paired OOF error decomposition, support-only predictor, signed vulnerability supervision).
- C2: Counterfactual utility-guided retrieval (strict single-candidate gain vs support-only; dynamic teacher; gradient to actual coarse query maps and matcher; bounded uncertainty objective).
- C3: Horizon-adaptive risk-controlled transfer (horizon-conditioned candidate weights, temporal tokens in frozen STLAformer suffix, harm/uncertainty-aware gate, exact NULL return).
- Every full training automatically runs **Validation** innovation audit, before any formal Test evaluation. Selected stages also run small fixed validation audits; these are exploratory, not publication proof.
- Formal verification requires independent re-trained capacity matched C1/C2/C3 controls, repeated seeds, cross-network and severity/failure-family analysis. Fixed-checkpoint audit must report `INCONCLUSIVE` otherwise.

## Changes from v5.0.3

1. **Model**: `predict_support_only()` and mode `support_only` do not retrieve anything. Null output uses the support-only branch. `backbone_only` is a separate control. New support vulnerability head and zero-initialized query-only delta.
2. **Counterfactual Teacher**: `candidate_single` now forces the selected candidate to unit mass on valid horizons, bypassing learned NULL and Risk attenuation (`alpha_override=1`). On invalid horizons it reverts exactly to support-only. All Teacher futures are **train-only**. Teacher samples rotate with seeded random state, stratified by failure/severity where cache metadata exists, and avoid chronologically impossible OOF cases.
3. **Utility loss**: bounded Student-t style uncertainty loss, smooth-L1 Gain regression and Pairwise/Listwise/Harm terms, all FP32 for sensitive arithmetic. Logs disaggregate losses and actual replay label counts. Improves numerical stability, not a guarantee that utility learns.
4. **Retriever supervision**: Training Teacher records store source Query, candidate IDs and node IDs. At replay, differentiable scoring calls the same state/capability query projections and matching module used for online retrieval; pairwise supervised ranks update the **actual retrieval scoring parameters**. The discrete candidate discovery operator is not differentiated. Teacher labels only cover checked candidate pools, not the complete Memory.
5. **Memory**: reuse v5.0.3 schema-6 paired OOF Memory. `residual_complete=residual_incomplete-failure_delta` is computed lazily. Normal transfer uses complete-residual branch plus sensing-delta branch. `ordinary_residual`, `no_failure`, and `shuffled_sensing` are available fixed-checkpoint controls. No re-running the four OOF producers is needed.
6. **Risk & horizon**: top K shortlist set is chosen using peak per-horizon candidate score rather than global-only average (compared via `global_selection`); weights are per horizon with NULL. Risk sees selected utility, harm and uncertainty. `temporal_mean`, `no_risk_control`, `no_hidden_injection` interventions cover mechanisms.
7. **Stages**: `support:6`, `utility:8`, `joint:24`, `risk:8`, `calibration:5` default. Support-only stage does not access Memory. Stage-specific checkpoints and `best.pt` for full-model stages. All FARE modules start with freshly seeded weights. Loading v5 FARE best/last is forbidden. Explicit `--resume` requires a compatible v6 checkpoint from the SAME v6 experiment.
8. **Efficiency**: bounded GPU cache (default 5 GiB) for value tables and compact keys, automatic CPU fallback; persistent DataLoader workers; GPU tensor epoch loss accumulator; epoch & train timing. Batch stays 32 until benchmarking on real V100.
9. **Audit**: `innovation_audit.py` records canonical MAE/RMSE intervention controls on Val, paired day-block resampling if >=3 distinct blocks, limited strict single-candidate gains and candidate oracle **for diagnosis only**. Writes `innovation_verdict.json` etc. Missing independent retrained ablations always => `INCONCLUSIVE`, not fabricated `SUPPORTED`.

## Files / reproducibility

- Project structure and all pre-existing OOF/Baseline interfaces retained to minimize data migration.
- This ZIP includes FARE source, configs, PBS, scripts, tests and docs. It excludes **Backbone/vendor**, trained checkpoints, Base_SupportBench datasets, generated OOF predictions and large Memory tensors. On Gadi, continue to use the existing `Backbone/`, `Base_SupportBench/` and the already-built Memory.
- Memory and checkpoint hashes remain provenance checked in `fare.pipeline.prepare`.
- Requires exact TraffiDent cache names from Base_SupportBench; no data values, N=388 or graph source identities are hardcoded in the model. H=P=12 follows the research protocol.

## Gadi deployment

From `/scratch/mh20/zw1165/Speed_LLM_RAG`:

```bash
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/FARE:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
python -m pytest -q FARE/tests
```

Run independent CPU pre-submission verification **for each target**:

```bash
python -u FARE/scripts/submit_gadi.py --target speed \
  --config FARE/configs/formal.yaml \
  --checkpoint "$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt" \
  --oos-forecast-file "$PWD/FARE_cache/oof_speed_paired.npz" --check-only
```

Repeat with `--target flow`, the flow STLA checkpoint and `oof_flow_paired.npz`.

After checks pass, submit two independent GPU tasks. **These jobs freshly initialize every trainable FARE module**; only historical paired OOF data and the separate frozen pretrained STLAformer backbone are reused:

```bash
python -u FARE/scripts/submit_gadi.py --target speed \
  --config FARE/configs/formal.yaml \
  --checkpoint "$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt" \
  --oos-forecast-file "$PWD/FARE_cache/oof_speed_paired.npz"

python -u FARE/scripts/submit_gadi.py --target flow \
  --config FARE/configs/formal.yaml \
  --checkpoint "$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt" \
  --oos-forecast-file "$PWD/FARE_cache/oof_flow_paired.npz"
```

**DO NOT submit twice**. Each command submits one 1×V100, 12×CPU, 32GB, 48h PBS task. The original OOF jobs are already complete. The 48-hour limit might still be exceeded; verify runtime logs and checkpoint resume.

After training, look for:

```text
run_dir/best.pt
run_dir/stage_support_best.pt
run_dir/stage_utility_best.pt
run_dir/stage_joint_best.pt
run_dir/stage_risk_best.pt
run_dir/train_metrics.jsonl
run_dir/innovation_audit/evaluation_manifest.json
run_dir/innovation_audit/mode_comparison.csv
run_dir/innovation_audit/candidate_utility.json
run_dir/innovation_audit/paired_statistics.json
run_dir/innovation_audit/innovation_verdict.json
run_dir/innovation_audit/innovation_report.md
run_dir/evaluation/performance_summary.csv
```

To independently re-run Validation audit:

```bash
python -u FARE/scripts/innovation_audit.py --target speed \
 --checkpoint /absolute/path/to/FARE_v6_best.pt \
 --backbone-checkpoint /absolute/path/to/STLAformer_best.pt
```

## Limits of verified implementation

- The source ZIP was unit tested with an actual CPU forward/backward and mock Backbone; it has **not** been measured on the actual V100 hardware or real Net-06 35k Memory in this offline environment. Pre-submit check tests the actual STLAformer checkpoint + vendor on Gadi. Start with `--check-only` and consider the built-in `--smoke` option before the full jobs.
- This release implements the three new mechanisms and a **preliminary automated** method audit. It does not automatically train the entire full-factorial independent ablation matrix. The audit explicitly marks all TRC claims `INCONCLUSIVE` until that additional evidence is run.
- Validation currently uses the existing formal `val` cache (rather than an invented full s20/s40/s60/s90 validation stress suite); fixed-checkpoint s20/40/60/90 formal performance is still reported separately on Test at pipeline completion. Development decisions must not use Test.
- Performance targets and speed-up estimates are **unproven**. Legacy FARE weight initialization is forbidden in this release; training is from fresh FARE parameters.
