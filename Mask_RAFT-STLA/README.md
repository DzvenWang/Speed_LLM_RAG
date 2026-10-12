# Mask_RAFT-STLA v2.3.0

Independent, complete forecast-space historical-residual retrieval project for incomplete traffic sensing. **Breaking release**: old v2.1/v2.2 checkpoints and caches are deliberately unsupported. The project does **not** depend on Baseline, FARE or Support_Oracle; it depends on `Backbone` and its `Base_SupportBench` shared cache.

## New method

1. Mask-aware frozen STLA queries three scales of observed history for legal, non-overlapping historical Train-only windows.
2. `forecast_pool.py` computes a **forecast-conditioned shortlist**: historical similarity plus correspondence between **query STLA prediction** and **historical Train future**. It removes repeated Train indices, keeping `final_candidates=12` of the top-36 initial entries. At no point does it access query validation/test labels. Its selection key is the initial frozen STLA, not the fine-tuned student.
3. `normal_teacher.py` saves frozen-model normal prediction residuals. `replay_cache.py` computes the same Teacher's current-mask historical residuals *for the finalized shortlist only*. Both caches have new v2.3 names and strict file/backbone/index fingerprints.
4. `model.py` separately weights normal/failure residual components, ranks cases per node/horizon, computes disagreement, and applies a learned **risk gate** to the actual historical residual aggregate. There is **no unconstrained future head** in Full mode. The unrestricted adapter exists only in the separately trained No-retrieval control.
5. `trainer.py` supervises positive/negative candidate utility relative to None plus risk of the *combined* candidate correction, in addition to MAE/RMSE. The loss uses training truth for supervision only.

## Protocol

- `TraffiDent_Net_01/02/03/04/06` supported by dynamic node count; smoke/real formal available where the corresponding Backbone/SupportBench cache and pretrained STLA checkpoint exist.
- `speed` and `flow`, history/horizon `12/12`, Train s40; Test `s20/s40/s60/s90`, Seed 10 by default.
- Train-only `train_x_full_norm` plus `train_y_norm` is legal historical evidence. Evaluation query *future* labels are used only for metrics and offline audit, never to choose Retrieval candidates or produce predictions.
- Forecast-conditioned retrieval changes the historical candidate ranking; the improvement target of **≥6% Avg(s20–s90) MAE relative to RAFT-STLA** is a research objective, not an established result.
- Still uses the single s40 Validation MAE for checkpoint selection. An all-severity Validation selector would require actual Val views under a separate protocol; never select on Test.

## Gadi

Place this folder directly under `/scratch/mh20/zw1165/Speed_LLM_RAG`. Jobs need **1 V100, 12 CPUs, 32 GB host memory, 48h**, PyTorch environment `TFP`.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export STLA_S40_SPEED="$(pwd)/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export STLA_S40_FLOW="$(pwd)/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
python -u Mask_RAFT-STLA/scripts/preflight.py --target speed --backbone-checkpoint "$STLA_S40_SPEED"
qsub -N MRAFT23_speed -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_SPEED" Mask_RAFT-STLA/pbs/formal.pbs
qsub -N MRAFT23_flow  -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_FLOW" Mask_RAFT-STLA/pbs/formal.pbs
# Independently fitted no-retrieval controls (separate best.pt outputs)
qsub -N MRAFT23_noret_s -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,ABLATION=no_retrieval,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_SPEED" Mask_RAFT-STLA/pbs/formal.pbs
qsub -N MRAFT23_noret_f -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,ABLATION=no_retrieval,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_FLOW" Mask_RAFT-STLA/pbs/formal.pbs
```

A Formal run executes Train → `best.pt` / `last.pt` → four-severity Evaluate → Retrieval Audit. You may resume **this same v2.3 experiment only** using PBS `RESUME_CHECKPOINT=/absolute/path/last.pt`. Logs carry timestamps and GPU CSV; test metrics are under `Mask_RAFT-STLA_exps/<target>/<dataset>/formal/<stamp>/evaluation`.

## Re-running the independent attribution audit

The automatic audit has no independent control until that control is trained. Supply it afterward:

```bash
python -u Mask_RAFT-STLA/scripts/evaluate_retrieval.py \
  --checkpoint /absolute/full_run/best.pt \
  --no-retrieval-checkpoint /absolute/noret_run/best.pt \
  --severities 20 40 60 90 --reuse
```

The audit generates `retrieval_audit/paired_summary.csv`, `paired_horizon_family.csv` and `audit_manifest.json`. It tests `full`, `no_evidence`, `no_mask_replay`, **deterministic cross-history wrong Train cases**, and separately trained `no_retrieval`. `shuffled` remains as an additional same-query rank-mismatch diagnostic. Direct-masked and observed subsets cover all 12 prediction horizons (fixed from prior release).

## Repository files

- `mask_raft_stla/forecast_pool.py`: forecast-conditioned Train-future candidate selection and fingerprints.
- `mask_raft_stla/retrieval.py`: three-scale masked Pearson temporal-safe initial retrieval.
- `mask_raft_stla/evidence.py`, `normal_teacher.py`, `replay_cache.py`: legal historical future and fixed mask-conditioned residual banks.
- `mask_raft_stla/model.py`: per-node/per-horizon residual decomposition, scoring, risk gate.
- `mask_raft_stla/trainer.py`: progressive STLA finetuning, utility/risk losses, AMP, checkpointing.
- `scripts/evaluate_retrieval.py`: causal-intervention diagnostics and external No-retrieval comparison.

## Tested / limitations

Run `python -m pytest Mask_RAFT-STLA/tests -q`. Unit tests cover v2.3 shortlist, Train legal mismatch, mask-replay, neural gradients and toy cache generation. Real full Net-06 training and V100 peak memory have **not** been run in this release environment. v2.3 caches should not be mixed with earlier versions. In-sample normal residuals can still be optimistically biased; blocked out-of-fold historical residuals remain a research ablation, not a feature claimed here.


## Parallel STLA initialization modes (2026-10-11)

Submit with `BACKBONE_MODE=incomplete` (STLA path contains `/formal/`) or
`BACKBONE_MODE=complete` (STLA path contains `/formal_complete/`).
Both modes use the **same incomplete-sensing Base_SupportBench inputs**; only
the initial STLA checkpoint changes. Experiment roots, live logs and
retrieval caches
are separated by the mode component. Run the provided repository-root
`submit_mask_retrieval_dual_backbone.sh` to submit eight independent PBS jobs.
Do not interpret differences between complete/incomplete initializations as
Retrieval-only gains: compare independently trained NoRetrieval controls.
