# Mask_RAST-STLA v2.0.0 — Observability-Conditioned Regional Future Evidence Retrieval

**Complete v2 replacement of v1.1.2.** Deploy alongside `Backbone` and `Base_SupportBench` inside `Speed_LLM_RAG`. The project does **not** import FARE, Support_Oracle, STLA_DASR or Baseline, and has no history-imputation head.

## Research mechanism

This design differs materially from Mask_RAFT-STLA: it transfers **historical future evolution** at the target node/neighbourhood, not a query-mask-conditioned historical STLA residual.

1. Build a **train-only statistical sensor association graph** by correlating normalized training histories. This is explicitly *not* claimed to be directed physical road connectivity.
2. Encode current observable history and mask using a genuine multi-head attention over statistical graph neighbours (`spatial_heads` is active), as well as a temporal encoder.
3. Retrieve global Temporal and Spatial candidate pools by learned keys and excluding temporal overlap; fetch the **actual historical future trajectories** from the train memory.
4. Represent each candidate's future movement by $Y_k-X_{k,last}$ and align it with the current last observed level (or observable graph boundary when the target node has no history). The current STLA forecast remains the unchanged reference forecast.
5. Use a query-conditioned, node/horizon-level differentiable reranker with observed boundary support, candidate state compatibility, candidate future and two retrieval-branch scores. The candidate attention includes a learnable **None** option.
6. Correct the STLA forecast only with bounded, weighted historic future-evolution evidence. No unconstrained posthoc residual network is used in the Full model.

Temporal, Spatial and None probabilities are returned separately. The candidate-ranking target is calculated using **Train/Validation future ground truth for loss only**. No Test `x_full` is ever read in the query Dataset or passed to the model.

## Experimental scope

- `TraffiDent_Net_01/02/03/04/06` (dynamic sensor number, Train-derived graph), Speed + Flow.
- History=12, horizon=12; Train s40 incomplete-sensing adaptation, Test s20/s40/s60/s90, all five failure families.
- A clean-pretrained (complete) STLA with frozen backbone is a secondary and separately labeled protocol; pass that exact checkpoint rather than silently switching Backbones.
- Train-only history+future; temporal-overlap filtering. No future target leakage, and positions with invalid Train future labels are excluded from candidate weighting.

## Folder structure

`mask_rast_stla/graph.py`: train-only statistical association graph with cache fingerprint.
`mask_rast_stla/encoders.py`: mask-aware temporal features, observable-neighbour multi-head spatial attention.
`mask_rast_stla/memory.py`: learnable temporal+spatial keys, nonoverlapping index search.
`mask_rast_stla/data.py`: per-network Train-only future repository, no Test x_full usage.
`mask_rast_stla/model.py`: temporal/regional history future transfer, outcome-aware node-level ranking, None action.
`mask_rast_stla/trainer.py`: index refresh, MAE model selection, differentiable forecast/ranking loss, RMSE risk, Frozen/Partial/Full STLA schedule.
`scripts/train.py`: Formal training with automatic Evaluate; `scripts/evaluate.py`: four severities and all failure families.
`pbs/formal.pbs`: Gadi gpuvolta, 1 GPU+12 CPUs, 32GB, 48h, timestamped live logs.
`tests/test_rast_v2.py`: isolated model, temporal index, graph and no-x_full Test dataset checks.

## Prerequisites

Same as existing STLAformer experiments: project `Backbone` including STLAformer vendor implementation and checkpoint, Base_SupportBench shared caches, PyTorch, NumPy, pandas, PyYAML. **v1 checkpoints and store indexes cannot be loaded as v2 checkpoints.** The best index saved with the v2 Checkpoint must travel with it for manual evaluation.

## Local commands

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:TRAFFIDENT_TARGET = 'speed'
$env:BACKBONE_CHECKPOINT = 'C:\path\to\s40_STLAformer\best.pt'
python -u Mask_RAST-STLA\scripts\preflight.py --target speed
python -u Mask_RAST-STLA\scripts\build_store.py --target speed --smoke
python -u Mask_RAST-STLA\scripts\train.py --target speed --seed 10 --smoke
python -u Mask_RAST-STLA\scripts\train.py --target speed --seed 10
```

For Flow, change target and use a matching Flow checkpoint. For additional networks pass `--dataset TraffiDent_Net_01` etc., build the respective shared cache and use the correctly dimensioned Backbone checkpoint.

## Gadi Formal

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
qsub -N MRAST2_speed -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,BACKBONE_CHECKPOINT=/absolute/path/to/speed_s40_STLA_best.pt Mask_RAST-STLA/pbs/formal.pbs
qsub -N MRAST2_flow -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,BACKBONE_CHECKPOINT=/absolute/path/to/flow_s40_STLA_best.pt Mask_RAST-STLA/pbs/formal.pbs
```

`pbs/formal.pbs` passes the selected dataset, target, seed and ablation to the trainer. Outputs use `Mask_RAST-STLA_exps/<target>/<dataset>/formal/<timestamp>_seed<seed>_<ablation>` and `Mask_RAST-STLA_logs/<target>`.

## Evaluate and attribution

```bash
python -u Mask_RAST-STLA/scripts/evaluate.py --checkpoint /path/to/best.pt --severities 20 40 60 90
python -u Mask_RAST-STLA/scripts/evaluate.py --checkpoint /path/to/best.pt --severities 60 --evidence-mode no_future
python -u Mask_RAST-STLA/scripts/evaluate.py --checkpoint /path/to/best.pt --severities 60 --evidence-mode shuffled
python -u Mask_RAST-STLA/scripts/evaluate.py --checkpoint /path/to/best.pt --severities 60 --evidence-mode no_regional
python -u Mask_RAST-STLA/scripts/train.py --target speed --seed 10 --ablation no_retrieval
python -u Mask_RAST-STLA/scripts/train.py --target speed --seed 10 --ablation no_regional
python -u Mask_RAST-STLA/scripts/train.py --target speed --seed 10 --ablation no_none
```

Important: evaluate on a relocated machine using identical Train cache/memory order and `--backbone-checkpoint` if the saved absolute STLA path does not exist. Evaluation verifies the STLA SHA256 and checks that Train Memory start times exactly match those of the saved index. Results contain per-severity, per-missing-family MAE/RMSE, Backbone and Full predictions and correction magnitude.

The adapter-only `no_retrieval` run trains an independent correction model. A shuffled or zeroed evidence inference run tests reliance on retrieval but does not replace a retrained control. Because the adapter has a different computational path, match effective parameter budget for strict causal attribution.

## Training details

- Candidate index refresh during training according to `store_refresh_epochs`; a fresh index is always constructed **after each training epoch for validation**, preventing checkpoint selection on stale keys.
- Forecast loss uses real-scale MAE by default. Ranking loss supervises candidate utility with training future labels, RMSE term discourages excessive errors.
- Frozen (warmup), partial and full STLA parameter schedules are included. Early stopping does not terminate a progressive schedule before the full phase starts.
- Test models never use the query's complete historical window. Test `y_raw` and `y_norm` are consumed by evaluation only.

## Validation status

This release passed unit tests and an end-to-end **synthetic** Backbone/Cache CLI smoke (preflight → two training epochs → saved checkpoint/index → Test evaluation). That does not demonstrate improvement on Net-06 or compatibility with the unprovided real Backbone vendor. The research claim must be established by real GPU Formal runs and evidence/no-retrieval ablations.

## v2.1.0 performance addendum

See `../MASK_RETRIEVAL_v2.1.0_RELEASE_NOTES.md` from the combined package for deployment, new AMP and fixed-replay controls, whole-run estimates and resume instructions. Existing Train-only evidence, no historical imputation, explicit s40 checkpoint, four-severity automatic evaluation and 12 CPUs per V100 are preserved. Use `--resume /absolute/path/to/last.pt` to recover an interrupted run. Default 32-batch/AMP is **not** identical numerical optimization to the prior 16-batch FP32 version.
