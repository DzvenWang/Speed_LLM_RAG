# Mask_RAFT-STLA v2.0.0 — Failure-Conditioned Predictive Residual Retrieval

**A full replacement of Mask_RAFT-STLA v1.1.2.** The project is intended to live directly next to `Backbone` and `Base_SupportBench` in the `Speed_LLM_RAG` repository. It does **not** use FARE, Support_Oracle, STLA_DASR, the Baseline project, or a historical-imputation head.

## Research mechanism

For an incomplete query $(X_q^{obs},M_q)$, STLAformer produces $\hat Y_q^0$. Multi-resolution observed-feature Pearson search retrieves nonoverlapping **Train-only** examples $(X_k^{full},Y_k)$. We apply the **query's mask** to each shortlisted historical example, replay the current STLAformer and calculate the historical failure-conditional residual:

$$R_{k\mid q}=Y_k-f_{\mathrm{STLA}}(M_q\odot X_k^{full},M_q).$$

A differentiable horizon/node-level candidate reranker uses query observable context, historical masked distance, candidate future, residual, coverage and STLA features. A learnable **None** action can reject all candidates. The final correction is a bounded weighted sum of actual retrieved residuals, not an unconstrained correction MLP:

$$\hat Y_q=\hat Y_q^0 + c\tanh\!\bigl(\sum_k w_{q,k}R_{k\mid q}/c\bigr).$$

The saved config controls correction bound, ranking loss, MAE/RMSE balance, STLA finetuning, Top-K and replay chunk size. This is a **forecasting** model: no historical-reconstruction target or loss.

## Supported experimental protocol

- Datasets: `TraffiDent_Net_01/02/03/04/06`; node counts are resolved dynamically from STLA checkpoint and cache. Backbone shared-cache selection is made through its installed protocol config; generate the respective shared caches before running.
- Tasks: `speed` and `flow`; history/horizon 12/12, complete **training** history+future is used **only** as retrieval memory; incomplete query inputs and observation mask come from Base_SupportBench.
- Primary fair comparison: same **s40 incomplete-trained STLA checkpoint** as RAFT-STLA and RAST-STLA, Test s20/s40/s60/s90, five missingness families, seed 10 initially.
- Secondary study: complete-trained STLA, frozen/pretrained Backbone and the **same** retrieval-adaptation budget for all competing methods. Pass its exact checkpoint explicitly; do not mix primary/secondary tables.
- Every cached candidate search excludes overlapping temporal windows. Future targets in the evaluation split are used for metrics only, not retrieval or model forward.
- The Train Memory may contain rows with invalid future targets; those candidate positions are masked out of retrieval scoring.

## Folder structure

`mask_raft_stla/evidence.py`: Train-only memory, candidate shortlist, query-mask historical replay.
`mask_raft_stla/retrieval.py`: multi-resolution complete-history masked-Pearson search and time-overlap exclusion.
`mask_raft_stla/model.py`: horizon/node-level outcome-aware scoring, None action, evidence-only correction.
`mask_raft_stla/trainer.py`: train/val/eval losses, progressive or frozen schedule, MAE checkpoint selection, candidate ranking, RMSE risk term.
`scripts/build_retrieval_cache.py`: versioned/cache-input fingerprint validation.
`scripts/train.py`, `scripts/evaluate.py`: end-to-end formal run and automatic evaluation.
`pbs/formal.pbs`, `pbs/evaluate.pbs`: Gadi V100 submission.
`tests/test_raft_v2.py`: unit tests using dummy STLA only.

## Prerequisites

- Installed project `Backbone` (including the official STLAformer vendor module), pre-trained STLA checkpoint, and initialized shared Base_SupportBench dataset caches.
- PyTorch, NumPy, pandas, PyYAML and all dependencies required by the existing Backbone.
- STLA and cache dimensions, normalization and dataset ID must match. `preflight.py` verifies main shapes.
- Existing v1 Checkpoints are **not** compatible with v2. Old retrieval caches are rebuilt upon version or input change.

## Local PowerShell example

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:TRAFFIDENT_TARGET = 'speed'
$env:BACKBONE_CHECKPOINT = 'C:\path\to\your\s40_STLAformer\best.pt'
python -u Mask_RAFT-STLA\scripts\preflight.py --target speed
python -u Mask_RAFT-STLA\scripts\train.py --target speed --seed 10 --smoke
python -u Mask_RAFT-STLA\scripts\train.py --target speed --seed 10
```

For Flow, change `TRAFFIDENT_TARGET`, `BACKBONE_CHECKPOINT` and `--target` to `flow`. For other networks use `--dataset TraffiDent_Net_01` etc., with the matching STLA checkpoint/cache. `--batch-size` and `--num-workers` override defaults.

## Gadi commands

`pbs/formal.pbs` requests gpuvolta, **1 V100 + 12 CPUs**, 32GB, 48h; every output line is timestamped in the project's `*_logs/<target>` directory. Formal training automatically evaluates all four severities and saves independent per-severity predictions and metrics.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
qsub -N MRAFT2_speed -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,BACKBONE_CHECKPOINT=/absolute/path/to/s40_STLA_best.pt Mask_RAFT-STLA/pbs/formal.pbs
qsub -N MRAFT2_flow -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,BACKBONE_CHECKPOINT=/absolute/path/to/flow_s40_STLA_best.pt Mask_RAFT-STLA/pbs/formal.pbs
```

If the exact checkpoint is already the correct `LATEST_CHECKPOINT.txt`, `BACKBONE_CHECKPOINT` may be omitted, but explicitly specifying it is strongly recommended when complete/s40 variants coexist.

## Manual evaluation and ablations

```bash
python -u Mask_RAFT-STLA/scripts/evaluate.py --checkpoint /path/to/best.pt --severities 20 40 60 90
python -u Mask_RAFT-STLA/scripts/evaluate.py --checkpoint /path/to/best.pt --severities 40 --evidence-mode no_evidence
python -u Mask_RAFT-STLA/scripts/evaluate.py --checkpoint /path/to/best.pt --severities 40 --evidence-mode shuffled
python -u Mask_RAFT-STLA/scripts/evaluate.py --checkpoint /path/to/best.pt --severities 40 --evidence-mode no_mask_replay
python -u Mask_RAFT-STLA/scripts/train.py --target speed --seed 10 --ablation no_retrieval
python -u Mask_RAFT-STLA/scripts/train.py --target speed --seed 10 --ablation no_rerank
python -u Mask_RAFT-STLA/scripts/train.py --target speed --seed 10 --ablation standard_future
```

Supported training ablations: `full`, `no_retrieval` (separately trained forecast adapter), `no_evidence`, `no_mask_replay`, `no_rerank`, `standard_future`, `shuffled`. Inference-only evidence removal/shuffling proves dependence on evidence but **does not** substitute for the train-matched `no_retrieval` control. For an equivalent parameter-budget control, adjust the adapter width and report parameter counts explicitly.

Best Checkpoint includes the exact v2 config and SHA256 of the referenced Backbone checkpoint. If evaluating on another machine, set `SPEED_LLM_RAG_ROOT` and provide `--backbone-checkpoint /local/equivalent/best.pt` (matching SHA256). Evaluation writes `severity_summary.<evidence-mode>.csv` and per-severity family metrics and prediction NPZ files.

## What is tested and what is not

`pytest -q Mask_RAFT-STLA/tests` exercises evidence replay, candidate scoring and gradients. Additionally, the v2 code was executed against a **synthetic** STLA/Cache with the real CLI (`preflight`, two-epoch smoke, evaluation). These tests do **not** validate actual checkpoint compatibility with your unprovided Backbone source, Gadi V100 runtime, or improvements over the original RAFT/RAST-STLA. Formal Speed/Flow results and retrieval attribution are still required.

## Research caution

Residual replay uses the installed STLA backbone on training memory, which may have in-sample residual bias. It is a **working retrieval research implementation**, not an unbiased out-of-fold estimator. For publication-quality statements about historical residual distributions, add temporally blocked out-of-fold Backbone models and compare their utility. A future-derived ranking target is used **only in Train/Validation loss**; future truth is never supplied to the model forward in Test.
