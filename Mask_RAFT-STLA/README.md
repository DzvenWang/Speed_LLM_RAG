# Mask_RAFT-STLA

Full mask-aware RAFT + STLAformer for TraffiDent incomplete sensing. This project is independent of `Baseline`; it uses only the durable `Base_SupportBench` data contract and the official STLAformer source/checkpoint managed by `Backbone`.

## Architecture

- **Mask-aware STLAformer**: zero-initialized mask context (`observed`, node coverage, time coverage, missing age) is injected into the official STLAformer embedding while preserving the pretrained model at initialization.
- **Mask-aware periodic RAFT retrieval**: periods `[4,2,1]`; Pearson similarity is computed only on observed query evidence. Scores are coverage-regularized and overlapping train windows are excluded by true `window_start` intervals.
- **Full-train memory**: queries are incomplete, retrieval keys and futures are complete Train histories/futures.
- **Reliability-aware fusion**: Top-K similarity, margin, entropy confidence and observation coverage drive period routing and a node × horizon gate.
- **End-to-end optimization**: pretrained STLAformer initialization, then warm-up → partial joint training → full joint fine-tuning. Final checkpoints contain the whole model.

## Required layout

Place this directory under `Speed_LLM_RAG/` beside `Base_SupportBench/` and `Backbone/`. The official STLAformer source must exist at `Backbone/vendor/STLAformer/model/STLformer.py`, and a formal STLAformer checkpoint must exist under `Backbone_exps/<target>/STLAformer/TraffiDent_Net_06/formal/` or be supplied with `MASK_RAFT_STLA_BACKBONE_CHECKPOINT`.

## Commands (PowerShell)

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:TRAFFIDENT_TARGET = "speed"

python -u Mask_RAFT-STLA\scripts\preflight.py --target speed
python -u Mask_RAFT-STLA\scripts\build_retrieval_cache.py --target speed --split train
python -u Mask_RAFT-STLA\scripts\build_retrieval_cache.py --target speed --split val
python -u Mask_RAFT-STLA\scripts\train.py --target speed --seed 10
```

Evaluation after training:

```powershell
python -u Mask_RAFT-STLA\scripts\evaluate.py --target speed --checkpoint <best.pt>
```

Smoke:

```powershell
python -u Mask_RAFT-STLA\scripts\build_retrieval_cache.py --target speed --split train --smoke --force
python -u Mask_RAFT-STLA\scripts\build_retrieval_cache.py --target speed --split val --smoke --force
python -u Mask_RAFT-STLA\scripts\train.py --target speed --seed 10 --smoke
```

For flow, replace `speed` with `flow`.
