# Migration: PreSuRe v0.7.6 -> v0.8.0

v0.8.0 stops expanding the numerical support-value router and introduces the first **Need-Triggered Qwen Support-Value Reasoning Probe**.

## What is reused unchanged

- Frozen PatchSTG backbone and M0--M5 artifacts.
- Need-q90 detector from v0.7.2.
- Generic Utility-A, Hard-q90, Pred10, and Pred20 experts.
- v0.7.6 proposal Meta-Ranker, retained only as a numerical baseline/proposal source.
- Train-derived predicted-Need thresholds and M3 Top4 retrieval artifacts.

No old experiment directory or checkpoint is overwritten.

## New local Qwen caches

The project is configured for the user-provided local HuggingFace cache roots:

```text
/scratch/mh20/zw1165/cache/huggingface/models--Qwen--Qwen2.5-0.5B-Instruct
/scratch/mh20/zw1165/cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct
```

`presure.qwen_reasoner.resolve_hf_snapshot()` resolves `snapshots/<revision>` locally. `local_files_only=true`; the v0.8.0 code never downloads models from the Internet.

## v0.8.0 task

For predicted-Need Top20 cells with at least two distinct frozen expert proposals, Qwen receives only deployment-safe evidence and returns one structured decision:

- `USE(P0..P3)`
- `REJECT`
- `ABSTAIN`

Qwen is **not** a traffic-speed forecaster and does not predict the future residual.

Train future is used only to construct supervised action/proposal labels. Validation future is used only for reporting. `direct_mask`, `support_mask`, true-q90, current-query future, and true residual are never prompt features.

## Dataset and evaluation

The dataset builder creates:

- `reason_train.jsonl`
- `reason_dev.jsonl`
- `reason_final_val.jsonl`

Training is query-disjoint within Train `utility_dev`. Final Validation remains untouched by SFT. The final probe reports action accuracy, USE/REJECT/ABSTAIN precision/recall, decision utility, selection regret, selected beneficial/harmful rate, and harmful-regret rate.

The project compares:

1. Generic proposal, always use.
2. v0.7.6 Meta proposal, always use.
3. Qwen zero-shot.
4. Qwen LoRA-SFT.
5. Privileged oracle action.

The v0.7.6 absolute pairwise-rank thresholds are intentionally not reused because the candidate universe changed in v0.7.6. v0.8.0 decisions are made with decision utility/regret and harmful-support metrics.

## Recommended run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_static.py
python tests/smoke_qwen_support_value_reasoning.py
python scripts/qwen_reasoning_preflight.py --stage development
bash scripts/submit_qwen_support_value_suite.sh
```

The suite submits the dataset build, 0.5B/1.5B zero-shot probes, 0.5B/1.5B LoRA training, LoRA probes, and final summary with PBS dependencies.
