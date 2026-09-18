# PreSuRe v0.8.0 -> v0.9.0 (v0.9A Predictive-Utility Retrieval)

## Why this revision exists

The v0.7.x diagnostics showed that useful support is often present in the broad candidate pool, while deployed support selection remains close to a beneficial/harmful split.  Additional specialist/router/meta-router layers gave only marginal gains.  v0.8.0 further showed that Qwen can be precise when it elects to USE a support, but its coverage collapses when the numerical evidence is weak.

v0.9A therefore moves capacity *before* Qwen and removes the early M3 Top-4 information bottleneck.

## New mainline probe

`PredictiveUtilityRetrieval` scores the same leakage-safe 24-candidate pool directly at node x horizon granularity:

- future-grounded Train target: best-alpha relative predictive utility;
- shared missing-aware query/support state encoder (warm-started from M1, then fine-tuned);
- full historical residual trajectory as a Train-memory input;
- expected utility `mu`;
- learned uncertainty `sigma`;
- auxiliary beneficial/harmful probability;
- hard negative mining biased toward high-similarity but harmful candidates;
- risk score `mu - beta*sigma`, with beta selected on Train-dev only.

## Intentionally unchanged

- PatchSTG backbone;
- leakage-safe coarse candidate construction (64 -> first 24 for this probe);
- existing M1/M3/M4/specialist/router/meta/Qwen code and checkpoints, retained as baselines;
- no Dual Memory yet;
- no new Qwen training yet;
- no Predictive-Need feature is injected into support identity ranking.

## Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python scripts/predictive_utility_reranker_preflight.py --stage development
bash scripts/submit_predictive_utility_reranker_probe.sh
```

## GO target

On the held-out validation diagnostic, the selected risk-aware variant targets:

- Masked pairwise rank accuracy >= 0.67;
- selected beneficial rate >= 0.60;
- selected harmful rate <= 0.35;
- mean selection-regret reduction >= 20% relative to coarse similarity.

Do not proceed to Dual Memory / Qwen redesign if the v0.9A support-value identification layer remains below these thresholds.
