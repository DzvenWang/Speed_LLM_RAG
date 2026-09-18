# PreSuRe v0.9.0 -> v0.9.1 (v0.9A.1 Safe Top-Selection)

v0.9A established that node-horizon predictive-utility supervision can raise Masked pairwise ranking from approximately 0.50 (coarse retrieval) to 0.86, while reducing strong harmful selection.  However, its final development run remained NO_GO because selected beneficial support was only about 52%, selected harmful support remained about 47%, and selection-regret reduction was only about 4.2%.

v0.9A.1 therefore does **not** change the candidate pool, PatchSTG backbone, support memory, or Qwen.  It keeps the same leakage-safe K=24 candidates and warm-starts the full v0.9A reranker checkpoint.  The change is isolated to top-selection learning and safe deployment policy.

## New training objectives

- listwise best-candidate loss on cells with a beneficial candidate;
- focal beneficial/harmful sign classification under the natural candidate distribution;
- an absolute score barrier that pushes clearly beneficial support above a positive margin and clearly harmful support below a negative margin;
- retained node-horizon utility regression, pairwise ordering, and similar-but-harmful ranking.

The deployment score is

`mu + lambda * beneficial_logit - beta * sigma`.

`beta` and `lambda` are selected using Train-dev only.  Checkpoint selection is no longer dominated by pairwise rank; it emphasizes beneficial selection, harmful/strong-harm suppression, and regret.

## Safe no-support policy

After selecting the best score form, v0.9A.1 calibrates an ABSTAIN threshold on Train-dev only.  If the top support score falls below that threshold, PreSuRe retains the protected PatchSTG forecast.  Calibration uses an explicit minimum coverage of 35%, so the system cannot obtain a good harmful rate simply by rejecting every query.

## Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_predictive_utility_safe_selection.py
python scripts/predictive_utility_safe_selection_preflight.py --stage development
bash scripts/submit_predictive_utility_safe_selection_probe.sh
```

The development run requires the completed v0.9A checkpoint referenced by:

`PreSuRe_exps/PredictiveUtilityRetrieval/TraffiDent_Net_06/development/LATEST.json`.

## Main outputs

The new experiment root is:

`PreSuRe_exps/PredictiveUtilitySafeSelection/TraffiDent_Net_06/development/<stamp>_seed10/`

Key files are `summary.json`, `comparison.csv`, `policy_comparison.csv`, and `DECISION.txt`.
