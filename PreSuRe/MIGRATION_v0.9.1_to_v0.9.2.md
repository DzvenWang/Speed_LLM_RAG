# Migration: PreSuRe v0.9.1 -> v0.9.2 (v0.9A.2 Rank-then-Veto)

## Why this revision exists

The completed v0.9A.1 experiment showed that jointly fine-tuning ranking, Top-1 classification, sign separation, and abstention degraded the useful v0.9A ranking representation.  However, its strong-harm rate dropped substantially.  v0.9A.2 therefore separates the two responsibilities:

1. **Utility ranking is frozen** from the completed v0.9A checkpoint.
2. **Safety is learned independently** by a strong-harm veto classifier.

The veto is not allowed to reorder candidates.  At inference it checks candidates in the frozen v0.9A order.  A candidate is skipped only when its strong-harm probability exceeds the Train-dev-calibrated veto threshold.  If all candidates within the calibrated inspection depth are vetoed, the policy abstains and keeps the protected PatchSTG forecast.

## Controlled experimental contract

- Reuses the exact v0.9A K=24 candidate pool.
- Reuses the exact v0.9A `train_ids.npy`, `dev_ids.npy`, and `final_val_ids.npy`.
- Loads the exact v0.9A best checkpoint from `PredictiveUtilityRetrieval/.../LATEST.json`.
- Freezes every v0.9A ranker parameter.
- Strong-harm supervision is `relative_utility <= -0.02` on Train only.
- Veto threshold and maximum inspection depth are selected on Train-dev only.
- Qwen and Dual Memory remain disabled.

## New files

- `presure/harm_veto.py`
- `scripts/run_predictive_utility_rank_then_veto_probe.py`
- `scripts/predictive_utility_rank_then_veto_preflight.py`
- `scripts/submit_predictive_utility_rank_then_veto_probe.sh`
- `pbs/predictive_utility_rank_then_veto_probe.pbs`
- `tests/smoke_predictive_utility_rank_then_veto.py`

`PredictiveUtilityRerankerV09A.forward_features()` and `StrongHarmVetoHead` were added in `presure/models.py` without adding parameters to the old v0.9A ranker, so existing v0.9A checkpoints remain state-dict compatible.

## Recommended run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_predictive_utility_rank_then_veto.py
python scripts/predictive_utility_rank_then_veto_preflight.py --stage development
bash scripts/submit_predictive_utility_rank_then_veto_probe.sh
```

Main outputs are written under:

```text
PreSuRe_exps/PredictiveUtilityRankThenVeto/TraffiDent_Net_06/development/<timestamp>_seed10/
```

Review `summary.json`, `policy_comparison.csv`, and `DECISION.txt`.

## Hotfix1 after job 179261672

The original v0.9.2 runner dropped `rscore` while unpacking `model_outputs()` inside `policy_grid_eval()` and `fixed_policy_eval()`, causing `NameError: rscore is not defined`. Hotfix1 explicitly retains the frozen v0.9A rank score in both paths and centralizes ordering in `presure.harm_veto.frozen_rank_order()`. This is an implementation-only correction; the v0.9A.2 experimental design is unchanged.
