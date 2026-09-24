# PreSuRe v0.7.5 -> v0.7.6

v0.7.5 showed a strong gap between the learned Expert-Preference Router and the privileged Oracle Router. The four frozen experts (`Generic`, `Hard-q90`, `Pred10`, `Pred20`) are complementary, but classification-based expert routing is misaligned with the real objective: selecting a support with high predictive utility and low harmful-regret.

## New diagnostic

**PreSuRe v0.7.6 — Regret-Aware Support Proposal Meta-Ranking Probe**

1. Reuse all four v0.7.5/v0.7.4/v0.7.1 utility experts without retraining them.
2. Each expert contributes its Top1 support proposal for each node x horizon cell.
3. Duplicate proposals are merged, while expert agreement is retained as a feature.
4. A proposal meta-ranker sees only deployment-safe information:
   - query/support node context;
   - predicted Need-q90;
   - frozen expert scores and margins;
   - M3 confidence;
   - historical support residual summaries;
   - expert proposal agreement/disagreement;
   - horizon identity.
5. Train only on the Train-side `utility_dev` split. `direct_mask` is used only as Train weighting/evaluation view and is never an inference feature.
6. The training objective combines:
   - proposal pairwise ranking;
   - SmoothL1 proposal-utility regression;
   - clear beneficial/harmful classification;
   - expected selection-regret minimization.
7. Ambiguous samples are **not dropped**. If two proposals have nearly equal utility, their regret is naturally small.
8. Evaluate once on the v0.7.2 final Validation IDs with frozen Train thresholds.
9. Report a privileged `oracle_proposal` selector only as a diagnostic ceiling.

## Leakage contract

- Current-query future is never an inference input.
- Proposal utility uses future only as Train supervision or final Validation reporting.
- Need-q90 is predicted from deployment-safe state.
- Historical residuals come from Train support memory only.
- `direct_mask` / `support_mask` remain weighting/evaluation views only.
- Validation true-q90 and `oracle_proposal` are diagnostic only.

## Main outputs

- `proposal_meta_ranker_best.pt`
- `meta_ranker_train_only_split.json`
- `proposal_meta_validation_summary.json`
- `proposal_meta_comparison.csv`
- `summary.json`
- `DECISION.txt`

## Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_regret_aware_support_proposal_meta_probe.py
python scripts/proposal_meta_ranker_preflight.py --stage development
bash scripts/submit_regret_aware_support_proposal_meta_probe.sh
```

The decisive slice is **Masked / frozen Train-threshold PredTop10**. The main decision metrics are proposal pairwise rank accuracy, selected beneficial/harmful rates, mean selection regret, P90 regret, near-oracle rate, and harmful-regret rate.
