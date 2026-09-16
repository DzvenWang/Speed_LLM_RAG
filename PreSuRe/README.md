# PreSuRe v0.7.5 — Deployment-Aligned Specialist + Expert-Preference Routing Probe

v0.7.5 is the current development diagnostic release. v0.7.4 established an important asymmetry: the Hard-q90 specialist improves utility discrimination when the true hard regime is known, but direct routing by predicted Need-q90 does not preserve that advantage. v0.7.5 therefore tests whether the remaining bottleneck is **deployment-distribution alignment and expert preference**, rather than another Need threshold.

The probe compares four deployable utility experts (`Generic`, `Hard-q90`, `Pred10`, `Pred20`) and an Expert-Preference Router. Pred10/Pred20 are trained on cells selected by Train-derived predicted-Need thresholds. The router is trained only on Train `utility_dev` to learn which expert's selected support has higher true utility; current-query future is never an inference feature. A privileged `oracle_router` is reported only as a ceiling.

## Run on Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_static.py
python tests/smoke_deployment_aligned_utility_router_probe.py
python scripts/deployment_aligned_router_preflight.py --stage development
bash scripts/submit_deployment_aligned_utility_router_probe.sh
```

Primary outputs are `deployment_router_validation_summary.json`, `deployment_router_comparison.csv`, `train_only_predicted_need_thresholds.json`, `expert_preference_router_best.pt`, `summary.json`, and `DECISION.txt`.

The decisive deployment slice is **Masked / frozen Train-threshold PredTop10**. `true_q90` and `oracle_router` are privileged diagnostics only. See `MIGRATION_v0.7.4_to_v0.7.5.md`.

# PreSuRe v0.7.6 — Regret-Aware Support Proposal Meta-Ranking Probe

v0.7.6 is the current development diagnostic release. v0.7.5 showed that the expert pool has substantial privileged complementarity, but ordinary expert classification does not translate into better support selection. v0.7.6 therefore removes discrete expert classification as the primary objective.

For every node x horizon cell, the frozen `Generic`, `Hard-q90`, `Pred10`, and `Pred20` experts each submit one Top1 support proposal. Duplicates are merged. A new proposal-level meta-ranker estimates which proposal is actually useful using only deployment-safe query/support context, predicted Need-q90, expert score/confidence summaries, M3 confidence, historical residual summaries, and expert agreement.

The loss is explicitly utility aligned: `pairwise ranking + utility regression + beneficial/harmful classification + expected selection regret`. Unlike v0.7.5, ambiguous expert cases are retained instead of discarded. Small utility differences naturally create small regret, whereas selecting a harmful proposal when a beneficial one exists creates a large penalty.

`direct_mask` is allowed only as Train weighting and evaluation view; it is not a model feature. Current-query future is used only for Train proposal-utility supervision and final Validation reporting.

## Run on Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_static.py
python tests/smoke_regret_aware_support_proposal_meta_probe.py
python scripts/proposal_meta_ranker_preflight.py --stage development
bash scripts/submit_regret_aware_support_proposal_meta_probe.sh
```

Primary outputs are `proposal_meta_validation_summary.json`, `proposal_meta_comparison.csv`, `proposal_meta_ranker_best.pt`, `summary.json`, and `DECISION.txt`. The decisive deployment slice is **Masked / PredTop10**. See `MIGRATION_v0.7.5_to_v0.7.6.md`.
