# PreSuRe v0.7.4 -> v0.7.5

v0.7.4 showed that the Hard-q90 specialist improves support discrimination when the **true** hard regime is known, but predicted Need-q90 is not a sufficiently precise hard routing variable. v0.7.5 therefore aligns specialist training with the actual deployable predicted-Need distribution and learns expert preference directly.

## New diagnostic

**PreSuRe v0.7.5 — Deployment-Aligned Specialist + Expert-Preference Routing Probe**

1. Estimate horizon-specific PredNeed thresholds from **Train predicted Need-q90 only**.
2. Train `Pred10` and `Pred20` utility specialists on Train cells selected by those frozen predicted-Need thresholds.
3. Reuse the v0.7.4 Hard-q90 specialist and v0.7.1 Generic Utility-A.
4. Train an Expert-Preference Router on Train-only `utility_dev` to choose among:
   - Generic Utility-A
   - Hard-q90 Specialist
   - Pred10 Specialist
   - Pred20 Specialist
5. Evaluate once on v0.7.2 final Validation IDs using the frozen Train thresholds.
6. Report a privileged `oracle_router` ceiling only as a diagnostic.

## Leakage contract

- Current-query future is never an inference feature.
- Pred10/Pred20 membership is defined from predicted Need-q90, with thresholds estimated from Train only.
- True support utility is Train-only supervision for specialist training.
- Expert-preference labels use Train-only `utility_dev` future supervision.
- Validation true-q90 and oracle expert routing are diagnostic only.
- direct/support masks are evaluation views only and are not inference features.

## Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_deployment_aligned_utility_router_probe.py
python scripts/deployment_aligned_router_preflight.py --stage development
bash scripts/submit_deployment_aligned_utility_router_probe.sh
```

Main outputs:

- `train_only_predicted_need_thresholds.json`
- `pred10_specialist_best.pt`
- `pred20_specialist_best.pt`
- `expert_preference_router_best.pt`
- `router_train_only_split.json`
- `deployment_router_validation_summary.json`
- `deployment_router_comparison.csv`
- `summary.json`
- `DECISION.txt`
