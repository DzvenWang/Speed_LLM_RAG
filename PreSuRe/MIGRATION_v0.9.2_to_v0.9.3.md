# Migration: PreSuRe v0.9.2 -> v0.9.3

## Why v0.9A.3 exists

The completed v0.9A.2 development probe showed that strong-harm detection is learnable (AUROC about 0.71 and high recall), but the sequential veto policy was too aggressive: it vetoed most v0.9A Top-1 candidates, fell back through many lower-ranked candidates, nearly eliminated strong harm but reduced beneficial capture and worsened policy regret. Therefore v0.9A.3 removes deep fallback and asks a deployment-aligned question only on the support the frozen v0.9A ranker would actually use.

## New controlled experiment

`v0.9A.3 Selective Utility Acceptance`:

1. Load the completed v0.9A `PredictiveUtilityRerankerV09A` checkpoint.
2. Freeze every v0.9A parameter and reuse its stored risk beta.
3. Reuse the exact v0.9A `train_ids`, `dev_ids`, and `final_val_ids`.
4. Reuse the same leakage-safe K=24 candidate pool.
5. Determine frozen v0.9A Top-1 independently for every node-horizon cell.
6. Train `SelectedSupportAcceptanceHead` **only on those Top-1 selections**.
7. Target is `selected predictive utility > 0`.
8. Near-zero utility receives reduced loss weight because its sign is unstable.
9. Calibrate a single acceptance threshold on Train-dev.
10. At inference: USE the frozen Top-1 or ABSTAIN. There is no candidate fallback and no re-ranking.

## New rank-context features

The acceptance head receives the frozen 96-D Top-1 role representation plus:

- `mu`, `sigma`, beneficial logit and frozen risk score;
- Top1-Top2 and Top1-Top3 frozen score margins;
- candidate-score mean and standard deviation;
- Top-1 score z-score;
- normalized frozen-score entropy;
- fraction of available candidates;
- relative uncertainty `sigma / (|mu| + 0.05)`.

No current-query future value is used as an inference feature.

## New files

- `presure/selective_acceptance.py`
- `scripts/run_predictive_utility_selective_acceptance_probe.py`
- `scripts/predictive_utility_selective_acceptance_preflight.py`
- `scripts/submit_predictive_utility_selective_acceptance_probe.sh`
- `pbs/predictive_utility_selective_acceptance_probe.pbs`
- `tests/smoke_predictive_utility_selective_acceptance.py`

`presure/models.py` adds `SelectedSupportAcceptanceHead`, and `configs/presure.yaml` adds `predictive_utility_selective_acceptance`.

## Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_predictive_utility_selective_acceptance.py
python scripts/predictive_utility_selective_acceptance_preflight.py --stage development
bash scripts/submit_predictive_utility_selective_acceptance_probe.sh
```

## What to inspect

The first diagnostic question is whether the selected-support classifier has meaningful Masked AUROC on the frozen Top-1 distribution. The policy question is whether selective acceptance can reduce harmful exposure and safe regret **without changing which candidate v0.9A ranks first**.

The important final metrics are:

- acceptance AUROC / Brier score;
- coverage;
- accepted positive-utility rate;
- accepted beneficial rate (utility >= 0.02);
- accepted harmful rate (utility < 0);
- mean policy relative utility;
- mean safe policy regret;
- regret reduction relative to frozen v0.9A Top-1.
