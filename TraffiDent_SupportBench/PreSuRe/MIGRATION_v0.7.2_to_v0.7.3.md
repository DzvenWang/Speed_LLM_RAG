# PreSuRe v0.7.2 -> v0.7.3

## Purpose

v0.7.2 established that future failure **need** is identifiable (especially q90), while future error direction is substantially weaker. v0.7.3 therefore stops trying to reconstruct a complete predictive-need state and tests a narrower deployable decomposition:

- **Need-q90** answers **WHEN / WHERE** external support is likely needed.
- frozen **Utility-A** answers **WHICH** M3 Top-4 historical support appears useful.

No M0-M5 model is retrained in v0.7.3.

## New diagnostic

`NeedGatedUtilityProbe` reuses:

1. the latest v0.7.1 `PredictiveNeedProbe` Utility-A checkpoint;
2. the latest v0.7.2 Need-q90 checkpoint/probabilities;
3. the v0.7.2 validation Top-4 candidate cache;
4. the v0.7.1 Train-only `utility_dev` Top-4 cache for policy calibration.

It reports Utility-A ranking conditioned on q90 Need score for:

- all valid cells;
- top 50% Need;
- top 20% Need;
- top 10% Need;
- top 5% Need.

## Train-only deployable policy calibration

A small grid is calibrated on v0.7.1 `utility_dev` only:

`Need-q90 threshold x Utility-A threshold x fixed residual alpha`.

The chosen fixed policy is then applied once to the v0.7.2 validation subset. Validation future is used only for final diagnostic metrics, never to select thresholds/alpha.

Outputs include:

- Overall / Masked / Support MAE and relative gains;
- activation rates;
- activated harmful / beneficial fractions;
- conditional Utility-A pairwise ranking and support benefit rates.

## Commands

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python scripts/need_gated_probe_preflight.py --stage development
bash scripts/submit_need_gated_utility_probe.sh
```

## Decision interpretation

- `GO_V08_NEED_GATED`: high-Need conditioning improves Utility-A reliability and the Train-calibrated policy improves masked forecasting without unacceptable side effects.
- `CONDITIONAL_V08_NEED_GATED`: one of ranking concentration or direct policy gain is positive, but evidence is not yet strong enough for a full v0.8 claim.
- `NO_GO_V08_NEED_GATED`: Need gating does not make the support reasoner materially safer/more useful.
