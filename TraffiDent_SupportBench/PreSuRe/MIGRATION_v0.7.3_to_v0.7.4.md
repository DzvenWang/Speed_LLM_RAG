# PreSuRe v0.7.3 -> v0.7.4

v0.7.3 showed that Need-q90 reliably detects future PatchSTG hard cases, but the generic Utility-A ranker becomes *less* reliable as predicted Need increases.  The high-Need regime also has more oracle support opportunity, so v0.7.4 tests the narrower hypothesis that support utility requires **regime-specialized learning** rather than another gate threshold.

## New diagnostic

`Need-q90 -> Generic vs Hard-q90 vs Need-weighted Utility -> Train-only General+Hard routing -> Validation comparison`

Four matched variants are reported:

1. **Generic** — frozen v0.7.1 Utility-A.
2. **Hard** — initialized from Utility-A and fine-tuned only on Train true-q90 cells.
3. **Weighted** — initialized from Utility-A and trained on all cells with deployable predicted Need-q90 weighting.
4. **Two-Expert** — Generic + Hard; routing threshold and mixture weight calibrated on Train-only `utility_dev`.

The specialist objective intentionally excludes utility regression, support mass and forecast loss. It uses only clear beneficial-vs-harmful classification plus pairwise support ranking. Ambiguous near-zero-utility candidates are excluded from the classification term.

True q90 is derived from the v0.7.2 Train-only horizon-specific thresholds. It is allowed only as Train supervision and as a privileged Validation subset for diagnostics. Inference uses predicted Need-q90 probability only.

## Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_need_conditioned_utility_specialist_probe.py
python scripts/need_conditioned_specialist_preflight.py --stage development
bash scripts/submit_need_conditioned_utility_specialist_probe.sh
```

## Main outputs

Under `PreSuRe_exps/NeedConditionedUtilitySpecialistProbe/...`:

- `hard_specialist_best.pt`
- `weighted_specialist_best.pt`
- `two_expert_train_only_calibration.json`
- `specialist_validation_summary.json`
- `specialist_comparison.csv`
- `summary.json`
- `DECISION.txt`

The primary deployable readout is Masked `pred_top10` Need-q90 ranking and harmful selection. `true_q90` is a privileged diagnostic ceiling only.
