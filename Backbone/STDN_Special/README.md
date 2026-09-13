# STDN_Special v4.1

`STDN_Special` is a diagnostic harness for STDN training-recipe attribution. It is **not** a separate data pipeline.

## v4.1 change

Temporal-feature handling is now owned by `backbone/time_features.py` for every backbone. The old `install_backbone_time_fix.py` mechanism has been removed. `STDN_Special/time_features.py` is only a compatibility re-export of the shared implementation.

The diagnostic runner still:

- overrides only the in-memory `cfg['models']['STDN']` recipe;
- never edits `vendor/STDN`;
- asserts effective batch size/LR/weight decay/patience;
- blocks training if the shared cache time-feature audit fails;
- disables fallback batch sizes during controlled diagnostic runs.

## Required cache state

From the Backbone root:

```bash
python scripts/prepare_data.py --stage screening --force
python scripts/validate_cache.py --stage screening
python STDN_Special/audit_time_features.py --stage screening
```

The audit must PASS before any diagnostic GPU job is submitted.

## Priority diagnostics

```bash
python STDN_Special/run_diagnostic.py \
  --experiment e2_pat30 \
  --stage screening \
  --seed 10 \
  --preflight-only

bash STDN_Special/submit_diagnostics.sh
```

Current priority experiments remain:

- `e2_pat30`: batch 64, LR 2e-4, WD 1e-4, patience 30.
- `e5_wd0`: batch 64, LR 2e-4, WD 0, patience 10.

No diagnostic recipe is automatically promoted into the official benchmark configuration. Promotion requires an explicit edit to `configs/backbone_selection.yaml` followed by a normal official STDN rerun.
