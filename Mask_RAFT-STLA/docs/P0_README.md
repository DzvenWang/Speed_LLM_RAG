# P0 Retrieval Value Diagnostics · v2.3.0 · standalone project

This is a strict **post-hoc diagnostic + independently trained no_retrieval control** on the original v2.3 model. No changes were made to the model forward/training loss, existing PBS formal runner, Evaluate, checkpoint layouts, or Train Memory rules.

## Key distinctions

1. `Full vs base/no_evidence/no_future/cross_history` are **within the same trained Full checkpoint**; they are not independently trained baseline gains.
2. `Full vs no_retrieval` is an independent comparison **only** after training and providing a second `no_retrieval` checkpoint with identical Backbone SHA, data/seed, optimizer/epoch schedule (it has its own learned no-retrieval adapter).
3. `p0_candidate_probe.py` measures post-hoc best **single** retrieved candidate-or-zero, using test labels **only for analysis**, not for candidate selection. It is not a practical oracle and not a mathematical upper bound on mixtures of evidence.
4. Family/support stratifications and per-query gains are computed on the exact same rows, horizons and masks. Element-weighted P0 summary is separated from the query-equal-weight summary. Bootstrap query CIs are descriptive: overlapping windows are not independent.
5. The P0 gate calibration target is the sign of the raw **ungated** correction improvement, matching the training risk-label definition. Calibration is about this proposal, not final full prediction calibration.

## Usage

```bash
# Full v2.3 checkpoint already present; no Full retraining necessary.
python -u Mask_RAFT-STLA/scripts/p0_diagnose.py \
  --checkpoint "$FULL_CHECKPOINT" --ensure-audit --severities 20 40 60 90

# Hindsight candidate/oracle and gate/router calibration; GPU V100 recommended.
python -u Mask_RAFT-STLA/scripts/p0_candidate_probe.py \
  --checkpoint "$FULL_CHECKPOINT" --severities 20 40 60 90 --batch-size 32

# Independent control: submit as a new 1xV100 PBS job; same initializer.
qsub -N P0_RAFT_NR -v TARGET="$TARGET",SEED=10,DATASET=TraffiDent_Net_06,BATCH_SIZE=64,BACKBONE_MODE=incomplete,BACKBONE_CHECKPOINT="$BACKBONE_CHECKPOINT" \
  Mask_RAFT-STLA/pbs/p0_no_retrieval.pbs

# After control completes, rerun paired P0 analysis with --no-retrieval-checkpoint.
python -u Mask_RAFT-STLA/scripts/p0_diagnose.py \
  --checkpoint "$FULL_CHECKPOINT" --no-retrieval-checkpoint "$NO_RET_CHECKPOINT" --ensure-audit
```

The analysis will require Base_SupportBench/Backbone test caches and Train-only Memory as the original v2.3 evaluator does. Never use labels to fetch/select historical candidates in deployment. The probe is expensive: it requires new inference but not training. `--max-queries 256` is useful for pilot; `0` means the full formal Test set.

### Outputs

Under `<full_checkpoint_dir>/p0_retrieval/`:
- `p0_query_level.csv` / `p0_group_summary.csv`: per-query errors, benefit/harm, masked vs unmasked, failure family and coverage
- `p0_element_weighted_summary.csv`: MAE using the original valid-element protocol
- `p0_severity_summary.csv` / `p0_descriptive_bootstrap.csv`: query means and descriptive uncertainty
- `p0_manifest.json`: hashes, protocol, whether an independent control exists
- `p0_candidate_queries.csv` / `p0_candidate_summary.csv`: candidate utility-oracle, None, Gate, region breakdown
- `p0_risk_calibration.csv` / `p0_risk_ece.csv`: risk gate probability vs empirical benefit frequencies
- `p0_candidate_manifest.json`: oracle definition and checkpoint provenance

Do not interpret `oracle_candidate_gain` as a deployable method or as evidence that model selection would obtain this gain without ground truth.
