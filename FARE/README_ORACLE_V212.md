# FARE v2.1.2 — Native Candidate-Level Oracle Audit

**Base:** The uploaded `FARE.zip` FARE **v2.1.1**, NOT v2.0.4. This is a **complete updated project**, with diagnostic-only extension to the v2.1.1 graph/node retriever. Original Stage Evidence → Retrieval → Integration → Control, STLAformer backbone, loss functions, optimizer stages, inference outputs, cache contracts and training PBS remain unchanged. A model trained by 2.1.1 is loadable strictly (no new trainable model parameters).

## Scientific purpose

At each sampled query and target region, compare actual FARE predicted future from:

- **Real**: the trained node-specific Top-8 evidence mixture.
- **Null**: target-region evidence turned off, with all other regions unchanged.
- **Retriever selected Top-1**: trained node-specific first-ranked evidence forced to weight 1 per target node (a diagnostic intervention).
- **Forced singleton candidates**: each **causally legal** historical window in the region's actual mixed **Top-64 coarse pool**, replayed through the *trained* MaskConditionedEvidenceAdapter, Utility head, STLAformer retrieval injection and Final Control. Every target-region node uses the same historical episode; invalid-history-future target nodes are nulled.
- **Best evaluated singleton / Null option**: retrospective label-based diagnostic, NEVER an inference policy.
- **Uniform random singleton expectation**: mean actual MAE gain over evaluated singletons, NOT a newly trained random retriever.

The single-region conditional utility is:

`gain(q,g,k) = MAE(target-region null baseline) - MAE(full FARE with candidate k in region g)`

Positive gain means the candidate helps the current region relative to switching off **only this region's retrieval evidence**. `std` is applied to normalized MAE to show Speed/Flow in original units.

**Do not call this a global Oracle@64 upper bound.** The candidate set is Top64 after existing *diversified* coarse recall, one candidate is forced over all target-region nodes, and all other regions retain their Real evidence. Blended Real Top8 may beat the best singleton. Label-informed oracle selection is only for audit, never inference. The logged `ranker_candidate_spearman`/`ranker_predicted_top1_gain` permit separating poor ranking from poor evidence availability.

## Installation

From `Speed_LLM_RAG` repository root on Gadi, **back up FARE first** and unzip this archive's `FARE/` over the existing `FARE/` directory. Do **not** overwrite the entire repository, Backbone, Base_SupportBench or trained experiment directories.

```bash
cp -a FARE "FARE.backup.pre_v212.$(date +%Y%m%d_%H%M%S)"
# Example: unzip -o FARE_v2.1.2_Native_Oracle_Audit.zip -d .

conda activate TFP
python -m compileall -q FARE/fare FARE/scripts
python -m pytest -q FARE/tests/test_candidate_oracle_v212.py
bash -n FARE/pbs/oracle_audit.pbs
python -u FARE/scripts/check_metric_protocol.py
```

Gadi's real Base_SupportBench, official graph, STLAformer checkpoint, original historical memory and FARE trained model are required for the audit itself. All historical data are TRAIN-only; Test labels are not passed to model forward, candidate scoring or memory retrieval.

## Find previously trained checkpoint

```bash
find FARE_exps/speed/TraffiDent_Net_06/pilot -name candidate_best.pt -printf '%T@ %p\n' | sort -nr | head -5
find FARE_exps/flow/TraffiDent_Net_06/pilot -name candidate_best.pt -printf '%T@ %p\n' | sort -nr | head -5
```

Assign the exact paths for the v2.1.1 runs:

```bash
export FARE211_SPEED="$(pwd)/FARE_exps/speed/TraffiDent_Net_06/pilot/<run_id>/candidate_best.pt"
export FARE211_FLOW="$(pwd)/FARE_exps/flow/TraffiDent_Net_06/pilot/<run_id>/candidate_best.pt"
ls -lh "$FARE211_SPEED" "$FARE211_FLOW"
```

### Smoke (Speed first)

```bash
qsub -N FARE212_S_smoke \
  -v "TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,FARE_CHECKPOINT=$FARE211_SPEED,FARE_ORACLE_SMOKE=1" \
  FARE/pbs/oracle_audit.pbs
```

Smoke runs **2 query windows × 1 region × 4 coarse candidates** for Validation and Test s40. Check PBS log for `FARE212_AUDIT_PBS_PASS` and `FARE_ORACLE_COMPLETE`. If Checkpoint/load issues occur, the script fails clearly. **Do not disable pool replay consistency in normal operation.**

### Full diagnostic Pilot on prior v2.1.1 checkpoints

```bash
qsub -N FARE212_speed \
  -v "TARGET=speed,SEED=10,FARE_CHECKPOINT=$FARE211_SPEED,MAX_QUERIES=12,MAX_REGIONS=2,MAX_CANDIDATES=64" \
  FARE/pbs/oracle_audit.pbs

qsub -N FARE212_flow \
  -v "TARGET=flow,SEED=10,FARE_CHECKPOINT=$FARE211_FLOW,MAX_QUERIES=12,MAX_REGIONS=2,MAX_CANDIDATES=64" \
  FARE/pbs/oracle_audit.pbs
```

Default evaluations: one Validation split and four Test severities (s20/s40/s60/s90). Each group evaluates a *deterministic small sample* (12 query windows, 2 regions). This is **not** the comprehensive Test MAE/RMSE used for publication.

To inspect more queries after a first smoke, raise MAX_QUERIES / MAX_REGIONS. Top64 singleton replay can be expensive; use the run's progress log to estimate resource requirements. 1 V100 / 12 CPUs / 32GB RAM / 24h PBS is requested. Audit batch size is **1 by design**, unrelated to BS32 training.

### Direct command (useful for local/Gadi debugging)

```bash
python -u FARE/scripts/run_oracle_audit.py \
  --checkpoint "$FARE211_SPEED" --expect-target speed \
  --split both --severities 20 40 60 90 \
  --max-queries 12 --max-regions 2 --max-candidates 64
```

## Outputs

`FARE_exps/<target>/<dataset>/oracle_audit/<run>/`

- `audit_manifest.json`: exact Checkpoint and Memory SHA-256, sampling, causality and scope.
- `oracle_records.jsonl`: per-query/region singleton records, full trained ranker scores, positive/harmful candidate indicators and direct-mask MAE.
- `oracle_summary.csv` / `.json` / `.md`: split- and severity-specific aggregate diagnostics.

The JSONL is append-only and can be resumed with `--output <existing_folder> --resume`; mismatching Checkpoint/Memory or protocol settings are rejected.

## Interpretation

1. **Weak singleton oracle and weak Real**: available candidate evidence or transfer mechanism may be deficient. The diagnostic only tests shared-episode singleton transfer, so this is not proof that a better adapter/multicandidate combination is impossible.
2. **Strong oracle, weak Real, poor Ranker Spearman**: retrieval scoring or its supervision likely needs redesign.
3. **Strong ranker, weak Real**: candidate integration, transport and correction control are possible bottlenecks.
4. **Speed s90**: compare positive candidate fraction, masked gain and candidate eligibility with lower severities.

No Train/Validation/Test data split is modified. The source has no dependency on deleted projects STLA_DASR or Support_Oracle. No new learnable architecture, Residual Memory, or counterfactual training objective is implemented in this **diagnostic release**; those should be decided *after* Oracle evidence is measured.

## Tests and limitations

- New native integration tests check exact forward replay, safe label separation, sparse-node Adapter parity, condition-local interventions, causal legality and reports.
- FARE's original training gradient, causal retrieval and forecast-forward regression tests have also been run.
- Some original end-to-end/metrics tests require the live `Base_SupportBench.experiment_common.metrics` module, which is not bundled in `FARE.zip`. Never change official metrics to make local tests pass. Run the full suite in the deployed repository.
- The updated project has **not** been run on a real Gadi STLAformer v2.1.1 checkpoint here. PBS smoke is still mandatory.
