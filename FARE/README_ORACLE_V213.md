# FARE v2.1.3 — Weight-Preserving Counterfactual Audit

## Exact code base and scope

This is the **complete FARE v2.1.1 native project**, carried through v2.1.2 and upgraded to v2.1.3. It is not the prior v2.0.4 demonstration code. `FARE/fare/model.py`, historical Train Memory, the learned STLAformer injection, the four training stages and their checkpoint parameters are **unchanged**. The new components are audit-only; no retraining and no changes to checkpoint schema are required. It works with the previous validation-selected 20261012_212838_seed10 checkpoints, subject to Gadi smoke testing.

v2.1.2 singleton Oracle forced the *same episode* with *unit weight* into every node of the target Region. This caused large negative singleton gains even where the learned Top-8 mixture was useful. v2.1.3 retains the original diagnostics but adds counterfactual interventions that preserve the original multi-evidence context and probabilities.

### New interventions

**LOO — leave one rank slot out.** For each of the Top-8 slots, remove that slot from every node in the selected Region. Move its exact original weight to the node's NULL option, without renormalizing the other seven slots. Then recompute the *real FARE* integration, retrieved-token attention and final controller. A positive `keep_gain_vs_removing = MAE(LOO)-MAE(Real)` means keeping that slot is beneficial. *Rank slot k contains potentially different historical memory IDs at different nodes.* This is not one-episode global leave-one-out.

**Same-weight replacement.** Replace the historical episode in a chosen Top-8 rank slot (`--replacement-slot 0` by default) with each causally legal Region Top-64 candidate. Keep the **original per-node slot weight**, original other slots and untouched other regions. Recompute candidate evidence, its Utility/Risk features, full FARE prediction. If the replacement episode is invalid for a node, assign its original slot mass to NULL rather than forcing an illegal history or renormalizing other evidence. This intervention isolates *candidate identity conditional on the original weight*, but still includes trained downstream utility/risk features: it is **not** evidence-content-only ablation. Candidate ID validity is verified before use.

**Node-specific LOO.** At most 4 nodes per Region by default: remove one selected rank slot for **one node only** and measure that node's horizon-wise MAE difference. Other nodes' evidence and other Regions stay unchanged. Set `MAX_NODES=0` to skip. `max_nodes=0` does **not** trigger a full-network node audit.

**Correction scaling.** On final prediction tensors, form `Y_alpha = Y_null + alpha*(Y_real-Y_null)` and measure the target Region MAE for alphas `[0,.25,.5,.75,1,1.25]`. Alpha 0 exactly reproduces target-region NULL; alpha 1 exactly reproduces Real. This is a **retrospective prediction-space interpolation/extrapolation**, **not** changing the existing gate or a deployable learned calibration model. Best alpha uses future truth **only for diagnostic**.

**Prior singleton Oracle remains available.** Its v2.1.2 output fields and `oracle_summary.*` remain; v2.1.3 adds `counterfactual_summary.*` and nested intervention records. Do not compare singleton and weight-preserving Oracle as if they were the same upper bound.

## Training, inference, and causality guarantees

- Model parameters unchanged: frozen validation-selected checkpoint, actual trained STLAformer, trained historical evidence adapter, retrieval injection and final control.
- Current future `y_norm`, `y_raw` and validity labels are excluded from `inference_only()`; they enter *only* after prediction for retrospective scoring.
- Only historical Train memory episodes whose outcomes were completed before the current prediction issue time are eligible. Same standard Region coarse recall and legality checks as v2.1.2.
- Other Regions keep their Real Top-8 prediction path; a target Region NULL control is recomputed for each Query–Region. A target Region intervention may affect information propagated beyond the Region, but **only the target Region's future MAE** is scored.
- Neither Test MAE, best scaling factor, retrospective highest-gain candidate, nor Oracle is used to train, choose the checkpoint or tune a deployed policy.

## Gadi deployment

From `/scratch/mh20/zw1165/Speed_LLM_RAG`:

```bash
cp -a FARE "FARE.backup.pre_v213.$(date +%Y%m%d_%H%M%S)"
unzip -o FARE_v2.1.3_Weight_Preserving_Counterfactual_Audit_Full.zip -d .
conda activate TFP
python -m compileall -q FARE/fare FARE/scripts
python -m pytest -q FARE/tests/test_weight_preserving_audit_v213.py FARE/tests/test_candidate_oracle_v212.py
bash -n FARE/pbs/oracle_audit.pbs
python -u FARE/scripts/check_metric_protocol.py
```

Your Gadi Base_SupportBench, Backbone, STLAformer checkpoints, Train Memory and the exact deployed MAPE metric extension must remain in place. The ZIP does not include these shared projects.

Define the two **already trained v2.1.1** checkpoint paths:

```bash
export FARE211_SPEED="$(pwd)/FARE_exps/speed/TraffiDent_Net_06/pilot/20261012_212838_seed10/candidate_best.pt"
export FARE211_FLOW="$(pwd)/FARE_exps/flow/TraffiDent_Net_06/pilot/20261012_212838_seed10/candidate_best.pt"
ls -lh "$FARE211_SPEED" "$FARE211_FLOW"
```

First Speed smoke (2 Queries per split × 1 Region × Top4, Node audit 1):

```bash
qsub -N FA213_smoke \
  -v "TARGET=speed,SEED=10,FARE_CHECKPOINT=$FARE211_SPEED,FARE_ORACLE_SMOKE=1" \
  FARE/pbs/oracle_audit.pbs
```

On successful `FARE_ORACLE_COMPLETE` and `FARE213_AUDIT_PBS_PASS`, run both pilots:

```bash
qsub -N FA213_speed \
  -v "TARGET=speed,SEED=10,FARE_CHECKPOINT=$FARE211_SPEED,MAX_QUERIES=12,MAX_REGIONS=2,MAX_CANDIDATES=64,MAX_NODES=4" \
  FARE/pbs/oracle_audit.pbs

qsub -N FA213_flow \
  -v "TARGET=flow,SEED=10,FARE_CHECKPOINT=$FARE211_FLOW,MAX_QUERIES=12,MAX_REGIONS=2,MAX_CANDIDATES=64,MAX_NODES=4" \
  FARE/pbs/oracle_audit.pbs
```

Default: `val` and `test` s20/s40/s60/s90, fixed 12 Queries and 2 Regions per evaluation group, up to 64 causal candidates. 1 V100, 12 CPU, 32GB RAM, 24:00 walltime. Audit uses *one Query* per forward, independent of training BS32. This release repeats the candidate interventions and will take longer than v2.1.2; measure the actual job rather than assuming a fixed duration.

A quick LOO + scaling-only audit without the heavier singleton/replacement loop:

```bash
qsub -N FA213_quick \
  -v "TARGET=speed,SEED=10,FARE_CHECKPOINT=$FARE211_SPEED,AUDIT_MODES=loo:scaling:node_loo,MAX_NODES=2" \
  FARE/pbs/oracle_audit.pbs
```

To resume an interrupted run, set `FARE_ORACLE_OUTPUT=<absolute-existing-output-path>` and `FARE_ORACLE_RESUME=1`, preserving **all** configuration parameters. The audit manifest verifies checkpoint/memory SHA and parameters. Do not resume into a previous v2.1.2 audit folder.

## Files and outputs

```
FARE/fare/weight_preserving_audit.py   # new faithful interventions
FARE/fare/candidate_oracle.py         # existing singleton + new audit integration
FARE/fare/retrieval_diagnostics.py    # both reports
FARE/scripts/run_oracle_audit.py      # mode, alpha, slot and nodes CLI
FARE/pbs/oracle_audit.pbs             # same Gadi entry, v213 options
FARE/configs/oracle_audit.yaml        # documented defaults
FARE/tests/test_weight_preserving_audit_v213.py
```

Output path: `FARE_exps/<target>/<dataset>/oracle_audit/<run>/`:

- `audit_manifest.json` — checkpoint SHA, Memory SHA, modes and exact protocol;
- `oracle_records.jsonl` — original 2.1.2 per-region singleton fields **plus** all v2.1.3 per-slot and per-node counterfactuals;
- `oracle_summary.{csv,json,md}` — backward-compatible singleton summary;
- `counterfactual_summary.{csv,json,md}` — new, separate severity-level LOO, replacement, node and correction-strength summaries.

Key primary fields:

| Field | Scientific meaning |
|---|---|
| `loo_mean_keep_gain` | Average benefit of keeping a learned selected slot; positive is good |
| `loo_harmful_slot_fraction` | Fraction of per-Region slots whose removal is beneficial |
| `loo_best_removal_improvement` | **Hindsight** best improvement from removing ONE rank slot |
| `replacement_best_improvement_over_real` | **Hindsight** best one-slot identity replacement within legal coarse Top64 |
| `replacement_random_mean_gain_vs_real` | Mean same-weight replacement improvement across legal candidates |
| `replacement_ranker_spearman` | Correlation of original candidate rank score with same-weight replacement gain |
| `best_scaling_alpha` | **Hindsight** best Region Real/NULL prediction-space interpolation strength |
| `best_scaling_improvement_over_real` | Diagnostic potential of strength calibration |
| `node_loo_mean_keep_gain` | Node-level benefit of retaining a particular selected evidence slot |
| `*_horizon_group_*` | Horizon thirds (1–4, 5–8, 9–12 when P=12) |
| `*_direct_mean_*` | Directly masked node gains when the protocol provides masks |

## Important interpretation

- **Many beneficial LOOs / best alpha <1** → evidence over-use and risk calibration. Do not conclude that the retrieved candidate is inherently useless.
- **Same-weight replacement much better than Real with weak rank-score correlation** → target candidate identity and query-conditioned utility ranking.
- **Node LOO disagrees with Region LOO** → node-specific assignment / transport rather than a single Region score.
- **Same-weight replacement little benefit, scaling large** → correction/control more important than candidate ranking.
- A single high-gain Oracle on 12 selected Queries is *not* a validated full Test advantage. Enlarge the audit and do query/time-block paired uncertainty analysis before design selection.

## Local validation and limits

The new v2.1.3 plus old v2.1.2 integration tests run against the real native small PyTorch FARE graph fixture. The two original end-to-end tests requiring `Base_SupportBench.experiment_common.metrics` cannot run outside the user's shared repository. Do not substitute an ad hoc MAPE implementation into the delivered project. A real Gadi Checkpoint smoke remains necessary.
