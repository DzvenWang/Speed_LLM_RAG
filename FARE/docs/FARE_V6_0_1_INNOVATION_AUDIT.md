# FARE v6.0.1 — Read-only Innovation Audit Hotfix

## Goal and limits

Evaluate the **existing** FARE v6.0.0 `best.pt` models on an untouched Validation
split without changing any model parameter, optimizer state, OOF source or
Test metrics. No fine-tuning, no OOF recomputation, no old FARE initialization.

The package retains v6.0.0 **checkpoint schema**, uses strict state-dict
loading and checks dataset, target, seed, original frozen STLAformer SHA256,
and the OOF memory manifest SHA256. `FARE/VERSION` is intentionally not bumped
because model/checkpoint ABI is unchanged; this is audit revision 6.0.1.

## P0 audit corrections

* Final fixed-weight Validation mode study: **512 queries**, sampled over
  failure-family, available severity and time blocks, versus 32 previously.
  Stage studies sample 32 queries rather than six.
* Candidate probe: **128 queries**, 32 nodes sampled per batch, up to 8
  candidates per node; stage probes use 16 queries, 16 nodes and 4 candidates
  to limit training overhead.
* **Same shortlist** learned, similarity, deterministic random and historical
  candidate-prior selection. Historical prior uses only stored OOF episode
  residual coverage/error summaries, not the query-conditioned proposal.
* Per-candidate utility with fixed strength **alpha = 0, 0.1, 0.25, 0.5, 1**,
  plus the actual learned risk gate. Oracle is `max(0, best legal examined
  candidate gain)` so refusing all evidence is a valid option.
* Positive-evidence Recall@1/2/4/8 within **examined** candidates, harmonic
  sign, learned-vs-random gain and time-block confidence intervals.
* Strict cross-horizon-shared **candidate, branch and NULL mass** ablation.
  Only source branches valid over every horizon can participate. The learned
  risk gate remains horizon-specific. As availability changes, report results
  as a strict intervention, not a perfect architecture ablation.
* Per-horizon and per-available-severity/failure-family MAE/RMSE outputs.
* Explicit refusal to audit `test` or `train` datasets. Test labels are not
  used to decide which architecture change to make.

Important: Same-pool **full-policy** interventions retain the trained NULL/Gate
and may induce different policy scores/weights, not an isolated rank-only
counterfactual. The **fixed-alpha top-1** curves hold injection strength and
shortlist constant for the clearest evidence-identity test. Forced evidence
is applied simultaneously to nodes; network interactions can contribute.

## Deploy on Gadi (from Speed_LLM_RAG root)

1. Back up the source folder, unzip `FARE_v6.0.1_InnovationAudit_Full.zip`
   into the root (creates `FARE/`). Existing `FARE_exps`, Backbone and OOF
   folders are **never** overwritten by this ZIP.
2. `conda activate TFP`
3. `export SPEED_LLM_RAG_ROOT="$PWD"`
4. `export PYTHONPATH="$PWD/FARE:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"`
5. `python -m pytest -q FARE/tests`

### Run a low-cost read-only GPU smoke first

```bash
python -u FARE/scripts/submit_audit.py --target speed \
  --run-dir FARE_exps/speed/TraffiDent_Net_06/formal/20261020_233722_seed10 \
  --smoke
python -u FARE/scripts/submit_audit.py --target flow \
  --run-dir FARE_exps/flow/TraffiDent_Net_06/formal/20261020_233756_seed10 \
  --smoke
```

These jobs create `innovation_audit_v601_smoke/` under their original run
directories. They do not update `best.pt`.

### Submit full audits after smoke passes

```bash
python -u FARE/scripts/submit_audit.py --target speed \
  --run-dir FARE_exps/speed/TraffiDent_Net_06/formal/20261020_233722_seed10
python -u FARE/scripts/submit_audit.py --target flow \
  --run-dir FARE_exps/flow/TraffiDent_Net_06/formal/20261020_233756_seed10
```

Each submits a 1-GPU V100 PBS job. A completed report is under
`innovation_audit_v601/`. Do not re-submit a job into an already populated
output directory. A standalone direct invocation is also supported:

```bash
python -u FARE/scripts/innovation_audit.py \
  --target speed \
  --checkpoint FARE_exps/speed/TraffiDent_Net_06/formal/20261020_233722_seed10/best.pt \
  --validation-queries 512 --candidate-queries 128 --candidate-nodes 32 \
  --candidate-slots 8
```

The standalone action automatically obtains the STLAformer and OOF Memory
paths from the original checkpoint provenance; it never imports v5 FARE
parameters. For a new output location pass `--output` explicitly.

## Result files

* `mode_comparison.csv` — Full, support-only, backbone, same-pool controls,
  strict shared horizon, old legacy controls and C3 interventions.
* `paired_statistics.json` — block-bootstrap MAE gains (positive means the
  controlled model was **worse**), without pretending overlapping windows are
  independent.
* `horizon_metrics.csv` — 12 horizons, MAE/RMSE for selected modes.
* `severity_family_metrics.csv` — only groups **actually observed** in the
  formal Validation set; no fake s20/s60/s90 Validation values.
* `candidate_utility.json` — fixed-alpha and learned-gate response curves,
  NULL oracle, positive evidence Recall@K in the *examined* shortlist,
  paired candidate gains.
* `candidate_node_details.csv` — each audited query/node, candidates selected
  by each policy, alpha-specific gains and original historical episode IDs.
* `innovation_verdict.json` — conservative `INCONCLUSIVE` until truly
  capacity-matched independently retrained ablations and multi-seed networks.
* `evaluation_manifest.json` — complete definitions, sample sizes, split
  coverage, limitations.
* `audit_source.json` — exact checkpoint hash, provenance and no-training
  declaration.

## Decision matrix after new audits

* **C1:** support-only vs backbone remains useful, but whether Failure-aware
  Modeling beats an independent capacity-matched Mask-only adapter **cannot**
  be established using fixed-checkpoint interventions alone.
* **C2:** examine alpha=0.1, .25, .5 and learned-gate candidate gains, Top-1
  learned vs random/similarity/static prior, within-pool Recall@K, and
  full-policy same-pool MAE. If improvement appears only at low alpha, focus
  on transfer scaling and risk calibration, not just retrieval ranking.
* **C3:** compare `audit_shared_horizon`, no hidden injection, no risk,
  no forecast correction and mean pooling by horizon and failure family.
  Disagreement between fixed-alpha and learned-gate effects points to Gate.
* Do NOT rerun long Joint Training until these probes identify an actual
  mechanism and the ranking/transfer bottleneck.

## Validation boundaries

Run the full 55-test suite and GPU smoke before declaring the package
validated on Gadi. Local CPU tests are not a substitute for real Net-06
STLAformer, complete OOF tensor memory and GPU-evaluation results. Audit
reports must not be represented as independent-training evidence.
