# TraffiDent-SupportBench 1.2.3

Support-oriented benchmark for **historical RAG, LLM predictive-support reasoning and grounded explanation under incomplete traffic sensing**.

## Fixed paths

```text
PROJECT_ROOT   /scratch/mh20/zw1165/Speed_LLM_RAG
PROCESS_ROOT   /scratch/mh20/zw1165/Speed_LLM_RAG/TraffiDent_SupportBench
DATA_ROOT      /scratch/mh20/zw1165/Speed_LLM_RAG/data
BENCHMARK_ROOT /scratch/mh20/zw1165/Speed_LLM_RAG/data/processed/traffident/supportbench
DERIVED_ROOT   /scratch/mh20/zw1165/Speed_LLM_RAG/artifacts/supportbench
```

## Design principle

The canonical benchmark is not tuned with Test labels. It deliberately emphasises the scientific setting in which the proposed method should matter:

```text
structured incomplete sensing
  -> missing predictive context
  -> accessible Train historical support
  -> ambiguous multi-view candidates
  -> support-value reasoning
  -> grounded explanation
```

Test challenge membership never uses Test future or Test forecast gain.

## Three benchmark tracks

### Track A — General Incomplete Sensing

Broad baseline comparison over:

```text
entry_random / sensor_outage / temporal_outage / regional_outage / directional_outage
20% / 40% / 60%
```

Exact files:

```text
protocols/resolved/track_general_{development,full}.csv
```

### Track B — Main Predictive-Support Setting

Primary paper setting:

```text
40% temporal_outage
40% regional_outage
40% directional_outage
```

Exact files:

```text
protocols/resolved/track_main_support_{development,full}.csv
```

### Track C — Support Challenge

Source views combine main structured masks with support-oriented predictive regional/directional masks. Support Opportunity v2 is two-stage:

```text
Stage A (future-free):
network-mean predictive-context loss
+ support-dependent scope
+ Train-memory coverage

Stage B (Validation outcome calibration):
future-free Stage-A features -> realised safe support headroom
```

Retrieval ambiguity is retained as a difficulty diagnostic but is excluded from the default SO-v2 calibration. Test applies the frozen Validation calibrator without Test refitting or re-quantiling.

## Forecast task

```text
history  = 12 × 5 min = 60 min
horizon  = 12 × 5 min = 60 min
split    = 70% / 10% / 20% chronological
full     = stride 1 for Train/Val/Test
develop  = 12k / 5k / 5k deterministic temporal subsets
```

## Predictive dependency

`graph/predictive_dependency.npz` is fitted **only on Train** using physical directed neighbours and raw-resolution source-leading-target lagged speed correlation. Correlation anchors may be sparsely sampled for efficiency, but lag duration remains at the original 5-min resolution. It is used for:

- predictive regional/directional sensing layouts;
- predictive-context loss;
- support-dependent target masks;
- propagation-aware retrieval features;
- LLM reasoning/explanation evidence.

Validation/Test future is never used.



### v1.2.3 predictive-context metric semantics

The formal Chapter-3 query-level predictive-context loss is

```text
network_mean_context_loss = mean(per_target_context_loss)
```

and is also exposed as `predictive_context_loss`.  The historical v1.2.2 `aggregate_context_loss` was actually the configured target quantile (default q=0.75); it is retained only as a backward-compatible diagnostic alias and must not be used as the formal C_q^loss.

### v1.2.2 lag semantics (retained from v1.2.1)

`dependency.sample_stride=12` means correlation anchors are sampled every 60 minutes for efficiency. It does **not** mean lag 1 is 60 minutes. With 5-min TraffiDent data and `max_lag_steps=3`, the dependency builder tests exactly:

```text
0 / 5 / 10 / 15 min
```

The build audit verifies this explicitly. It also reports a future-free predictive-context diagnostic comparing general versus predictive regional/directional masks on matched Validation windows.

## Support-oriented sensing

In addition to the five general sensing families, v1.2.2 retains:

```text
predictive_regional_outage
predictive_directional_outage
```

These are generated without Test outcomes. In v1.2.2 the layout constructor no longer uses a high-centrality centre followed by ordinary BFS. It directly optimises the **Train-only predictive-context loss** objective under compact-regional or directional constraints. ID and OOD remain fixed matched-severity layout catalogues.

The Validation audit compares each predictive family against its general counterpart. The default diagnostic target is a predictive/general context-loss ratio of at least `1.05`; this is a scientific diagnostic, not a Test-label selection rule.

## Historical support memory

Main archive:

```text
episode_stride = 6 raw steps = 30 min
```

Formal experiments use the full Train memory archive. Development memory is a deterministic subset of the same full episode IDs. Historical future values are legal because they already happened in Train.

## Multi-view control retrieval

`derived/build_candidate_pool.py` now uses a future-free union of:

```text
temporal retrieval
spatial retrieval
propagation retrieval
```

Default final score:

```text
0.30 temporal + 0.30 spatial + 0.40 propagation
```

Memory features are automatically cached under:

```text
artifacts/supportbench/<NET>/retrieval_cache/
```

Example:

```bash
python derived/build_candidate_pool.py \
  --subgraph TraffiDent_Net_06 \
  --views-file \
  /scratch/mh20/zw1165/Speed_LLM_RAG/data/processed/traffident/supportbench/TraffiDent_Net_06/protocols/resolved/challenge_val_development.csv \
  --budget development \
  --memory-budget development \
  --k 64 \
  --output /scratch/mh20/zw1165/Speed_LLM_RAG/artifacts/supportbench/TraffiDent_Net_06/val_challenge_pool.npz
```


## I1 training mining pool

Formal retrieval evaluation remains full-memory retrieval with Top-K reporting. Training-label mining is deliberately larger so Innovation 1 is not capped by the control Top-64 pool:

```bash
python derived/build_training_mining_pool.py \
  --subgraph TraffiDent_Net_06 \
  --views-file <TRAIN_PROTOCOL.csv> \
  --memory-budget full \
  --k 256 \
  --output <TRAIN_MINING_POOL.npz>
```

Default mining composition:

```text
256 total candidates
192 multi-view hard candidates
 64 temporally diverse legal Train-memory negatives
```

The exact quotas are configurable. The formal evaluator still retrieves from the full Train archive and reports Beneficial Recall@8/16/32/64.

## Build Support Opportunity challenge

### Validation calibration

```bash
python derived/build_support_opportunity.py \
  --subgraph TraffiDent_Net_06 \
  --candidate-pool <VAL_POOL.npz> \
  --fit-calibrator \
  --output <VAL_OPPORTUNITY.csv>
```

### Apply the same thresholds to Test

```bash
python derived/build_support_opportunity.py \
  --subgraph TraffiDent_Net_06 \
  --candidate-pool <TEST_POOL.npz> \
  --calibrator <VAL_OPPORTUNITY.calibrator.json> \
  --output <TEST_OPPORTUNITY.csv>
```

The script also writes `<...>__high.csv`, the exact High-Support-Opportunity challenge queries. No Test future is read.

## Forecast evaluation

Predictions:

```text
predictions.npz
  view_id [Q]
  pred    [Q,12,N]
```

Evaluate:

```bash
python derived/evaluate_forecast.py \
  --subgraph TraffiDent_Net_06 \
  --predictions predictions.npz \
  --protocol <exact_views.csv>
```

Reported metrics include:

- all-node MAE/RMSE;
- directly-masked target MAE;
- support-dependent target MAE;
- H3/H6/H12 MAE;
- H3/H6/H12 support-dependent MAE.

## Multi-granularity residual support utility

Use one alpha chosen on Validation only:

```bash
python derived/select_support_alpha.py ...
python derived/build_support_utility.py ...
```

Primary support intervention:

```text
historical residual = historical future - backbone prediction on historical episode
augmented forecast  = current backbone forecast + alpha × historical residual
```

v1.2.2 stores multiple realised utility views for every query/candidate pair:

```text
gain_all
gain_masked
gain_support_dependent   # primary reasoning target
gain_h3 / gain_h6 / gain_h12
safe_support             # support gain > 0 with no global degradation beyond tolerance
```

`forecast_gain` is retained as a backward-compatible alias of `gain_all`. Utility is **backbone-conditioned realised benefit**, not an intrinsic ground-truth property of a historical episode. `raw_future` remains an ablation only.

## LLM reasoning data

```bash
python derived/build_reasoning_pairs.py --utility <UTILITY.npz>
python derived/build_reasoning_records.py --subgraph TraffiDent_Net_06 --utility <UTILITY.npz>
```

Hard pairs prioritise:

- beneficial vs harmful support;
- similarity conflicts;
- temporal-match but propagation-mismatch cases.

MLP / small Transformer / Qwen receive the same structured evidence.

## Grounded + contrastive explanation

Single-candidate evidence:

```bash
python derived/build_explanation_evidence.py ...
```

Contrastive evidence:

```bash
python derived/build_contrastive_explanation.py \
  --subgraph TraffiDent_Net_06 \
  --reasoning-pairs <PAIRS.csv>
```

This supports the question: **why should candidate B be trusted over a numerically similar candidate A?** No reference prose is stored in the benchmark.

## Build on Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/TraffiDent_SupportBench
conda activate TFP
export PYTHONPATH=$PWD:$PYTHONPATH
python preflight.py
qsub pbs/build_supportbench.pbs
```

Live log:

```bash
tail -f logs/SupportBenchBuild.<JOBID>.gadi-pbs.live.log
```

Build stages:

```text
00 preflight
01 canonical core + Train predictive dependency + fixed sensing catalogues
02 dense windows + development budgets
03 general + support-oriented sensing views
04 Train-only full/development memory
05 three-track + GAP resolved protocols
06 stratified audit
07 API smoke
```

## Canonical vs derived boundary

Canonical benchmark contains only facts and reproducible future-free protocol information. The following remain outside the benchmark under `artifacts/supportbench/`:

```text
learned embeddings
FAISS / retrieval cache
candidate pools
backbone predictions/residuals
support utility labels
reasoning pairs
Qwen outputs
natural-language explanations
```

## Recommended next experiment

After a successful build, start on Net06 with:

1. `track_main_support_development.csv` and `challenge_{val,test}_development.csv`;
2. multi-view candidate pools;
3. Validation-calibrated Support Opportunity groups;
4. future compatibility + Similarity/Multi-view RAG vs Oracle headroom;
5. only then train MLP / Transformer / Qwen reasoners.

## Explanation leakage barrier

Generator-visible evidence and evaluator-only outcome labels are physically separated:

```bash
python derived/build_explanation_inputs.py  --candidate-pool <POOL.npz> ...
python derived/build_explanation_labels.py  --utility <UTILITY.npz> ...
```

Contrastive explanation data are likewise written as separate `*_inputs.jsonl` and `*_labels.jsonl`. No forecast gain or oracle preference is stored in the generator input file.

## RAG Go/No-Go diagnostics

Before training Qwen, run:

```bash
python derived/analyze_rag_headroom.py --utility <UTILITY.npz> --ks 8 16 32 64
```

It reports Beneficial Recall@K, safe-support Recall@K, retrieval Top-1 harmful rate, oracle gain within the candidate pool and selection headroom. Low candidate recall indicates an I1/acquisition problem; large selection headroom indicates a genuine I2 reasoning opportunity.
