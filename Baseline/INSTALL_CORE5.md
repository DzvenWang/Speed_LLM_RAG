# Baseline v1.1.0 Core-5 — Gadi run order

## Contract

- Train/Val: s40 only
- Evaluate: s20 / s40 / s60 / s90
- Families: entry_random, sensor_outage, temporal_outage, regional_outage, directional_outage
- Protocol: v2.1.0
- ExperimentCommon: v1.2.0
- Frozen PatchSTG producer: Backbone v1.0.10

## 0. Freeze the new Backbone first

After the three Core-5 PatchSTG formal seeds have completed:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
/scratch/mh20/zw1165/envs/TFP/bin/python scripts/summarize_formal.py
```

This must create/update:

```text
/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_exps/FROZEN_BACKBONE.json
```

## 1. Validate Baseline Core-5 deployment

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
/scratch/mh20/zw1165/envs/TFP/bin/python tests/smoke_core5_contract.py
/scratch/mh20/zw1165/envs/TFP/bin/python scripts/baseline_status.py
```

## 2. Validate/prepare s40 Formal cache

```bash
qsub pbs/prepare_formal.pbs
```

The formal cache must resolve to `protocol_core5_s40`, with 50,000 train rows = 10,000 per family and 5,000 validation rows = 1,000 per family.

## 3. Rerun baselines on the new Core-5 data identity

Old Baseline v1.0.0 results are intentionally not selectable.

```bash
bash scripts/submit_primary_development.sh
bash scripts/submit_ginar_stability_probe.sh 10
bash scripts/submit_primary_formal_seed10.sh
```

After seed10 is healthy:

```bash
bash scripts/submit_primary_formal_remaining.sh
```

## 4. Direct post-Formal evaluation

Baseline-level freeze/selection is no longer required. `pbs/methods/formal_primary.pbs` defaults to `AUTO_EVAL=1`: after a successful Formal run, it resolves that method/seed run directory and evaluates the exact `best.pt` on s20/s40/s60/s90.

The same live log ends with:

```text
FINAL SEVERITY SUMMARY
FINAL SCENARIO SUMMARY
BASELINE MULTI-SEVERITY EVALUATION PASS
```

Optional evaluation-cache validation:

```bash
qsub pbs/prepare_evaluation.pbs
```

Standalone re-evaluation of a completed Formal run is also supported through `pbs/evaluation_main.pbs`; it resolves the latest valid Formal run for the requested method/seed.
