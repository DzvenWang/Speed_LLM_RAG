# Baseline v1.1.0 — Core-5 Primary-7 Experiment Project

This is the clean-slate Baseline project for Task 1. Previous Baseline logs and experiment artifacts are intentionally not reused.


## Core-5 benchmark contract

All Primary baselines are retrained on the same frozen Core-5 s40 Formal cache used by the current Backbone. Train/Val use `entry_random`, `sensor_outage`, `temporal_outage`, `regional_outage`, and `directional_outage` at s40. Final evaluation uses the same five families on paired s20/s40/s60/s90 caches. Predictive outage families are not part of the main Baseline benchmark.

## Primary Baselines

The main experiment contains exactly seven external baselines:

1. **ImputeFormer** — neural reconstruction-first forecasting
2. **LRTC-TMCP** — TRC 2024 low-rank tensor completion + the same frozen PatchSTG
3. **GinAR** — direct variable-missing forecasting
4. **RAFT** — generic retrieval-augmented forecasting
5. **RAST** — traffic-specific retrieval-augmented forecasting
6. **SARAF** — reliability/stationarity-aware retrieval
7. **ReMFT** — TRC 2026 LLM-enhanced traffic forecasting

`PatchSTG` is the frozen numerical reference, not an external Baseline entry.

## Secondary / controlled experiments

Existing implementations are retained but are not submitted by the primary workflow:

- SPRINT
- SLPF
- MoGERNN
- LEAF
- LEAF-Qwen1.5B
- RAST-PatchSTG

## Clean-slate policy

- Only results produced by **Baseline v1.1.0** from the current Core-5 Formal cache are eligible for evaluation.
- There is no result-version compatibility layer and no old-result migration.
- Every Primary method is rerun from the current Development/Formal caches.
- Baseline does **not** require `FORMAL_SELECTION.json` or `FROZEN_BASELINE_PROTOCOL.json`: after a successful Formal run, that exact run's `best.pt` is evaluated directly on paired s20/s40/s60/s90 Core-5 caches.

## Primary workflow

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
conda activate TFP

# 0. Install/freeze official source trees and local ReMFT BERT
bash scripts/setup_official_models.sh
bash scripts/setup_remft_bert.sh   # REMFT_BERT_PATH must point to the local BERT checkpoint if not already installed

# 1. Prepare/validate shared caches and frozen PatchSTG reference
qsub pbs/prepare_data.pbs
qsub pbs/prepare_formal.pbs

# 2. Cheap contract smoke for all seven Primary baselines
bash scripts/submit_primary_smoke.sh

# 3. Development experiments
bash scripts/submit_primary_development.sh

# 4. GinAR only: full-Formal-population stability probe before any Formal GinAR job
bash scripts/submit_ginar_stability_probe.sh 10

# 5a. First Formal pass
bash scripts/submit_primary_formal_seed10.sh

# 5b. Final multi-seed Formal pass (use after seed10 is confirmed healthy)
bash scripts/submit_primary_formal_remaining.sh  # seeds 20/30 after seed10 sanity check

# 6. Formal jobs automatically evaluate their own completed best.pt
#    AUTO_EVAL=1 is the default in pbs/methods/formal_primary.pbs.
#    Each job runs s20 -> s40 -> s60 -> s90 after training.
python scripts/primary_run_status.py

# Optional: validate all evaluation caches manually
qsub pbs/prepare_evaluation.pbs

# Optional: standalone re-evaluation of an already completed Formal run
# qsub -v METHOD=ImputeFormer,SEED=10 pbs/evaluation_main.pbs
```

LRTC-TMCP is deterministic under the benchmark adapter and therefore uses seed 10 only.

## Validity rules that matter

### GinAR

A Formal run that triggers validation divergence is **invalid**. The runner exits non-zero before writing a selectable Formal artifact. The current clean configuration uses a conservative Formal LR and gradient clipping; the Formal-population stability probe must pass first.

### ReMFT

The local pretrained BERT checkpoint and runtime encoder architecture must match. The official notebook can instantiate more layers than a standard local BERT checkpoint; the adapter removes runtime-only random layers and then enforces an exact frozen checkpoint/runtime depth contract. A remaining mismatch is a hard failure.

### LRTC-TMCP

For the main table, LRTC-TMCP is restricted to the query's current incomplete H12. It cannot use the forecast horizon, other validation/test queries, or extra historical days. Observed entries are protected before the completed history enters the same frozen PatchSTG used by ImputeFormer.

## Secondary experiments

Secondary code is preserved and can be run after the Primary Baseline is complete:

```bash
bash scripts/submit_secondary_development.sh SPRINT 10
bash scripts/submit_secondary_development.sh MoGERNN 10
bash scripts/submit_secondary_development.sh LEAF 10
bash scripts/submit_secondary_development.sh LEAF-Qwen1.5B 10
```

They are not part of `formal_selection_methods` and cannot enter the main-table selection artifact.
