# FARE v2.0.1 — Full-Training Performance Optimization

This is an **implementation optimization** of FARE v2.0.0, not a new forecasting method or a completed accuracy experiment. The STLAformer family, all five training phases, Forecast Utility ranking, state/latent retrieval, control, historical-future memory and evaluation protocols are retained.

## Verified implementation changes

1. **Retrieval coarse Top-64:** memory lookups gather only the region's anchor nodes, not every node of every 64-candidate full-network historical window (`fare/memory.py`, `fare/retrieval.py`). The retrieved candidate IDs, scores and weights were compared numerically with the v2.0.0 code.
2. **Historical-future evidence:** valid-only per-region future means are cached when the retrieval index is refreshed. Only a small `[B,G,K,P]` view is fetched for ranking. The cache is rebuilt on a new Memory object, with a different source pointer, or a different region index. Memory values are assumed immutable during training.
3. **Recovery:** gather each node's evidence from its *own* graph region, rather than building full `[B,G,K,H,N]` candidate states and selecting regions afterwards. Float32 numerical agreement with v2.0.0 was within ~1e-7 on a synthetic comparison.
4. **Ranking:** vectorize target-state and forecast-utility candidate teachers across padded, disjoint region node sets. Padded nodes are explicitly excluded. Teachers still use labels only as offline training supervision, never as inference features.
5. **Phase-aware work:** in the retrieval phase, stop before the second STLAformer, Innovation and Controller passes, and do not compute the two Teacher forwards that never contributed to that phase's optimized loss. For later-stage checkpoint validation, compute only forecast MAE, since that is the checkpoint selection criterion. The final phase metrics still use the unchanged full inference/evaluation code.
6. **Training overhead:** reuse train/validation DataLoaders with persistent workers; remove per-parameter finite-gradient Python polling in the normal path while retaining `error_if_nonfinite=True` global clipping and an offending-parameter failure investigation; aggregate detached statistics on device and copy them after each epoch.
7. **Key index refresh:** remove the extra refresh at the beginning of each epoch when an immediately preceding post-training validation refresh already contains the same key-encoder weights. On phase-best reload, rebuild the index. Stage-level evaluator can reuse the fresh index. One post-update refresh per epoch remains.
8. **P0.1 / fixed-pool diagnostic parity:** v2.0.0 stand-alone diagnostic retrieval omitted Support Demand and Backbone Forecast conditioning. The two scripts now call the same retrieval helper as the full model, so selected candidate pools correspond to the actual model.
9. **Diagnostics:** `FARE_EPOCH` adds `train_seconds`, `validation_seconds`, `memory_refresh_seconds` to support measurement on Gadi.

## Defaults and controls

| Setting | v2.0.0 | v2.0.1 |
|---|---:|---:|
| `training.batch_size` | 8 | **16** |
| `training.num_workers` | 0 | **2** |
| `OMP_NUM_THREADS` (PBS) | 12 | **3** |
| Max formal epochs | 115 | 115 |
| Retrieval Top-K | 64 → 8 | 64 → 8 |
| Train-memory capacity | 50,000 | 50,000 |
| Speed / Flow | Supported | Supported |
| Severity evaluation | s20/s40/s60/s90 | Unchanged |
| 1 V100 / 12 CPU / 32GB / 48h | Unchanged | Unchanged |

To revert throughput-sensitive hyperparameters without editing YAML:

```bash
qsub -N FARE_speed -v TARGET=speed,SEED=10,BATCH_SIZE=8,NUM_WORKERS=0 FARE/pbs/train_formal.pbs
```

Default full run:

```bash
qsub -N FARE_v201_speed -v TARGET=speed,SEED=10 FARE/pbs/train_formal.pbs
qsub -N FARE_v201_flow  -v TARGET=flow,SEED=10 FARE/pbs/train_formal.pbs
```

Run the existing `scripts/preflight.py` and `scripts/train.py --smoke` on Gadi before formal submission. The PBS script still runs automatic evaluation in the same allocation, provided training completes within the PBS walltime.

## Validation performed

- Python compile and standalone CPU smoke/model-gradient tests.
- Six new optimized-index/region-cache/ranking/fast-path/trainer tests.
- Numerical comparison to the uploaded v2.0.0 implementation: identical retrieval candidate IDs, scores and weights; negligible floating-point differences in analog means, Recovery output and ranking targets/losses.
- Entire repository test suite: 39 passed using a **local, test-only stub for** `experiment_common.metrics` because the external Base_SupportBench project is not included in the uploaded zip. **No metric stub is included with this deliverable.**

**Not yet validated:** genuine Net-06 STLAformer end-to-end smoke, V100 GPU peak memory, Gadi throughput and full MAE/RMSE. No claim of a specific speed-up is warranted until a before/after Gadi benchmark.

## Compatibility

The new region-node tensors are **non-persistent** module buffers. The model's learned parameter names and sizes have not changed; existing v2.0.0 model state dictionaries can be loaded strictly. The optimizer and training trajectory may nevertheless change when using batch size 16 or a different number of DataLoader workers. For exact hyperparameter parity, run batch 8 and workers 0; environment/software reproducibility still matters.

**Important:** Backup running experiments and preserve their logs/checkpoints. Do not hot-swap a `FARE/` installation while a PBS job is running; use the optimized package for newly submitted jobs. The `--resume` CLI path enforces checkpoint configuration, so a v2.0.0 resume normally continues with its saved batch-size and workers, not the new YAML defaults.

To enable detailed tensor checks during troubleshooting:

```bash
export FARE_DEBUG_NONFINITE=1
```

All masking, Retrieval causality constraints, Auto Evaluate routing and official Base_SupportBench metrics remain active.
