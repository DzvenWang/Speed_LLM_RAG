# TCRR v5.3.0 CPU / DataLoader throughput tuning (no algorithm changes)

- `TCRR_PBS_MEM` accepts a Gadi-style memory reservation (default `32GB`, e.g. `96GB`). The launcher passes `qsub -l mem=...` to TRAIN/DIAGNOSTICS/TEST and Smoke. PBS directives inside the job scripts remain fallback defaults.
- `TCRR_PBS_NCPUS` (default `12`) permits 12, 24, 36, or 48 cores. **One V100 on gpuvolta is normally paired with 12 cores**; requests above 12 may not be schedulable or cost-effective with `ngpus=1` and should not be assumed supported. Start with 12 cores.
- `TCRR_TRAIN_WORKERS=4` enables four concurrent DataLoader workers for S0 / C12 / C3; `TCRR_VAL_WORKERS=0` (default) avoids keeping three additional persistent validation worker pools alive. `TCRR_PERSISTENT_WORKERS=1` avoids restarting workers each epoch. `TCRR_PREFETCH_FACTOR=2` retains two prefetched batches per worker.
- PBS runtime defaults OMP/MKL/OPENBLAS threads to 1 to avoid multiplying CPU threads in worker processes. Explicit environment overrides remain supported.
- The model, optimizer, LR, dataset, Top-K, replay cache and training budgets are unchanged.
- Increased memory reservation does **not** itself accelerate training; it enables more workers without memory kills.

## Recommended benchmark

With the PBS Batch Hotfix installed, try `BATCH_SIZE=64 TCRR_TRAIN_WORKERS=4 TCRR_PERSISTENT_WORKERS=1 TCRR_PREFETCH_FACTOR=2 TCRR_PBS_MEM=96GB TCRR_PBS_NCPUS=12`. Compare `train_queries_per_second` and CPU/PBS memory use to the `TCRR_TRAIN_WORKERS=0` baseline. Smokes isolate peak memory but use duplicated samples, so do not infer full-dataset throughput from smoke.
