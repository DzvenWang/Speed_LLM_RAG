# v5.3.0 Default Throughput Profile (overridable)

Applies to the v5.3.0 AMP + PBS Batch Hotfix + CPU/DataLoader Speed release.
This is a configuration-only change. **Model version remains 5.3.0**.

## Defaults

| Setting | Default | Override example |
|---|---|---|
| `BATCH_SIZE` | `64` | `BATCH_SIZE=32` |
| `TCRR_PBS_NCPUS` | `12` | `TCRR_PBS_NCPUS=24` (if scheduler permits) |
| `TCRR_PBS_MEM` | `96GB` | `TCRR_PBS_MEM=32GB` |
| `TCRR_TRAIN_WORKERS` | `4` | `TCRR_TRAIN_WORKERS=0` |
| `TCRR_VAL_WORKERS` | `0` | `TCRR_VAL_WORKERS=2` |
| `TCRR_PERSISTENT_WORKERS` | `1` | `TCRR_PERSISTENT_WORKERS=0` |
| `TCRR_PREFETCH_FACTOR` | `2` | `TCRR_PREFETCH_FACTOR=1` |

All defaults are set in `pbs/submit_pipeline_v5.sh`, forwarded via `qsub -v`, and validated before submission. The requested CPU/memory allocation uses `qsub -l`; the job header has matching 96GB fallback. `configs/tcrr_stla_v5.yaml` also matches these Python training defaults for direct invocations. Externally set values always take precedence. Sample overrides:

```bash
TARGET=speed BACKBONE_CHECKPOINT="$SPEED_CKPT" bash TCRR-STLA/pbs/submit_pipeline_v5.sh

BATCH_SIZE=32 TCRR_PBS_MEM=32GB TCRR_TRAIN_WORKERS=0 \
  TCRR_PERSISTENT_WORKERS=0 \
  TARGET=speed BACKBONE_CHECKPOINT="$SPEED_CKPT" \
  bash TCRR-STLA/pbs/submit_pipeline_v5.sh
```

`SMOKE=1` without `SMOKE_PROFILE_FULL_BATCH=1` still uses three actual training samples for the ordinary smoke even when the chosen configured batch size is 64. With `SMOKE_PROFILE_FULL_BATCH=1` it uses the configured full batch size and submits a **single** GPU Smoke job. New default batch size changes the S0 optimizer-step schedule and can invalidate a 32-batch S0 adapter; this is expected and does not mean old retrieval/mask caches need deletion. Compare experiments with batch size recorded and keep existing running jobs unchanged.
