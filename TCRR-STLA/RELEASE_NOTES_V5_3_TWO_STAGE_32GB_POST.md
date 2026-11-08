# v5.3.0 Two-Stage PBS Pipeline (TRAIN 96GB / POST 32GB)

Only the **submission and postprocessing orchestration** changes. Model code, C1/C2/C3 mathematics, `VERSION=5.3.0`, training defaults, checkpoint identity, replay protocol, data splits and automatic mechanism comparisons remain unchanged.

## Stages

1. `T53_<target>_tr`: TRAIN, GPU ×1, 12 CPUs, default 96GB CPU RAM, 48h.
2. `T53_<target>_pp`: POSTPROCESS, GPU ×1, 12 CPUs, default **32GB CPU RAM**, default 48h. This one job runs validation diagnostics **first**, then the locked test sequence using unchanged evaluate commands. A diagnostics failure prevents any test run. Logs remain separated in `RUN_DIR/postprocess_logs/`.

The two jobs use `depend=afterok:<TRAIN_ID>`. No DIAGNOSTICS → TEST scheduler dependency remains, because those two procedures execute sequentially in a single PBS job.

## Overridable parameters

- `TCRR_PBS_MEM=96GB` affects **only TRAIN** (and SMOKE).
- `TCRR_POST_PBS_MEM=32GB` affects **only POSTPROCESS**.
- `TCRR_POST_WALLTIME=48:00:00` affects **only POSTPROCESS**.
- `TCRR_PBS_NCPUS=12` applies to both. Defaults for `BATCH_SIZE=64`, `TCRR_TRAIN_WORKERS=4` are unchanged.

```bash
TARGET=speed BACKBONE_CHECKPOINT="$SPEED_CKPT" bash TCRR-STLA/pbs/submit_pipeline_v5.sh
# Example override, if the actual PBS postprocess memory requirement exceeds 32GB:
TCRR_POST_PBS_MEM=64GB TARGET=flow BACKBONE_CHECKPOINT="$FLOW_CKPT" bash TCRR-STLA/pbs/submit_pipeline_v5.sh
```

**IMPORTANT: existing queued PBS jobs are NOT modified retroactively.** For already submitted TRAIN jobs, cancel only their held DIAGNOSTICS and TEST jobs and submit `postprocess_v5.pbs` with `-W depend=afterok:<existing_train_id>` and the exact `RUN_DIR` inherited from that TRAIN submission.

**Caveat:** 32GB has not been verified on real V100 postprocessing. Earlier TRAIN main-process RSS was ~23.3GiB under Batch 32; multi-process and evaluation working-set may differ. Monitor job-wide memory usage and increase `TCRR_POST_PBS_MEM` if a 32GB request fails. The maximum walltime is a request, not a runtime estimate.

For currently queued/running jobs, preserve each TRAIN ID and submit a replacement combined postprocessing job:

```bash
bash TCRR-STLA/pbs/submit_post_for_existing_train_v5.sh 182459670.gadi-pbs
bash TCRR-STLA/pbs/submit_post_for_existing_train_v5.sh 182459674.gadi-pbs
# Only AFTER the two new POST jobs are confirmed:
qdel 182459671.gadi-pbs 182459672.gadi-pbs 182459676.gadi-pbs 182459677.gadi-pbs
```

This helper uses `qstat -f -F json` to read the **exact original** `RUN_DIR`, and will refuse to submit if it cannot verify that directory. If your PBS version lacks JSON output, obtain the `RUN_DIR` from `qstat -f <TRAIN_ID>` and explicitly prefix the helper invocation with `RUN_DIR=...`. It does not modify/cancel TRAIN.
