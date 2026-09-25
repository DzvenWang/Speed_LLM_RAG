# PatchSTG legacy spatial-layout recovery

Historical PatchSTG checkpoints created before Backbone v1.0.5 store only
`adapter_metadata.spatial_layout_fingerprint`. The official PatchSTG model keeps
`ori_parts_idx`, `reo_parts_idx`, and `reo_all_idx` outside `state_dict`, so an exact
historical evaluation requires those arrays to be reconstructed from the benchmark
metadata used at training time.

## Local recovery

From `Speed_LLM_RAG`:

```powershell
python Backbone\scripts\recover_patchstg_layout.py --checkpoint Backbone_exps\PatchSTG\TraffiDent_Net_06\formal\20260913_150736_seed10\best.pt
```

The recovery tool scans common `data/` and `artifacts/` locations. If an archived
SupportBench root lives elsewhere, add either:

```powershell
--benchmark-root C:\path\to\historical\supportbench
```

or:

```powershell
--search-root C:\path\to\archives
```

A successful exact match writes `spatial_layout.npz` beside `best.pt`. The normal
checkpoint evaluation command then discovers it automatically.

## Gadi recovery

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
qsub -v CHECKPOINT=Backbone_exps/PatchSTG/TraffiDent_Net_06/formal/20260913_150736_seed10/best.pt Backbone/pbs/recover_patchstg_layout.pbs
```

If the old benchmark copy is outside the default search roots:

```bash
qsub -v CHECKPOINT=Backbone_exps/PatchSTG/TraffiDent_Net_06/formal/20260913_150736_seed10/best.pt,SEARCH_ROOT=/scratch/mh20/zw1165/old_archives Backbone/pbs/recover_patchstg_layout.pbs
```

The PBS writes a timestamped live log in `Backbone_logs/`.

## Diagnostic-only current-layout mode

If the original benchmark metadata no longer exists, evaluation can be forced with the
current layout for debugging only:

```powershell
python Backbone\scripts\run_checkpoint_evaluations.py --checkpoint Backbone_exps\PatchSTG\TraffiDent_Net_06\formal\20260913_150736_seed10\best.pt --expected-seed 10 --allow-current-layout-mismatch
```

Those outputs contain `officially_comparable: false` and must not be used as the
historical model's official result.

## New checkpoints

Backbone v1.0.5 stores the full spatial layout inside `best.pt` and also writes
`spatial_layout.npz` beside the checkpoint. New-model evaluation therefore does not
need recovery even if SupportBench metadata later changes.
