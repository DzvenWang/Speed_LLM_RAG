# Mask_RAST-STLA

Full incomplete-sensing-aware RAST + STLAformer. It is independent of `Baseline` and depends only on `Base_SupportBench` plus the official STLAformer source/checkpoint managed by `Backbone`.

## Architecture

- **Mask-aware STLAformer** with zero-initialized mask context.
- **Asymmetric incomplete→full dual encoder**: incomplete query states and complete Train states have learned temporal/spatial representations.
- **Full-Train latent store**: temporal and spatial keys are rebuilt periodically from complete Train histories using the current memory encoder.
- **Multi-positive alignment**: incomplete query keys align to the corresponding complete-state keys using `view_group_id`, avoiding false negatives between multiple sensing views of the same underlying window.
- **Temporal/spatial retrieval**: independent top-K searches retrieve complete historical states.
- **Failure-adaptive routing**: node coverage, missing age, temporal coverage morphology and retrieval confidence determine temporal-vs-spatial support per node.
- **Latent + forecast support**: retrieved support is injected into the STLAformer latent representation and also drives a node × horizon forecast correction gate.
- **End-to-end optimization**: pretrained STLAformer initialization, warm-up → partial joint → full joint fine-tuning.

## Required layout

Place this directory under `Speed_LLM_RAG/` beside `Base_SupportBench/` and `Backbone/`. `Backbone/vendor/STLAformer/model/STLformer.py` and a formal STLAformer checkpoint are required. Set `MASK_RAST_STLA_BACKBONE_CHECKPOINT` to override automatic checkpoint discovery.

## Commands (PowerShell)

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:TRAFFIDENT_TARGET = "speed"

python -u Mask_RAST-STLA\scripts\preflight.py --target speed
python -u Mask_RAST-STLA\scripts\train.py --target speed --seed 10
```

The training runner refreshes the full-state latent store automatically and writes `best_memory_index.npz` beside `best.pt`.

Evaluation:

```powershell
python -u Mask_RAST-STLA\scripts\evaluate.py --target speed --checkpoint <best.pt>
```

Optional store diagnostic/build:

```powershell
python -u Mask_RAST-STLA\scripts\build_store.py --target speed --checkpoint <best.pt>
```

Smoke:

```powershell
python -u Mask_RAST-STLA\scripts\train.py --target speed --seed 10 --smoke
```

For flow, replace `speed` with `flow`.


## v1.1 Gadi throughput profile
- Formal batch size: **32**.
- DataLoader: 4 workers, pinned memory, persistent workers, prefetch=2. Smoke keeps workers=0 for portability.
- Full training complete-history repository is preloaded to the V100 when possible; OOM automatically falls back to CPU mmap.
- Temporal and spatial retrieved histories are encoded in one memory-encoder call.
- Latent-store refresh batch size is **256** (was 64); refresh time is logged separately.
- PBS keeps **ncpus=12 per V100** as required on Gadi gpuvolta.
- Each formal job writes a sibling `*.gpu.csv` sampled every 30 seconds. `history.csv` and `EPOCH` logs include samples/s and peak CUDA memory.
- Optional runtime overrides: `--batch-size N --num-workers N`.
- `runtime.amp` remains false, so v1.1 changes throughput rather than numerical precision.
