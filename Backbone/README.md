# Backbone v0.6.0

Backbone v0.6.0 extends the TraffiDent numerical-backbone benchmark with **STLAformer (TRC 2026)** while preserving the shared ExperimentCommon v1.0.0 protocol and the existing PatchSTG downstream reference.

## Active models

- STAEformer
- HimNet
- BigST
- STDN
- PatchSTG
- STLAformer (TRC 2026)

Six-model screening is the comparison/verification experiment. `PatchSTG` is explicitly configured as the downstream numerical reference model and is trained on the formal stage with seeds 10/20/30. Seed 10 is the fixed downstream reference checkpoint after the three-seed formal run is summarized.

## Shared Experiment Contract

Backbone no longer owns a private materialized data cache. `scripts/prepare_data.py`, training, summaries, and evaluation all read **directly** from the canonical ExperimentCache:

```text
ExperimentCache/TraffiDent_Net_06/development  <- Backbone screening
ExperimentCache/TraffiDent_Net_06/formal       <- Backbone formal
ExperimentCache/TraffiDent_Net_06/evaluation   <- test after freeze
```

`Backbone_exps/` contains model/run artifacts only. No `Backbone_exps/_cache` directory or symlink is created or required.

The common contract fixes:

- sample membership from frozen Shared Protocol;
- one TOD/DOW resolver and one temporal fingerprint;
- one cache/data identity;
- one point-level micro MAE/RMSE implementation;
- one prediction NPZ contract `[M, P, N]`.

Per-query metrics are still written for diagnostics, but model selection and reported aggregate metrics use the common point-level evaluator.

## Recommended rerun

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
conda activate TFP

python scripts/preflight.py --stage screening
python scripts/prepare_data.py --stage screening --force
python scripts/validate_cache.py --stage screening
bash scripts/submit_screening.sh
```

After all six screening runs finish:

```bash
python scripts/summarize_screening.py
```

Prepare and run the configured reference model formally:

```bash
python scripts/preflight.py --model PatchSTG --stage formal
python scripts/prepare_data.py --stage formal
python scripts/validate_cache.py --stage formal
bash scripts/submit_reference_formal.sh
```

After PatchSTG seeds 10/20/30 finish:

```bash
python scripts/summarize_formal.py
```

This creates the Backbone-owned `Backbone_exps/FROZEN_BACKBONE.json`. Baseline and future OwnModel projects are read-only consumers of that artifact.

## Important

v0.6.0 adds STLAformer without changing the original five adapters, their recipes, or the Shared Protocol/cache identity. Screening summarization therefore explicitly permits current-contract v0.5.0 results for the original five models and requires v0.6.0 for STLAformer. PatchSTG remains the configured downstream reference; its compatible v0.5.0 formal/reference evidence remains reusable.
## Remove legacy `_cache` aliases

After deploying this direct-cache revision, old symlink trees can be removed safely with:

```bash
python scripts/cleanup_legacy_cache_aliases.py
```

The cleanup refuses to delete real files or non-empty real data directories.



## STLAformer integration

STLAformer is pinned from the official TRC 2026 repository and integrated without changing its architecture. The adapter maps canonical TraffiDent history into the model's official input semantics:

```text
channel 0 = normalized traffic state
channel 1 = (day_of_week * 288 + time_of_day_slot) / 2016
```

No future TOD/DOW or target value is used. The official class is `STLformer`; project-facing method name is `STLAformer`. PatchSTG remains `selection.reference_model`.

After deploying v0.6.0, install/pin the new source and refresh the source lock:

```bash
bash scripts/setup_official_models.sh
python scripts/verify_official_sources.py
python scripts/smoke_recent_model_contracts.py
```

Run only STLAformer screening if the previous five screening models already have current-compatible results:

```bash
qsub -v MODEL=STLAformer,STAGE=screening,SEED=10 pbs/backbone_select.pbs
```

The compatibility rule is explicit rather than silent: v0.5.0 screening evidence is reusable only for the unchanged original five models, while STLAformer must be produced by v0.6.0 under the same frozen Shared Protocol/cache identity.
